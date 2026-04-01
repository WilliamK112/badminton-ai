"""
Court Annotation Tool - Draw rectangle then adjust corners (Fixed + Skip)
"""
import cv2
import numpy as np
from pathlib import Path

FRAMES = sorted(Path("data/training_frames").glob("*.jpg"))
LABELS = Path("data/training_labels")
LABELS.mkdir(exist_ok=True)

corners = []
dragging = False
drag_idx = -1

def draw_court(img, corners, drag_idx):
    display = img.copy()
    
    # Draw filled polygon first
    if len(corners) >= 3:
        overlay = display.copy()
        cv2.fillPoly(overlay, [np.array(corners)], (0, 255, 0))
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
    
    # Draw lines
    if len(corners) > 1:
        for i in range(len(corners)-1):
            cv2.line(display, corners[i], corners[i+1], (0, 255, 0), 3)
    
    if len(corners) == 4:
        cv2.line(display, corners[3], corners[0], (0, 255, 0), 3)
    
    # Draw corner points
    for i, (x, y) in enumerate(corners):
        color = (0, 0, 255) if i == drag_idx else (0, 255, 0)
        cv2.circle(display, (x, y), 12, color, -1)
        cv2.putText(display, str(i+1), (x+15, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return display

def mouse_event(event, x, y, flags, param):
    global dragging, drag_idx
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_idx = -1
        for i, (cx, cy) in enumerate(corners):
            if abs(x - cx) < 25 and abs(y - cy) < 25:
                drag_idx = i
                dragging = True
                break
        
        if drag_idx == -1 and len(corners) < 4:
            corners.append((x, y))
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and drag_idx >= 0:
            corners[drag_idx] = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        drag_idx = -1

print("="*50)
print("Court Annotation - Fixed + Skip")
print("="*50)
print("1. Click 4 corners")
print("2. Drag corners to adjust")
print("Commands:")
print("  's' = save & next")
print("  'r' = reset")
print("  'd' = skip (no court in this frame)")
print("  'q' = quit")
print("="*50)

current = 0
while current < len(FRAMES):
    img = cv2.imread(str(FRAMES[current]))
    h, w = img.shape[:2]
    corners = []
    
    print(f"\n[{current+1}/{len(FRAMES)}] {FRAMES[current].name}")
    
    while True:
        display = draw_court(img, corners, drag_idx)
        
        status = f"Points: {len(corners)}/4"
        if len(corners) == 4:
            status += " - Press 's' to save"
        else:
            status += " - Draw 4 points or 'd' to skip"
        
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.setMouseCallback("Court", mouse_event, None)
        cv2.imshow("Court", display)
        
        key = cv2.waitKey(20) & 0xFF
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit()
        
        if key == ord('r'):
            corners = []
        
        if key == ord('d'):
            # Skip - delete label file if exists
            label_file = LABELS / f"{FRAMES[current].stem}.txt"
            if label_file.exists():
                label_file.unlink()
            print("  Skipped!")
            break
        
        if key == ord('s') and len(corners) == 4:
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            
            label_file = LABELS / f"{FRAMES[current].stem}.txt"
            with open(label_file, 'w') as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            
            print(f"  Saved!")
            break
    
    current += 1

cv2.destroyAllWindows()
print("\n✅ Done!")
