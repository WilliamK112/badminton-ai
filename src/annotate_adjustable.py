"""
Court Annotation Tool - Draw rectangle then adjust corners
"""
import cv2
import numpy as np
from pathlib import Path

FRAMES = sorted(Path("data/training_frames").glob("*.jpg"))
LABELS = Path("data/training_labels")
LABELS.mkdir(exist_ok=True)

# State
corners = []  # 4 corner points
drawing = False
drag_idx = -1
start_x, start_y = -1, -1

def draw_court(img, corners):
    display = img.copy()
    
    if len(corners) > 0:
        # Draw points
        for i, (x, y) in enumerate(corners):
            color = (0, 255, 0) if i != drag_idx else (255, 0, 0)
            cv2.circle(display, (x, y), 10, color, -1)
            cv2.putText(display, str(i+1), (x+12, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw polygon
        if len(corners) >= 2:
            for i in range(len(corners)-1):
                cv2.line(display, corners[i], corners[i+1], (0, 255, 0), 2)
        
        if len(corners) == 4:
            cv2.line(display, corners[3], corners[0], (0, 255, 0), 2)
            # Fill
            overlay = display.copy()
            cv2.fillPoly(overlay, [np.array(corners)], (0, 255, 0))
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
    
    return display

def mouse_event(event, x, y, flags, param):
    img, h, w = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicking near existing corner
        drag_idx = -1
        for i, (cx, cy) in enumerate(corners):
            if abs(x - cx) < 20 and abs(y - cy) < 20:
                drag_idx = i
                break
        
        if drag_idx == -1 and len(corners) < 4:
            # Start new corner
            corners.append((x, y))
            drag_idx = len(corners) - 1
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_idx >= 0:
            # Drag corner
            corners[drag_idx] = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drag_idx = -1

print("="*50)
print("Court Annotation - Draw & Adjust")
print("="*50)
print("1. Click 4 corners to draw court")
print("2. Drag corners to adjust position")
print("Commands:")
print("  's' = save & next frame")
print("  'r' = reset/redo")
print("  'q' = quit")
print("="*50)

current = 0
while current < len(FRAMES):
    img = cv2.imread(str(FRAMES[current]))
    h, w = img.shape[:2]
    corners = []
    
    print(f"\n[{current+1}/{len(FRAMES)}] {FRAMES[current].name}")
    
    while True:
        display = draw_court(img, corners)
        cv2.putText(display, f"Points: {len(corners)}/4", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Court Annotation", display)
        cv2.setMouseCallback("Court Annotation", mouse_event, (img, h, w))
        
        key = cv2.waitKey(20) & 0xFF
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit()
        
        if key == ord('r'):
            corners = []
            continue
        
        if key == ord('s') and len(corners) == 4:
            # Save as YOLO bounding box
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
print(f"\n✅ Done!")
