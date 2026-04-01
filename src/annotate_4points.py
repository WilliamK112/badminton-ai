"""
4-Point Court Annotation Tool
Click 4 corners of the court (not a rectangle)
"""
import cv2
import numpy as np
from pathlib import Path

FRAMES = sorted(Path("data/training_frames").glob("*.jpg"))
LABELS = Path("data/training_labels")
LABELS.mkdir(exist_ok=True)

points = []
current_frame = 0

def mouse_handler(event, x, y, flags, img):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
        
        # Draw
        display = img.copy()
        
        # Draw clicked points
        for i, (px, py) in enumerate(points):
            cv2.circle(display, (px, py), 8, (0, 255, 0), -1)
            cv2.putText(display, str(i+1), (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        # Draw lines between points
        if len(points) > 1:
            for i in range(len(points)-1):
                cv2.line(display, points[i], points[i+1], (0, 255, 0), 2)
        
        # Draw closed polygon if 4 points
        if len(points) == 4:
            cv2.line(display, points[3], points[0], (0, 255, 0), 2)
            # Fill
            overlay = display.copy()
            cv2.fillPoly(overlay, [np.array(points)], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        
        cv2.imshow("Annotate 4 Points", display)

print("="*50)
print("4-Point Court Annotation")
print("="*50)
print("Click 4 corners of the court:")
print("  1 = Top-Left")
print("  2 = Top-Right") 
print("  3 = Bottom-Right")
print("  4 = Bottom-Left")
print("")
print("Commands:")
print("  's' = save & next frame")
print("  'r' = reset/redo")
print("  'q' = quit")
print("="*50)

while current_frame < len(FRAMES):
    img = cv2.imread(str(FRAMES[current_frame]))
    h, w = img.shape[:2]
    
    points = []
    display = img.copy()
    
    print(f"\n[{current_frame+1}/{len(FRAMES)}] {FRAMES[current_frame].name}")
    
    cv2.imshow("Annotate 4 Points", display)
    cv2.setMouseCallback("Annotate 4 Points", lambda e,x,y,f,img=display: mouse_handler(e,x,y,f,img))
    
    while True:
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            print("\nQuit!")
            exit()
        
        elif key == ord('r'):
            points = []
            display = img.copy()
            cv2.imshow("Annotate 4 Points", display)
        
        elif key == ord('s') and len(points) == 4:
            # Save as YOLO format (use smallest bounding box)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            
            # Normalize to YOLO format
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            
            # Save
            label_file = LABELS / f"{FRAMES[current_frame].stem}.txt"
            with open(label_file, 'w') as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            
            print(f"  Saved: {label_file.name}")
            break
        
        elif len(points) == 4:
            # Show preview
            display = img.copy()
            overlay = display.copy()
            cv2.fillPoly(overlay, [np.array(points)], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            cv2.polylines(display, [np.array(points)], True, (0, 255, 0), 3)
            cv2.imshow("Annotate 4 Points", display)
    
    current_frame += 1

cv2.destroyAllWindows()
print(f"\n✅ Done! Labeled {current_frame} frames")
