"""
手动标注羽毛球场角点工具
运行后会打开一张图片，点击4个角点后保存
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys


# 全局变量
corners = []
window_name = "Click 4 court corners - TopLeft, TopRight, BottomRight, BottomLeft"


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corners) < 4:
            corners.append((x, y))
            print(f"Corner {len(corners)}: ({x}, {y})")


def annotate_court(video_path=None, image_path=None, output_path="data/court_corners.json"):
    """
    手动标注场地角点
    """
    global corners
    
    # 读取图片
    if image_path and Path(image_path).exists():
        img = cv2.imread(str(image_path))
    elif video_path:
        cap = cv2.VideoCapture(str(video_path))
        ret, img = cap.read()
        cap.release()
        if not ret:
            print("无法读取视频")
            return None
    else:
        print("没有提供图片或视频")
        return None
    
    # 创建窗口
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    h, w = img.shape[:2]
    print(f"\n图片尺寸: {w}x{h}")
    print("\n请按顺序点击4个角点:")
    print("  1. Top-Left (左上角)")
    print("  2. Top-Right (右上角)")
    print("  3. Bottom-Right (右下角)")
    print("  4. Bottom-Left (左下角)")
    print("\n按 'r' 重新开始")
    print("按 's' 保存并退出")
    print("按 'q' 退出不保存\n")
    
    while True:
        display = img.copy()
        
        # 已点击的点
        for i, (x, y) in enumerate(corners):
            color = (0, 255, 0)  # 绿色
            cv2.circle(display, (x, y), 10, color, -1)
            cv2.putText(display, str(i+1), (x+15, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 已有的点之间画线
        if len(corners) >= 2:
            for i in range(len(corners) - 1):
                cv2.line(display, corners[i], corners[i+1], (0, 255, 0), 2)
        
        # 如果4个点都点好了，画完整多边形
        if len(corners) == 4:
            pts = np.array(corners, np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 3)
            
            # 填充
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        
        # 显示指令
        cv2.putText(display, f"点击: {len(corners)}/4", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('r'):
            corners = []
            print("重置")
        elif key == ord('s') and len(corners) == 4:
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    
    # 保存角点
    result = {
        "corners": corners,
        "image_size": {"width": w, "height": h}
    }
    
    # 确保目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2))
    
    print(f"\n✅ 角点已保存到: {output_path}")
    print(f"   角点坐标: {corners}")
    
    return result


def apply_saved_corners(video_path, corners_path="data/court_corners.json", output_dir="reports"):
    """
    使用保存的角点处理视频，生成带场地标注的帧
    """
    import os
    from ultralytics import YOLO
    
    # 读取角点
    corners_data = json.loads(Path(corners_path).read_text())
    corners = corners_data["corners"]
    img_w, img_h = corners_data["image_size"]["width"], corners_data["image_size"]["height"]
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    model = YOLO("yolov8n-pose.pt")
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_idx = 0
    saved = 0
    
    while frame_idx < 30:  # 只处理前30帧demo
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # 调整角点到当前帧尺寸（如果不同）
        scale_x = w / img_w
        scale_y = h / img_h
        
        # 保持透视比例
        corners_scaled = []
        for x, y in corners:
            corners_scaled.append((
                int(x * scale_x),
                int(y * scale_y)
            ))
        
        # 创建场地多边形
        pts = np.array(corners_scaled, np.int32)
        
        # 绘制
        display = frame.copy()
        
        # 半透明填充
        overlay = display.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
        
        # 边框
        cv2.polylines(display, [pts], True, (0, 255, 0), 4)
        
        # 角点标注
        for i, (x, y) in enumerate(corners_scaled):
            cv2.circle(display, (x, y), 12, (0, 255, 255), -1)
            labels = ["TL", "TR", "BR", "BL"]
            cv2.putText(display, labels[i], (x+15, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 检测球员脚点
        results = model(frame, verbose=False)
        
        if results and len(results) > 0 and results[0].keypoints is not None:
            kpts_all = results[0].keypoints.xy.cpu().numpy()
            
            for person_kpts in kpts_all:
                # 获取脚点
                if len(person_kpts) > 16:
                    left_ankle = person_kpts[15]
                    right_ankle = person_kpts[16]
                    valid = []
                    if left_ankle[0] > 0: valid.append(left_ankle)
                    if right_ankle[0] > 0: valid.append(right_ankle)
                    
                    if valid:
                        foot = (sum(p[0] for p in valid)/len(valid), 
                               sum(p[1] for p in valid)/len(valid))
                    else:
                        foot = None
                else:
                    foot = None
                
                if foot:
                    # 检查是否在场内
                    inside = cv2.pointPolygonTest(pts.astype(np.float32), foot, False) >= 0
                    
                    if inside:
                        color = (0, 255, 0)  # 绿色
                        cv2.circle(display, (int(foot[0]), int(foot[1])), 15, color, -1)
                    else:
                        color = (0, 0, 255)  # 红色
                        cv2.circle(display, (int(foot[0]), int(foot[1])), 15, color, -1)
        
        # 保存
        cv2.imwrite(f"{output_dir}/court_annotated_{saved:02d}.jpg", display)
        saved += 1
        frame_idx += 1
    
    cap.release()
    print(f"✅ 已保存 {saved} 张标注图片到 {output_dir}/")


if __name__ == "__main__":
    import sys
    
    # 找视频
    video = Path.home() / "Desktop" / "badminton_sample.mp4"
    if not video.exists():
        video = Path.home() / "Desktop" / "badminton_hd.mp4"
    
    if not video.exists():
        print("找不到视频文件")
        sys.exit(1)
    
    print(f"使用视频: {video}")
    
    # 步骤1: 手动标注
    print("\n" + "="*50)
    print("步骤1: 手动标注场地角点")
    print("="*50)
    
    result = annotate_court(video_path=video)
    
    if result:
        # 步骤2: 应用到视频
        print("\n" + "="*50)
        print("步骤2: 生成标注 demo")
        print("="*50)
        
        apply_saved_corners(video)
        print(f"\n查看图片: reports/court_annotated_*.jpg")
