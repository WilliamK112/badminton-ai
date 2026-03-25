"""
Pose Tracker - Extends player tracking with skeleton detection
Integrates with Badminton-Analysis project
"""
from ultralytics import YOLO
import cv2
import numpy as np
import json

class PoseTracker:
    def __init__(self, pose_model_path='yolo11n-pose.pt'):
        self.pose_model = YOLO(pose_model_path)
        
        # COCO keypoint connections
        self.skeleton = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # left arm
            (0, 5), (5, 6), (6, 7),            # right arm  
            (0, 8), (8, 9), (9, 10),          # left leg
            (0, 11), (11, 12), (12, 13),      # right leg
            (8, 11), (8, 14), (11, 14),       # torso
            (14, 15), (15, 16)                 # face
        ]
        
    def detect_pose(self, frame, player_bbox=None):
        """
        Detect pose in a frame, optionally limited to a player bounding box
        
        Args:
            frame: Image frame
            player_bbox: Optional [x1, y1, x2, y2] to limit detection area
            
        Returns:
            List of keypoints for each person detected
        """
        if player_bbox is not None:
            # Crop to player region for faster detection
            x1, y1, x2, y2 = map(int, player_bbox)
            # Add some padding
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1-20), max(0, y1-20)
            x2, y2 = min(w, x2+20), min(h, y2+20)
            crop = frame[y1:y2, x1:x2]
            
            results = self.pose_model.predict(crop, conf=0.5, verbose=False)
            
            # Adjust keypoints back to original coordinates
            keypoints = []
            if results[0].keypoints is not None:
                for kpts in results[0].keypoints.data:
                    adjusted = []
                    for kp in kpts:
                        if kp[2] > 0.3:  # confidence threshold
                            adjusted.append([kp[0] + x1, kp[1] + y1, kp[2]])
                        else:
                            adjusted.append([0, 0, 0])
                    keypoints.append(adjusted)
            return keypoints
        else:
            # Full frame detection
            results = self.pose_model.predict(frame, conf=0.5, verbose=False)
            keypoints = []
            if results[0].keypoints is not None:
                for kpts in results[0].keypoints.data:
                    keypoints.append([[kp[0], kp[1], kp[2]] for kp in kpts])
            return keypoints
    
    def extract_features(self, keypoints):
        """
        Extract features from keypoints for analysis
        
        Returns dict with:
        - shoulder_angle: angle between shoulders
        - arm_angles: angles of both arms
        - torso_angle: angle of torso
        - leg_angles: angles of both legs
        - reach_distance: distance from center to hand
        """
        if not keypoints or len(keypoints[0]) < 17:
            return None
            
        kpts = keypoints[0]  # First person
        
        features = {}
        
        # Keypoint indices (COCO format):
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        # 5: left_shoulder, 6: right_shoulder
        # 7: left_elbow, 8: right_elbow
        # 9: left_wrist, 10: right_wrist
        # 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee
        # 15: left_ankle, 16: right_ankle
        
        # Shoulder angle (between shoulders)
        if all(kpts[5][2] > 0.3 and kpts[6][2] > 0.3):
            l_shoulder = np.array([kpts[5][0], kpts[5][1]])
            r_shoulder = np.array([kpts[6][0], kpts[6][1]])
            shoulder_vec = r_shoulder - l_shoulder
            features['shoulder_angle'] = np.arctan2(shoulder_vec[1], shoulder_vec[0])
            features['shoulder_width'] = np.linalg.norm(shoulder_vec)
        
        # Left arm angle (shoulder -> elbow -> wrist)
        if all(kpts[5][2] > 0.3 and kpts[7][2] > 0.3 and kpts[9][2] > 0.3):
            l_shoulder = np.array([kpts[5][0], kpts[5][1]])
            l_elbow = np.array([kpts[7][0], kpts[7][1]])
            l_wrist = np.array([kpts[9][0], kpts[9][1]])
            v1 = l_shoulder - l_elbow
            v2 = l_wrist - l_elbow
            features['l_arm_angle'] = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        
        # Right arm angle
        if all(kpts[6][2] > 0.3 and kpts[8][2] > 0.3 and kpts[10][2] > 0.3):
            r_shoulder = np.array([kpts[6][0], kpts[6][1]])
            r_elbow = np.array([kpts[8][0], kpts[8][1]])
            r_wrist = np.array([kpts[10][0], kpts[10][1]])
            v1 = r_shoulder - r_elbow
            v2 = r_wrist - r_elbow
            features['r_arm_angle'] = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        
        # Torso angle (shoulder center to hip center)
        if all(kpts[5][2] > 0.3 and kpts[6][2] > 0.3 and kpts[11][2] > 0.3 and kpts[12][2] > 0.3):
            shoulder_center = (np.array([kpts[5][0], kpts[5][1]]) + np.array([kpts[6][0], kpts[6][1]])) / 2
            hip_center = (np.array([kpts[11][0], kpts[11][1]]) + np.array([kpts[12][0], kpts[12][1]])) / 2
            torso_vec = hip_center - shoulder_center
            features['torso_angle'] = np.arctan2(torso_vec[1], torso_vec[0])
            features['torso_height'] = np.linalg.norm(torso_vec)
        
        # Left leg angle
        if all(kpts[11][2] > 0.3 and kpts[13][2] > 0.3 and kpts[15][2] > 0.3):
            l_hip = np.array([kpts[11][0], kpts[11][1]])
            l_knee = np.array([kpts[13][0], kpts[13][1]])
            l_ankle = np.array([kpts[15][0], kpts[15][1]])
            v1 = l_hip - l_knee
            v2 = l_ankle - l_knee
            features['l_leg_angle'] = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        
        # Right leg angle
        if all(kpts[12][2] > 0.3 and kpts[14][2] > 0.3 and kpts[16][2] > 0.3):
            r_hip = np.array([kpts[12][0], kpts[12][1]])
            r_knee = np.array([kpts[14][0], kpts[14][1]])
            r_ankle = np.array([kpts[16][0], kpts[16][1]])
            v1 = r_hip - r_knee
            v2 = r_ankle - r_knee
            features['r_leg_angle'] = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
        
        # Reach distance (shoulder to wrist)
        if all(kpts[5][2] > 0.3 and kpts[9][2] > 0.3):
            shoulder = np.array([kpts[5][0], kpts[5][1]])
            wrist = np.array([kpts[9][0], kpts[9][1]])
            features['l_reach'] = np.linalg.norm(wrist - shoulder)
        
        if all(kpts[6][2] > 0.3 and kpts[10][2] > 0.3):
            shoulder = np.array([kpts[6][0], kpts[6][1]])
            wrist = np.array([kpts[10][0], kpts[10][1]])
            features['r_reach'] = np.linalg.norm(wrist - shoulder)
        
        return features
    
    def draw_skeleton(self, frame, keypoints, color=(0, 255, 255), thickness=3):
        """Draw skeleton on frame"""
        if not keypoints:
            return frame
            
        for kpts in keypoints:
            # Draw keypoints
            for i, kp in enumerate(kpts):
                if kp[2] > 0.3:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(frame, (x, y), 5, color, -1)
            
            # Draw skeleton connections
            for i, j in self.skeleton:
                if i < len(kpts) and j < len(kpts):
                    if kpts[i][2] > 0.3 and kpts[j][2] > 0.3:
                        pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                        pt2 = (int(kpts[j][0]), int(kpts[j][1]))
                        cv2.line(frame, pt1, pt2, color, thickness)
        
        return frame


def process_video_with_pose(input_video, output_video=None):
    """
    Process a badminton video with pose detection
    """
    from utils import read_video, save_video
    from tracker import PlayerTracker, ShuttleTracker
    
    # Read video
    frames = read_video(input_video)
    print(f"Processing {len(frames)} frames...")
    
    # Player detection (from original project)
    player_tracker = PlayerTracker("train/player_output/models/weights/best.pt")
    player_detections = player_tracker.detect_player(frames, last_detect=True,
                                                       path_of_last_detect="last_detect/list_player_dict.pkl")
    
    # Shuttle detection
    shuttle_tracker = ShuttleTracker("train/shuttle_output/models/weights/best.pt")
    shuttle_detect = shuttle_tracker.detect_shuttle(frames, last_detect=True,
                                                     path_of_last_detect="last_detect/list_shuttle_dict.pkl")
    shuttle_interpolate = shuttle_tracker.interpolate_shuttle_position(shuttle_detect)
    
    # Pose tracking
    pose_tracker = PoseTracker()
    
    # Process each frame
    pose_data = []
    output_frames = []
    
    for i, (frame, players, shuttle) in enumerate(zip(frames, player_detections, shuttle_interpolate)):
        # Get player bounding boxes
        for player_id, bbox in players.items():
            # Detect pose in player region
            keypoints = pose_tracker.detect_pose(frame, bbox)
            
            # Extract features
            if keypoints:
                features = pose_tracker.extract_features(keypoints)
                features['frame'] = i
                features['player_id'] = player_id
                features['bbox'] = bbox
                pose_data.append(features)
                
                # Draw skeleton
                frame = pose_tracker.draw_skeleton(frame, keypoints)
        
        # Draw player bboxes (original)
        frame = player_tracker.draw_single_frame(frame, players)
        
        # Draw shuttle
        if shuttle:
            frame = shuttle_tracker.draw_single_shuttle(frame, shuttle)
        
        output_frames.append(frame)
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(frames)} frames")
    
    # Save pose data
    with open('pose_data.json', 'w') as f:
        json.dump(pose_data, f, indent=2)
    
    # Save output video
    if output_video:
        save_video(output_frames, input_video, output_video)
    
    return pose_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
        output_video = sys.argv[2] if len(sys.argv) > 2 else 'output_pose.mp4'
        process_video_with_pose(input_video, output_video)
    else:
        print("Usage: python pose_tracker.py <input_video> [output_video]")