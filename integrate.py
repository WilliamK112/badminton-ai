#!/usr/bin/env python3
"""
Badminton Analysis Integration
Combines:
1. Player detection (from Badminton-Analysis project)
2. Shuttle detection (from Badminton-Analysis project)  
3. Pose detection (our yolo11n-pose)
4. Point outcome prediction

Usage:
    python integrate.py <input_video> [output_video]
"""
import sys
import os
import json
import cv2
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pose_tracker import PoseTracker
from point_predictor import PointOutcomePredictor, analyze_rally_features


def create_integrated_pipeline(
    player_model_path="train/player_output/models/weights/best.pt",
    shuttle_model_path="train/shuttle_output/models/weights/best.pt",
    pose_model_path="yolo11n-pose.pt"
):
    """Create integrated analysis pipeline"""
    
    # Import from original project (if available)
    try:
        from tracker import PlayerTracker, ShuttleTracker
        from utils import read_video, save_video
        has_original = True
    except ImportError:
        print("Original project not found, using standalone mode")
        has_original = False
    
    class IntegratedAnalyzer:
        def __init__(self):
            self.has_original = has_original
            if has_original:
                self.player_tracker = PlayerTracker(player_model_path)
                self.shuttle_tracker = ShuttleTracker(shuttle_model_path)
            self.pose_tracker = PoseTracker(pose_model_path)
            self.predictor = PointOutcomePredictor()
            
            self.pose_data = []
            self.shuttle_positions = []
            
        def process_video(self, input_path, output_path=None):
            """Process video with full analysis pipeline"""
            
            if self.has_original:
                frames = read_video(input_path)
            else:
                cap = cv2.VideoCapture(input_path)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
            
            print(f"Processing {len(frames)} frames...")
            
            # Detect players and shuttle (if available)
            if self.has_original:
                player_detections = self.player_tracker.detect_player(frames, last_detect=True,
                                                                      path_of_last_detect="last_detect/list_player_dict.pkl")
                shuttle_detections = self.shuttle_tracker.detect_shuttle(frames, last_detect=True,
                                                                           path_of_last_detect="last_detect/list_shuttle_dict.pkl")
                shuttle_positions = self.shuttle_tracker.interpolate_shuttle_position(shuttle_detections)
            else:
                # Use dummy detections for demo
                player_detections = [{} for _ in frames]
                shuttle_positions = [None for _ in frames]
            
            # Process each frame with pose detection
            output_frames = []
            
            for i, (frame, players, shuttle) in enumerate(zip(frames, player_detections, shuttle_positions)):
                # Store shuttle position
                if shuttle:
                    self.shuttle_positions.append({'frame': i, 'x': shuttle[0], 'y': shuttle[1]})
                
                # Detect poses for each player
                for player_id, bbox in players.items():
                    # Get pose keypoints
                    keypoints = self.pose_tracker.detect_pose(frame, bbox)
                    
                    if keypoints:
                        # Extract features
                        features = self.pose_tracker.extract_features(keypoints)
                        if features:
                            features['frame'] = i
                            features['player_id'] = player_id
                            self.pose_data.append(features)
                            
                            # Draw skeleton
                            frame = self.pose_tracker.draw_skeleton(frame, keypoints)
                
                # Draw player boxes
                if self.has_original and players:
                    frame = self.player_tracker.draw_single_frame(frame, players)
                
                # Draw shuttle
                if shuttle and self.has_original:
                    frame = self.shuttle_tracker.draw_single_shuttle(frame, shuttle)
                
                output_frames.append(frame)
                
                if i % 100 == 0:
                    print(f"Processed {i}/{len(frames)} frames")
            
            # Save pose data
            with open('pose_data.json', 'w') as f:
                json.dump(self.pose_data, f, indent=2)
            
            # Analyze rally patterns
            self.analyze_rallies()
            
            # Generate win probability timeline
            self.generate_timeline()
            
            # Save output video
            if output_path:
                if self.has_original:
                    save_video(output_frames, input_path, output_path)
                else:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, 30, (frames[0].shape[1], frames[0].shape[0]))
                    for f in output_frames:
                        out.write(f)
                    out.release()
            
            return output_frames
            
        def analyze_rallies(self):
            """Analyze features per rally"""
            if not self.pose_data:
                print("No pose data to analyze")
                return
            
            # Find rally boundaries (gaps in frame sequence)
            df = pd.DataFrame(self.pose_data)
            frames = sorted(df['frame'].unique())
            
            gaps = []
            for i in range(1, len(frames)):
                if frames[i] - frames[i-1] > 50:  # Gap > 50 frames
                    gaps.append((frames[i-1], frames[i]))
            
            rally_summaries = []
            for start, end in gaps:
                summary = analyze_rally_features(self.pose_data, end)
                if summary:
                    rally_summaries.append({
                        'start_frame': start,
                        'end_frame': end,
                        'features': summary
                    })
            
            # Save rally analysis
            with open('rally_analysis.json', 'w') as f:
                json.dump(rally_summaries, f, indent=2)
            
            print(f"Analyzed {len(rally_summaries)} rallies")
            
        def generate_timeline(self):
            """Generate win probability timeline"""
            if not self.shuttle_positions:
                return
            
            timeline = []
            for sp in self.shuttle_positions:
                frame = sp['frame']
                y = sp['y'] / 1080  # Normalize to court height
                
                # Simple model: shuttle in bottom half = X advantage
                # Net at ~40% from top
                net_y = 0.4
                if y > net_y:
                    prob_X = 0.5 + (y - net_y) / (1 - net_y) * 0.5
                else:
                    prob_X = 0.5 - (net_y - y) / net_y * 0.5
                
                timeline.append({'frame': frame, 'prob_X': prob_X})
            
            with open('win_prob_timeline.json', 'w') as f:
                json.dump(timeline, f, indent=2)
            
            print(f"Generated timeline with {len(timeline)} points")
            
        def save_outputs(self):
            """Save all analysis outputs"""
            outputs = {
                'pose_data': self.pose_data,
                'shuttle_positions': self.shuttle_positions,
                'num_frames': len(self.pose_data),
                'num_players': len(set(p['player_id'] for p in self.pose_data if 'player_id' in p))
            }
            
            with open('analysis_outputs.json', 'w') as f:
                json.dump(outputs, f, indent=2)
            
            print("Saved all outputs")
    
    return IntegratedAnalyzer()


def main():
    if len(sys.argv) < 2:
        print("Usage: python integrate.py <input_video> [output_video]")
        print("\nThis script integrates pose detection with Badminton-Analysis project")
        print("Expected to be run in the Badminton-Analysis directory")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else 'output_integrated.mp4'
    
    analyzer = create_integrated_pipeline()
    analyzer.process_video(input_video, output_video)
    analyzer.save_outputs()
    
    print(f"\nAnalysis complete!")
    print(f"Outputs:")
    print(f"  - {output_video} (video with pose overlay)")
    print(f"  - pose_data.json (skeletal features)")
    print(f"  - rally_analysis.json (rally summaries)")
    print(f"  - win_prob_timeline.json (win probability)")
    print(f"  - analysis_outputs.json (summary)")


if __name__ == '__main__':
    main()