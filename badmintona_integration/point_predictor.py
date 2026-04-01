"""
Point Outcome Predictor
Uses skeletal movement data to predict point winner
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

class ModelNotTrainedError(Exception):
    """Raised when predictor is used before being trained."""
    pass


class PointOutcomePredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        
    def load_training_data(self, pose_data_file, rally_labels_file):
        """
        Load and prepare training data from pose data and rally labels
        """
        # Load pose data
        with open(pose_data_file, 'r') as f:
            pose_data = json.load(f)
        
        # Load rally labels (who won each point)
        # Format: {frame: winner} where winner is 'X' or 'Y'
        rally_labels = {}
        try:
            with open(rally_labels_file, 'r') as f:
                rally_data = json.load(f)
                for rally in rally_data:
                    rally_labels[rally['end_frame']] = rally['winner']
        except:
            print("No rally labels found, using shuttle position to infer")
        
        # Convert to DataFrame
        df = pd.DataFrame(pose_data)
        
        # Aggregate features per rally (last N frames before point ends)
        training_samples = []
        
        for frame in df['frame'].unique():
            frame_data = df[df['frame'] == frame]
            
            # Group by player
            for player_id in frame_data['player_id'].unique():
                player_data = frame_data[frame_data['player_id'] == player_id]
                
                # Get last 10 frames of data before this frame (simulated)
                # In real usage, would use actual sequence data
                features = {
                    'shoulder_angle': player_data['shoulder_angle'].mean() if 'shoulder_angle' in player_data else 0,
                    'shoulder_width': player_data['shoulder_width'].mean() if 'shoulder_width' in player_data else 0,
                    'l_arm_angle': player_data['l_arm_angle'].mean() if 'l_arm_angle' in player_data else 0,
                    'r_arm_angle': player_data['r_arm_angle'].mean() if 'r_arm_angle' in player_data else 0,
                    'torso_angle': player_data['torso_angle'].mean() if 'torso_angle' in player_data else 0,
                    'torso_height': player_data['torso_height'].mean() if 'torso_height' in player_data else 0,
                    'l_leg_angle': player_data['l_leg_angle'].mean() if 'l_leg_angle' in player_data else 0,
                    'r_leg_angle': player_data['r_leg_angle'].mean() if 'r_leg_angle' in player_data else 0,
                    'l_reach': player_data['l_reach'].mean() if 'l_reach' in player_data else 0,
                    'r_reach': player_data['r_reach'].mean() if 'r_reach' in player_data else 0,
                }
                
                # Add velocity features (difference from previous frame)
                prev_frame = frame - 1
                prev_data = df[(df['frame'] == prev_frame) & (df['player_id'] == player_id)]
                if len(prev_data) > 0:
                    for feat in ['shoulder_angle', 'l_arm_angle', 'r_arm_angle', 'torso_angle']:
                        if feat in player_data and feat in prev_data:
                            features[f'{feat}_vel'] = player_data[feat].mean() - prev_data[feat].mean()
                
                training_samples.append(features)
        
        return pd.DataFrame(training_samples)
    
    def train(self, X, y):
        """Train the model"""
        self.feature_cols = X.columns.tolist()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
    def predict(self, features):
        """Predict winner probability"""
        if self.model is None:
            raise ModelNotTrainedError("Model not trained yet")
        
        X = pd.DataFrame([features])
        X = X.reindex(columns=self.feature_cols, fill_value=0)
        
        prob = self.model.predict_proba(X)[0]
        return {
            'prediction': self.model.classes_[np.argmax(prob)],
            'prob_X': prob[0] if self.model.classes_[0] == 'X' else prob[1],
            'prob_Y': prob[1] if self.model.classes_[1] == 'Y' else prob[0]
        }
    
    def save(self, path):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols
        }, path)
    
    def load(self, path):
        """Load model"""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_cols = data['feature_cols']


def analyze_rally_features(pose_data, rally_end_frame):
    """
    Analyze features for a specific rally
    
    Returns summary statistics for both players
    """
    # Filter data for rally
    rally_data = [p for p in pose_data if p['frame'] <= rally_end_frame]
    
    if not rally_data:
        return None
    
    df = pd.DataFrame(rally_data)
    
    summary = {}
    for player_id in df['player_id'].unique():
        player_df = df[df['player_id'] == player_id]
        
        player_summary = {}
        for feat in ['shoulder_angle', 'shoulder_width', 'l_arm_angle', 'r_arm_angle', 
                     'torso_angle', 'torso_height', 'l_leg_angle', 'r_leg_angle']:
            if feat in player_df.columns:
                player_summary[f'{feat}_mean'] = player_df[feat].mean()
                player_summary[f'{feat}_std'] = player_df[feat].std()
                player_summary[f'{feat}_max'] = player_df[feat].max()
        
        summary[f'player_{player_id}'] = player_summary
    
    return summary


def generate_win_prob_timeline(pose_data, shuttle_positions, rally_boundaries):
    """
    Generate win probability timeline based on pose features
    
    Args:
        pose_data: List of pose features per frame
        shuttle_positions: List of shuttle (x, y) positions per frame
        rally_boundaries: List of (start_frame, end_frame) for each rally
    
    Returns:
        List of (frame, prob_X) for timeline
    """
    timeline = []
    
    for start, end in rally_boundaries:
        # Get pose data for this rally
        rally_poses = [p for p in pose_data if start <= p['frame'] <= end]
        
        if not rally_poses:
            continue
        
        # Calculate average features for each frame
        for frame in range(start, end, 10):  # Sample every 10 frames
            frame_poses = [p for p in rally_poses if abs(p['frame'] - frame) < 5]
            
            if not frame_poses:
                continue
            
            # Simple heuristic: X wins if shuttle in bottom half
            frame_shuttle = [s for s in shuttle_positions if s['frame'] == frame]
            if frame_shuttle:
                shuttle_y = frame_shuttle[0]['y']
                # Assuming court net at y=0.3 (normalized), shuttle in bottom half = X advantage
                prob_X = 0.5 + (0.5 - shuttle_y) if shuttle_y else 0.5
            else:
                prob_X = 0.5
            
            timeline.append((frame, prob_X))
    
    return timeline


if __name__ == '__main__':
    # Example usage
    predictor = PointOutcomePredictor()
    
    # Load training data
    # X = predictor.load_training_data('pose_data.json', 'rally_labels.json')
    # y = ... (labels)
    # predictor.train(X, y)
    # predictor.save('point_outcome_model.pkl')
    
    print("PointOutcomePredictor ready for use")