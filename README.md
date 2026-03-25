# Badminton Analysis Integration

Integrates pose detection and skeletal analysis with [Badminton-Analysis](https://github.com/ToanNguyenKhanh/Badminton-Analysis) project.

## 🎯 Demo Outputs

| Frame Analysis | Detection Result |
|----------------|------------------|
| ![Demo Frame](./demo_frame.jpg) | ![Demo Screenshot](./demo_screenshot.png) |

## Features

- **Player Tracking**: Uses original project's YOLO model for player bounding boxes
- **Pose Detection**: Uses YOLO11n-pose for skeleton detection
- **Feature Extraction**: 
  - Shoulder angle/width
  - Arm angles (left/right)
  - Torso angle/height
  - Leg angles (left/right)
  - Reach distance
  - Motion velocities
- **Point Prediction**: Predicts winner based on skeletal movement patterns

## Files

| File | Description |
|------|-------------|
| `pose_tracker.py` | Pose detection and feature extraction |
| `point_predictor.py` | Point outcome prediction model |
| `integrate.py` | Main integration pipeline |

## Usage

```bash
# In Badminton-Analysis directory
python integrate.py input_video.mp4 output_video.mp4
```

## Requirements

```bash
pip install ultralytics opencv-python pandas scikit-learn
```

## Output Files

- `pose_data.json` - Skeletal features per frame
- `rally_analysis.json` - Rally-level feature summaries
- `win_prob_timeline.json` - Win probability over time
- `analysis_outputs.json` - Analysis summary
