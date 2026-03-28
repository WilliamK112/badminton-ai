# Badminton AI Tracking & Analysis

A practical badminton video analysis pipeline focused on **stable 2-player tracking + shuttle tracking + pose visualization**.

## About

This project is built for demo-ready, reproducible badminton analysis outputs from match videos.
Current focus:
- Keep only on-court 2 players (avoid audience / sideline jumps)
- Keep shuttle trajectory continuous and visible
- Preserve human pose (skeleton) data and visualization in the final output

## Tech Stack / Models Used

### Core Libraries
- Python
- OpenCV
- Ultralytics YOLO
- NumPy / Pandas / SciPy

### Detection & Tracking Models
- **Player detection/tracking:** `yolo11n.pt` (person class + temporal constraints)
- **Pose estimation:** `yolo11n-pose.pt`
- **Shuttle primary model:**
  - `Badminton-Analysis/train/shuttle_output/models/weights/best.pt`
- **Shuttle fallback model:** `yolo11n.pt` (sports-ball class fallback)

### Tracking/Filtering Strategies
- ROI filtering (court-focused region)
- Top/Bottom 2-player slot assignment
- Temporal anti-jump constraints for player boxes
- Shuttle missing-frame recovery via interpolation + velocity-based prediction

## Project Structure

```text
badminton-ai/
├── README.md
├── PROJECT_STATUS.md
├── src/
│   ├── pipeline/
│   │   └── runner.py                # main entry
│   ├── integration/
│   │   └── pipeline.py              # player + shuttle + pose integration
│   ├── player/
│   │   └── tracker.py               # 2-player constrained tracker
│   ├── shuttle/
│   │   └── tracker.py               # rewritten shuttle tracker (primary+fallback)
│   └── ...                          # historical experiments / training scripts
├── demo/
│   └── gif/
│       ├── 截屏2026-03-25 01.40.31.png
│       ├── frame_0010.jpg
│       └── badminton_github_long.gif
└── output_*/                         # local generated outputs (ignored in git)
```

## Current Output Style

Output video includes:
- Player 1 / Player 2 labels (on top of boxes)
- Skeleton keypoints + limbs
- Shuttle marker + `SHUTTLE` label

## Quick Start

```bash
cd /Users/William/.openclaw/workspace/projects/badminton-ai
source .venv/bin/activate

python -m src.pipeline.runner thisone.mp4 -o output_demo -i 2 -m 3000
```

## Demo (Latest)

<table>
  <tr>
    <td><img src="./demo/gif/%E6%88%AA%E5%B1%8F2026-03-25%2001.40.31.png" alt="screenshot" width="320"/></td>
    <td><img src="./demo/gif/frame_0010.jpg" alt="frame_0010" width="320"/></td>
    <td><img src="./demo/gif/badminton_github_long.gif" alt="badminton_github_long" width="320"/></td>
  </tr>
</table>

## Notes

- This repo keeps the latest demo-facing pipeline and visuals clean.
- Large temporary artifacts (outputs/reports/trash) are local and excluded from git.
