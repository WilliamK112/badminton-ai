from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def pick_player_model(cli_value: str | None = None) -> str:
    if cli_value:
        return cli_value
    return "yolo11n.pt"


def pick_pose_model(cli_value: str | None = None) -> str:
    if cli_value:
        return cli_value
    return "yolo11n-pose.pt"


def pick_shuttle_model(cli_value: str | None = None) -> str:
    if cli_value:
        return cli_value
    local_best = ROOT / "models" / "weights" / "shuttle_best.pt"
    if local_best.exists():
        return str(local_best)
    preferred = ROOT.parent / "Badminton-Analysis" / "train" / "shuttle_output" / "models" / "weights" / "best.pt"
    if preferred.exists():
        return str(preferred)
    return "yolo11n.pt"

