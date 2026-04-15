#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_defaults import pick_player_model, pick_pose_model, pick_shuttle_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run current badminton models on labeled validation frames")
    p.add_argument("--manifest", required=True, help="manifest.jsonl or labels.jsonl")
    p.add_argument("--out", default=None, help="Predictions jsonl path")
    p.add_argument("--player-model", default=None)
    p.add_argument("--pose-model", default=None)
    p.add_argument("--shuttle-model", default=None)
    p.add_argument("--fallback-shuttle-model", default="yolo11n.pt")
    p.add_argument("--with-pose", action="store_true")
    return p.parse_args()


def as_builtin(value):
    if isinstance(value, dict):
        return {str(k): as_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_builtin(v) for v in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return as_builtin(value.tolist())
        except Exception:
            pass
    return value


def main() -> int:
    args = parse_args()
    try:
        from src.player.tracker import PlayerTracker
        from src.shuttle.tracker import ShuttleTracker
        from badmintona_integration.pose_tracker import PoseTracker
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing dependency: {exc}")

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else manifest_path.with_name("predictions.jsonl")
    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    grouped = defaultdict(list)
    for row in records:
        grouped[row["video_path"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: r["frame_idx"])

    player_model = pick_player_model(args.player_model)
    pose_model = pick_pose_model(args.pose_model)
    shuttle_model = pick_shuttle_model(args.shuttle_model)

    player_tracker = PlayerTracker(model_path=player_model)
    shuttle_tracker = ShuttleTracker(model_path=shuttle_model, fallback_model_path=args.fallback_shuttle_model)
    pose_tracker = PoseTracker(pose_model) if args.with_pose else None

    with out_path.open("w", encoding="utf-8") as fp:
        for video_path, rows in grouped.items():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise SystemExit(f"Failed to open video: {video_path}")
            for row in rows:
                cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame_idx"])
                ret, frame = cap.read()
                if not ret or frame is None:
                    pred = {
                        "video_path": video_path,
                        "frame_idx": row["frame_idx"],
                        "image_path": row.get("image_path"),
                        "image_width": row.get("image_width"),
                        "image_height": row.get("image_height"),
                        "players": {},
                        "shuttle": {"visible": False, "x": None, "y": None},
                        "pose": {},
                        "error": "failed_to_read_frame",
                    }
                    fp.write(json.dumps(pred, ensure_ascii=False) + "\n")
                    continue

                players = player_tracker.track_frame(frame, row["frame_idx"])
                shuttle_xy = shuttle_tracker.detect_frame(frame, row["frame_idx"])
                pose = {}
                for slot_id, info in players.items():
                    keypoints = []
                    if pose_tracker is not None:
                        try:
                            keypoints = pose_tracker.detect_pose(frame, info["bbox"])
                        except Exception:
                            keypoints = []
                    pose[str(slot_id)] = as_builtin(keypoints)

                pred = {
                    "video_path": video_path,
                    "frame_idx": row["frame_idx"],
                    "image_path": row.get("image_path"),
                    "image_width": row.get("image_width"),
                    "image_height": row.get("image_height"),
                    "players": {
                        str(slot_id): {
                            "bbox": [float(v) for v in info["bbox"]],
                            "conf": float(info.get("conf", 0.0)),
                            "source_track_id": int(info.get("source_track_id", -1)),
                        }
                        for slot_id, info in players.items()
                    },
                    "shuttle": {
                        "visible": shuttle_xy[0] is not None and shuttle_xy[1] is not None,
                        "x": float(shuttle_xy[0]) if shuttle_xy[0] is not None else None,
                        "y": float(shuttle_xy[1]) if shuttle_xy[1] is not None else None,
                    },
                    "pose": pose,
                }
                fp.write(json.dumps(as_builtin(pred), ensure_ascii=False) + "\n")
            cap.release()

    print(f"player model -> {player_model}")
    print(f"pose model -> {pose_model}")
    print(f"shuttle model -> {shuttle_model}")
    print(f"saved predictions -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
