#!/usr/bin/env python3
"""Live badminton overlay demo.

MVP goals:
- read webcam or video stream
- detect players
- detect pose skeletons in player boxes
- detect shuttle
- preview annotated frames live
- optionally record raw + annotated video and frame metadata JSONL

Examples:
  python scripts/live_overlay_demo.py --source 0 --output-dir live_output
  python scripts/live_overlay_demo.py --source badminton_sample.mp4 --no-display
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_defaults import pick_player_model, pick_pose_model, pick_shuttle_model

LOGGER = logging.getLogger("live_overlay_demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live badminton overlay demo")
    parser.add_argument("--source", default="0", help="Camera index or video path, default: 0")
    parser.add_argument("--output-dir", default="live_output", help="Directory for recordings and metadata")
    parser.add_argument("--player-model", default=None)
    parser.add_argument("--pose-model", default=None)
    parser.add_argument("--shuttle-model", default=None, help="Primary shuttle model path")
    parser.add_argument("--fallback-shuttle-model", default="yolo11n.pt")
    parser.add_argument("--pose-every", type=int, default=1, help="Run pose detection every N frames")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--no-display", action="store_true", help="Disable live preview window")
    parser.add_argument("--no-save", action="store_true", help="Disable raw/annotated/mp4/jsonl outputs")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")
    return parser.parse_args()


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def open_source(source_arg: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    source: int | str
    if source_arg.isdigit():
        source = int(source_arg)
    else:
        source = source_arg
        source_path = Path(source_arg).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"video source does not exist: {source_path}")
        source = str(source_path)

    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def make_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    safe_fps = fps if fps and fps > 0 else 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, safe_fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return writer


def as_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): as_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return as_builtin(value.tolist())
        except Exception:
            pass
    return value


def draw_players_and_pose(frame: np.ndarray, players: dict[int, dict], pose_tracker: Any, pose_cache: dict[int, list], run_pose: bool) -> tuple[np.ndarray, dict[int, dict]]:
    frame_payload: dict[int, dict] = {}
    for slot_id, info in players.items():
        bbox = info["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        color = (255, 0, 255) if int(slot_id) == 1 else (0, 165, 255)
        label = f"P{slot_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        keypoints = pose_cache.get(slot_id, [])
        if run_pose:
            try:
                keypoints = pose_tracker.detect_pose(frame, bbox)
            except Exception:
                keypoints = pose_cache.get(slot_id, [])
            pose_cache[slot_id] = keypoints

        if keypoints:
            frame = pose_tracker.draw_skeleton(frame, keypoints, color=color, thickness=2)

        frame_payload[int(slot_id)] = {
            "bbox": [float(v) for v in bbox],
            "keypoints": as_builtin(keypoints),
            "conf": float(info.get("conf", 0.0)),
        }

    return frame, frame_payload


def draw_shuttle(frame: np.ndarray, shuttle_xy: tuple[float | None, float | None]) -> tuple[np.ndarray, dict | None]:
    x, y = shuttle_xy
    if x is None or y is None:
        return frame, None

    cx, cy = int(x), int(y)
    cv2.circle(frame, (cx, cy), 7, (0, 255, 0), -1)
    cv2.rectangle(frame, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 255, 0), 2)
    cv2.putText(frame, "shuttle", (cx + 12, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, {"x": float(x), "y": float(y)}


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    player_model = pick_player_model(args.player_model)
    pose_model = pick_pose_model(args.pose_model)
    shuttle_model = pick_shuttle_model(args.shuttle_model)

    try:
        from src.player.tracker import PlayerTracker
        from src.shuttle.tracker import ShuttleTracker
        from badmintona_integration.pose_tracker import PoseTracker
    except ModuleNotFoundError as exc:
        print(
            "Missing dependency or local module. For this live demo, install at least: ultralytics opencv-python numpy scipy pandas",
            file=sys.stderr,
        )
        print(f"detail: {exc}", file=sys.stderr)
        return 1

    cap = None
    raw_writer = None
    annotated_writer = None
    meta_fp = None
    output_dir = ROOT / args.output_dir

    try:
        cap = open_source(args.source, args.camera_width, args.camera_height, args.camera_fps)
        if not cap.isOpened():
            LOGGER.error("Failed to open source: %s", args.source)
            return 1

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.camera_width
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.camera_height
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or args.camera_fps or 30

        if not args.no_save:
            output_dir.mkdir(parents=True, exist_ok=True)
            raw_writer = make_writer(output_dir / "raw.mp4", actual_fps, (actual_width, actual_height))
            annotated_writer = make_writer(output_dir / "annotated.mp4", actual_fps, (actual_width, actual_height))
            meta_fp = (output_dir / "frames.jsonl").open("w", encoding="utf-8")

        player_tracker = PlayerTracker(model_path=player_model)
        shuttle_tracker = ShuttleTracker(
            model_path=shuttle_model,
            fallback_model_path=args.fallback_shuttle_model,
        )
        pose_tracker = PoseTracker(pose_model)
        pose_cache: dict[int, list] = {}

        frame_idx = 0
        started_at = time.time()
        smoothed_fps = 0.0

        LOGGER.info("%s", "=" * 60)
        LOGGER.info("Live badminton overlay demo")
        LOGGER.info("source: %s", args.source)
        LOGGER.info("player model: %s", player_model)
        LOGGER.info("pose model: %s", pose_model)
        LOGGER.info("shuttle model: %s", shuttle_model)
        LOGGER.info("fallback shuttle model: %s", args.fallback_shuttle_model)
        if not args.no_save:
            LOGGER.info("output dir: %s", output_dir)
        LOGGER.info("press q to quit")
        LOGGER.info("%s", "=" * 60)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                LOGGER.info("Input ended at frame %d", frame_idx)
                break

            loop_t0 = time.time()
            raw_frame = frame.copy()

            try:
                players = player_tracker.track_frame(frame, frame_idx)
            except Exception as exc:
                LOGGER.warning("player tracking error at frame %d: %s", frame_idx, exc)
                players = {}

            try:
                shuttle_xy = shuttle_tracker.detect_frame(frame, frame_idx)
            except Exception as exc:
                LOGGER.warning("shuttle tracking error at frame %d: %s", frame_idx, exc)
                shuttle_xy = (None, None)

            run_pose = frame_idx % max(1, args.pose_every) == 0
            annotated, player_payload = draw_players_and_pose(frame.copy(), players, pose_tracker, pose_cache, run_pose)
            annotated, shuttle_payload = draw_shuttle(annotated, shuttle_xy)

            instant_fps = 1.0 / max(time.time() - loop_t0, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0 else (0.85 * smoothed_fps + 0.15 * instant_fps)
            cv2.putText(annotated, f"FPS {smoothed_fps:.1f}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated, f"frame {frame_idx}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if raw_writer is not None:
                raw_writer.write(raw_frame)
            if annotated_writer is not None:
                annotated_writer.write(annotated)
            if meta_fp is not None:
                payload = {
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(time.time() - started_at, 4),
                    "players": player_payload,
                    "shuttle": shuttle_payload,
                }
                meta_fp.write(json.dumps(as_builtin(payload), ensure_ascii=False) + "\n")

            if not args.no_display:
                cv2.imshow("Badminton Live Overlay", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            frame_idx += 1
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

        LOGGER.info("done, processed %d frames", frame_idx)
        if not args.no_save:
            LOGGER.info("saved raw video -> %s", output_dir / "raw.mp4")
            LOGGER.info("saved annotated video -> %s", output_dir / "annotated.mp4")
            LOGGER.info("saved metadata -> %s", output_dir / "frames.jsonl")
        return 0
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        return 130
    except Exception as exc:
        LOGGER.exception("live overlay demo failed: %s", exc)
        return 1
    finally:
        if cap is not None:
            cap.release()
        if raw_writer is not None:
            raw_writer.release()
        if annotated_writer is not None:
            annotated_writer.release()
        if meta_fp is not None:
            meta_fp.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
