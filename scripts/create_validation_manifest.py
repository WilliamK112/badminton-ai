#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract benchmark frames and create a validation manifest")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--out-dir", default="validation/session", help="Output directory")
    p.add_argument("--every", type=int, default=30, help="Sample every N frames")
    p.add_argument("--max-frames", type=int, default=None, help="Optional max sampled frames")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--prefix", default=None, help="Optional output frame filename prefix")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prefix = args.prefix or video_path.stem

    sampled = 0
    frame_idx = 0
    with manifest_path.open("w", encoding="utf-8") as fp:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx < args.start_frame:
                frame_idx += 1
                continue
            if (frame_idx - args.start_frame) % max(1, args.every) != 0:
                frame_idx += 1
                continue

            image_name = f"{prefix}_f{frame_idx:06d}.jpg"
            image_path = frames_dir / image_name
            cv2.imwrite(str(image_path), frame)
            record = {
                "video_path": str(video_path),
                "frame_idx": frame_idx,
                "image_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "video_fps": fps,
                "video_total_frames": total_frames,
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            sampled += 1
            if args.max_frames is not None and sampled >= args.max_frames:
                break
            frame_idx += 1

    cap.release()
    print(f"saved manifest -> {manifest_path}")
    print(f"saved frames -> {frames_dir}")
    print(f"sampled frames -> {sampled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

