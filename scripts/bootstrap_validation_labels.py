#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap validation labels from model predictions for later human/agent review")
    p.add_argument("--manifest", required=True, help="manifest.jsonl")
    p.add_argument("--predictions", required=True, help="predictions.jsonl")
    p.add_argument("--out", default=None, help="Output labels jsonl path")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    predictions_path = Path(args.predictions).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else manifest_path.with_name("labels.bootstrap.jsonl")

    manifest = load_jsonl(manifest_path)
    predictions = {
        (row["video_path"], int(row["frame_idx"])): row
        for row in load_jsonl(predictions_path)
    }

    with out_path.open("w", encoding="utf-8") as fp:
        for row in manifest:
            pred = predictions.get((row["video_path"], int(row["frame_idx"])) , {})
            label = {
                "video_path": row["video_path"],
                "frame_idx": row["frame_idx"],
                "image_path": row["image_path"],
                "image_width": row["image_width"],
                "image_height": row["image_height"],
                "players": {
                    "P1": {"bbox": pred.get("players", {}).get("1", {}).get("bbox")},
                    "P2": {"bbox": pred.get("players", {}).get("2", {}).get("bbox")},
                },
                "shuttle": pred.get("shuttle", {"visible": False, "x": None, "y": None}),
                "pose": {
                    "P1": pred.get("pose", {}).get("1", []),
                    "P2": pred.get("pose", {}).get("2", []),
                },
                "bootstrap": {
                    "source": "model_predictions",
                    "reviewed": False,
                },
            }
            fp.write(json.dumps(label, ensure_ascii=False) + "\n")

    print(f"saved bootstrap labels -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

