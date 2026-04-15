#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a reusable badminton benchmark bundle from video to review outputs")
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--out-dir", required=True, help="Benchmark output directory")
    p.add_argument("--every", type=int, default=45, help="Sample every N frames")
    p.add_argument("--max-frames", type=int, default=24, help="Maximum sampled frames")
    p.add_argument("--indices", default=None, help="Optional comma-separated row indices to keep after sampling")
    p.add_argument("--player-model", default=None)
    p.add_argument("--pose-model", default=None)
    p.add_argument("--shuttle-model", default=None)
    p.add_argument("--fallback-shuttle-model", default="yolo11n.pt")
    p.add_argument("--with-pose", action="store_true")
    p.add_argument("--python", default=None, help="Python executable to use, default prefers repo-local .venv")
    return p.parse_args()


def pick_python(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    local = ROOT / ".venv" / "bin" / "python"
    if local.exists():
        return str(local)
    return sys.executable


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_predictions(path: Path) -> dict:
    rows = load_jsonl(path)
    return {
        "frames": len(rows),
        "shuttle_visible": sum(1 for r in rows if r.get("shuttle", {}).get("visible")),
        "p1_detected": sum(1 for r in rows if r.get("players", {}).get("1")),
        "p2_detected": sum(1 for r in rows if r.get("players", {}).get("2")),
    }


def main() -> int:
    args = parse_args()
    python_bin = pick_python(args.python)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    predictions_path = out_dir / "predictions.jsonl"
    bootstrap_path = out_dir / "labels.bootstrap.jsonl"
    review_dir = out_dir / "review"

    run([
        python_bin,
        "scripts/create_validation_manifest.py",
        "--video",
        str(Path(args.video).expanduser().resolve()),
        "--out-dir",
        str(out_dir),
        "--every",
        str(args.every),
        "--max-frames",
        str(args.max_frames),
    ])

    if args.indices:
        subset_manifest = out_dir / "manifest.subset.jsonl"
        run([
            python_bin,
            "scripts/select_validation_subset.py",
            "--input",
            str(manifest_path),
            "--out",
            str(subset_manifest),
            "--indices",
            args.indices,
        ])
        shutil.move(str(subset_manifest), str(manifest_path))

    infer_cmd = [
        python_bin,
        "scripts/run_validation_inference.py",
        "--manifest",
        str(manifest_path),
        "--out",
        str(predictions_path),
        "--fallback-shuttle-model",
        args.fallback_shuttle_model,
    ]
    if args.player_model:
        infer_cmd += ["--player-model", args.player_model]
    if args.pose_model:
        infer_cmd += ["--pose-model", args.pose_model]
    if args.shuttle_model:
        infer_cmd += ["--shuttle-model", args.shuttle_model]
    if args.with_pose:
        infer_cmd.append("--with-pose")
    run(infer_cmd)

    run([
        python_bin,
        "scripts/bootstrap_validation_labels.py",
        "--manifest",
        str(manifest_path),
        "--predictions",
        str(predictions_path),
        "--out",
        str(bootstrap_path),
    ])

    run([
        python_bin,
        "scripts/render_validation_review.py",
        "--input",
        str(predictions_path),
        "--out-dir",
        str(review_dir),
    ])

    summary = {
        "video": str(Path(args.video).expanduser().resolve()),
        "out_dir": str(out_dir),
        "sampling": {
            "every": args.every,
            "max_frames": args.max_frames,
            "indices": args.indices,
        },
        "models": {
            "player_model": args.player_model,
            "pose_model": args.pose_model,
            "shuttle_model": args.shuttle_model,
            "fallback_shuttle_model": args.fallback_shuttle_model,
        },
        "prediction_summary": summarize_predictions(predictions_path),
        "artifacts": {
            "manifest": str(manifest_path),
            "predictions": str(predictions_path),
            "bootstrap_labels": str(bootstrap_path),
            "review_dir": str(review_dir),
            "review_sheet": str(review_dir / f"{predictions_path.stem}.sheet.jpg"),
        },
    }
    summary_path = out_dir / "bundle.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved bundle summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

