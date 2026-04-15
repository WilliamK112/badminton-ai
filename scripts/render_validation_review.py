#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render visual review sheets for validation predictions or labels")
    p.add_argument("--input", required=True, help="JSONL file with predictions or labels")
    p.add_argument("--out-dir", default=None, help="Output directory")
    p.add_argument("--columns", type=int, default=4)
    p.add_argument("--tile-width", type=int, default=480)
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def draw_row(row: dict) -> np.ndarray:
    img_path = Path(row["image_path"])
    frame = cv2.imread(str(img_path))
    if frame is None:
        raise FileNotFoundError(img_path)

    players = row.get("players", {})
    color_map = {
        "P1": (255, 0, 255),
        "P2": (0, 165, 255),
        "1": (255, 0, 255),
        "2": (0, 165, 255),
    }
    for pid, pdata in players.items():
        box = pdata.get("bbox") if isinstance(pdata, dict) else None
        if not box:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = color_map.get(pid, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, str(pid), (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    shuttle = row.get("shuttle", {})
    if shuttle.get("visible") and shuttle.get("x") is not None and shuttle.get("y") is not None:
        x, y = int(shuttle["x"]), int(shuttle["y"])
        cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
        cv2.putText(frame, "shuttle", (x + 8, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    label = f"f={row['frame_idx']}"
    cv2.rectangle(frame, (0, 0), (180, 36), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def make_contact_sheet(images: list[np.ndarray], columns: int, tile_width: int) -> np.ndarray:
    if not images:
        raise ValueError("no images")
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = tile_width / w
        resized.append(cv2.resize(img, (tile_width, int(h * scale))))
    tile_height = max(img.shape[0] for img in resized)
    normalized = []
    for img in resized:
        if img.shape[0] < tile_height:
            pad = np.zeros((tile_height - img.shape[0], img.shape[1], 3), dtype=img.dtype)
            img = np.vstack([img, pad])
        normalized.append(img)
    rows = math.ceil(len(normalized) / columns)
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for idx, img in enumerate(normalized):
        r = idx // columns
        c = idx % columns
        y1 = r * tile_height
        x1 = c * tile_width
        sheet[y1:y1 + tile_height, x1:x1 + tile_width] = img
    return sheet


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else input_path.parent / "review"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    rendered = [draw_row(row) for row in rows]
    sheet = make_contact_sheet(rendered, columns=args.columns, tile_width=args.tile_width)
    sheet_path = out_dir / f"{input_path.stem}.sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet)

    for idx, img in enumerate(rendered):
        cv2.imwrite(str(out_dir / f"{input_path.stem}.{idx:03d}.jpg"), img)

    print(f"saved review sheet -> {sheet_path}")
    print(f"saved frame renders -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

