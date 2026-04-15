#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate validation frames for players and shuttle")
    p.add_argument("--manifest", required=True, help="manifest.jsonl from create_validation_manifest.py")
    p.add_argument("--labels", default=None, help="Output labels jsonl path, default next to manifest")
    return p.parse_args()


class Annotator:
    def __init__(self, manifest_path: Path, labels_path: Path):
        self.records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        self.labels_path = labels_path
        self.labels = {}
        if labels_path.exists():
            for line in labels_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self.labels[(row["video_path"], row["frame_idx"])] = row
        self.index = 0
        self.mode = "shuttle"
        self.drag_start = None
        self.current_box = None
        self.window = "Validation Annotator"

    def current_key(self):
        rec = self.records[self.index]
        return (rec["video_path"], rec["frame_idx"])

    def current_label(self):
        rec = self.records[self.index]
        key = self.current_key()
        if key not in self.labels:
            self.labels[key] = {
                "video_path": rec["video_path"],
                "frame_idx": rec["frame_idx"],
                "image_path": rec["image_path"],
                "image_width": rec["image_width"],
                "image_height": rec["image_height"],
                "players": {
                    "P1": {"bbox": None},
                    "P2": {"bbox": None},
                },
                "shuttle": {"visible": False, "x": None, "y": None},
                "pose": {},
            }
        return self.labels[key]

    def save(self):
        with self.labels_path.open("w", encoding="utf-8") as fp:
            rows = sorted(self.labels.values(), key=lambda r: (r["video_path"], r["frame_idx"]))
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    def on_mouse(self, event, x, y, flags, param):
        label = self.current_label()
        if self.mode == "shuttle":
            if event == cv2.EVENT_LBUTTONDOWN:
                label["shuttle"] = {"visible": True, "x": int(x), "y": int(y)}
                self.save()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.current_box = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.current_box = [self.drag_start[0], self.drag_start[1], x, y]
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x1, y1 = self.drag_start
            x2, y2 = x, y
            box = [float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))]
            label["players"][self.mode]["bbox"] = box
            self.drag_start = None
            self.current_box = None
            self.save()

    def draw(self):
        rec = self.records[self.index]
        label = self.current_label()
        img = cv2.imread(rec["image_path"])
        if img is None:
            raise FileNotFoundError(rec["image_path"])

        colors = {"P1": (255, 0, 255), "P2": (0, 165, 255)}
        for pid, pdata in label["players"].items():
            box = pdata.get("bbox")
            if box:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), colors[pid], 2)
                cv2.putText(img, pid, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[pid], 2)

        shuttle = label.get("shuttle", {})
        if shuttle.get("visible") and shuttle.get("x") is not None and shuttle.get("y") is not None:
            cv2.circle(img, (int(shuttle["x"]), int(shuttle["y"])), 8, (0, 255, 0), -1)
            cv2.putText(img, "shuttle", (int(shuttle["x"]) + 10, int(shuttle["y"]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.current_box:
            x1, y1, x2, y2 = map(int, self.current_box)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors.get(self.mode, (255, 255, 255)), 1)

        status = f"[{self.index + 1}/{len(self.records)}] mode={self.mode} | keys: 1=P1 2=P2 s=shuttle o=occluded c=clear n=next p=prev w=save q=quit"
        cv2.putText(img, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return img

    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.on_mouse)
        while True:
            cv2.imshow(self.window, self.draw())
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                self.index = min(self.index + 1, len(self.records) - 1)
            elif key == ord("p"):
                self.index = max(self.index - 1, 0)
            elif key == ord("1"):
                self.mode = "P1"
            elif key == ord("2"):
                self.mode = "P2"
            elif key == ord("s"):
                self.mode = "shuttle"
            elif key == ord("o"):
                self.current_label()["shuttle"] = {"visible": False, "x": None, "y": None}
                self.save()
            elif key == ord("c"):
                label = self.current_label()
                if self.mode in {"P1", "P2"}:
                    label["players"][self.mode]["bbox"] = None
                else:
                    label["shuttle"] = {"visible": False, "x": None, "y": None}
                self.save()
            elif key == ord("w"):
                self.save()
        self.save()
        cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve() if args.labels else manifest_path.with_name("labels.jsonl")
    Annotator(manifest_path, labels_path).run()
    print(f"saved labels -> {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

