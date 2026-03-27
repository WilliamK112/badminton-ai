"""
球员追踪模块 - 使用 YOLO + 强时序约束，稳定只保留场上两名球员
"""
import numpy as np
from ultralytics import YOLO
import pandas as pd


class PlayerTracker:
    """球员追踪器（ROI + top/bottom 双槽位 + 防跳变）"""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        roi_x_min: float = 0.14,
        roi_x_max: float = 0.86,
        roi_y_min: float = 0.10,
        roi_y_max: float = 0.94,
        min_area_ratio: float = 0.004,
        max_jump_norm: float = 0.12,   # 单帧允许的最大中心跳变（相对宽高归一化）
        hold_frames: int = 10,         # 丢失后最多保留历史框的帧数
    ):
        self.model = YOLO(model_path)
        self.tracks = {}

        self.roi_x_min = roi_x_min
        self.roi_x_max = roi_x_max
        self.roi_y_min = roi_y_min
        self.roi_y_max = roi_y_max
        self.min_area_ratio = min_area_ratio
        self.max_jump_norm = max_jump_norm
        self.hold_frames = hold_frames

        # 两个固定槽位：1=上场球员，2=下场球员
        self.slot_state = {
            1: {"bbox": None, "cx": None, "cy": None, "miss": 0},
            2: {"bbox": None, "cx": None, "cy": None, "miss": 0},
        }

    def _in_roi(self, cx: float, cy: float, w: int, h: int) -> bool:
        nx, ny = cx / max(w, 1), cy / max(h, 1)
        return self.roi_x_min <= nx <= self.roi_x_max and self.roi_y_min <= ny <= self.roi_y_max

    def _clip_bbox_to_roi(self, bbox, w: int, h: int):
        x1, y1, x2, y2 = bbox
        rx1, ry1 = self.roi_x_min * w, self.roi_y_min * h
        rx2, ry2 = self.roi_x_max * w, self.roi_y_max * h
        x1 = max(rx1, min(rx2 - 2, x1))
        x2 = max(rx1 + 2, min(rx2, x2))
        y1 = max(ry1, min(ry2 - 2, y1))
        y2 = max(ry1 + 2, min(ry2, y2))
        if x2 <= x1 + 1:
            x2 = x1 + 2
        if y2 <= y1 + 1:
            y2 = y1 + 2
        return [float(x1), float(y1), float(x2), float(y2)]

    def _detect_candidates(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        results = self.model.track(frame, persist=True, verbose=False)[0]
        cands = []

        if results.boxes is None:
            return cands

        for box in results.boxes:
            cls_id = int(box.cls.item()) if box.cls is not None else -1
            if cls_id != 0:  # only person
                continue
            if box.id is None:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if area < self.min_area_ratio * (w * h):
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if not self._in_roi(cx, cy, w, h):
                continue

            conf = float(box.conf.item()) if box.conf is not None else 0.0
            cands.append({
                "track_id": int(box.id.item()),
                "bbox": [x1, y1, x2, y2],
                "class": results.names[cls_id],
                "conf": conf,
                "cx": cx,
                "cy": cy,
                "area": area,
            })

        return cands

    def _init_slots(self, candidates):
        if not candidates:
            return
        c_sorted = sorted(candidates, key=lambda c: c["cy"])
        top = c_sorted[0]
        bot = c_sorted[-1] if len(c_sorted) > 1 else None

        self.slot_state[1].update({"bbox": top["bbox"], "cx": top["cx"], "cy": top["cy"], "miss": 0})
        if bot is not None and bot is not top:
            self.slot_state[2].update({"bbox": bot["bbox"], "cx": bot["cx"], "cy": bot["cy"], "miss": 0})

    def _assign_slot(self, slot_id: int, candidates, used, w: int, h: int):
        state = self.slot_state[slot_id]
        if state["cx"] is None or state["cy"] is None:
            return None

        prev = np.array([state["cx"] / w, state["cy"] / h], dtype=float)
        best_i, best_score, best_dist = None, 1e9, 1e9

        for i, c in enumerate(candidates):
            if i in used:
                continue
            cur = np.array([c["cx"] / w, c["cy"] / h], dtype=float)
            dist = float(np.linalg.norm(cur - prev))

            # 上场更偏上、下场更偏下（软约束）
            side_pen = 0.0
            if slot_id == 1 and c["cy"] > 0.62 * h:
                side_pen = 0.08
            if slot_id == 2 and c["cy"] < 0.38 * h:
                side_pen = 0.08

            # score 越小越好：距离主导，置信度辅助
            score = dist + side_pen - 0.03 * c["conf"]
            if score < best_score:
                best_score = score
                best_i = i
                best_dist = dist

        if best_i is None:
            return None

        # 跳变太大则拒绝（避免跳到场边/观众）
        if best_dist > self.max_jump_norm:
            return None

        used.add(best_i)
        return candidates[best_i]

    def track_frame(self, frame: np.ndarray, frame_idx: int) -> dict:
        h, w = frame.shape[:2]
        candidates = self._detect_candidates(frame)

        # 初次建立槽位
        if self.slot_state[1]["cx"] is None and self.slot_state[2]["cx"] is None:
            self._init_slots(candidates)

        used = set()
        assigned = {}

        # 先按时序给两个槽位匹配
        for sid in [1, 2]:
            c = self._assign_slot(sid, candidates, used, w, h)
            if c is not None:
                bbox = self._clip_bbox_to_roi(c["bbox"], w, h)
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                self.slot_state[sid].update({"bbox": bbox, "cx": cx, "cy": cy, "miss": 0})
                assigned[sid] = {
                    "bbox": bbox,
                    "class": c["class"],
                    "conf": c["conf"],
                    "source_track_id": c["track_id"],
                }
            else:
                # 未匹配到，短时间保留上一帧位置，避免抖动/跳变
                self.slot_state[sid]["miss"] += 1
                if self.slot_state[sid]["bbox"] is not None and self.slot_state[sid]["miss"] <= self.hold_frames:
                    assigned[sid] = {
                        "bbox": self.slot_state[sid]["bbox"],
                        "class": "person",
                        "conf": 0.01,
                        "source_track_id": -1,
                    }

        self.tracks[frame_idx] = assigned
        return assigned

    def get_player_x_and_y(self) -> pd.DataFrame:
        data = []
        for frame_idx, tracks in self.tracks.items():
            for track_id, info in tracks.items():
                x1, y1, x2, y2 = info["bbox"]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                player_label = "X" if int(track_id) == 1 else "Y"
                data.append({
                    "frame": frame_idx,
                    "track_id": track_id,
                    "player": player_label,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": cx, "cy": cy,
                })
        return pd.DataFrame(data)


class SmoothFilter:
    """平滑滤波器 - 减少追踪抖动"""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history = {}

    def smooth(self, frame_idx: int, track_id: int, position: tuple) -> tuple:
        if track_id not in self.history:
            self.history[track_id] = []

        self.history[track_id].append(position)
        if len(self.history[track_id]) > self.window_size:
            self.history[track_id].pop(0)

        positions = self.history[track_id]
        n = len(positions)
        return (
            sum(p[0] for p in positions) / n,
            sum(p[1] for p in positions) / n,
        )
