"""
羽毛球追踪模块（重写版）
思路参考 Badminton-Analysis：
1) 用专用 shuttle 模型做逐帧检测/跟踪
2) 记录 bbox 序列
3) 对缺失帧做时序插值补点
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.interpolate import interp1d


class ShuttleTracker:
    """羽毛球追踪器（基于专用模型 + 时序插值）"""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.05,
        # 主模型（专用 shuttle）通常类 id=0；辅助 COCO sports ball 是 32
        class_candidates: Tuple[int, ...] = (0,),
        use_track_mode: bool = False,
        fallback_model_path: str = "yolo11n.pt",
    ):
        self.model = YOLO(model_path)
        self.fallback_model = YOLO(fallback_model_path)
        self.conf_threshold = conf_threshold
        self.class_candidates = set(class_candidates)
        self.use_track_mode = use_track_mode

        # 保存每帧中心点（像素）
        self.positions: Dict[int, Tuple[float, float]] = {}
        # 保存每帧 bbox（像素）
        self.boxes_by_frame: Dict[int, List[float]] = {}

        # 最近观测，给缺失补点用
        self._recent_obs = deque(maxlen=12)  # [(frame_idx, cx, cy)]
        self._last_frame_idx: Optional[int] = None
        self.prev_xy: Optional[np.ndarray] = None
        self.prev_v = np.array([0.0, 0.0], dtype=float)

        # 画面 ROI（避免角落误检）
        self.play_x_min = 0.03
        self.play_x_max = 0.97
        self.play_y_min = 0.02
        self.play_y_max = 0.98

    def _in_play_region(self, cx: float, cy: float, w: int, h: int) -> bool:
        nx, ny = cx / max(w, 1), cy / max(h, 1)
        return self.play_x_min <= nx <= self.play_x_max and self.play_y_min <= ny <= self.play_y_max

    def _detect_bbox(self, frame: np.ndarray, model: YOLO, class_set: set, use_track_mode: bool = False) -> Optional[Tuple[List[float], float]]:
        """返回最佳 shuttle bbox 和 conf"""
        if use_track_mode:
            result = model.track(frame, persist=True, verbose=False)[0]
        else:
            result = model(frame, verbose=False)[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes))
        clss = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros(len(boxes), dtype=int)

        h, w = frame.shape[:2]
        candidates = []
        for box, conf, cls_id in zip(boxes, confs, clss):
            if cls_id not in class_set:
                continue
            if float(conf) < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(float, box)
            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if not self._in_play_region(cx, cy, w, h):
                continue

            area = max((x2 - x1) * (y2 - y1), 1.0)
            # 分数：高置信度优先，小面积优先（羽毛球框通常小）
            score = float(conf) - 0.00001 * area
            candidates.append((score, [x1, y1, x2, y2], float(conf)))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, bbox, conf = candidates[0]
        return bbox, conf

    def _downup_variant(self, frame: np.ndarray, scale: float = 0.8) -> np.ndarray:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def _jpeg_variant(self, frame: np.ndarray, quality: int = 95) -> Optional[np.ndarray]:
        ok, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def _detect_with_rescue_variants(self, frame: np.ndarray) -> Optional[Tuple[List[float], float]]:
        """在原始帧 miss 时，用轻量变体做二次抢救。"""
        for variant in (self._downup_variant(frame, scale=0.8), self._jpeg_variant(frame, quality=95)):
            if variant is None:
                continue
            det = self._detect_bbox(variant, self.model, self.class_candidates, use_track_mode=False)
            if det is not None:
                return det
        return None

    def _interpolate_center(self, frame_idx: int) -> Optional[Tuple[float, float]]:
        """参考 Badminton-Analysis 思路：对缺失中心点线性插值。"""
        if len(self._recent_obs) < 2:
            return None

        df = pd.DataFrame(self._recent_obs, columns=["frame", "x", "y"]).sort_values("frame")
        # 若 frame 不在已有范围，做轻微外推
        min_f, max_f = int(df["frame"].min()), int(df["frame"].max())
        if frame_idx < min_f - 5 or frame_idx > max_f + 5:
            return None

        fx = interp1d(df["frame"], df["x"], kind="linear", fill_value="extrapolate")
        fy = interp1d(df["frame"], df["y"], kind="linear", fill_value="extrapolate")

        x = float(fx(frame_idx))
        y = float(fy(frame_idx))
        return x, y

    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[Optional[float], Optional[float]]:
        """检测单帧羽毛球，返回中心点 (cx, cy) 像素坐标。"""
        # 1) 主模型（Badminton-Analysis 专用 shuttle）
        det = self._detect_bbox(frame, self.model, self.class_candidates, use_track_mode=self.use_track_mode)
        # 2) 主模型 miss 时，尝试轻量 rescue 变体，补 raw 帧首帧/压缩差异 miss
        if det is None:
            det = self._detect_with_rescue_variants(frame)
        # 3) 仍 miss 时，辅助模型兜底（COCO sports ball=32）
        if det is None:
            det = self._detect_bbox(frame, self.fallback_model, {32}, use_track_mode=False)

        if det is not None:
            bbox, _ = det
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            self.boxes_by_frame[frame_idx] = bbox
            self.positions[frame_idx] = (cx, cy)
            self._recent_obs.append((frame_idx, cx, cy))

            # 更新速度状态
            cur = np.array([cx, cy], dtype=float)
            if self.prev_xy is not None:
                dv = cur - self.prev_xy
                sp = float(np.linalg.norm(dv))
                if sp > 120:  # 限制异常大跳
                    dv = dv / (sp + 1e-6) * 120
                self.prev_v = 0.7 * self.prev_v + 0.3 * dv
            self.prev_xy = cur

            self._last_frame_idx = frame_idx
            return cx, cy

        # 无检测：先用速度预测补点（保证轨迹连续）
        h, w = frame.shape[:2]
        if self.prev_xy is not None:
            pred = self.prev_xy + self.prev_v
            pred[0] = float(np.clip(pred[0], 0, w - 1))
            pred[1] = float(np.clip(pred[1], 0, h - 1))
            self.prev_xy = pred
            self.prev_v = 0.9 * self.prev_v
            cx, cy = float(pred[0]), float(pred[1])
            self.positions[frame_idx] = (cx, cy)
            self._last_frame_idx = frame_idx
            return cx, cy

        # 仍无法预测时，再尝试历史线性插值
        interp = self._interpolate_center(frame_idx)
        if interp is not None:
            cx, cy = interp
            cx = float(np.clip(cx, 0, w - 1))
            cy = float(np.clip(cy, 0, h - 1))
            self.positions[frame_idx] = (cx, cy)
            self.prev_xy = np.array([cx, cy], dtype=float)
            self._last_frame_idx = frame_idx
            return cx, cy

        return None, None

    def to_dataframe(self) -> pd.DataFrame:
        data = [{"frame": f, "x": x, "y": y} for f, (x, y) in self.positions.items()]
        if not data:
            return pd.DataFrame(columns=["frame", "x", "y"])
        return pd.DataFrame(data).sort_values("frame")


class ShuttleInterpolator:
    """羽毛球轨迹插值器 - 填补漏检"""

    def __init__(self, method: str = "linear"):
        self.method = method

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 2:
            return df

        min_frame = int(df["frame"].min())
        max_frame = int(df["frame"].max())
        full_frames = range(min_frame, max_frame + 1)

        fx = interp1d(df["frame"], df["x"], kind=self.method, fill_value="extrapolate")
        fy = interp1d(df["frame"], df["y"], kind=self.method, fill_value="extrapolate")

        result_df = pd.DataFrame({"frame": list(full_frames)})
        result_df["x"] = fx(result_df["frame"])
        result_df["y"] = fy(result_df["frame"])
        result_df["is_interpolated"] = ~result_df["frame"].isin(df["frame"])
        return result_df

    def refine_temporal(self, df: pd.DataFrame, court_top: float = 0.4) -> pd.DataFrame:
        df = df.copy()
        df["dx"] = df["x"].diff()
        df["dy"] = df["y"].diff()
        df["speed"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

        max_speed = 50
        df.loc[df["speed"] > max_speed, "x"] = np.nan
        df.loc[df["speed"] > max_speed, "y"] = np.nan
        df["x"] = df["x"].interpolate()
        df["y"] = df["y"].interpolate()
        return df
