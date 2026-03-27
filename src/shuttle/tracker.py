"""
羽毛球追踪模块
"""
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from scipy.interpolate import interp1d


class ShuttleTracker:
    """羽毛球追踪器（YOLO class=32 + 时序预测 + motion fallback + 球场ROI）"""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        class_id: int = 32,  # COCO sports ball
        conf_threshold: float = 0.10,
        max_jump_norm: float = 0.22,
    ):
        self.model = YOLO(model_path)
        self.class_id = class_id
        self.conf_threshold = conf_threshold
        self.max_jump_norm = max_jump_norm

        self.positions = {}  # frame_idx -> (x, y)
        self.prev_xy = None  # normalized [0,1]
        self.prev_v = np.array([0.0, 0.0], dtype=float)
        self.prev_gray = None
        self.miss_count = 0
        self.max_pred_hold = 8  # 无观测时最多预测补点帧数

        # 主比赛区域（归一化）
        self.play_x_min = 0.10
        self.play_x_max = 0.90
        self.play_y_min = 0.05
        self.play_y_max = 0.96

    def _in_play_region(self, nxy) -> bool:
        nx, ny = float(nxy[0]), float(nxy[1])
        return self.play_x_min <= nx <= self.play_x_max and self.play_y_min <= ny <= self.play_y_max

    @staticmethod
    def _in_bad_corner(nx: float, ny: float) -> bool:
        return (
            (nx > 0.90 and ny > 0.82) or
            (nx < 0.10 and ny > 0.82) or
            (nx > 0.93 and ny < 0.20) or
            (nx < 0.08 and ny < 0.20)
        )

    def _motion_candidates(self, frame: np.ndarray):
        """从帧差+亮度提取羽毛球候选点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        if self.prev_gray is None:
            self.prev_gray = gray
            return [], gray

        diff = cv2.absdiff(gray, self.prev_gray)
        _, mot = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        mot = cv2.medianBlur(mot, 3)

        _, bri = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        cand_map = cv2.bitwise_and(mot, bri)

        num, _, stats, cents = cv2.connectedComponentsWithStats(cand_map, connectivity=8)
        cands = []
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < 1 or area > 120:
                continue
            cx, cy = cents[i]
            if not (80 <= cx <= w - 80 and 50 <= cy <= h - 40):
                continue
            nxy = np.array([float(cx) / w, float(cy) / h], dtype=float)
            if not self._in_play_region(nxy):
                continue
            if self._in_bad_corner(float(nxy[0]), float(nxy[1])):
                continue
            # 弱置信度给 fallback 用
            pseudo_conf = 0.18 if area <= 45 else 0.12
            cands.append((nxy, pseudo_conf, float(area), float(cx), float(cy)))

        self.prev_gray = gray
        return cands, gray

    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> tuple:
        """检测单帧羽毛球，返回像素坐标 (cx, cy) 或 (None, None)"""
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return None, None

        candidates = []

        # 1) YOLO 候选
        results = self.model(frame, verbose=False)[0]
        if results.boxes is not None and len(results.boxes) > 0:
            cls = results.boxes.cls.cpu().numpy().astype(int)
            bxy = results.boxes.xyxy.cpu().numpy()
            conf = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else np.ones(len(bxy))

            for c, b, s in zip(cls, bxy, conf):
                if c != self.class_id or s < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = b
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                area = float(max((x2 - x1) * (y2 - y1), 1.0))
                nxy = np.array([cx / w, cy / h], dtype=float)
                if not self._in_play_region(nxy):
                    continue
                if self._in_bad_corner(float(nxy[0]), float(nxy[1])):
                    continue
                candidates.append((nxy, float(s), area, cx, cy, 'yolo'))

        # 2) fallback 候选（仅当 yolo 少或无时）
        mot_cands, _ = self._motion_candidates(frame)
        if len(candidates) <= 1:
            for nxy, s, area, cx, cy in mot_cands:
                candidates.append((nxy, s, area, cx, cy, 'motion'))

        if not candidates:
            # 无候选时，短时间用时序预测补点，避免轨迹断裂/可视化消失
            if self.prev_xy is not None and self.miss_count < self.max_pred_hold:
                pred = self.prev_xy + self.prev_v
                pred[0] = float(np.clip(pred[0], self.play_x_min, self.play_x_max))
                pred[1] = float(np.clip(pred[1], self.play_y_min, self.play_y_max))
                self.prev_xy = pred
                self.prev_v = 0.85 * self.prev_v  # 逐帧衰减
                self.miss_count += 1
                cx = float(pred[0] * w)
                cy = float(pred[1] * h)
                self.positions[frame_idx] = (cx, cy)
                return cx, cy
            return None, None

        # 初始帧：优先小目标+高置信度
        if self.prev_xy is None:
            candidates.sort(key=lambda x: (x[2], -x[1], 0 if x[5] == 'yolo' else 1))
            picked = candidates[0]
        else:
            pred = self.prev_xy + self.prev_v
            scored = []
            for nxy, s, area, cx, cy, src in candidates:
                dist = float(np.linalg.norm(nxy - pred))
                # 轨迹一致性为主，小面积/高置信度为辅；优先 yolo
                src_penalty = 0.03 if src == 'motion' else 0.0
                score = dist + 0.01 * area / (w * h) - 0.03 * s + src_penalty
                scored.append((score, dist, nxy, s, cx, cy))

            scored.sort(key=lambda x: x[0])
            _, dist, nxy, s, cx, cy = scored[0]
            if dist > self.max_jump_norm:
                return None, None
            picked = (nxy, s, 0.0, cx, cy, 'pick')

        nxy, _, _, cx, cy, _ = picked
        self.miss_count = 0

        # 更新速度状态
        if self.prev_xy is not None:
            new_v = nxy - self.prev_xy
            sp = float(np.linalg.norm(new_v))
            if sp > 0.08:  # 防止速度爆炸
                new_v = new_v / (sp + 1e-6) * 0.08
            self.prev_v = 0.65 * self.prev_v + 0.35 * new_v
        self.prev_xy = nxy

        self.positions[frame_idx] = (cx, cy)
        return cx, cy

    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        data = []
        for frame_idx, (x, y) in self.positions.items():
            data.append({'frame': frame_idx, 'x': x, 'y': y})

        if not data:
            return pd.DataFrame(columns=['frame', 'x', 'y'])

        df = pd.DataFrame(data).sort_values('frame')
        return df


class ShuttleInterpolator:
    """羽毛球轨迹插值器 - 填补漏检"""

    def __init__(self, method: str = 'linear'):
        self.method = method

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 2:
            return df

        min_frame = int(df['frame'].min())
        max_frame = int(df['frame'].max())
        full_frames = range(min_frame, max_frame + 1)

        fx = interp1d(df['frame'], df['x'], kind=self.method, fill_value='extrapolate')
        fy = interp1d(df['frame'], df['y'], kind=self.method, fill_value='extrapolate')

        result_df = pd.DataFrame({'frame': list(full_frames)})
        result_df['x'] = fx(result_df['frame'])
        result_df['y'] = fy(result_df['frame'])
        result_df['is_interpolated'] = ~result_df['frame'].isin(df['frame'])
        return result_df

    def refine_temporal(self, df: pd.DataFrame, court_top: float = 0.4) -> pd.DataFrame:
        df = df.copy()
        df['dx'] = df['x'].diff()
        df['dy'] = df['y'].diff()
        df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2)

        max_speed = 50
        df.loc[df['speed'] > max_speed, 'x'] = np.nan
        df.loc[df['speed'] > max_speed, 'y'] = np.nan
        df['x'] = df['x'].interpolate()
        df['y'] = df['y'].interpolate()
        return df
