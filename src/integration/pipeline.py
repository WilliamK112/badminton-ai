"""
整合模块 - 组合多个追踪器进行一体化分析
"""
import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from ..player.tracker import PlayerTracker, SmoothFilter
from ..shuttle.tracker import ShuttleTracker, ShuttleInterpolator
from ..pose.detector import PoseDetector, PoseVisualizer
from ..utils.io import save_json


@dataclass
class FrameData:
    """单帧数据结构"""
    frame_idx: int
    players: Dict[int, dict]  # track_id -> {bbox, keypoints}
    shuttle: Optional[Tuple[float, float]]  # (x, y)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass  
class RallyData:
    """单个回合数据结构"""
    rally_id: int
    start_frame: int
    end_frame: int
    winner: Optional[str]  # 'X' or 'Y'
    features: dict
    
    def to_dict(self) -> dict:
        return asdict(self)


class IntegratedPipeline:
    """整合分析流水线"""
    
    def __init__(
        self,
        player_model: str = "yolo11n.pt",
        shuttle_model: str = "yolo11n.pt",
        pose_model: str = "yolo11n-pose.pt",
        use_bbox_filter: bool = True
    ):
        # 初始化各个模块
        self.player_tracker = PlayerTracker(player_model)
        self.shuttle_tracker = ShuttleTracker(shuttle_model)
        self.pose_detector = PoseDetector(pose_model)
        self.pose_visualizer = PoseVisualizer()
        
        # 可选: 平滑滤波
        self.smooth_filter = SmoothFilter() if use_bbox_filter else None
        
        # 数据存储
        self.frame_data: List[FrameData] = []
        self.rallies: List[RallyData] = []
        
    def process_video(
        self,
        video_path: str,
        sample_interval: int = 10,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> List[FrameData]:
        """
        处理视频
        Args:
            video_path: 视频路径
            sample_interval: 采样间隔 (每N帧处理一次)
            output_path: 可选输出路径
        Returns:
            List[FrameData]
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 采样处理
            if frame_idx % sample_interval == 0:
                frame_data = self._process_frame(frame, frame_idx)
                self.frame_data.append(frame_data)
                
                if frame_idx % 100 == 0:
                    print(f"Processed frame {frame_idx}/{total_frames}")
            
            frame_idx += 1
            
            # 限制最大处理帧数（可选）
            if max_frames is not None and frame_idx >= max_frames:
                break
        
        cap.release()
        
        # 后处理: 检测回合
        self._detect_rallies()
        
        # 保存结果
        if output_path:
            self.save(output_path)
        
        return self.frame_data
    
    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> FrameData:
        """处理单帧"""
        # 1. 球员追踪
        player_tracks = self.player_tracker.track_frame(frame, frame_idx)
        
        # 2. 羽毛球追踪
        shuttle_pos = self.shuttle_tracker.detect_frame(frame, frame_idx)
        
        # 3. 骨架检测 (只在球员框内)
        players_with_pose = {}
        for track_id, info in player_tracks.items():
            bbox = info['bbox']
            keypoints = self.pose_detector.detect_in_box(frame, bbox)

            # 无论是否检测到骨骼，都保留球员框；keypoints 为空数组表示该帧骨骼缺失
            players_with_pose[track_id] = {
                'bbox': bbox,
                'keypoints': keypoints if keypoints else []
            }
        
        return FrameData(
            frame_idx=frame_idx,
            players=players_with_pose,
            shuttle=shuttle_pos if shuttle_pos[0] is not None else None
        )
    
    def _detect_rallies(self):
        """检测回合边界"""
        if not self.frame_data:
            return
        
        # 简单策略: 羽毛球位置在画面下方时为回合开始/结束
        # TODO: 更复杂的回合检测逻辑
        
        rally_id = 0
        rally_start = None
        
        for frame_data in self.frame_data:
            if frame_data.shuttle:
                y = frame_data.shuttle[1]
                
                # 假设画面下方是落点区域
                if y > 700:  # 落点阈值 (1080p)
                    if rally_start is not None:
                        # 回合结束
                        self.rallies.append(RallyData(
                            rally_id=rally_id,
                            start_frame=rally_start,
                            end_frame=frame_data.frame_idx,
                            winner=None,  # 需要计分系统
                            features={}
                        ))
                        rally_id += 1
                        rally_start = None
                else:
                    if rally_start is None:
                        rally_start = frame_data.frame_idx
    
    def visualize_frame(self, frame: np.ndarray, frame_data: FrameData) -> np.ndarray:
        """可视化单帧"""
        # 绘制球员框 + 标签（框上方）
        for track_id, info in frame_data.players.items():
            bbox = info['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            color = (255, 0, 255) if int(track_id) == 1 else (255, 120, 0)
            label = f"Player {int(track_id)}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 标签放在框上方
            tx, ty = x1, max(24, y1 - 10)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # 绘制骨架
            if 'keypoints' in info:
                frame = self.pose_visualizer.draw_skeleton(frame, info['keypoints'])

        # 绘制羽毛球 + 标签（点上方）
        if frame_data.shuttle:
            cx, cy = map(int, frame_data.shuttle)
            h, w = frame.shape[:2]
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            cv2.rectangle(frame, (cx - 11, cy - 11), (cx + 11, cy + 11), (0, 255, 0), 2)

            s_label = "SHUTTLE"
            (tw, th), _ = cv2.getTextSize(s_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            # 默认放在点上方偏右；若会出界则改到左侧
            stx = cx + 12
            if stx + tw > w - 4:
                stx = max(4, cx - tw - 12)
            sty = max(th + 4, cy - 14)

            cv2.putText(frame, s_label, (stx, sty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(frame, s_label, (stx, sty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame
    
    def save(self, path: str):
        """保存结果"""
        data = {
            'frames': [fd.to_dict() for fd in self.frame_data],
            'rallies': [r.to_dict() for r in self.rallies]
        }
        save_json(path, data)


class RallyAnalyzer:
    """回合分析器 - 提取特征用于预测"""
    
    @staticmethod
    def extract_features(frame_data_list: List[FrameData]) -> dict:
        """从帧数据中提取回合特征"""
        if not frame_data_list:
            return {}
        
        # 统计
        num_frames = len(frame_data_list)
        num_players_detected = sum(1 for fd in frame_data_list if fd.players)
        num_shuttle_detected = sum(1 for fd in frame_data_list if fd.shuttle)
        
        # 羽毛球轨迹统计
        shuttle_x = [fd.shuttle[0] for fd in frame_data_list if fd.shuttle]
        shuttle_y = [fd.shuttle[1] for fd in frame_data_list if fd.shuttle]
        
        features = {
            'num_frames': num_frames,
            'player_detection_rate': num_players_detected / num_frames,
            'shuttle_detection_rate': num_shuttle_detected / num_frames,
            'shuttle_x_mean': np.mean(shuttle_x) if shuttle_x else None,
            'shuttle_x_std': np.std(shuttle_x) if shuttle_x else None,
            'shuttle_y_mean': np.mean(shuttle_y) if shuttle_y else None,
            'shuttle_y_std': np.std(shuttle_y) if shuttle_y else None,
        }
        
        return features