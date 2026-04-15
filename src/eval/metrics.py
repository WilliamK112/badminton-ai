from __future__ import annotations

import math
from typing import Iterable


def box_iou_xyxy(box_a: list[float] | tuple[float, float, float, float], box_b: list[float] | tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def bbox_center(box: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def normalized_point_error(p1: tuple[float, float], p2: tuple[float, float], image_width: int, image_height: int) -> float:
    diag = math.hypot(max(image_width, 1), max(image_height, 1))
    return euclidean(p1, p2) / diag if diag > 0 else 0.0


def pck(gt_keypoints: Iterable[list[float]], pred_keypoints: Iterable[list[float]], scale: float, threshold: float = 0.1) -> tuple[int, int]:
    hits = 0
    total = 0
    if scale <= 0:
        return 0, 0
    for gt, pred in zip(gt_keypoints, pred_keypoints):
        if len(gt) < 3 or len(pred) < 3:
            continue
        if gt[2] <= 0:
            continue
        total += 1
        dist = euclidean((gt[0], gt[1]), (pred[0], pred[1]))
        if dist <= threshold * scale:
            hits += 1
    return hits, total

