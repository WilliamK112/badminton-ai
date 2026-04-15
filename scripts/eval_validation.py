#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import bbox_center, box_iou_xyxy, euclidean, normalized_point_error, pck


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate predictions against validation labels")
    p.add_argument("--labels", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-md", default=None)
    return p.parse_args()


def load_index(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[(row["video_path"], int(row["frame_idx"]))] = row
    return rows


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    s = sorted(values)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    return {
        "count": n,
        "mean": sum(s) / n,
        "median": med,
        "min": s[0],
        "max": s[-1],
    }


def match_player_boxes(gt_players: dict, pred_players: dict) -> dict[str, tuple[str | None, list[float] | None, float | None]]:
    gt_ids = [pid for pid in ["P1", "P2"] if gt_players.get(pid, {}).get("bbox")]
    pred_ids = [pid for pid, pdata in pred_players.items() if pdata.get("bbox")]
    if not gt_ids:
        return {}

    assignments = {pid: (None, None, None) for pid in ["P1", "P2"]}
    if len(gt_ids) == 1:
        gt_id = gt_ids[0]
        gt_box = gt_players[gt_id]["bbox"]
        best = None
        for pred_id in pred_ids:
            pred_box = pred_players[pred_id]["bbox"]
            iou = box_iou_xyxy(gt_box, pred_box)
            if best is None or iou > best[2]:
                best = (pred_id, pred_box, iou)
        assignments[gt_id] = best if best is not None else (None, None, None)
        return assignments

    p1_box = gt_players["P1"]["bbox"]
    p2_box = gt_players["P2"]["bbox"]

    pred1 = pred_players.get("1", {}).get("bbox")
    pred2 = pred_players.get("2", {}).get("bbox")
    if pred1 and pred2:
        direct = box_iou_xyxy(p1_box, pred1) + box_iou_xyxy(p2_box, pred2)
        swapped = box_iou_xyxy(p1_box, pred2) + box_iou_xyxy(p2_box, pred1)
        if swapped > direct:
            assignments["P1"] = ("2", pred2, box_iou_xyxy(p1_box, pred2))
            assignments["P2"] = ("1", pred1, box_iou_xyxy(p2_box, pred1))
        else:
            assignments["P1"] = ("1", pred1, box_iou_xyxy(p1_box, pred1))
            assignments["P2"] = ("2", pred2, box_iou_xyxy(p2_box, pred2))
        return assignments

    used = set()
    for gt_id in ["P1", "P2"]:
        gt_box = gt_players[gt_id].get("bbox")
        if not gt_box:
            continue
        best = None
        for pred_id in pred_ids:
            if pred_id in used:
                continue
            pred_box = pred_players[pred_id]["bbox"]
            iou = box_iou_xyxy(gt_box, pred_box)
            if best is None or iou > best[2]:
                best = (pred_id, pred_box, iou)
        if best is not None:
            used.add(best[0])
            assignments[gt_id] = best
    return assignments


def main() -> int:
    args = parse_args()
    labels = load_index(Path(args.labels).expanduser().resolve())
    preds = load_index(Path(args.predictions).expanduser().resolve())

    shuttle_errors_px = []
    shuttle_errors_norm = []
    shuttle_visible_total = 0
    shuttle_visible_pred = 0
    shuttle_hit_5 = shuttle_hit_10 = shuttle_hit_20 = 0

    player_ious = {"P1": [], "P2": []}
    player_center_errors = {"P1": [], "P2": []}
    player_recall_05 = {"P1": 0, "P2": 0}
    player_recall_075 = {"P1": 0, "P2": 0}
    player_total = {"P1": 0, "P2": 0}

    pose_hits_01 = pose_total_01 = 0
    pose_hits_02 = pose_total_02 = 0

    for key, gt in labels.items():
        pred = preds.get(key, {})
        width = int(gt.get("image_width") or pred.get("image_width") or 1)
        height = int(gt.get("image_height") or pred.get("image_height") or 1)

        gt_shuttle = gt.get("shuttle", {})
        pred_shuttle = pred.get("shuttle", {})
        if gt_shuttle.get("visible"):
            shuttle_visible_total += 1
            if pred_shuttle.get("visible") and pred_shuttle.get("x") is not None and pred_shuttle.get("y") is not None:
                shuttle_visible_pred += 1
                err = euclidean((gt_shuttle["x"], gt_shuttle["y"]), (pred_shuttle["x"], pred_shuttle["y"]))
                shuttle_errors_px.append(err)
                shuttle_errors_norm.append(normalized_point_error((gt_shuttle["x"], gt_shuttle["y"]), (pred_shuttle["x"], pred_shuttle["y"]), width, height))
                shuttle_hit_5 += int(err <= 5)
                shuttle_hit_10 += int(err <= 10)
                shuttle_hit_20 += int(err <= 20)

        pred_players = pred.get("players", {})
        gt_players = gt.get("players", {})
        matched_players = match_player_boxes(gt_players, pred_players)
        for pid in ["P1", "P2"]:
            gt_box = gt_players.get(pid, {}).get("bbox")
            pred_key, pred_box, matched_iou = matched_players.get(pid, (None, None, None))
            if gt_box:
                player_total[pid] += 1
                if pred_box:
                    iou = matched_iou if matched_iou is not None else box_iou_xyxy(gt_box, pred_box)
                    player_ious[pid].append(iou)
                    player_center_errors[pid].append(euclidean(bbox_center(gt_box), bbox_center(pred_box)))
                    player_recall_05[pid] += int(iou >= 0.5)
                    player_recall_075[pid] += int(iou >= 0.75)

        gt_pose = gt.get("pose", {})
        pred_pose = pred.get("pose", {})
        for pid in ["P1", "P2"]:
            gt_entry = gt_pose.get(pid)
            pred_slot, _pred_box, _matched_iou = matched_players.get(pid, (None, None, None))
            pred_entry = pred_pose.get(pred_slot) if pred_slot else None
            gt_box = gt_players.get(pid, {}).get("bbox")
            if not gt_entry or not pred_entry or not gt_box:
                continue
            if isinstance(pred_entry, list) and pred_entry and isinstance(pred_entry[0], list) and pred_entry and pred_entry and len(pred_entry[0]) and isinstance(pred_entry[0][0], list):
                pred_entry = pred_entry[0]
            scale = max(gt_box[2] - gt_box[0], gt_box[3] - gt_box[1])
            h01, t01 = pck(gt_entry, pred_entry, scale=scale, threshold=0.1)
            h02, t02 = pck(gt_entry, pred_entry, scale=scale, threshold=0.2)
            pose_hits_01 += h01
            pose_total_01 += t01
            pose_hits_02 += h02
            pose_total_02 += t02

    summary = {
        "shuttle": {
            "visible_total": shuttle_visible_total,
            "visible_predicted": shuttle_visible_pred,
            "visibility_recall": shuttle_visible_pred / shuttle_visible_total if shuttle_visible_total else None,
            "pixel_error": summarize(shuttle_errors_px),
            "normalized_error": summarize(shuttle_errors_norm),
            "accuracy_at_5px": shuttle_hit_5 / shuttle_visible_total if shuttle_visible_total else None,
            "accuracy_at_10px": shuttle_hit_10 / shuttle_visible_total if shuttle_visible_total else None,
            "accuracy_at_20px": shuttle_hit_20 / shuttle_visible_total if shuttle_visible_total else None,
        },
        "players": {
            pid: {
                "frames_labeled": player_total[pid],
                "mean_iou": summarize(player_ious[pid])["mean"],
                "iou_summary": summarize(player_ious[pid]),
                "center_error_px": summarize(player_center_errors[pid]),
                "recall_at_iou_0_5": player_recall_05[pid] / player_total[pid] if player_total[pid] else None,
                "recall_at_iou_0_75": player_recall_075[pid] / player_total[pid] if player_total[pid] else None,
            }
            for pid in ["P1", "P2"]
        },
        "pose": {
            "pck_at_0_1": pose_hits_01 / pose_total_01 if pose_total_01 else None,
            "pck_at_0_2": pose_hits_02 / pose_total_02 if pose_total_02 else None,
            "labeled_keypoints": pose_total_02,
        },
    }

    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else Path(args.predictions).with_suffix(".eval.json")
    out_md = Path(args.out_md).expanduser().resolve() if args.out_md else Path(args.predictions).with_suffix(".eval.md")
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    md = []
    md.append("# Validation Summary")
    md.append("")
    md.append("## Shuttle")
    md.append(f"- visible gt frames: {summary['shuttle']['visible_total']}")
    md.append(f"- visibility recall: {summary['shuttle']['visibility_recall']}")
    md.append(f"- mean pixel error: {summary['shuttle']['pixel_error']['mean']}")
    md.append(f"- accuracy@5px: {summary['shuttle']['accuracy_at_5px']}")
    md.append(f"- accuracy@10px: {summary['shuttle']['accuracy_at_10px']}")
    md.append(f"- accuracy@20px: {summary['shuttle']['accuracy_at_20px']}")
    md.append("")
    md.append("## Players")
    for pid in ["P1", "P2"]:
        info = summary['players'][pid]
        md.append(f"### {pid}")
        md.append(f"- labeled frames: {info['frames_labeled']}")
        md.append(f"- mean IoU: {info['mean_iou']}")
        md.append(f"- recall@0.5: {info['recall_at_iou_0_5']}")
        md.append(f"- recall@0.75: {info['recall_at_iou_0_75']}")
        md.append(f"- mean center error px: {info['center_error_px']['mean']}")
        md.append("")
    md.append("## Pose")
    md.append(f"- PCK@0.1: {summary['pose']['pck_at_0_1']}")
    md.append(f"- PCK@0.2: {summary['pose']['pck_at_0_2']}")
    md.append(f"- labeled keypoints: {summary['pose']['labeled_keypoints']}")
    out_md.write_text("\n".join(md))

    print(f"saved summary json -> {out_json}")
    print(f"saved summary md -> {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
