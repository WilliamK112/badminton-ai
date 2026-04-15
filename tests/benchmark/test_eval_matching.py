from scripts.eval_validation import match_player_boxes


def test_match_player_boxes_swaps_when_iou_is_better():
    gt = {
        "P1": {"bbox": [0, 0, 10, 10]},
        "P2": {"bbox": [20, 0, 30, 10]},
    }
    pred = {
        "1": {"bbox": [20, 0, 30, 10]},
        "2": {"bbox": [0, 0, 10, 10]},
    }
    matched = match_player_boxes(gt, pred)
    assert matched["P1"][0] == "2"
    assert matched["P2"][0] == "1"


def test_match_player_boxes_single_gt_uses_best_available_prediction():
    gt = {
        "P1": {"bbox": [0, 0, 10, 10]},
        "P2": {"bbox": None},
    }
    pred = {
        "1": {"bbox": [50, 50, 60, 60]},
        "2": {"bbox": [1, 1, 10, 10]},
    }
    matched = match_player_boxes(gt, pred)
    assert matched["P1"][0] == "2"
