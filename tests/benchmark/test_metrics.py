from src.eval.metrics import bbox_center, box_iou_xyxy, euclidean, normalized_point_error, pck


def test_box_iou_xyxy_identical():
    assert box_iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_box_iou_xyxy_non_overlap():
    assert box_iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_bbox_center():
    assert bbox_center([0, 0, 10, 20]) == (5.0, 10.0)


def test_euclidean():
    assert euclidean((0, 0), (3, 4)) == 5.0


def test_normalized_point_error():
    value = normalized_point_error((0, 0), (3, 4), 100, 100)
    assert 0 < value < 0.1


def test_pck():
    gt = [[10, 10, 1], [20, 20, 1], [0, 0, 0]]
    pred = [[11, 10, 1], [40, 20, 1], [0, 0, 0]]
    hits, total = pck(gt, pred, scale=20, threshold=0.1)
    assert hits == 1
    assert total == 2
