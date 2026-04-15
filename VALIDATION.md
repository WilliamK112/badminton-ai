# Validation Framework

This repo now includes a full validation path for player, shuttle, and optional pose evaluation.

## What is included

- benchmark frame extraction
- manual annotation workflow for player boxes and shuttle points
- model inference on labeled frames
- automatic metric calculation and report generation

## Files

- `scripts/create_validation_manifest.py`
- `scripts/annotate_validation.py`
- `scripts/run_validation_inference.py`
- `scripts/bootstrap_validation_labels.py`
- `scripts/eval_validation.py`
- `scripts/render_validation_review.py`
- `scripts/run_benchmark_bundle.py`
- `scripts/select_validation_subset.py`
- `src/eval/metrics.py`
- `src/model_defaults.py`
- `validation/schema/frame_labels.schema.json`
- `validation/examples/frame_label.example.json`

## Recommended workflow

### 1. Extract benchmark frames

```bash
python scripts/create_validation_manifest.py \
  --video badminton_sample.mp4 \
  --out-dir validation/sample \
  --every 30 \
  --max-frames 20
```

Outputs:
- `validation/sample/manifest.jsonl`
- `validation/sample/frames/*.jpg`

### 2. Annotate player boxes and shuttle

```bash
python scripts/annotate_validation.py --manifest validation/sample/manifest.jsonl
```

Controls:
- `1` annotate `P1` box by drag
- `2` annotate `P2` box by drag
- `s` shuttle mode, left click to place shuttle
- `o` mark shuttle as occluded / not visible
- `c` clear current mode annotation
- `n` next frame
- `p` previous frame
- `w` save
- `q` quit

Default output:
- `validation/sample/labels.jsonl`

### 3. Run model inference on the labeled frames

```bash
.venv/bin/python scripts/run_validation_inference.py \
  --manifest validation/sample/labels.jsonl \
  --with-pose
```

Default output:
- `validation/sample/predictions.jsonl`

### 4. Score the predictions

```bash
python scripts/eval_validation.py \
  --labels validation/sample/labels.jsonl \
  --predictions validation/sample/predictions.jsonl
```

Outputs:
- `validation/sample/predictions.eval.json`
- `validation/sample/predictions.eval.md`

## Optional bootstrap workflow

If you want a faster review loop, first generate model predictions, then turn them into prefilled labels for later correction:

```bash
.venv/bin/python scripts/bootstrap_validation_labels.py \
  --manifest validation/sample/manifest.jsonl \
  --predictions validation/sample/predictions.jsonl
```

Output:
- `validation/sample/labels.bootstrap.jsonl`

Then review that file inside the annotator by copying it to `labels.jsonl` or passing it as your working labels file.

## Visual review output

You can also render a contact sheet for quick inspection:

```bash
.venv/bin/python scripts/render_validation_review.py \
  --input validation/sample/predictions.jsonl
```

Outputs:
- `validation/sample/review/*.jpg`

This is useful for fast regression review before doing detailed manual correction.

## One-command benchmark bundle

If you want a repeatable regression bundle from a video sample, run:

```bash
.venv/bin/python scripts/run_benchmark_bundle.py \
  --video badminton_sample.mp4 \
  --out-dir validation/bundles/fullcourt12 \
  --every 45 \
  --max-frames 24 \
  --indices 7,8,9,10,11,12,13,14,20,21,22,23 \
  --with-pose
```

Outputs include:
- sampled manifest
- predictions
- bootstrap labels
- review contact sheet
- `bundle.summary.json`

This is the fastest way to regenerate a benchmark package after tracker or model changes.

## Curating a cleaner benchmark subset

If a sampled benchmark includes close-ups, logo cards, or other non-rally frames, you can carve out a cleaner subset:

```bash
python scripts/select_validation_subset.py \
  --input validation/sample/manifest.jsonl \
  --out validation/sample/manifest.rally.jsonl \
  --indices 0,1,4,5,6
```

## Default model selection

The repo now prefers the local shuttle specialized weight when present:

- `models/weights/shuttle_best.pt`

Player detection still defaults to generic `yolo11n.pt` unless you explicitly pass `--player-model`, because the local `player_best.pt` may underperform on some sample videos.

## Metrics

### Shuttle
- visibility recall
- mean / median pixel error
- normalized error by image diagonal
- accuracy at 5px / 10px / 20px

### Players
- mean IoU for `P1` and `P2`
- recall at IoU 0.5 and 0.75
- mean center error in pixels

### Pose
- PCK@0.1
- PCK@0.2

Pose metrics are only computed when ground truth pose keypoints are present in the labels file.

## Design notes

- Ground truth labels use semantic player ids: `P1`, `P2`
- Predictions use tracker slots `1`, `2`, and the evaluator maps them to `P1`, `P2`
- This lets the current pipeline be evaluated without changing its runtime output format
- The framework is intended for repeated regression checks after model or tracker changes

## Practical recommendation

Start by labeling shuttle + players on 100 to 300 frames across several hard scenes.
That already gives a useful benchmark before investing in full pose keypoint annotation.
