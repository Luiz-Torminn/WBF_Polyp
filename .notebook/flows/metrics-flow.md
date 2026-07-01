# Metrics Flow → summary.csv
> How Precisão/Recall/mAP50/mAP50-95 are computed and written, and what `dynamic_metrics` toggles

Entry: `ensemble/pipeline.py:run_pipeline()`
Flow: per-model infer → `metrics.py:evaluate()` → fuse (WBF) → `metrics.py:evaluate()` on ensemble → `pipeline.py:_write_summary_csv()`

## `dynamic_metrics` flag
- Default `True` (`ensemble/config.py:129` DEFAULT_DYNAMIC_METRICS). Settable via CLI (`cli.py:69,181`) + YAML (`config_file.py:86`).
- Controls ONLY the 3 standalone-model rows of `summary.csv`:
  - `True`  → rows use live `EvalResult` from this run (`pipeline.py:388-395`)
  - `False` → rows copied from frozen `config.py:156` HARDCODED_METRICS (can be `"-"` for RFDETR/DEIMv2 P/R)
- ENSEMBLE row ALWAYS live-computed regardless of flag (`pipeline.py:405-415`).
- Note: `evaluate()` runs per model unconditionally (`pipeline.py:262`); flag only decides if result is *used*.

## The 4 numbers — `metrics.py:evaluate()` (L149)
predictions → `predictions_to_coco_results()` (L47, maps class_idx→cat_id via bundle) → `_build_coco_eval()` (L74, pycocotools COCOeval, loadRes + evaluate + accumulate)
- **mAP50-95** = `coco_eval.stats[0]`; **mAP50** = `stats[1]` (after `.summarize()`, L168-170)
- **P / R** = `_precision_recall_at_best_f1()` (L91): NOT from COCO precision matrix (would quantize recall to 101-pt grid). Walks `coco_eval.evalImgs` raw per-detection (score, TP@IoU0.50, ignore) + total_gt, sorts by desc score, cumulative TP/FP curve, returns (P,R) at **F1-argmax** point. Matches Ultralytics box.mp/box.mr. Area 'all', largest maxDets.
- No detections or no GT → returns (0.0, 0.0).

## Ensemble row — `pipeline.py:_run_ensemble()` (L306)
WBF-fuses per-model preds (`fusion.py:fuse_image`) → same `evaluate()` → ENSEMBLE row.
- <2 active models → short-circuits to zeroed metrics (L313-322).

## Config values that move the numbers (all via RunConfig)
- `predict_threshold` = 0.001 (`config.py:110`): tiny on purpose → full PR curve survives for honest mAP/F1. Raising it truncates curve.
- `skip_models`: skipped model = no row + excluded from fusion.
- `wbf_iou` (0.7), `wbf_skip_box_thr` (0.5), `wbf_weights`: affect ENSEMBLE row only. Weights length must == active count or falls back to equal (`pipeline.py:328`).
- `yolo_iou_threshold` (0.75): YOLO NMS; affects YOLO standalone (dynamic) + ENSEMBLE rows.
- dataset_dir/annotations (`config.py:195`): defines GT → total_gt + class_idx_to_cat_id.
- Resolution order: defaults → YAML → CLI. Provenance dumped to PARAMETER_VALUES.txt (`pipeline.py:152`).

## Gotcha
- CSV header + concept labels are Portuguese ("Modelo","Precisão") — conflicts with Rule 73 (English naming). Existing name, flag-as-cleanup not change.
- metrics.py is SINGLE source of truth for standalone rows; upstream native evaluators run only for cross-check, don't feed CSV (see `metrics.py` docstring L18-21).

Updated: 2026-07-01
