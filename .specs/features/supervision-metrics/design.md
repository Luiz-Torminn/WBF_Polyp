# Design: Supervision metrics + validation-default solo baselines

## Components touched

1. **`ensemble/metrics_old.py`** (new, gitignored) — verbatim copy of current
   `metrics.py` for reference.
2. **`ensemble/metrics.py`** (rewrite) — supervision-based `evaluate()`; drops
   pycocotools, `predictions_to_coco_results`, `write_coco_results_json`.
3. **`ensemble/config.py`** — add `VALIDATION_DEFAULTS` constant.
4. **`ensemble/pipeline.py`** — `_instantiate_adapter` gains `overrides`; model loop
   runs the gated second (validation-default) pass; COCO writes + `coco_results_path`
   removed; `_run_ensemble` COCO write removed.
5. **`tests/test_metrics.py`** (new) — toy-data `evaluate()` + override tests.
6. **`tests/test_summary_and_params.py`** — drop the dead `coco_results_path` kwarg.
7. **`.gitignore`** — add `ensemble/metrics_old.py`.

## metrics.py (new)

```python
def evaluate(predictions: dict[int, Prediction], bundle: CocoBundle) -> EvalResult:
    image_ids = sorted(bundle.targets.keys())
    if not image_ids:
        return EvalResult(0.0, 0.0, 0.0, 0.0)
    targets = [bundle.targets[i] for i in image_ids]
    preds = [_prediction_to_detections(predictions.get(i)) for i in image_ids]
    m = MeanAveragePrecision(class_agnostic=False).update(preds, targets).compute()
    p = Precision().update(preds, targets).compute()
    r = Recall().update(preds, targets).compute()
    return EvalResult(
        precision=_nonneg(p.precision_at_50), recall=_nonneg(r.recall_at_50),
        map50=_nonneg(m.map50), map50_95=_nonneg(m.map50_95),
    )
```

Helpers: `_empty_detections()` (xyxy/confidence/class_id all `(0,)`-shaped);
`_prediction_to_detections(pred)` → empty when `pred is None or len(pred)==0`, else
`sv.Detections(xyxy, confidence=scores, class_id=class_ids)` (no score filtering —
R3); `_nonneg(v) = max(0.0, float(v))`.

## config.py

```python
# Per-model VALIDATION-mode defaults — the params each upstream repo uses to
# report its published benchmark metrics (RFDETR/main.py, YOLO_model/main.py
# model.val(), DEIMv2 postprocessor/det_engine). Keys map to adapter kwargs.
VALIDATION_DEFAULTS: dict[str, dict[str, float | int]] = {
    "rfdetr": {"predict_threshold": 0.001},
    "yolo": {"predict_threshold": 0.001, "iou_threshold": 0.7, "imgsz": 640},
    "deimv2": {"score_threshold": 0.0},
}
```

## pipeline.py

`_instantiate_adapter(model_key, run, overrides=None)` — `overrides = overrides or {}`;
each adapter kwarg becomes `overrides.get("<kwarg>", run.<field>)`. YOLO gains an
explicit `imgsz=overrides.get("imgsz", 640)`.

Model loop per `spec`:
```python
adapter = _instantiate_adapter(spec.key, run)          # config pass (feeds WBF)
adapter.load(run.device)
try: predictions = _run_inference(adapter, bundle, run.batch_size)
finally: adapter.unload()

if run.dynamic_metrics:                                 # solo baseline (R6/R8)
    base = _instantiate_adapter(spec.key, run, VALIDATION_DEFAULTS[spec.key])
    base.load(run.device)
    try: base_preds = _run_inference(base, bundle, run.batch_size)
    finally: base.unload()
    metrics = evaluate(base_preds, bundle)
else:
    metrics = EvalResult(0.0, 0.0, 0.0, 0.0)            # unused; CSV quotes HARDCODED

model_results[spec.key] = ModelRunResult(spec=spec, predictions=predictions, metrics=metrics)
```

`ModelRunResult`: remove `coco_results_path`. Remove `predictions_to_coco_results` /
`write_coco_results_json` imports and all call sites (per-model + ensemble, incl. the
empty-ensemble write). `_run_ensemble` returns `(ensemble_metrics, ensemble_predictions)`
— drop the vestigial `ensemble_path` (verify no caller uses it; if `predictions_dir`
becomes unused, leave it — harmless, out of scope).

## Notes / accepted consequences

- **Double inference when dynamic.** Two passes per model. For RFDETR the two passes
  use identical params (both `0.001`) so results coincide; kept for uniformity and
  simplicity (grilled decision). YOLO (iou 0.7 vs config 0.75) and DEIMv2 (0.0 vs
  config `predict_threshold`) genuinely differ.
- **Low precision on solo rows.** At `conf=0.001` the full low-confidence tail is kept,
  so supervision's single-point weighted precision is lower than the published
  F1-max precision. This is the accepted apples-to-apples supervision semantics (AD-002).
- **class_id alignment** preserved from the old path (model 0-indexed class == bundle
  `class_idx`); no remapping.

## Test strategy

`tests/test_metrics.py` builds toy `CocoBundle`-like inputs. `evaluate()` only needs
`bundle.targets`, so tests pass a lightweight stub object exposing `.targets`
(`dict[int, sv.Detections]`) — no real dataset/models. Adapter-override test calls
`_instantiate_adapter(key, run, VALIDATION_DEFAULTS[key])` and asserts the private
threshold/iou/imgsz attributes. Reuse `parse_run_config` for a `run`. No GPU needed
(adapters aren't `.load()`-ed).
