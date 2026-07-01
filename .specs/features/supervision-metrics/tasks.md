# Tasks: Supervision metrics + validation-default solo baselines

Ordered so each commit leaves `pytest` green (no broken-import window). One atomic
commit per task. Gate = the stated verification passes in the `ensemble-method` env.

## T1 — Preserve old metrics (R12, AC8)
- Copy `ensemble/metrics.py` → `ensemble/metrics_old.py` (verbatim).
- Add `ensemble/metrics_old.py` to `.gitignore`; import it nowhere.
- **Gate:** `git check-ignore ensemble/metrics_old.py` succeeds; `grep -r metrics_old ensemble/ | grep import` empty; `pytest` green.
- **Commit:** `chore(metrics): preserve pycocotools evaluator as gitignored metrics_old.py`

## T2 — Config: VALIDATION_DEFAULTS (R6)
- Add `VALIDATION_DEFAULTS` constant to `ensemble/config.py` (rfdetr/yolo/deimv2 → adapter-kwarg overrides).
- **Gate:** `python -c "from ensemble.config import VALIDATION_DEFAULTS"` ok; `pytest` green.
- **Commit:** `feat(config): add per-model validation-mode default params`

## T3 — Pipeline: adapter overrides + gated second pass (R6, R7, R8, AC5)
- `_instantiate_adapter(model_key, run, overrides=None)`; each kwarg = `overrides.get(...) or run.<field>`. Add YOLO `imgsz` override.
- Model loop: keep config pass (feeds WBF); when `run.dynamic_metrics`, run a second
  validation-default pass and `evaluate()` it → solo `metrics`; else `EvalResult(0,0,0,0)`.
  (metrics.py still pycocotools here — evaluate signature unchanged.)
- Add `tests/test_metrics.py::test_instantiate_adapter_validation_overrides` (AC5) —
  asserts private threshold/iou/imgsz per model.
- **Gate:** new override test + full `pytest` green.
- **Commit:** `feat(pipeline): second validation-default pass for standalone rows`

## T4 — metrics.py → supervision (R1–R5, AC1–AC4)
- Rewrite `evaluate()` to use `supervision.metrics` (MeanAveragePrecision/Precision/Recall,
  defaults). Helpers `_empty_detections`, `_prediction_to_detections` (no filtering), `_nonneg`.
  Keep `EvalResult` shape and `evaluate(predictions, bundle)` signature. Keep
  `predictions_to_coco_results` / `write_coco_results_json` for now (removed in T5).
- Add `evaluate()` toy tests to `tests/test_metrics.py`: perfect (AC1), partial (AC2),
  empty-preds + empty-targets (AC3), and assert no `pycocotools` import (AC4).
- **Gate:** `tests/test_metrics.py` + full `pytest` green.
- **Commit:** `feat(metrics): compute P/R/mAP via supervision instead of pycocotools`

## T5 — Drop COCO JSON (R10, R11, AC7)
- Remove `predictions_to_coco_results` + `write_coco_results_json` from `metrics.py`.
- In `pipeline.py`: remove their imports + all call sites (per-model + ensemble +
  empty-ensemble write), remove `ModelRunResult.coco_results_path`, drop the vestigial
  `ensemble_path` from `_run_ensemble` (verify no other use).
- Fix `tests/test_summary_and_params.py` `_model_results()` — drop `coco_results_path` kwarg.
- **Gate:** `grep -rn "predictions_to_coco_results\|write_coco_results_json\|coco_results_path\|predictions_.*json" ensemble/ tests/` empty; full `pytest` green.
- **Commit:** `refactor(pipeline): drop COCO results JSON artifacts (unused downstream)`

## Verification (always-on, after T5)
Fresh Verifier pass (author ≠ verifier): spec-anchored AC check + discrimination
sensor + `validation.md`. Then user's own self-review + CodeRabbit workflow. No merge.
