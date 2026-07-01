# Feature: Supervision-based metrics + validation-default solo baselines

Branch: `feat/supervision-metrics` (worktree off `main_working`).

## Goal

Replace `pycocotools` with the `supervision` (0.27.0.post1) package as the single
metric engine for the four `summary.csv` numbers — **Precision, Recall, mAP@50,
mAP@50-95** — and make each standalone-model row an **apples-to-apples baseline**:
the model evaluated at its *own* validation-mode defaults, versus the ENSEMBLE row
evaluated on the config/WBF-fused predictions. This lets us read how WBF influences
results against each model's honest solo baseline.

## Background (verified from upstream sources)

Validation-mode defaults come from the upstream `main.py` files that produce each
model's published benchmark numbers (Knowledge-chain Step 1, authoritative):

| Model  | Upstream validation call                                   | Validation-mode defaults           |
|--------|------------------------------------------------------------|------------------------------------|
| RFDETR | `RFDETR/main.py:35` `PREDICT_THRESHOLD=0.001`, no NMS knob  | `threshold=0.001`                  |
| YOLO   | `YOLO_model/main.py:29` `model.val(conf=0.001, iou=0.7)`    | `conf=0.001, iou=0.7, imgsz=640`   |
| DEIMv2 | postproc `num_top_queries=300`, `det_engine.py:148` no thr | top-300 queries, `score_thr=0.0`   |

Supervision metric semantics (verified empirically on the installed version):
- `MeanAveragePrecision(class_agnostic=False)` → `.map50`, `.map50_95` (IoU 0.50:0.95).
- `Precision()` / `Recall()` (default `averaging_method=WEIGHTED`) → `.precision_at_50`,
  `.recall_at_50`. These are `TP/(TP+FP)` and `TP/(TP+FN)` over **all** supplied
  detections at IoU=0.50 — there is **no F1-max confidence sweep** (deliberate
  departure from the old Ultralytics-style operating point; see AD-002).
- Empty targets → supervision returns the sentinel `-1`; clamp to `0.0`.
- Empty predictions must carry a real `confidence` array (build with
  `confidence=np.zeros((0,))`), not `sv.Detections.empty()`.

## Requirements

- **R1 — mAP via supervision.** `evaluate()` computes mAP@50 and mAP@50-95 with
  `supervision.metrics.MeanAveragePrecision(class_agnostic=False)` (`metric_target`
  default `BOXES`), reading `.map50` / `.map50_95`.
- **R2 — Precision/Recall via supervision.** `evaluate()` computes Precision and
  Recall with `supervision.metrics.Precision` / `Recall` at defaults
  (`averaging_method=WEIGHTED`), reading `.precision_at_50` / `.recall_at_50`.
- **R3 — Uniform calculation, predictions differ.** The metric *calculation* is
  identical (supervision defaults) for every row. Only the *predictions* fed in
  differ: solo rows use each model's validation-default pass; the ENSEMBLE row uses
  the config/WBF-fused predictions. `evaluate(predictions, bundle)` takes **no**
  threshold parameter.
- **R4 — In-memory ground truth.** Use `bundle.targets` (`dict[int, sv.Detections]`);
  do not use `pycocotools`, `bundle.annotations_path`, or re-read the annotations
  JSON. Build per-image prediction `sv.Detections` and align with `bundle.targets`
  by `image_id` over `sorted(bundle.targets.keys())`. An image with no predictions
  contributes an empty `sv.Detections` carrying a `(0,)` `confidence` array.
  `prediction.class_ids` are already in the same `class_idx` space as
  `bundle.targets.class_id` — preserve the existing assumption (no remapping;
  the previous pycocotools path relied on the same alignment). Clamp `-1` → `0.0`.
- **R5 — Stable public surface.** `EvalResult` keeps its exact shape
  (`precision, recall, map50, map50_95`). `evaluate()` keeps signature
  `evaluate(predictions: dict[int, Prediction], bundle: CocoBundle) -> EvalResult`.
- **R6 — Validation-default solo pass (second inference pass).** When
  `dynamic_metrics` is enabled, each standalone model is run a **second** time at
  its validation-mode defaults (table above) to produce its solo row. This is a
  real second inference pass (re-instantiate the adapter with default params), not
  a score-threshold filter of the config pass. Defaults live in a config constant
  keyed by model → adapter-kwarg overrides.
- **R7 — Config pass still feeds the ensemble.** The existing config pass
  (`RunConfig.predict_threshold`, `yolo_iou_threshold`, …) still runs once per model
  to produce the predictions that WBF fuses. The ENSEMBLE row is always computed
  from the fused predictions via `evaluate()`, regardless of `dynamic_metrics`.
- **R8 — dynamic_metrics gating.** When `dynamic_metrics=True`, run the per-model
  validation-default pass and populate live solo rows. When `False`, **skip** the
  per-model validation-default pass entirely (no extra inference) and quote
  `HARDCODED_METRICS` for the solo rows. The config pass and the ENSEMBLE row always
  run in both cases.
- **R9 — HARDCODED_METRICS unchanged.** Leave `HARDCODED_METRICS` exactly as-is,
  including the `-` placeholders for RFDETR/DEIMv2 P/R. Do not re-baseline.
- **R10 — Drop COCO JSON.** Remove `predictions_to_coco_results()` and
  `write_coco_results_json()` and their `predictions_*.json` artifact writes from the
  pipeline (verified: no downstream consumer — used only inside `pipeline.py`;
  `bayesian_optimization.py` reads only `ensemble_metrics.map50_95`; the
  `ModelRunResult.coco_results_path` field is dead/`"unused.json"`).
- **R11 — Solo pass output is metrics only.** The validation-default pass yields an
  `EvalResult` for the solo row; it persists no new predictions artifact.
- **R12 — Old file preserved.** Copy the current `ensemble/metrics.py` content
  verbatim to `ensemble/metrics_old.py`; add `ensemble/metrics_old.py` to
  `.gitignore`; import it nowhere.
- **R13 — Tests.** Toy-data unit tests for `evaluate()` (perfect, partial, empty
  predictions, empty targets), a test that the validation-default pass instantiates
  adapters with the correct override params, and updates keeping the existing
  summary/gating tests green.

## Out of scope

- Changing inference thresholds/WBF/CSV format/Optuna objective (beyond the metric
  swap and the added solo pass).
- Removing `pycocotools` from `requirements.txt` (upstream repos still need it).
- Re-baselining `HARDCODED_METRICS`.
- Renaming the Portuguese CSV headers (`Modelo`, `Precisão`) — existing name, flag
  as cleanup, do not change (naming Rule 73).

## Acceptance Criteria

- **AC1 (R1,R2,R4)** — `evaluate()` on a toy perfect-match set returns
  `precision=recall=map50=map50_95=1.0`.
- **AC2 (R1,R2)** — `evaluate()` on a toy set with 1 TP + 1 FP + 1 FN returns
  `precision≈0.5`, `recall≈0.5`, `map50≈0.5`.
- **AC3 (R4)** — `evaluate()` with predictions for images that have empty GT, and/or
  empty predictions, returns finite non-negative numbers (no crash, `-1` clamped to
  `0.0`).
- **AC4 (R5)** — `evaluate` imports `supervision`, not `pycocotools`; `EvalResult`
  shape and `evaluate` signature unchanged (no `score_threshold`).
- **AC5 (R6)** — the validation-default pass instantiates each adapter with the
  documented override params (RFDETR `predict_threshold=0.001`; YOLO
  `predict_threshold=0.001, iou_threshold=0.7, imgsz=640`; DEIMv2
  `score_threshold=0.0`).
- **AC6 (R8,R9)** — with `dynamic_metrics=False`, the summary quotes
  `HARDCODED_METRICS` (unchanged) and no per-model validation-default pass runs;
  the ENSEMBLE row is still computed.
- **AC7 (R10)** — `predictions_to_coco_results` / `write_coco_results_json` no longer
  exist and no `predictions_*.json` is written; `pytest` (incl.
  `test_summary_and_params`, `test_bayesian_optimization`) is green.
- **AC8 (R12)** — `ensemble/metrics_old.py` exists (verbatim old content) and is
  gitignored; nothing imports it.

## Done when

- `pytest` passes in the `ensemble-method` conda env (new + existing green).
- `ensemble/metrics.py` imports `supervision`, not `pycocotools`.
- Solo rows reflect validation-mode-default runs (when dynamic); ENSEMBLE reflects
  config+WBF; `dynamic_metrics=False` skips per-model default passes and quotes
  HARDCODED_METRICS.
- Atomic commits on `feat/supervision-metrics`.
