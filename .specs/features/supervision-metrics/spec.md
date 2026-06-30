# Feature: Supervision-based metrics

Replace `pycocotools` with the `supervision` (0.27) package as the source of
truth for `ensemble/metrics.py::evaluate`, producing the four numbers reported
in `summary.csv`: Precision, Recall, mAP@50, mAP@50-95.

Branch: `feat/supervision-metrics` (worktree off `main_working`).

## Requirements

- **R1 — mAP via supervision.** Compute mAP@50 and mAP@50-95 with
  `supervision.metrics.MeanAveragePrecision` (`class_agnostic=False`) on the same
  `score_threshold`-filtered prediction set used for P/R (see R3). Read `.map50`
  and `.map50_95`.
- **R2 — Precision/Recall via supervision.** Compute Precision and Recall with
  `supervision.metrics.Precision` / `Recall` using defaults
  (`averaging_method=WEIGHTED`); take the scalar at IoU=0.50
  (`precision_at_50` / `recall_at_50`).
- **R3 — Single operating-point filter (per-model baselines).** `evaluate()` takes
  one `score_threshold`; ALL four metrics (mAP + P/R) are computed on the
  predictions filtered to `score >= score_threshold`. Standalone models use their
  config-independent per-model `default_conf` (RFDETR=0.5, YOLO=0.25, DEIMv2=0.45)
  so baselines stay fixed regardless of ensemble tuning. The ensemble uses
  `score_threshold=0.0` because its predictions are already WBF-filtered at
  `wbf_skip_box_thr` (so the ensemble row reflects the config). Negative mAP
  (Supervision's empty-target sentinel `-1`) is clamped to `0.0`.
- **R4 — In-memory ground truth.** Use `bundle.targets` (`dict[int, sv.Detections]`).
  Do **not** use `pycocotools`, `bundle.annotations_path`, or re-read JSON for
  evaluation. Build per-image prediction `sv.Detections` and align with
  `bundle.targets` by `image_id` (an empty `sv.Detections` carrying a `(0,)`
  confidence array where a model produced none — `sv.Detections.empty()` leaves
  `confidence=None`, which the metrics reject for predictions).
  `prediction.class_ids` are already in the `class_idx` space of
  `bundle.targets.class_id` (preserve existing assumption).
- **R5 — Stable public surface.** `EvalResult` keeps the exact shape
  (`precision, recall, map50, map50_95`). `predictions_to_coco_results()` and
  `write_coco_results_json()` stay unchanged. `evaluate()` gains a single
  `score_threshold` parameter (default `0.0`) — see R3.
- **R6 — Caller wiring.** Add `default_conf: float` to each `ModelSpec`
  (RFDETR=0.5, YOLO=0.25, DEIMv2=0.45). `ensemble/pipeline.py` passes
  `spec.default_conf` to `evaluate()` for standalone models and `0.0` for the
  ensemble.
- **R7 — Old file preserved.** Move the current `ensemble/metrics.py` content to
  `ensemble/metrics_old.py`; add `ensemble/metrics_old.py` to `.gitignore`;
  import it nowhere.
- **R8 — Dependency.** Keep `pycocotools==2.0.11` in `requirements.txt`
  (upstream cross-checks still need it).
- **R9 — Tests.** Add toy-data unit tests for `evaluate()`: perfect-match,
  partial-match, empty-predictions, empty-targets — asserting all four numbers.

## Out of scope

- Changing inference thresholds, WBF fusion, CSV format, or the Optuna objective.
- Removing pycocotools from the environment.
- Matching the old F1-max operating point (deliberately replaced by native
  supervision P/R — see grilled design decisions).

## Done when

- `pytest` passes in the `ensemble-method` conda env (new + existing tests green).
- `ensemble/metrics.py` imports `supervision`, not `pycocotools`.
- Atomic commits on `feat/supervision-metrics`.
