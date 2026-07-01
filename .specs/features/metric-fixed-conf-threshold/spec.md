# Spec — Fixed-confidence operating point for Precision/Recall

## Problem

`summary.csv` reports near-zero precision for the DETR models (RFDETR 0.0036,
DEIMv2 0.0053) and depressed precision for YOLO (0.3241). Root cause: `evaluate()`
computes supervision Precision/Recall as a single point over the **entire**
unfiltered prediction set, and that set is produced at a near-zero confidence
threshold (validation defaults: RFDETR/YOLO `conf=0.001`, DEIMv2 `score_thr=0.0`).
The DETR models emit a fixed ~300 queries per image with no NMS, so precision
collapses. mAP (0.89–0.92) confirms the models themselves are healthy — only the
single-point P/R is uninformative. This supersedes the deliberate no-operating-point
choice recorded in AD-002.

## Decision (chosen by user)

Option B: apply a **fixed confidence threshold of 0.5** to predictions before
computing Precision and Recall, so both are read at a meaningful operating point.
mAP is left untouched (it needs the full low-confidence tail for the PR curve).

## Verified facts

- **Supervision has no confidence-threshold parameter and no default.**
  `Precision.__init__(metric_target, averaging_method)` /
  `Recall.__init__(...)` expose no threshold; confidence is used only to *sort*
  detections (`supervision/metrics/precision.py:249`), never to filter. The only
  lever is to filter the `sv.Detections` before `.update()`. (supervision 0.27.0)
- `sv.Detections` supports boolean-mask filtering: `det[det.confidence >= thr]`
  returns a `Detections` that preserves the `confidence` array (incl. `(0,)` when
  empty), which the metrics require.

## Success criteria (acceptance)

- **AC1** — Precision and Recall are computed only over predictions with
  `confidence >= 0.5` (per image), using `>=` so a box exactly at 0.5 is kept.
- **AC2** — mAP@50 and mAP@50-95 are unchanged: computed over the full,
  unfiltered prediction set.
- **AC3** — The threshold is a named module constant defaulting to 0.5, and
  `evaluate()` accepts an optional keyword to override it (for tests / future use).
  No new CLI flag, no config plumbing (fixed value, per the chosen option).
- **AC4** — The filter is applied inside `evaluate()`, so it governs BOTH the
  standalone rows and the ENSEMBLE row identically (uniform operating point).
- **AC5** — A high-confidence true positive plus a low-confidence (<0.5) false
  positive yields precision 1.0 (FP excluded from P/R) while mAP still counts the
  FP from the full set.
- **AC6** — A true positive whose confidence is <0.5 is excluded from Recall
  (recall drops) but still counts toward mAP.
- **AC7** — Existing edge cases hold: no targets → zeros; empty predictions →
  zeros; empty-GT image sentinel clamps to 0.

## Out of scope

- Changing the inference thresholds, WBF params, or `HARDCODED_METRICS`.
- Best-F1 sweep (Option A) or dropping the P/R columns (Option C).
- Any change to mAP computation.
