# Feature: Decouple Solo Model Metrics from the Ensemble Run

**Branch:** `feat/decouple-solo-ensemble-metrics`
**Status:** Specified
**Date:** 2026-06-30

## Problem

The `summary.csv` standalone (solo) rows change between config profiles
(`optuna_best.yaml` vs `map5095.yaml`) even though they are meant to be a
config-invariant baseline. Root cause: each model runs inference **once**, and
that single prediction set feeds **both** the solo metrics **and** the WBF
fusion input. The tuned `yolo_iou` (an ensemble-only NMS knob, e.g. 0.745 vs
0.303) therefore contaminates YOLO's solo row. RFDETR/DEIMv2 have no tuned
inference knob (only the shared `predict_threshold=0.001`), so their solo rows
are already invariant.

## Goal

Solo rows reflect each model's **native default** inference settings and are
**identical across config profiles**. Only the ENSEMBLE row reflects tuned
params. A real, apples-to-apples comparison summary.

## Success Criteria

Running the pipeline with `optuna_best.yaml` and with `map5095.yaml` on the same
dataset produces:

- **Identical solo rows for all three models** (RFDETR, YOLOv12, DEIMv2),
  including YOLO — verified by byte-equal solo rows in both `summary.csv` files.
- **ENSEMBLE rows that still differ** per config (tuned params still applied to
  fusion input).
- The Optuna objective value (`ensemble_metrics.map50_95`) is unchanged for an
  equivalent param set (no regression in the search target).
- `pytest` green for the changed areas (plus the two pre-existing
  `wbf_skip_box` failures reconciled — see R9).

## Requirements (traceable)

| ID | Requirement |
|----|-------------|
| **R1** | Solo inference uses each model's **native defaults**. For YOLO that is `iou = DEFAULT_YOLO_IOU_THRESHOLD (0.75)`. `predict_threshold` stays `0.001` for all models so solo mAP integrates the full PR curve. Solo settings are **not** config-driven. |
| **R2** | Ensemble fusion input uses the **configured/tuned** params. For YOLO that is `iou = run.yolo_iou_threshold`. |
| **R3** | Second inference pass is **conditional**: a model runs a solo pass only when its solo inference signature differs from its ensemble signature. When identical (RFDETR/DEIMv2 today), the single pass is reused for both. |
| **R4** | Persist `predictions_<key>.json` = **solo** set. When the solo and ensemble sets differ for a model (YOLO), also persist `predictions_<key>_ensemble.json` = the fusion-input set. When they are identical, write only `predictions_<key>.json`. |
| **R5** | Add `skip_solo_metrics: bool` (default `False`) to `RunConfig`. When `True`, skip the solo inference pass and solo metric evaluation entirely; only ensemble-relevant work runs. The Optuna objective sets it `True`. ENSEMBLE behavior/score is unaffected. |
| **R6** | Visualizations use the **solo** predictions (match the solo rows). When `skip_solo_metrics` is `True`, visualizations are already disabled in the Optuna path; keep that. |
| **R7** | Under `skip_solo_metrics=True`, `summary.csv` contains the **ENSEMBLE row only** (no solo rows). |
| **R8** | Mode-aware adapter construction: `_instantiate_adapter(key, run, mode)` with `mode ∈ {"solo","ensemble"}`, plus a typed `_inference_signature(key, run, mode)` returning the inference-affecting params. No stringly-typed param dicts. RFDETR/DEIMv2 solo numbers must be **bit-for-bit unchanged** from today. |
| **R9** | Reconcile the two pre-existing `tests/test_bayesian_optimization.py` failures (stale `wbf_skip_box_thr` assertions left over from `wbf_skip_box` being intentionally commented out). These live in the file this feature edits. |

## Out of Scope

- Re-enabling `wbf_skip_box` in the Optuna search (it was intentionally disabled).
- Generalizing solo/ensemble split to params beyond `iou` (none exist today;
  the signature mechanism is general but only `yolo_iou` diverges now).
- Avoiding YOLO's double model-load (nano loads fast; optimization deferred).

## Design Decisions (from grill session)

1. Solo semantics = **model native defaults**, `conf=0.001` retained. (R1)
2. Pass strategy = **conditional** via signature diff. (R3)
3. `yolo_iou` config key = **ensemble-only**; solo pinned to default. (R1/R2)
4. Persist **both** YOLO sets, clearly named. (R4)
5. **`skip_solo_metrics`** flag; Optuna sets `True`. (R5)
6. **Generalized** mode-aware factory + inference-signature diff. (R8)
7. Mechanism home = typed `_instantiate_adapter(key, run, mode)` +
   `_inference_signature(...)`. (R8)
