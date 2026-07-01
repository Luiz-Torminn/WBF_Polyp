# Tasks: Decouple Solo Model Metrics from the Ensemble Run

Legend: `[P]` = parallelizable. Each task lists Depends/Reuses/Done-when/Tests/Gate.

---

## T1 — Add `skip_solo_metrics` to config

- **What:** Add `skip_solo_metrics: bool = False` to `RunConfig`. Wire the default
  into the CLI config-build defaults so `RunConfig(**d)` keeps working.
- **Where:** `ensemble/config.py` (RunConfig), `ensemble/cli.py` (defaults dict ~L174).
- **Depends on:** —
- **Reuses:** existing default-field pattern in cli defaults dict.
- **Done when:** field exists with default `False`; `_serialize_run_config` /
  `_write_parameter_values` pick it up automatically via `fields(run)`; no
  construction site breaks.
- **Tests:** existing config/cli tests still pass; add assert that default is `False`.
- **Gate:** `pytest tests/test_cli_config.py tests/test_config_file.py -q`

## T2 — Mode-aware adapter factory + inference signature

- **What:** Change `_instantiate_adapter(model_key, run)` →
  `_instantiate_adapter(model_key, run, mode)` with `mode ∈ {"solo","ensemble"}`.
  YOLO: `iou = run.yolo_iou_threshold` when `mode=="ensemble"` else
  `DEFAULT_YOLO_IOU_THRESHOLD`. RFDETR/DEIMv2 ignore `mode`. Add
  `_inference_signature(model_key, run, mode) -> tuple` returning the
  inference-affecting params (YOLO: predict_threshold + iou; others:
  predict_threshold only → mode-independent).
- **Where:** `ensemble/pipeline.py`; import `DEFAULT_YOLO_IOU_THRESHOLD` from config.
- **Depends on:** —  **[P]** with T1.
- **Reuses:** existing per-model branch structure in `_instantiate_adapter`.
- **Done when:** both helpers exist; YOLO solo sig != ens sig when
  `yolo_iou_threshold != 0.75`; RFDETR/DEIMv2 sigs equal across modes.
- **Tests:** unit test on `_inference_signature` for the three keys × two modes.
- **Gate:** `pytest tests/ -q -k "signature or adapter"`

## T3 — Dual-pass model loop + conditional solo + prediction files

- **What:** Rework `ModelRunResult` to carry `ensemble_predictions` (always),
  `solo_predictions: dict|None`, `metrics: EvalResult|None`,
  `coco_results_path: Path|None`. Rewrite the per-model loop:
  1. Compute `sig_solo`, `sig_ens`; `distinct = sig_solo != sig_ens`.
  2. Always run the **ensemble** pass → `ens_preds`.
  3. If `run.skip_solo_metrics`: `solo_preds=None`, `metrics=None`, write no
     per-model file.
  4. Else: `solo_preds = (run a solo pass if distinct else ens_preds)`;
     `metrics = evaluate(solo_preds, bundle)`; write
     `predictions_<key>.json` (solo); if `distinct` also write
     `predictions_<key>_ensemble.json` (ens).
- **Where:** `ensemble/pipeline.py` (`ModelRunResult`, loop L244–278).
- **Depends on:** T1, T2.
- **Reuses:** `_run_inference`, `evaluate`, `predictions_to_coco_results`,
  `write_coco_results_json`.
- **Done when:** RFDETR/DEIMv2 run exactly one pass (reuse); YOLO runs two when
  `yolo_iou != 0.75`; correct files written per R4.
- **Tests:** covered by T6 integration assertions.
- **Gate:** `pytest tests/ -q`

## T4 — Update ensemble / summary / visualization consumers

- **What:**
  - `_run_ensemble`: read `model_results[k].ensemble_predictions` (was `.predictions`).
  - `_write_summary_csv`: wrap the solo-row loop in `if not run.skip_solo_metrics`
    so only the ENSEMBLE row is written when solo is skipped (R7); guard against
    `metrics is None`.
  - `_write_visualizations`: read `.solo_predictions`; skip models whose
    `solo_predictions is None`.
- **Where:** `ensemble/pipeline.py`.
- **Depends on:** T3.
- **Reuses:** existing writer bodies.
- **Done when:** normal run unchanged output shape; skip-solo run emits
  ENSEMBLE-only summary.
- **Gate:** `pytest tests/ -q`

## T5 — Optuna objective sets `skip_solo_metrics=True` [P]

- **What:** In the `dataclasses.replace(base_run, ...)` call inside the objective,
  add `skip_solo_metrics=True`. Confirm `result.ensemble_metrics.map50_95` still
  returned.
- **Where:** `bayesian_optimization.py` (~L82–96).
- **Depends on:** T1 (field must exist).  **[P]** with T3/T4.
- **Reuses:** existing `dataclasses.replace` call.
- **Done when:** trials run no solo pass; objective value path intact.
- **Gate:** `pytest tests/test_bayesian_optimization.py -q`

## T6 — Tests: solo invariance + reuse + skip-solo + R9 reconcile

- **What:**
  - New test: two `RunConfig`s differing only in `yolo_iou_threshold` produce
    **identical solo metrics** for YOLO (and RFDETR/DEIMv2) — the core
    acceptance. Use a lightweight fake adapter or small fixture; assert solo
    predictions independent of `yolo_iou_threshold`.
  - New test: `skip_solo_metrics=True` → summary has only the ENSEMBLE row and
    no per-model solo `predictions_*.json` written.
  - New test: `_inference_signature` reuse — RFDETR/DEIMv2 single pass.
  - **R9:** reconcile the two pre-existing `tests/test_bayesian_optimization.py`
    failures (stale `wbf_skip_box_thr` expectations from the intentional
    `wbf_skip_box` removal). Update the assertions to match the emitted config.
- **Where:** `tests/` (new + edit `test_bayesian_optimization.py`).
- **Depends on:** T3, T4, T5.
- **Gate:** `pytest tests/ -q` → all green.

## T7 — Docs/help for ensemble-only `yolo_iou` [P]

- **What:** Update CLI `--yolo-iou` help text and the config comments to state it
  applies to the **ENSEMBLE fusion input only**; solo uses the model default.
  Add a one-line note in the two config YAMLs.
- **Where:** `ensemble/cli.py` (help ~L116–120), `configs/optuna_best.yaml`,
  `configs/map5095.yaml`.
- **Depends on:** —  **[P]** with everything.
- **Gate:** `pytest -q` (no behavior change) + manual read.

---

## Execution order

```
T1 ─┐
T2 ─┼─→ T3 ─→ T4 ─→ T6
T5 ─┘ (needs T1)
T7  (independent)
```

## Acceptance verification (final)

Run the pipeline on the same dataset with both configs and diff solo rows:

```bash
conda activate ensemble-method
python main.py --config configs/optuna_best.yaml --dataset <.../test>
python main.py --config configs/map5095.yaml    --dataset <.../test>
# solo rows (RFDETR, YOLOv12, DEIMv2) must be byte-identical across the two summary.csv;
# ENSEMBLE row must differ.
```
