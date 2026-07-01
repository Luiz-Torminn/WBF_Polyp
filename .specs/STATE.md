# Project State — supervision-metrics

## Decisions log

- **AD-001 — Supervision replaces pycocotools as the metric engine.** `evaluate()`
  uses `supervision.metrics` (MeanAveragePrecision/Precision/Recall) at defaults for
  all four numbers, uniformly for solo and ensemble rows. *Why:* single, modern,
  consistent metric source; GT already `sv.Detections` in `bundle.targets`.
- **AD-002 — Supervision-native P/R accepted (no F1-max).** Precision/Recall are
  supervision's single-point `TP/(TP+FP)` / `TP/(TP+FN)` at IoU 0.50 over all
  detections — NOT the old Ultralytics-style F1-max operating point. Consequence:
  solo precision at conf=0.001 is lower than published F1-point precision. *Why:*
  user chose "supervision defaults, uniform" for apples-to-apples comparison.
- **AD-003 — Solo rows = second inference pass at validation-mode defaults.** When
  `dynamic_metrics=True`, each model is re-run at its own validation defaults
  (RFDETR thr 0.001; YOLO conf 0.001/iou 0.7/imgsz 640; DEIMv2 score_thr 0.0 — all
  verified from upstream `main.py`). NOT a filter of the config pass; NOT predict-mode
  defaults. *Why:* honest per-model benchmark baseline vs the WBF ensemble.
- **AD-004 — Config pass still feeds WBF; ENSEMBLE always computed.** The
  `predict_threshold` (0.001) config pass produces the predictions WBF fuses; the
  ENSEMBLE row is always `evaluate()`-d regardless of `dynamic_metrics`.
- **AD-005 — dynamic_metrics=False skips the per-model validation pass.** Solo rows
  quote `HARDCODED_METRICS` (left AS-IS incl. `-` placeholders; not re-baselined).
- **AD-006 — Drop COCO results JSON.** Remove `predictions_to_coco_results` /
  `write_coco_results_json` and `predictions_*.json` writes. *Why:* no downstream
  consumer (verified); pycocotools gone from the metric path.
- **AD-007 — Discarded prior attempt (was worktree HEAD 369c3fe).** Earlier take used
  single-pass score filter, predict-mode default_conf (0.25/0.5/0.45), kept COCO, no
  dynamic gating — superseded by today's grilled decisions. Recoverable via reflog.

## Handoff snapshot

- Branch `feat/supervision-metrics` off `main_working` (`680f3d7`), worktree at
  `../working_trees/supervision-metrics`, `.repo_root` → real repo root (Rule 22).
- Spec/design/tasks written under `.specs/features/supervision-metrics/`.
- Env: `conda activate ensemble-method`; run `pytest` from the worktree root.
- Next: execute T1→T5 inline (sequential, interdependent — fanout not beneficial),
  then always-on Verifier, then user self-review + CodeRabbit. No merge.
