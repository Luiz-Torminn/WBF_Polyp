# ENSEMBLE_METHOD_PRD

Operational reference for the Weighted Box Fusion (WBF) ensemble pipeline in
this repository. Audience: software and machine-learning engineers working
on inference, evaluation, or ensemble experimentation.

## 1. Scope and Executive Summary

The pipeline runs three pretrained object detectors on the same COCO test
split, normalizes their per-image predictions into a shared format, fuses
them with Weighted Box Fusion via the `ensemble-boxes` package, and reports
Precisão, Recall, mAP@0.50, and mAP@[0.50:0.95] for all three standalone
models and the ensemble in a single CSV.

The current detection backends are:

- RFDETR Nano (consumed via the `rfdetr` Python package; weights at
  `RFDETR/rfdetr_nano.pth`)
- YOLOv12 Nano (consumed via Ultralytics `YOLO`; weights at
  `YOLO_model/yolo12n_ZScan2_ZScanKCLG_normais-80_10_10.pt`)
- DEIMv2 Pico (consumed via DEIMv2's `YAMLConfig` + `PostProcessor`; weights
  at `DEIMv2/deimv2_pico.pth`)

Single-class dataset (`polyp`). 1624 test images in COCO format at
`/run/media/luizlima/ED_NVME/Desktop/Coding/ZSCAN/datasets/folders/COCO/ZScan2_ZScanKCLG_normais/test/`.

The entrypoint is `python main.py` from the repository root. All artifacts
(predictions JSON, CSV summary, visualizations, run snapshot, log) are
written under `.outputs/<run_name>/`. The folder is gitignored.

## 2. Architecture and Bounded Responsibilities

```
new_ensemble/                 (repo root after merge to main_working)
├── main.py                   thin CLI wrapper
├── ensemble/                 (this project's owned code)
│   ├── config.py             defaults + RunConfig dataclass + repo discovery
│   ├── cli.py                argparse + RunConfig builder
│   ├── data.py               COCO loader, ImageRecord manifest, target dict
│   ├── adapters/
│   │   ├── base.py           Prediction dataclass + Adapter Protocol
│   │   ├── rfdetr_adapter.py wraps rfdetr.RFDETRNano
│   │   ├── yolo_adapter.py   wraps ultralytics.YOLO via predict()
│   │   └── deimv2_adapter.py builds DEIMv2 via YAMLConfig + PostProcessor
│   ├── fusion.py             WBF wrapper (normalize → fuse → denormalize)
│   ├── metrics.py            pycocotools COCOeval + best-F1 P/R extractor
│   ├── pipeline.py           orchestration: infer per model → fuse → CSV
│   └── visualize.py          per-image supervision overlays
├── requirements.txt          pinned direct dependencies
├── AGENTS.md                 project rules (workflow, weights layout, threshold semantics)
├── ENSEMBLE_METHOD_PRD.md    this document
├── .outputs/                 gitignored runtime artifacts
├── DEIMv2/  RFDETR/  YOLO_model/   upstream backends (untouched)
```

This project owns:

- Loading COCO ground truth and building the test-image manifest.
- Per-model adapters that load weights, run inference, and emit the shared
  Prediction contract.
- Weighted Box Fusion of per-model predictions.
- The unified evaluator that produces every number in `summary.csv`.
- Run orchestration, artifact serialization, and visualizations.

This project delegates to:

- `rfdetr.RFDETRNano` (RFDETR/) for RFDETR inference.
- `ultralytics.YOLO` (YOLO_model/) for YOLO inference.
- `engine.core.YAMLConfig`, `engine.deim.deim.DEIM`, and `engine.deim.postprocessor.PostProcessor` (DEIMv2/) for DEIMv2 inference.
- `pycocotools.cocoeval.COCOeval` for mAP computation.
- `ensemble_boxes.weighted_boxes_fusion` for fusion.
- `supervision.BoxAnnotator` / `LabelAnnotator` for overlays.

This project does NOT control:

- Upstream model code (architectures, postprocessors, default thresholds).
- Upstream native evaluation scripts (`RFDETR/main.py`, `YOLO_model/main.py`,
  `DEIMv2/train.py --test-only`). They still run from their own directories
  for cross-checks but are NOT the source of `summary.csv` numbers.
- Dataset annotation contents or file layout.

## 3. Module / Folder Structure

| Path | Owner | Description |
|---|---|---|
| `main.py` | this project | CLI entrypoint. Calls `cli.parse_run_config()` then `pipeline.run_pipeline()`. |
| `ensemble/config.py` | this project | Defaults, `find_repo_root()`, `RunConfig`. |
| `ensemble/cli.py` | this project | argparse builder + RunConfig assembly. |
| `ensemble/data.py` | this project | `load_coco`, `ImageRecord`, `CocoBundle`. |
| `ensemble/adapters/base.py` | this project | `Prediction`, `Adapter` Protocol. |
| `ensemble/adapters/rfdetr_adapter.py` | this project | RFDETR adapter. |
| `ensemble/adapters/yolo_adapter.py` | this project | YOLO adapter; injects `YOLO_model/` on sys.path. |
| `ensemble/adapters/deimv2_adapter.py` | this project | DEIMv2 adapter; rewrites stale `__include__` paths. |
| `ensemble/fusion.py` | this project | `fuse_image(...)`. |
| `ensemble/metrics.py` | this project | `evaluate(...)`, `predictions_to_coco_results(...)`. |
| `ensemble/pipeline.py` | this project | `run_pipeline(...)` orchestrator. |
| `ensemble/visualize.py` | this project | `write_overlays(...)`. |
| `DEIMv2/` `RFDETR/` `YOLO_model/` | upstream | Inference backends, gitignored. |

## 4. Core Flow

```
parse_run_config (cli.py)
    └── RunConfig
        └── pipeline.run_pipeline(run)
              ├── _setup_logging → .outputs/<run>/logs/pipeline.log
              ├── data.load_coco → CocoBundle (image manifest + targets)
              ├── _serialize_run_config → .outputs/<run>/run.json
              ├── for each active ModelSpec:
              │     ├── _instantiate_adapter(spec.key, run)
              │     ├── adapter.load(device)
              │     ├── _run_inference(adapter, bundle, batch_size)
              │     ├── predictions_to_coco_results + write_coco_results_json
              │     ├── metrics.evaluate(predictions, bundle)
              │     └── adapter.unload()                     # frees VRAM
              ├── _run_ensemble
              │     ├── for each image_id: fusion.fuse_image(...)
              │     ├── predictions_to_coco_results
              │     └── metrics.evaluate(ensemble_predictions, bundle)
              ├── _write_summary_csv → .outputs/<run>/summary.csv
              └── _write_visualizations → .outputs/<run>/visualizations/*.jpg
```

Each model is fully loaded, used, and unloaded before the next. With the
current backbones this keeps peak VRAM well under 16 GB on the RTX 5080.

## 5. CLI Contract

`python main.py` is the only supported entrypoint. The CLI is described in
`ensemble/cli.py`. All flags are optional; defaults live in
`ensemble/config.py`.

Most-used flags:

- `--dataset PATH` – COCO test split directory containing
  `_annotations.coco.json`.
- `--output-dir PATH` – where `.outputs/<run_name>/` is created.
- `--device str` – `cuda:0` by default.
- `--batch-size int` – default 8.
- `--predict-threshold float` – per-model score threshold applied during
  inference. Default `0.001` (matches RFDETR/YOLO native conventions).
- `--wbf-iou float` – default `0.7`.
- `--wbf-skip-box float` – default `0.25`.
- `--yolo-iou float` – Ultralytics NMS IoU threshold applied during the
  ensemble's YOLO inference. **High value = looser NMS, more candidates
  survive into WBF.** Default `0.99` (effectively disables YOLO-side
  dedup so WBF can do it instead). Set to `0.7` to reproduce
  `YOLO_model/main.py`. Lower values are MORE aggressive NMS, not less.
- `--log-level {DEBUG|INFO|WARNING|ERROR}` – verbosity for the
  `ensemble.*` loggers. Resolved at startup in this order: process env
  `LOG_LEVEL` → project-local `.env` file `LOG_LEVEL=…` → CLI default
  (`INFO`). `DEBUG` enables per-batch detection summaries, sample
  detections (top-5 by score) from each adapter's first batch, and a
  manifest summary from the COCO loader.
- `--weights w1 w2 w3` – WBF per-model weights, ordered RFDETR YOLO DEIMv2.
  Default: equal `[1, 1, 1]`.
- `--skip-models {rfdetr|yolo|deimv2}+` – drop one or more models from the
  run; ensemble row only emitted when at least two are active.
- `--no-visualizations` / `--visualization-count int` – overlay controls.
- `--run-name str` – override the auto `YYYYMMDD-HHMMSS_ensemble` name.

## 6. Shared Prediction Contract

`ensemble.adapters.base.Prediction`:

```
image_id  : int
xyxy      : np.ndarray (N, 4) float32, PIXEL coords on the original image
scores    : np.ndarray (N,)   float32
class_ids : np.ndarray (N,)   int64, 0-indexed model class id
```

Why pixel-space, not normalized:

- The COCO results JSON serializer expects pixel-space xywh.
- The visualizer draws boxes on the original (un-resized) image.
- The unified evaluator and the upstream native evaluators all consume
  pixel-space boxes. Normalization happens once, inside `fusion.fuse_image`,
  and is denormalized before leaving fusion.

`ensemble.data.ImageRecord` carries `(image_id, file_name, path, width,
height)` so adapters always have the original size on hand.

## 7. Per-Adapter Notes (Important Details)

### RFDETRAdapter
- File: `ensemble/adapters/rfdetr_adapter.py`.
- Mirrors `RFDETR/main.py` (`RFDETRNano(pretrain_weights=...)`,
  `optimize_for_inference(compile=False)`, `predict(...)` returning
  `sv.Detections`). `compile=False` is important — variable last-batch
  sizes otherwise trigger graph retraces.
- Output: `sv.Detections.xyxy / .confidence / .class_id` mapped 1:1 into
  `Prediction`. Returns up to 300 predictions per image at
  `threshold=0.001`.

### YOLOAdapter
- File: `ensemble/adapters/yolo_adapter.py`.
- Uses `YOLO.predict(...)` rather than `YOLO.val(...)` so detections come
  back keyed by image and can be aligned with the COCO `image_id`. The
  native `model.val(...)` flow in `YOLO_model/main.py` is intentionally
  preserved and is the source of the cross-check report (not of the CSV).
- `iou_threshold` (the Ultralytics `iou` arg for NMS) is wired to
  `RunConfig.yolo_iou_threshold` / `--yolo-iou`. Default `0.99` so
  YOLO-side NMS is effectively a no-op and WBF gets a rich candidate set.
  See §13 for the inversion gotcha (low iou = MORE aggressive NMS).
- Stale editable install workaround: the adapter injects
  `YOLO_model/` on `sys.path` before `import ultralytics`. The
  `__editable___ultralytics_*_finder.py` shipped with the conda env points
  at a previous workspace path and would otherwise fail with
  `ModuleNotFoundError`.

### DEIMv2Adapter
- File: `ensemble/adapters/deimv2_adapter.py`.
- Builds the model directly with `engine.core.YAMLConfig` and the
  `PostProcessor`, skipping the train.py / `det_solver` plumbing. The
  upstream val transforms (`Resize` to 640x640 + `ConvertPILImage(scale=True)`)
  are reimplemented inline using `torchvision.transforms.functional.to_tensor`.
- The DEIMv2 config has stale absolute `__include__` paths pointing at a
  previous workspace ("/run/.../ensemble-method/DEIMv2/..."). The adapter's
  `_materialize_config(...)` rewrites them — and any relative include — to
  absolute paths under the actual `DEIMv2/` directory and writes a temp
  config in the system temp dir. The upstream config file is never
  modified.
- Checkpoint selection follows DEIMv2's own
  `engine/solver/_solver.py:174-177` convention: prefer `state['ema']['module']`
  if EMA weights are present, otherwise fall back to `state['model']`. The
  loader filters to keys whose shapes match the freshly built model and
  passes `strict=False` so EMA-only buffers do not raise.
- Postprocessor convention: `orig_target_sizes` is `(B, 2)` with each row
  `[W, H]`. Verified against `DEIMv2/engine/data/dataset/coco_dataset.py:174`
  (`target["orig_size"] = torch.as_tensor([int(w), int(h)])`) and
  `DEIMv2/engine/deim/postprocessor.py:55`
  (`bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)`). Boxes
  returned by the postprocessor are already in xyxy pixel space on the
  original image.
- Candidate budget per image is capped by `num_queries × num_classes`
  (`DEIMv2/engine/deim/postprocessor.py:59`, `torch.topk(..., num_top_queries)`).
  For the polyp dataset (`num_classes=1`) the effective ceiling is
  `num_queries`. The leaf config `deimv2_hgnetv2_pico_coco.yml` ships with
  both set to `200`, so DEIMv2 hands WBF up to 200 detections per image
  vs. RFDETR's 300 and YOLO's `max_det` (300 with NMS effectively off).
  Raising `num_top_queries` above `num_queries × num_classes` will throw
  `RuntimeError: selected index k out of range` at first inference.

### Adapter logging
- All three adapters share `ensemble.adapters.base.log_batch_predictions(...)`.
  At DEBUG level it emits one line per batch with detection count and
  score min/mean/max, plus the top-5 detections (by score) from the first
  batch only — keeps the `pipeline.log` readable on the 1624-image split.
- Each adapter holds a `_first_batch_logged` flag flipped after the first
  call to `infer_batch`; samples are dumped only on that first call per
  adapter run.

## 8. Weighted Box Fusion Integration

`ensemble/fusion.py::fuse_image`:

1. Iterates over `predictions_by_model` in insertion order. The iteration
   order is the per-model order WBF assigns weights to; the pipeline
   populates the dict in `MODEL_SPECS` order — RFDETR, YOLO, DEIMv2 — so the
   `--weights` flag also follows that order.
2. Each model's boxes are divided by `[W, H, W, H]` and clipped to
   `[0, 1]` (defense against off-by-one drift from upstream postprocessors).
3. `weighted_boxes_fusion(boxes_list, scores_list, labels_list,
   weights=..., iou_thr=..., skip_box_thr=...)` is called once per image.
4. Fused boxes are scaled back by `[W, H, W, H]` to pixel space and packed
   into a `Prediction`.

If every model contributes zero boxes for an image, the fused prediction
is empty.

## 9. Unified Evaluator (the source of `summary.csv`)

This is the single source of truth for every row in the CSV — all three
standalone rows AND the ensemble row.

`ensemble/metrics.py::evaluate(predictions, bundle)`:

1. `predictions_to_coco_results(...)` builds an in-memory COCO results
   list `[{image_id, category_id, bbox=[x,y,w,h], score}, ...]`.
2. `COCO(annotations_path)` loads ground truth; `gt.loadRes(results_list)`
   loads detections.
3. `COCOeval(gt, dt, 'bbox')` runs `evaluate()` then `accumulate()` over
   all image ids in the GT. `summarize()` produces the canonical 12 stats.
4. `stats[1]` is taken as `MAP 50` and `stats[0]` as `MAP 50-95`.
5. `_precision_recall_at_best_f1(coco_eval)` walks the **raw**
   `coco_eval.evalImgs` list (one entry per `imgId × catId × areaRng × maxDet`),
   filters to `areaRng='all'` and `maxDet=params.maxDets[-1]` (100, the cap
   used for mAP), and pulls per-detection `(score, TP@IoU=0.50, ignore)`
   triples plus the non-ignored GT count. Detections are sorted by
   descending score; cumulative TP/FP build precision and recall curves at
   **detection resolution**, and `F1 = 2PR/(P+R)` is argmaxed to pick the
   operating point. P and R are written into the CSV as continuous floats —
   neither is snapped to the 101-point `recThrs` grid.

   This is intentionally NOT the matrix-based shortcut
   (`coco_eval.eval['precision'][T,R,K,A,M]`) — that path was the original
   implementation, and it silently quantized Recall to multiples of 0.01
   because the R axis is the fixed `params.recThrs` grid. The evalImgs
   walk is what Ultralytics' `ap_per_class` does conceptually (modulo a
   1000-point conf grid + EMA `smooth(..., 0.1)` they apply before
   argmax — see `YOLO_model/ultralytics/utils/metrics.py:839-840`). The
   only remaining methodological gap between our Recall and Ultralytics'
   `box.mr` is that smoothing pass; numbers usually agree within a few
   thousandths.

Why this evaluator and not each model's native one:

- Each model's native evaluator uses different conventions for the
  precision/recall operating point (Ultralytics uses best-F1, RFDETR's
  `supervision.metrics.MeanAveragePrecision` does not expose P/R the same
  way, DEIMv2's `CocoEvaluator` only emits the 12 COCO summary numbers).
  Mixing those numbers in one CSV would be misleading.
- pycocotools is also exactly the evaluator the ensemble row is judged
  against. Using it for the standalone rows guarantees apples-to-apples
  comparison.
- Each model's native evaluator still runs unchanged from its upstream
  directory (`cd RFDETR && python main.py`, etc.) for sanity cross-checks.

## 10. Output Artifacts

For every run the pipeline writes:

```
.outputs/<run_name>/
├── run.json                       resolved RunConfig snapshot (paths,
│                                  thresholds, WBF params, yolo_iou,
│                                  log_level, env, num images)
├── summary.csv                    Modelo,Precisão,Recall,MAP 50,MAP 50-95
│                                  (Recall is continuous — see §9)
├── predictions_rfdetr.json        COCO results list
├── predictions_yolo.json
├── predictions_deimv2.json
├── predictions_ensemble.json
├── visualizations/                per-image supervision overlays
│   ├── <image_stem>_rfdetr.jpg      blue boxes
│   ├── <image_stem>_yolo.jpg        green boxes
│   ├── <image_stem>_deimv2.jpg      amber boxes
│   ├── <image_stem>_ensemble.jpg    red boxes (thicker stroke)
│   └── <image_stem>_combined.jpg    all four overlaid in their colors;
│                                    per-model layers drawn thin without
│                                    labels, ensemble drawn thick with
│                                    labels for visual contrast
└── logs/pipeline.log              log file; verbosity honors LOG_LEVEL
                                   from .env / process env / --log-level
```

Per-model color palette lives in `ensemble.visualize.MODEL_COLORS`. The
combined overlay is written by `write_combined_overlay(...)` and is useful
for eyeballing which raw model boxes WBF merged vs. dropped.

`run_name` defaults to `YYYYMMDD-HHMMSS_ensemble`. Re-runs do not clobber
previous artifacts.

## 11. Environment Requirements

Conda env: `/home/luizlima/miniconda3/envs/ensemble-method` (Python 3.11).
Direct dependencies are pinned in `requirements.txt`. Critical packages and
versions:

- `ensemble-boxes==1.0.9`
- `pycocotools==2.0.11`
- `supervision==0.27.0.post1`
- `rfdetr==1.6.0` (editable from `RFDETR/`)
- `ultralytics==8.4.37` (editable from `YOLO_model/`)
- `torch>=2.2,<3.0`, `torchvision>=0.17`
- `numpy`, `pandas`, `Pillow`, `PyYAML`, `tqdm`

DEIMv2 is consumed as a source tree (no pip install) — the adapter injects
`DEIMv2/` on `sys.path`.

Expected layout when the project lives on `main_working`:

```
new_ensemble/
├── main.py
├── ensemble/
├── DEIMv2/
├── RFDETR/
└── YOLO_model/
```

When the project is checked out as a git worktree (filesystem may forbid
symlinks), a single-line `.repo_root` file with the absolute path of the
sibling-layout root is honored. The `ENSEMBLE_REPO_ROOT` environment
variable is honored as a last fallback. `.repo_root` is gitignored.

### `.env` file
A project-local `.env` at the repo root is read at import time by
`ensemble/config.py::_read_env_file`. The only key the pipeline currently
reads is `LOG_LEVEL` (default file: `LOG_LEVEL=DEBUG`). Resolution order
for the effective log level is: process env `LOG_LEVEL` → `.env` value →
`--log-level` flag → `INFO`. The parser is a tiny inline `KEY=VALUE` reader
to avoid a hard dependency on `python-dotenv`. `.env` is gitignored.

## 12. Validations Performed

- Adapter smoke: each adapter loads weights, runs inference on a 2-image
  batch, and returns at least one `Prediction` with sane pixel coordinates.
- Fusion: two near-identical boxes from two models fuse to a single
  averaged box; an empty third model does not break the call.
- Metric correctness: feeding the ground truth itself as predictions
  yields Precisão = Recall = mAP50 = mAP50-95 = 1.0.
- End-to-end: `python main.py` on the full 1624-image test split emits
  `summary.csv` with four rows of plausible metrics, plus per-model COCO
  JSONs, visualizations, and a run snapshot.
- Non-regression: `git status` on `DEIMv2/`, `RFDETR/`, `YOLO_model/`
  shows no changes introduced by this work.

Native cross-check commands (still functional unchanged):

```
cd RFDETR     && python main.py
cd YOLO_model && python main.py
cd DEIMv2     && python train.py -c deimv2_hgnetv2_pico_coco.yml --test-only -r deimv2_pico.pth
```

## 13. Threshold Semantics

Three thresholds, not to be conflated:

- **Inference / predict threshold** (`--predict-threshold`, default `0.001`)
  — applied per model during `infer_batch`. Kept tiny so the full PR curve
  survives for mAP. Same value used by `RFDETR/main.py`
  (`PREDICT_THRESHOLD`) and `YOLO_model/main.py` (`conf=0.001`).
- **WBF `skip_box_thr`** (`--wbf-skip-box`, default `0.25`) — applied
  inside `weighted_boxes_fusion(...)` only. Filters per-model boxes
  *before* fusion. Independent of the inference threshold.
- **YOLO NMS IoU** (`--yolo-iou`, default `0.99`) — Ultralytics' `iou` arg
  for NMS. **The semantics are inverted from intuition:** it is the IoU
  *above which* a same-class box is suppressed. So `0.99` ≈ no
  suppression (almost everything survives); `0.01` ≈ very aggressive
  suppression (almost nothing survives). We default high because WBF is
  the dedup step we trust; YOLO's NMS would just remove candidates that
  WBF would have merged. Override with `--yolo-iou 0.7` to reproduce
  `YOLO_model/main.py`'s native eval conditions.

There is also a fusion IoU (`--wbf-iou`, default `0.7`) which is the
threshold used by `weighted_boxes_fusion` to decide which per-model boxes
cluster together for averaging. Higher = boxes need to overlap more to be
fused; lower = looser fusion.

## 14. Operational Notes

- Re-runs are non-destructive: each `python main.py` invocation creates a
  fresh `.outputs/<run_name>/` directory.
- VRAM: peak usage is the largest single model on device at a time.
  RFDETR Nano is the largest of the three (~365 MB checkpoint, several GB
  activation memory at batch 8, 640×640).
- WBF tuning: by default each model contributes equally. To bias toward a
  stronger model use, e.g., `--weights 2 1 1`. To experiment with looser
  fusion try `--wbf-iou 0.6`. To filter pre-fusion noise use
  `--wbf-skip-box 0.05`.
- YOLO standalone vs. ensemble tradeoff: with `--yolo-iou 0.99`, YOLO's
  standalone CSV row will under-report relative to its native
  `YOLO_model/main.py` eval because pycocotools' greedy match labels the
  surviving near-duplicate boxes as FPs. The ENSEMBLE row benefits
  because WBF re-merges those duplicates. If a side-by-side YOLO row vs.
  `metrics.json` comparison is needed, either rerun with
  `--yolo-iou 0.7`, or run YOLO twice (once at 0.7 for the standalone
  row, once at 0.99 for the WBF feed) — the pipeline does NOT currently
  do the double-run.
- Skip a model on the fly: `python main.py --skip-models yolo` produces
  a two-model ensemble (RFDETR + DEIMv2).
- Verbose diagnostics: set `LOG_LEVEL=DEBUG` in `.env` (or pass
  `--log-level DEBUG`) to see per-adapter sample detections, batch
  detection counts, score distributions, and the COCO manifest summary
  emitted by `ensemble/data.py::load_coco`.

## 15. Anti-patterns

- Do NOT compute the standalone CSV rows with each model's native
  evaluator. Mixing evaluator semantics inside one CSV is the failure
  mode this project explicitly works around.
- Do NOT modify the upstream DEIMv2 config to fix paths in-place. Use the
  adapter's `_materialize_config(...)` which writes a temp copy; the
  upstream repo must stay clean so its own native eval flow keeps
  working.
- Do NOT call `weighted_boxes_fusion` on pixel-space boxes. The library
  requires normalized [0, 1] inputs; pass pixel boxes and you will get
  silent garbage.
- Do NOT pass DEIMv2 `orig_target_sizes` as `[H, W]`. The convention is
  `[W, H]`; flipping that flips x/y in the rescaled boxes.
- Do NOT load all three models simultaneously. The pipeline's
  load/infer/unload cadence is deliberate.

## 16. Alignment Rules and Ownership Boundaries

| Concern | Owned By | Notes |
|---|---|---|
| Inference correctness per model | upstream repo | We trust their forward + postprocessor outputs. |
| Box format compatibility | this project | Adapters convert to the shared pixel-space xyxy contract. |
| Normalization for WBF | this project | `fusion.py` only. |
| Score / class semantics | upstream repo | Single-class polyp dataset; class id is identity. |
| mAP computation | pycocotools | We call it identically for every row. |
| Best-F1 P/R selection | this project | `_precision_recall_at_best_f1` in `metrics.py`. |
| Output artifact schema | this project | CSV header is the contract; do not reorder columns. |
| Upstream repo cleanliness | this project | Adapters must not write into `DEIMv2/`, `RFDETR/`, `YOLO_model/`. |

## 17. Debugging Notes

- "Could not locate the upstream model directories" → either run the
  pipeline from the merged location where `DEIMv2/`, `RFDETR/`,
  `YOLO_model/` are siblings of `main.py`, or write the absolute path of
  that location into `.repo_root`, or export `ENSEMBLE_REPO_ROOT`.
- `ModuleNotFoundError: No module named 'ultralytics'` → the editable
  install pointer is stale; the YOLO adapter already injects
  `YOLO_model/` on sys.path. Verify the adapter ran `_ensure_ultralytics_importable(...)`
  (check logs).
- `FileNotFoundError: ...ensemble-method/DEIMv2/configs/...` → the
  DEIMv2 config's stale `__include__` paths were not rewritten. Verify
  the regex in `_materialize_config(...)` still matches the upstream
  prefix; if the upstream config was edited to use different absolute
  paths, extend the regex.
- DEIMv2 produces very wide or shifted boxes → `orig_target_sizes` is
  probably flipped to `[H, W]`. Confirm the call site in the adapter and
  cross-check `DEIMv2/engine/data/dataset/coco_dataset.py:174`.
- WBF produces zero fused boxes despite per-model predictions → check
  that `boxes_list` is a list of N lists (one per model) and that each
  inner list contains 4-float boxes in `[0, 1]`. Empty per-model lists
  are fine; mixed shapes are not.
- Recall in `summary.csv` looks suspiciously round (multiples of 0.01) →
  you are on an older revision of `ensemble/metrics.py` that used the
  precision-matrix shortcut (`coco_eval.eval['precision'][T,R,K,A,M]`).
  The current implementation walks `coco_eval.evalImgs` and returns
  continuous floats; if the Recall column shows e.g. exactly `0.8700`,
  re-pull and look for `_precision_recall_at_best_f1` referencing
  `evalImgs` (not `recThrs`).
- YOLO standalone Precision / Recall / mAP50 collapse vs. the previous
  run → expected if `--yolo-iou` was raised. With `iou=0.99` Ultralytics
  emits up to 300 near-duplicate boxes per image; pycocotools' greedy
  match marks all duplicates after the first as FPs, dragging the F1‑max
  operating point to a lower-recall region. The ENSEMBLE row should NOT
  collapse — if it does, WBF is not merging (verify `--wbf-iou` is not
  too high and that `wbf_skip_box_thr` is sane).
- DEIMv2 throws `RuntimeError: selected index k out of range` at first
  inference → `PostProcessor.num_top_queries` was raised above
  `DEIMTransformer.num_queries × num_classes`. Lower it to ≤ that
  product. For the polyp checkpoint that means ≤ `200 × 1 = 200`.
- No `[DEBUG]` lines in `logs/pipeline.log` despite `.env` setting
  `LOG_LEVEL=DEBUG` → the parser only honors the `KEY=VALUE` form. Check
  there's no leading whitespace, no surrounding quotes (or use matched
  quotes), and that the file is at the project root (next to `main.py`).
  The process env `LOG_LEVEL` takes precedence over the file — if it's
  set to something higher, `.env` won't win.
