"""End-to-end orchestration: load → infer → fuse → evaluate → write artifacts.

The pipeline loads one model at a time, runs inference for the whole test
split, writes that model's COCO results JSON, evaluates it with the unified
evaluator, and unloads it. This serialization keeps total VRAM use bounded
by the largest of the three models (RTX 5080, 16 GB).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm.auto import tqdm

from ensemble.adapters import DEIMv2Adapter, RFDETRAdapter, YOLOAdapter
from ensemble.adapters.base import Adapter, Prediction
from ensemble.config import (
    ENSEMBLE_DISPLAY_NAME,
    MODEL_SPECS,
    ModelSpec,
    RunConfig,
)
from ensemble.data import CocoBundle, iter_batches, load_coco
from ensemble.fusion import fuse_image
from ensemble.metrics import (
    EvalResult,
    evaluate,
    predictions_to_coco_results,
    write_coco_results_json,
)
from ensemble.visualize import (
    select_visualization_records,
    write_combined_overlay,
    write_overlays,
)

logger = logging.getLogger("ensemble.pipeline")


@dataclass
class ModelRunResult:
    spec: ModelSpec
    predictions: dict[int, Prediction]
    metrics: EvalResult
    coco_results_path: Path


def _instantiate_adapter(model_key: str, run: RunConfig) -> Adapter:
    if model_key == "rfdetr":
        return RFDETRAdapter(
            weights_path=run.rfdetr_weights,
            predict_threshold=run.predict_threshold,
        )
    if model_key == "yolo":
        return YOLOAdapter(
            weights_path=run.yolo_weights,
            predict_threshold=run.predict_threshold,
            iou_threshold=run.yolo_iou_threshold,
            yolo_dir=run.yolo_weights.parent,
        )
    if model_key == "deimv2":
        return DEIMv2Adapter(
            weights_path=run.deimv2_weights,
            config_path=run.deimv2_config,
            deimv2_dir=run.deimv2_dir,
            score_threshold=run.predict_threshold,
        )
    raise ValueError(f"Unknown model key: {model_key!r}")


def _run_inference(
    adapter: Adapter, bundle: CocoBundle, batch_size: int
) -> dict[int, Prediction]:
    predictions: dict[int, Prediction] = {}
    for batch in tqdm(
        list(iter_batches(bundle.image_records, batch_size)),
        desc=f"infer:{adapter.name}",
    ):
        batch_preds = adapter.infer_batch(batch)
        for prediction in batch_preds:
            predictions[prediction.image_id] = prediction
    return predictions


def _serialize_run_config(
    run: RunConfig, bundle: CocoBundle, output_path: Path
) -> None:
    snapshot = {
        "dataset_dir": str(run.dataset_dir),
        "annotations_path": str(run.annotations_path),
        "output_dir": str(run.output_dir),
        "run_dir": str(run.run_dir),
        "device": run.device,
        "batch_size": run.batch_size,
        "predict_threshold": run.predict_threshold,
        "wbf_iou": run.wbf_iou,
        "wbf_skip_box_thr": run.wbf_skip_box_thr,
        "wbf_weights": list(run.wbf_weights),
        "yolo_iou_threshold": run.yolo_iou_threshold,
        "log_level": run.log_level,
        "skip_models": list(run.skip_models),
        "save_visualizations": run.save_visualizations,
        "visualization_count": run.visualization_count,
        "rfdetr_weights": str(run.rfdetr_weights),
        "yolo_weights": str(run.yolo_weights),
        "deimv2_weights": str(run.deimv2_weights),
        "deimv2_config": str(run.deimv2_config),
        "deimv2_dir": str(run.deimv2_dir),
        "num_images": len(bundle.image_records),
        "num_classes": bundle.num_classes,
        "class_names": bundle.category_names,
        "config_path": run.extra.get("config_path"),
        "config_overrides": run.extra.get("config_overrides", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)


def _setup_logging(run: RunConfig) -> Path:
    run.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = run.logs_dir / "pipeline.log"

    # ``getLevelName`` of an unknown string returns "Level <name>" rather than
    # raising; guard against that so a typo in LOG_LEVEL falls back to INFO
    # and surfaces a warning instead of silently disabling logging.
    level_value = logging.getLevelName(run.log_level)
    if not isinstance(level_value, int):
        logger.warning("Unknown log level %r, falling back to INFO", run.log_level)
        level_value = logging.INFO

    root = logging.getLogger("ensemble")
    root.setLevel(level_value)
    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
        for h in root.handlers
    ):
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        file_handler.setLevel(level_value)
        root.addHandler(file_handler)
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        )
        stream_handler.setLevel(level_value)
        root.addHandler(stream_handler)

    return log_path


def run_pipeline(run: RunConfig) -> Path:
    run.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = _setup_logging(run)
    logger.info("Run directory: %s", run.run_dir)
    logger.info("Log file: %s", log_path)

    # Mirror the startup config banner into the log file for reproducibility.
    banner = run.extra.get("config_banner")
    if banner:
        for line in banner.splitlines():
            logger.info("%s", line)

    bundle = load_coco(run.annotations_path, run.dataset_dir)
    logger.info(
        "Loaded COCO bundle: %d images, %d annotations, %d classes",
        len(bundle.image_records),
        len(bundle.raw.get("annotations", [])),
        bundle.num_classes,
    )

    _serialize_run_config(run, bundle, run.run_dir / "run.json")

    skip_set = {key.lower() for key in run.skip_models}
    active_specs = [spec for spec in MODEL_SPECS if spec.key not in skip_set]
    if not active_specs:
        raise RuntimeError("All models were skipped — nothing to evaluate.")

    model_results: dict[str, ModelRunResult] = {}
    for spec in active_specs:
        logger.info("=== %s (%s) ===", spec.display_name, spec.key)
        start = time.perf_counter()
        adapter = _instantiate_adapter(spec.key, run)
        adapter.load(run.device)
        try:
            predictions = _run_inference(adapter, bundle, run.batch_size)
        finally:
            adapter.unload()

        coco_results = predictions_to_coco_results(
            predictions, bundle.class_idx_to_cat_id
        )
        coco_path = run.predictions_dir / f"predictions_{spec.key}.json"
        write_coco_results_json(coco_results, coco_path)
        logger.info("Wrote %d COCO detections to %s", len(coco_results), coco_path)

        metrics = evaluate(predictions, bundle)
        logger.info(
            "%s metrics: P=%.4f R=%.4f mAP50=%.4f mAP50-95=%.4f (%.1fs)",
            spec.display_name,
            metrics.precision,
            metrics.recall,
            metrics.map50,
            metrics.map50_95,
            time.perf_counter() - start,
        )

        model_results[spec.key] = ModelRunResult(
            spec=spec,
            predictions=predictions,
            metrics=metrics,
            coco_results_path=coco_path,
        )

    ensemble_metrics, ensemble_predictions, ensemble_path = _run_ensemble(
        run=run,
        bundle=bundle,
        model_results=model_results,
        active_specs=active_specs,
    )

    summary_path = _write_summary_csv(
        run=run,
        active_specs=active_specs,
        model_results=model_results,
        ensemble_metrics=ensemble_metrics,
    )
    logger.info("Wrote summary CSV: %s", summary_path)

    if run.save_visualizations:
        _write_visualizations(
            run=run,
            bundle=bundle,
            model_results=model_results,
            ensemble_predictions=ensemble_predictions,
        )

    return summary_path


def _run_ensemble(
    *,
    run: RunConfig,
    bundle: CocoBundle,
    model_results: dict[str, ModelRunResult],
    active_specs: list[ModelSpec],
) -> tuple[EvalResult, dict[int, Prediction], Path]:
    if len(active_specs) < 2:
        logger.warning(
            "Ensemble requested but only %d active model(s); skipping fusion.",
            len(active_specs),
        )
        empty_predictions: dict[int, Prediction] = {}
        empty_metrics = EvalResult(precision=0.0, recall=0.0, map50=0.0, map50_95=0.0)
        empty_path = run.predictions_dir / "predictions_ensemble.json"
        write_coco_results_json([], empty_path)
        return empty_metrics, empty_predictions, empty_path

    logger.info("=== %s ===", ENSEMBLE_DISPLAY_NAME)
    start = time.perf_counter()

    weights = run.wbf_weights
    if len(weights) != len(active_specs):
        logger.info(
            "WBF weights length (%d) != active models (%d); falling back to equal weights.",
            len(weights),
            len(active_specs),
        )
        weights = tuple(1.0 for _ in active_specs)

    ensemble_predictions: dict[int, Prediction] = {}
    for record in tqdm(bundle.image_records, desc="fuse"):
        per_model = {
            spec.key: model_results[spec.key].predictions.get(
                record.image_id, Prediction.empty(record.image_id)
            )
            for spec in active_specs
        }
        ensemble_predictions[record.image_id] = fuse_image(
            image_id=record.image_id,
            width=record.width,
            height=record.height,
            predictions_by_model=per_model,
            weights=weights,
            iou_thr=run.wbf_iou,
            skip_box_thr=run.wbf_skip_box_thr,
        )

    coco_results = predictions_to_coco_results(
        ensemble_predictions, bundle.class_idx_to_cat_id
    )
    ensemble_path = run.predictions_dir / "predictions_ensemble.json"
    write_coco_results_json(coco_results, ensemble_path)
    logger.info(
        "Wrote %d COCO ensemble detections to %s", len(coco_results), ensemble_path
    )

    ensemble_metrics = evaluate(ensemble_predictions, bundle)
    logger.info(
        "%s metrics: P=%.4f R=%.4f mAP50=%.4f mAP50-95=%.4f (%.1fs)",
        ENSEMBLE_DISPLAY_NAME,
        ensemble_metrics.precision,
        ensemble_metrics.recall,
        ensemble_metrics.map50,
        ensemble_metrics.map50_95,
        time.perf_counter() - start,
    )
    return ensemble_metrics, ensemble_predictions, ensemble_path


def _write_summary_csv(
    *,
    run: RunConfig,
    active_specs: list[ModelSpec],
    model_results: dict[str, ModelRunResult],
    ensemble_metrics: EvalResult,
) -> Path:
    csv_path = run.run_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Modelo", "Precisão", "Recall", "MAP 50", "MAP 50-95"])
        for spec in active_specs:
            metrics = model_results[spec.key].metrics
            writer.writerow(
                [
                    spec.display_name,
                    f"{metrics.precision:.4f}",
                    f"{metrics.recall:.4f}",
                    f"{metrics.map50:.4f}",
                    f"{metrics.map50_95:.4f}",
                ]
            )
        writer.writerow(
            [
                ENSEMBLE_DISPLAY_NAME,
                f"{ensemble_metrics.precision:.4f}",
                f"{ensemble_metrics.recall:.4f}",
                f"{ensemble_metrics.map50:.4f}",
                f"{ensemble_metrics.map50_95:.4f}",
            ]
        )
    return csv_path


def _write_visualizations(
    *,
    run: RunConfig,
    bundle: CocoBundle,
    model_results: dict[str, ModelRunResult],
    ensemble_predictions: dict[int, Prediction],
) -> None:
    records = select_visualization_records(
        bundle.image_records, bundle.targets, run.visualization_count
    )
    if not records:
        logger.info("No images with ground truth — skipping visualizations.")
        return

    logger.info("Writing visualizations for %d images", len(records))
    for record in records:
        per_model = {
            spec.key: model_results[spec.key].predictions.get(
                record.image_id, Prediction.empty(record.image_id)
            )
            for spec in MODEL_SPECS
            if spec.key in model_results
        }
        per_model["ensemble"] = ensemble_predictions.get(
            record.image_id, Prediction.empty(record.image_id)
        )
        write_overlays(
            record=record,
            predictions=per_model,
            class_names=bundle.category_names,
            output_dir=run.visualizations_dir,
        )
        write_combined_overlay(
            record=record,
            predictions=per_model,
            class_names=bundle.category_names,
            output_dir=run.visualizations_dir,
        )
