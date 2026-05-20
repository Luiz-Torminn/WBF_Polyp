"""Argument parser and RunConfig builder for the ensemble pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from ensemble.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_DIR,
    DEFAULT_DEVICE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREDICT_THRESHOLD,
    DEFAULT_VISUALIZATION_COUNT,
    DEFAULT_WBF_IOU,
    DEFAULT_WBF_SKIP_BOX_THR,
    DEFAULT_YOLO_IOU_THRESHOLD,
    DEIMV2_CONFIG,
    DEIMV2_DIR,
    DEIMV2_WEIGHTS,
    MODEL_SPECS,
    RFDETR_WEIGHTS,
    RunConfig,
    YOLO_WEIGHTS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ensemble",
        description=(
            "Run RFDETR Nano, YOLOv12 Nano, and DEIMv2 Pico on a COCO test split, "
            "fuse their predictions with Weighted Box Fusion, and write a unified "
            "Modelo,Precisão,Recall,MAP 50,MAP 50-95 summary to .outputs/."
        ),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--predict-threshold",
        type=float,
        default=DEFAULT_PREDICT_THRESHOLD,
        help="Score threshold applied during per-model inference (default %(default)s).",
    )
    parser.add_argument("--wbf-iou", type=float, default=DEFAULT_WBF_IOU)
    parser.add_argument("--wbf-skip-box", type=float, default=DEFAULT_WBF_SKIP_BOX_THR)
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=DEFAULT_YOLO_IOU_THRESHOLD,
        help=(
            "IoU threshold passed to Ultralytics NMS during the ensemble's YOLO "
            "inference. High value (default %(default)s) = looser NMS, more "
            "candidates survive into WBF. Set to 0.7 to match Ultralytics' "
            "stock default; lower values are MORE aggressive NMS, not less."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        help=(
            "Logging verbosity (DEBUG/INFO/WARNING/ERROR). Resolved from the "
            "process env, then the project .env file, then this default."
        ),
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="WBF per-model weights in the order RFDETR YOLO DEIMv2 (default: equal).",
    )
    parser.add_argument(
        "--skip-models",
        type=str,
        nargs="+",
        choices=[spec.key for spec in MODEL_SPECS],
        default=[],
        help="Skip one or more standalone models; ensemble requires at least 2 active.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disable per-image overlay rendering.",
    )
    parser.add_argument(
        "--visualization-count",
        type=int,
        default=DEFAULT_VISUALIZATION_COUNT,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override the auto-generated run name (default: timestamp_ensemble).",
    )

    # Weight overrides — kept as opt-in flags rather than required arguments
    # so `python main.py` works out of the box from the project root.
    parser.add_argument("--rfdetr-weights", type=Path, default=RFDETR_WEIGHTS)
    parser.add_argument("--yolo-weights", type=Path, default=YOLO_WEIGHTS)
    parser.add_argument("--deimv2-weights", type=Path, default=DEIMV2_WEIGHTS)
    parser.add_argument("--deimv2-config", type=Path, default=DEIMV2_CONFIG)
    parser.add_argument("--deimv2-dir", type=Path, default=DEIMV2_DIR)
    return parser


def parse_run_config(argv: list[str] | None = None) -> RunConfig:
    args = build_parser().parse_args(argv)
    run_name = args.run_name or _default_run_name()
    weights = tuple(args.weights) if args.weights is not None else tuple(1.0 for _ in MODEL_SPECS)
    return RunConfig(
        dataset_dir=args.dataset,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        predict_threshold=args.predict_threshold,
        wbf_iou=args.wbf_iou,
        wbf_skip_box_thr=args.wbf_skip_box,
        wbf_weights=weights,
        skip_models=tuple(args.skip_models),
        save_visualizations=not args.no_visualizations,
        visualization_count=args.visualization_count,
        run_name=run_name,
        rfdetr_weights=args.rfdetr_weights,
        yolo_weights=args.yolo_weights,
        deimv2_weights=args.deimv2_weights,
        deimv2_config=args.deimv2_config,
        deimv2_dir=args.deimv2_dir,
        yolo_iou_threshold=args.yolo_iou,
        log_level=str(args.log_level).upper(),
    )


def _default_run_name() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "_ensemble"
