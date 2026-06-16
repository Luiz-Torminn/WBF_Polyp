"""Argument parser and RunConfig builder for the ensemble pipeline.

Resolution order for every tunable value is::

    config.py default  ->  --config YAML file  ->  explicit CLI flag

The CLI flag always wins. To tell an explicitly-passed flag from one left at
its default, every override flag defaults to the private ``_UNSET`` sentinel
rather than to the ``config.py`` constant; the real defaults are layered in by
``ensemble.config_file.resolve``.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from ensemble.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_DIR,
    DEFAULT_DEVICE,
    DEFAULT_DYNAMIC_METRICS,
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
from ensemble.config_file import _format_value, load_config_file, render_banner, resolve

# Sentinel marking "flag not supplied on the command line".
_UNSET: Any = object()


def _bool_flag(value: str) -> bool:
    """Parse a ``true``/``false`` CLI value (case-insensitive)."""
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError(
        f"expected 'true' or 'false', got {value!r}"
    )

# Maps an argparse dest to (RunConfig field, transform-from-argparse-value).
_CLI_FIELD_MAP: dict[str, tuple[str, Any]] = {
    "dataset": ("dataset_dir", lambda v: v),
    "output_dir": ("output_dir", lambda v: v),
    "device": ("device", lambda v: v),
    "batch_size": ("batch_size", lambda v: v),
    "predict_threshold": ("predict_threshold", lambda v: v),
    "wbf_iou": ("wbf_iou", lambda v: v),
    "wbf_skip_box": ("wbf_skip_box_thr", lambda v: v),
    "yolo_iou": ("yolo_iou_threshold", lambda v: v),
    "weights": ("wbf_weights", tuple),
    "skip_models": ("skip_models", tuple),
    "dynamic_metrics": ("dynamic_metrics", lambda v: v),
    "visualization_count": ("visualization_count", lambda v: v),
    "run_name": ("run_name", lambda v: v),
    "rfdetr_weights": ("rfdetr_weights", lambda v: v),
    "yolo_weights": ("yolo_weights", lambda v: v),
    "deimv2_weights": ("deimv2_weights", lambda v: v),
    "deimv2_config": ("deimv2_config", lambda v: v),
    "deimv2_dir": ("deimv2_dir", lambda v: v),
    "log_level": ("log_level", lambda v: str(v).upper()),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ensemble",
        description=(
            "Run RFDETR Nano, YOLOv12 Nano, and DEIMv2 Pico on a COCO test split, "
            "fuse their predictions with Weighted Box Fusion, and write a unified "
            "Modelo,Precisão,Recall,MAP 50,MAP 50-95 summary to .outputs/."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a YAML config file whose values override the config.py "
            "defaults. A bare filename is also looked up under configs/. "
            "Explicit CLI flags still win over the file."
        ),
    )
    parser.add_argument("--dataset", type=Path, default=_UNSET,
                        help=f"COCO test split dir (default {DEFAULT_DATASET_DIR}).")
    parser.add_argument("--output-dir", type=Path, default=_UNSET,
                        help=f"Output root (default {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--device", type=str, default=_UNSET,
                        help=f"Torch device (default {DEFAULT_DEVICE}).")
    parser.add_argument("--batch-size", type=int, default=_UNSET,
                        help=f"Inference batch size (default {DEFAULT_BATCH_SIZE}).")
    parser.add_argument(
        "--predict-threshold", type=float, default=_UNSET,
        help=f"Per-model inference score threshold (default {DEFAULT_PREDICT_THRESHOLD}).",
    )
    parser.add_argument("--wbf-iou", type=float, default=_UNSET,
                        help=f"WBF IoU match threshold (default {DEFAULT_WBF_IOU}).")
    parser.add_argument("--wbf-skip-box", type=float, default=_UNSET,
                        help=f"WBF skip-box score threshold (default {DEFAULT_WBF_SKIP_BOX_THR}).")
    parser.add_argument(
        "--yolo-iou", type=float, default=_UNSET,
        help=(
            "IoU threshold for Ultralytics NMS during the ensemble's YOLO "
            f"inference (default {DEFAULT_YOLO_IOU_THRESHOLD}). High value = "
            "looser NMS, more candidates survive into WBF. Set to 0.7 to match "
            "Ultralytics' stock default; lower values are MORE aggressive NMS."
        ),
    )
    parser.add_argument(
        "--log-level", type=str, default=_UNSET,
        help=(
            "Logging verbosity (DEBUG/INFO/WARNING/ERROR). Resolved from the "
            f"process env, then the project .env file (default {DEFAULT_LOG_LEVEL})."
        ),
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", default=_UNSET,
        help="WBF per-model weights in the order RFDETR YOLO DEIMv2 (default: equal).",
    )
    parser.add_argument(
        "--skip-models", type=str, nargs="+",
        choices=[spec.key for spec in MODEL_SPECS], default=_UNSET,
        help="Skip one or more standalone models; ensemble requires at least 2 active.",
    )
    parser.add_argument(
        "--dynamic-metrics", type=_bool_flag, metavar="true|false", default=_UNSET,
        help=(
            "Compute the summary.csv standalone-model rows from this run "
            f"(default {str(DEFAULT_DYNAMIC_METRICS).lower()}). Set to false to "
            "emit the published HARDCODED_METRICS instead. The ENSEMBLE row is "
            "always computed."
        ),
    )
    parser.add_argument(
        "--no-visualizations", action="store_const", const=True, default=None,
        help="Disable per-image overlay rendering.",
    )
    parser.add_argument("--visualization-count", type=int, default=_UNSET)
    parser.add_argument(
        "--run-name", type=str, default=_UNSET,
        help="Override the auto-generated run name (default: timestamp_ensemble).",
    )
    parser.add_argument("--rfdetr-weights", type=Path, default=_UNSET)
    parser.add_argument("--yolo-weights", type=Path, default=_UNSET)
    parser.add_argument("--deimv2-weights", type=Path, default=_UNSET)
    parser.add_argument("--deimv2-config", type=Path, default=_UNSET)
    parser.add_argument("--deimv2-dir", type=Path, default=_UNSET)
    return parser


def _default_field_values() -> dict[str, Any]:
    """Every RunConfig field paired with its config.py default."""
    return {
        "dataset_dir": DEFAULT_DATASET_DIR,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "device": DEFAULT_DEVICE,
        "batch_size": DEFAULT_BATCH_SIZE,
        "predict_threshold": DEFAULT_PREDICT_THRESHOLD,
        "wbf_iou": DEFAULT_WBF_IOU,
        "wbf_skip_box_thr": DEFAULT_WBF_SKIP_BOX_THR,
        "yolo_iou_threshold": DEFAULT_YOLO_IOU_THRESHOLD,
        "wbf_weights": tuple(1.0 for _ in MODEL_SPECS),
        "skip_models": (),
        "save_visualizations": True,
        "dynamic_metrics": DEFAULT_DYNAMIC_METRICS,
        "visualization_count": DEFAULT_VISUALIZATION_COUNT,
        "run_name": _default_run_name(),
        "rfdetr_weights": RFDETR_WEIGHTS,
        "yolo_weights": YOLO_WEIGHTS,
        "deimv2_weights": DEIMV2_WEIGHTS,
        "deimv2_config": DEIMV2_CONFIG,
        "deimv2_dir": DEIMV2_DIR,
        "log_level": DEFAULT_LOG_LEVEL,
    }


def _collect_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for dest, (field, transform) in _CLI_FIELD_MAP.items():
        value = getattr(args, dest)
        if value is not _UNSET:
            overrides[field] = transform(value)
    if args.no_visualizations is not None:  # store_const => True when passed
        overrides["save_visualizations"] = not args.no_visualizations
    return overrides


def _resolve_config_path(raw: str) -> Path:
    """Resolve --config: as given, else fall back to configs/<name>."""
    path = Path(raw).expanduser()
    if path.is_file():
        return path
    if path.parent == Path("."):
        alt = Path("configs") / path.name
        if alt.is_file():
            return alt
    return path  # let load_config_file raise a clear not-found error


def parse_run_config(argv: list[str] | None = None) -> RunConfig:
    args = build_parser().parse_args(argv)

    config_path: Path | None = None
    yaml_overrides: dict[str, Any] = {}
    if args.config is not None:
        config_path = _resolve_config_path(args.config)
        yaml_overrides = load_config_file(config_path)

    cli_overrides = _collect_cli_overrides(args)
    final, overrides = resolve(_default_field_values(), yaml_overrides, cli_overrides)

    # When a config file drives the run and the user did not explicitly set a
    # run name (CLI or YAML), name the output folder after the config file:
    # ``{config_stem}_{timestamp}``. An explicit run_name always wins.
    run_name_overridden = any(o.field == "run_name" for o in overrides)
    if config_path is not None and not run_name_overridden:
        final["run_name"] = f"{config_path.stem}_{_timestamp()}"

    override_summary = [
        {
            "field": o.field,
            "key": o.yaml_key,
            "value": _format_value(o.value),
            "source": o.source,
        }
        for o in overrides
    ]
    banner = render_banner(str(config_path) if config_path else None, overrides)
    print(banner)

    return RunConfig(
        **final,
        extra={
            "config_path": str(config_path) if config_path else None,
            "config_overrides": override_summary,
            "config_banner": banner,
        },
    )


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _default_run_name() -> str:
    return _timestamp() + "_ensemble"
