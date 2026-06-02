"""YAML config-file support for the ensemble pipeline.

A ``--config <file>.yaml`` supplies overrides for the ``config.py`` defaults.
Resolution order is ``config.py default -> YAML file -> explicit CLI flag``
(the CLI wins). This module is deliberately standalone — it does not import
``ensemble.config`` — so it stays trivially unit-testable and free of the
repo-root resolution side effect.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class ConfigError(Exception):
    """Raised for any problem loading or validating a YAML config file."""


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    # Reject bools and float-ish strings that aren't whole ints.
    if isinstance(value, bool):
        raise ValueError("expected an integer, got a boolean")
    return int(value)


def _as_str(value: Any) -> str:
    return str(value)


def _as_upper_str(value: Any) -> str:
    return str(value).upper()


def _as_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError("expected a boolean (true/false)")


def _as_float_tuple(value: Any) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of numbers")
    return tuple(float(item) for item in value)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("expected a list of strings")
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class FieldSpec:
    """Maps one YAML key to a ``RunConfig`` field plus its coercion."""

    field: str  # RunConfig constructor field name
    yaml_key: str  # key as written in the YAML file / CLI flag
    coerce: Callable[[Any], Any]


# Canonical override surface — full parity with the CLI flags. The ``yaml_key``
# matches the CLI flag spelling; ``field`` matches the RunConfig constructor.
SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("dataset_dir", "dataset", _as_path),
    FieldSpec("output_dir", "output_dir", _as_path),
    FieldSpec("device", "device", _as_str),
    FieldSpec("batch_size", "batch_size", _as_int),
    FieldSpec("predict_threshold", "predict_threshold", _as_float),
    FieldSpec("wbf_iou", "wbf_iou", _as_float),
    FieldSpec("wbf_skip_box_thr", "wbf_skip_box", _as_float),
    FieldSpec("yolo_iou_threshold", "yolo_iou", _as_float),
    FieldSpec("wbf_weights", "weights", _as_float_tuple),
    FieldSpec("skip_models", "skip_models", _as_str_tuple),
    FieldSpec("save_visualizations", "save_visualizations", _as_bool),
    FieldSpec("visualization_count", "visualization_count", _as_int),
    FieldSpec("run_name", "run_name", _as_str),
    FieldSpec("rfdetr_weights", "rfdetr_weights", _as_path),
    FieldSpec("yolo_weights", "yolo_weights", _as_path),
    FieldSpec("deimv2_weights", "deimv2_weights", _as_path),
    FieldSpec("deimv2_config", "deimv2_config", _as_path),
    FieldSpec("deimv2_dir", "deimv2_dir", _as_path),
    FieldSpec("log_level", "log_level", _as_upper_str),
)

_BY_YAML_KEY: dict[str, FieldSpec] = {spec.yaml_key: spec for spec in SPECS}


def load_config_file(path: Path) -> dict[str, Any]:
    """Parse a YAML config file into ``{runconfig_field: coerced_value}``.

    Strict: an unknown key, a wrong-type value, a missing file, or non-mapping
    content all raise :class:`ConfigError` so a typo can never silently fall
    back to a default.
    """
    import yaml

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {path}: {exc}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config file {path} must contain a key/value mapping, "
            f"got {type(raw).__name__}."
        )

    resolved: dict[str, Any] = {}
    for key, value in raw.items():
        spec = _BY_YAML_KEY.get(key)
        if spec is None:
            raise ConfigError(_unknown_key_message(key, path))
        try:
            resolved[spec.field] = spec.coerce(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Invalid value for '{key}' in {path}: {value!r} ({exc})."
            ) from exc
    return resolved


def _unknown_key_message(key: Any, path: Path) -> str:
    suggestions = difflib.get_close_matches(str(key), _BY_YAML_KEY, n=1)
    hint = f" (did you mean '{suggestions[0]}'?)" if suggestions else ""
    return f"Unknown config key '{key}' in {path}{hint}"


@dataclass(frozen=True)
class Override:
    """One resolved override and where it came from."""

    field: str  # RunConfig field name
    yaml_key: str  # display key (CLI/YAML spelling)
    value: Any
    source: str  # "yaml" or "cli"


def resolve(
    defaults: dict[str, Any],
    yaml_overrides: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> tuple[dict[str, Any], list[Override]]:
    """Layer ``default -> yaml -> cli`` and report the effective overrides.

    Returns the merged ``{field: value}`` dict plus an ordered list of
    :class:`Override` entries for every field that came from YAML or the CLI
    (i.e. not left at its ``config.py`` default), in ``SPECS`` declaration order.
    """
    final = dict(defaults)
    sources: dict[str, str] = {}
    for field, value in yaml_overrides.items():
        final[field] = value
        sources[field] = "yaml"
    for field, value in cli_overrides.items():
        final[field] = value
        sources[field] = "cli"

    overrides = [
        Override(
            field=spec.field,
            yaml_key=spec.yaml_key,
            value=final[spec.field],
            source=sources[spec.field],
        )
        for spec in SPECS
        if spec.field in sources
    ]
    return final, overrides


_BANNER_WIDTH = 60


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ",".join(_format_value(item) for item in value)
    return str(value)


def render_banner(config_path: str | None, overrides: list[Override]) -> str:
    """Render the source-annotated load banner (printed and logged)."""
    rule = "=" * _BANNER_WIDTH
    count = len(overrides)
    plural = "override" if count == 1 else "overrides"
    if config_path:
        header = f"Config loaded: {config_path} ✓ ({count} {plural})"
    else:
        header = f"Config: built-in defaults ({count} {plural})"

    lines = [rule, header]
    if overrides:
        lines.append("-" * _BANNER_WIDTH)
        for override in overrides:
            value = _format_value(override.value)
            lines.append(
                f"  {override.yaml_key:<22}{value:<14}[{override.source}]"
            )
    lines.append(rule)
    return "\n".join(lines)
