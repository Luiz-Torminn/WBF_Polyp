"""Integration tests for --config wiring in cli.parse_run_config."""

from __future__ import annotations

import pytest

from ensemble.cli import parse_run_config
from ensemble.config import (
    DEFAULT_WBF_IOU,
    DEFAULT_WBF_SKIP_BOX_THR,
    DEFAULT_YOLO_IOU_THRESHOLD,
)
from ensemble.config_file import ConfigError


def _cfg(tmp_path, text: str):
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_yaml_overrides_applied(tmp_path):
    path = _cfg(tmp_path, "wbf_iou: 0.5\nwbf_skip_box: 0.1\nyolo_iou: 0.95\n")
    run = parse_run_config(["--config", path])
    assert run.wbf_iou == 0.5
    assert run.wbf_skip_box_thr == 0.1
    assert run.yolo_iou_threshold == 0.95


def test_cli_flag_beats_yaml(tmp_path):
    path = _cfg(tmp_path, "wbf_iou: 0.5\n")
    run = parse_run_config(["--config", path, "--wbf-iou", "0.6"])
    assert run.wbf_iou == 0.6


def test_yaml_beats_default_untouched_keys_keep_default(tmp_path):
    path = _cfg(tmp_path, "wbf_iou: 0.5\n")
    run = parse_run_config(["--config", path])
    assert run.wbf_iou == 0.5
    assert run.wbf_skip_box_thr == DEFAULT_WBF_SKIP_BOX_THR
    assert run.yolo_iou_threshold == DEFAULT_YOLO_IOU_THRESHOLD


def test_extra_records_config_path_and_sources(tmp_path):
    path = _cfg(tmp_path, "wbf_iou: 0.5\n")
    run = parse_run_config(["--config", path, "--batch-size", "16"])
    assert run.extra["config_path"].endswith("c.yaml")
    by_field = {o["field"]: o for o in run.extra["config_overrides"]}
    assert by_field["wbf_iou"]["source"] == "yaml"
    assert by_field["batch_size"]["source"] == "cli"


def test_no_config_uses_defaults(tmp_path):
    run = parse_run_config([])
    assert run.wbf_iou == DEFAULT_WBF_IOU
    assert run.extra["config_path"] is None
    assert run.extra["config_overrides"] == []


def test_unknown_yaml_key_raises(tmp_path):
    path = _cfg(tmp_path, "wbf_iuo: 0.5\n")
    with pytest.raises(ConfigError):
        parse_run_config(["--config", path])


def test_no_visualizations_flag_is_an_override(tmp_path):
    run = parse_run_config(["--no-visualizations"])
    assert run.save_visualizations is False
    by_field = {o["field"]: o for o in run.extra["config_overrides"]}
    assert by_field["save_visualizations"]["source"] == "cli"


def test_save_visualizations_true_by_default(tmp_path):
    run = parse_run_config([])
    assert run.save_visualizations is True


def test_banner_is_printed(tmp_path, capsys):
    path = _cfg(tmp_path, "wbf_iou: 0.5\n")
    parse_run_config(["--config", path])
    out = capsys.readouterr().out
    assert "Config loaded" in out
    assert "wbf_iou" in out
