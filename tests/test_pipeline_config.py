"""The run.json snapshot must record which config produced the run."""

from __future__ import annotations

import json
from types import SimpleNamespace

from ensemble.cli import parse_run_config
from ensemble.pipeline import _serialize_run_config


def _fake_bundle():
    return SimpleNamespace(
        image_records=[object(), object()],
        num_classes=1,
        category_names=["polyp"],
    )


def test_run_json_records_config_provenance(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("wbf_iou: 0.5\n", encoding="utf-8")
    run = parse_run_config(["--config", str(cfg), "--batch-size", "16"])

    out = tmp_path / "run.json"
    _serialize_run_config(run, _fake_bundle(), out)
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["config_path"].endswith("c.yaml")
    by_field = {o["field"]: o for o in data["config_overrides"]}
    assert by_field["wbf_iou"]["source"] == "yaml"
    assert by_field["batch_size"]["source"] == "cli"


def test_run_json_records_null_config_when_defaults(tmp_path):
    run = parse_run_config([])
    out = tmp_path / "run.json"
    _serialize_run_config(run, _fake_bundle(), out)
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["config_path"] is None
    assert data["config_overrides"] == []
