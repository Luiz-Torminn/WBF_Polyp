"""A bad --config should exit cleanly (no traceback), not crash."""

from __future__ import annotations

import sys

import main as main_module


def test_main_exits_2_on_bad_config(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "b.yaml"
    bad.write_text("wbf_iuo: 1\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", str(bad)])

    rc = main_module.main()

    assert rc == 2
    err = capsys.readouterr().err
    assert "wbf_iuo" in err
