"""CLI entrypoint for the ensemble pipeline.

Run with::

    python main.py
    python main.py --device cuda:0 --batch-size 8
    python main.py --config configs/aggressive.yaml

See ``python main.py --help`` for the full set of flags.
"""

from __future__ import annotations

import sys

from ensemble.cli import parse_run_config
from ensemble.config_file import ConfigError
from ensemble.pipeline import run_pipeline


def main() -> int:
    try:
        run = parse_run_config()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = run_pipeline(run)
    print(f"summary: {result.summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
