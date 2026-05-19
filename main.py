"""CLI entrypoint for the ensemble pipeline.

Run with::

    python main.py
    python main.py --device cuda:0 --batch-size 8

See ``python main.py --help`` for the full set of flags.
"""

from __future__ import annotations

import sys

from ensemble.cli import parse_run_config
from ensemble.pipeline import run_pipeline


def main() -> int:
    run = parse_run_config()
    summary_path = run_pipeline(run)
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
