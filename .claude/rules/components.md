---
description: Module and file sizing/organization
paths:
  - "./**"
---

## Module and File Rules

> Rule 28.

### Rule 28 — Keep Modules Focused and Small

Keep each Python module focused on a single responsibility and reasonably small
(around 250–350 lines when practical). Do not create giant catch-all modules.

### Organize the package by responsibility

The `ensemble/` package is split along its real internal seams — add new code on
the matching seam instead of widening an unrelated module. Existing axes:

- **Model adapters** (`ensemble/adapters/`) — one module per model, each
  normalizing that model's output into the shared `Prediction`.
- **Pipeline stage / responsibility** — data loading (`data.py`), fusion
  (`fusion.py`), evaluation (`metrics.py`), visualization (`visualize.py`),
  configuration (`config.py`, `config_file.py`, `cli.py`), and orchestration
  (`pipeline.py`).

When a module outgrows its seam, split it along these axes rather than letting it
sprawl.
