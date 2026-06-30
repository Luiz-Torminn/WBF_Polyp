---
description: Component sizing
paths:
  - "./**"
---

## Component Rules

> Rules 28–31, 75. Path-scoped to `apps/web`.

### Rule 28 — Keep Components Small

Keep components around 250–350 lines when practical. Do not create giant components.

### Group by what the component does

Structure the folder around the feature's real internal seams. Useful axes (combine as the block grows):

- **Business domain / sub-feature** — the slice of the product the component serves.
- **Workflow / responsibility** — a coherent flow or job (event creation/editing, attachments).
