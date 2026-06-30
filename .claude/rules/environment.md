## Environment and Tooling Rules

> Rules 47–52.

### Rule 47 — Clone Through SSH

Use SSH for repository access. Do not clone via HTTPS unless explicitly instructed.

### Rule 49 — Use Linux-Friendly Development Assumptions

Prefer commands and workflows that work cleanly in Linux environments. Avoid OS-specific assumptions unless required.

### Rule 50 — Use Repository Understanding Tools When Needed

When unfamiliar with the codebase structure, use repository understanding tools before modifying files. Identify imports, exports, callers, and ownership before making changes.

### Rule 51 — Respect Guardrails

Follow existing and upcoming guardrails such as lint rules, formatting rules, and project-specific checks. Do not bypass them unless explicitly instructed and technically justified.

### Rule 52 — Use the `ensemble-method` Conda Environment

On Linux, the environment for running the pipeline and prompting/executing the
On Linux, the environment for running the pipeline and prompting/executing the
test suite is the conda environment (typically located under your miniconda/anaconda installation, e.g., `~/miniconda3/envs/ensemble-method`).

Activate it by name (location-independent) before running or testing; the full
path is only needed if `conda` cannot resolve the name:

```bash
conda activate ensemble-method
# or, if conda cannot resolve the name, by full path:
conda activate /home/luizlima/miniconda3/envs/ensemble-method
```

Run `pytest` and any pipeline command (`python main.py`,
`python bayesian_optimization.py`) from within this environment so dependencies
match `requirements.txt`.
