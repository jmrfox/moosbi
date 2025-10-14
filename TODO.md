# TODO — moosbi

## Immediate next steps
- **Stabilize parameter management**: finalize API in `moosbi/scip.py` for parameter spaces, bounds, transforms.
- **Define pipeline API**: specify functions/classes for `optimize_with_pymoo()`, `build_prior_from_pareto()`, `infer_with_sbi()`, and a high-level `run_pipeline()`.
- **Pick initial working example**: simple simulator (e.g., Gaussian or Lotka–Volterra) + two objectives to exercise the pipeline.

## Milestones
- **M0 — Scaffolding**: packaging (`pyproject.toml`), basic docs, pre-commit/lint/test.
- **M1 — Parameter/space**: finalize `scip.py` shapes, validation, sampling utilities.
- **M2 — pymoo integration**: wrappers for defining objectives/constraints, running optimization, retrieving Pareto set/front.
- **M3 — prior from Pareto**: strategy interface (e.g., KDE, GMM, truncated box) to convert Pareto region into an informative prior.
- **M4 — sbi integration**: simulator adapter, posterior training (SNPE/SNLE/SNRE), conditioning on observed data.
- **M5 — E2E example & tests**: runnable example, unit/integration tests, reproducibility (seeds), CI.
- **M6 — Docs & benchmarks**: API docs, README examples, small benchmarks vs. baselines.

## Design decisions to settle
- **Model/simulator interface**: callable signature, batching, random seeds, device.
- **Objective API**: mapping user losses to `pymoo` objectives (vectorized, differentiability not required).
- **Constraints**: bounds/transform handling (log, softplus) and feasibility checks.
- **Pareto→prior strategies**: KDE vs GMM vs bounding boxes; pluggable strategy pattern.
- **Data IO**: observed data schema and summary statistics hooks.
- **Compute**: CPU/GPU handling for `sbi`, reproducibility controls.

## Backlog
- **pymoo wrapper**: minimal objective adapter and runner.
- **sbi runner**: minimal training loop with configurable algorithm (SNPE/SNLE/SNRE).
- **Prior strategies**: implement 1–2 (start with truncated box + KDE) behind a common interface.
- **Validation/diagnostics**: posterior predictive checks, coverage metrics, Pareto visualization.
- **Logging/telemetry**: simple structured logging; seeds in outputs.
- **Examples**: one notebook + one pure-Python script.
- **Packaging/CI**: `pyproject.toml`, GitHub Actions, test matrix.
- **Type hints & tests**: mypy/ruff/pytest.

## References
- pymoo: https://pymoo.org/
- sbi: https://www.mackelab.org/sbi/

## Status
- [x] Expand `README.md` with overview and workflow.
- [x] Create `TODO.md` with milestones and backlog.
- [ ] Decide initial example problem.
- [ ] Sketch `optimize_with_pymoo()` and `infer_with_sbi()` signatures.
