# disertatie_v2 — the LLM assistant context

## Current work: KBS revision (Knowledge-Based Systems, Revision 1, due 2026-12-01)

At the start of every session:
1. Read `revision/KBS_Revision_Plan.md` (phases F0-F5, tasks T0.x-T4.x, rules, calendar).
2. Read `revision/CIFRE_REVIZIE.md` (every figure produced so far, with script, seed, run date, and the "Status Faza N" sections).
3. State which phase is closed and which task is next, before doing anything.

## Rules (from the plan, non-negotiable)
- Canonical figures in `CIFRE_CANONICE.md` never change; all revision work is additive.
- Every new figure goes into `revision/CIFRE_REVIZIE.md` with script, seed 42 and run date before it is used anywhere.
- The test set (192 days) is touched once per experiment; tuning and selection use temporal CV on train only.
- Unpleasant results are reported, never hidden.
- Prefer the Edit tool over PowerShell heredocs for writing file content (heredocs have failed in this repo).
- Stage only `revision/` for revision commits; never stage `_referinte_upb/`, `eswa_architecture.py`, `outputs/figures/eswa_architecture/`.
- Keep one-off check scripts (`revision/scripts/_check_*.py`) as evidence; do not delete them.

## Layout
- `revision/scripts/` — revision scripts (common.py, save_test_predictions.py, persistence_baseline.py, test_t3_labels.py, _check_*)
- `revision/results/` — saved test predictions/probabilities (`test_predictions_*.csv`), metrics CSVs, `models/` (joblib)
- `revision/t3_labels.py` — parametrized labeling (thresholds 60/85/75 as parameters)
- `sanity_check_frozen.py` (repo root) — reproduces canonical RF figures from the frozen pipeline (2026-03-31)
- Frozen environment: Python 3.11.9 / scikit-learn 1.8.0 / numpy 2.4.3 / pandas 2.3.3
