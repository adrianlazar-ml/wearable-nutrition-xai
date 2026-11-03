# Revision 1 of the KBS manuscript: code, results and verification

This folder holds the scripts, result files, figure registers and verification reports of Revision 1 of the manuscript submitted to Knowledge-Based Systems:

- Manuscript: [title and authors to be completed at acceptance].
- Archived release: [Zenodo DOI to be added].
- License: [license to be chosen by the authors].

The canonical figures of the submitted manuscript are in `../CIFRE_CANONICE.md`; every figure produced in the revision is in `CIFRE_REVIZIE.md` with its script, seed, run date and note. Text decisions are in `TEXT_DECISIONS.md`, the audit of the revision in `audit/`.

## History

The git history of this repository starts on 26 May 2026, with the snapshot made for the dissertation defense. Design choices made before that date (the labeling thresholds 60 / 85 / 75, the temporal split, the feature set) are documented in the dissertation and in the submitted manuscript and cannot be dated from git. The revision work (September to October 2026) is committed task by task; the audit of the revision and its application are in `audit/` (`AUDIT_SUMMARY.md`, `AUDIT_DECISIONS.md`, `apply_summary.md`).

## Environment

Frozen environment of the submitted manuscript (`../models/frozen_meta_20260331.json`, verified at T0.1): Python 3.11.9, scikit-learn 1.8.0, numpy 2.4.3, pandas 2.3.3, imbalanced-learn 0.14.1, xgboost 3.2.0, scipy 1.17.1. Additional packages of the revision, pinned in `requirements.txt`: statsmodels 0.15.0, shap 0.51.0, torch 2.11.0 (CPU), lightgbm 4.6.0, catboost 1.2.10, pytorch-tabnet 4.1.0 (imported from `vendor/`, one-line scipy import patch, see `vendor/pytorch_tabnet/PATCH.md`), rtdl_revisiting_models 0.0.2, python-docx 1.2.0. All runs are CPU, single thread, seed 42.

## Reproduction

Run everything from the repository root with `python -W error` (every script was verified to emit no warning). Order of the phases, as in `KBS_Revision_Plan.md`:

1. T0, reproducibility gate: `python sanity_check_frozen.py` (canonical Random Forest, macro F1 0.5063); `python revision/scripts/save_test_predictions.py` (test predictions and probabilities of the three canonical models, saved models); `python revision/scripts/persistence_baseline.py`; `pytest revision/scripts/test_t3_labels.py` (parametrized labeling identical to the frozen one).
2. T1, analyses on the saved predictions: `extended_metrics.py` (T1.1, about 20 to 30 s), `statistical_tests.py` (T1.2), `calibration_rf.py --stage A` then `--stage B` and `safety_layer_calibrated.py` (T1.3, a few seconds each stage), `shap_stability.py --mode bootstrap` and `--mode seeds` (T1.4, about one minute per mode), `group_ablation.py` (T1.5, about 20 to 25 s).
3. T3, labels and time: `threshold_sensitivity.py` (T3.1), `label_agreement.py` (T3.2, a few seconds), `expanding_window.py` (T3.3, about 20 to 30 s).
4. T2, additional models: `extra_models.py --mode main` (T2.1, about 3 minutes) and `--mode seeds` (about 11 minutes); `hyperparameter_search.py --mode cv` (T2.2, about 28 minutes), `--mode test` (3 to 4 minutes) and `--mode seeds` (about 6 minutes).
5. F4, manuscript assembly: `assemble_manuscript.py` (needs the submitted v8 docx and the legacy figure folder, see its `--help`), then `_check_t41_numbers.py`, `_check_t41_outputs.py`, `_check_table_rounding.py`, `style_check.py`. The same run builds the sixth docx, `manuscript/KBS_Cover_Letter_R1.docx`, from `F4_cover_letter_R1.md` (T11.5; checked by `_check_t115_cover_letter.py`).

Determinism: every producing script was run twice in a row and the outputs compared byte for byte by its `_check_t*_outputs.py` script (sha256 on csv, md, npy, png, pdf; json compared without the run-date fields). The 1,000-resample bootstrap matrix (`results/bootstrap_indices_1000x192_seed42.npy`, `numpy.random.default_rng(42)`) is shared by all T1, T2 and T3 scripts, so differences between systems are paired. Unit tests: `python -m pytest revision/scripts/ -q`.

## Data

What is in the repository and what is not:

- Raw data are not published: the Apple Health exports (`../data/apple_health/`, including the native XML export and the daily Health Auto Export JSON files) and the Oura exports (`../data/Oura/`) contain minute-level personal health data of one participant. The public release is made from a filtered copy of the repository that excludes them (see `audit/hygiene_summary.md`, section Public release).
- `../models/dataset_snapshot_20260331.csv` holds the 958 labeled days with the 19 lag-1 features and the labels; it is the only input of the T1, T2 and T3 scripts. [TODO: publication of this file depends on the co-author's Option C decision: code only, or code plus aggregated features.]
- `../data/oura_combined.csv` (daily ring scores) is needed to regenerate the labels for other thresholds in T3.1, because the raw readiness and sleep scores are not in the snapshot. Same decision as above.
- `../apple_watch_api/apple_health_daily.csv` is the output of `parse_health_auto_export.py` on the 1,079 daily JSON files; the audit reproduced it on every date except 2026-03-29 (csv built before that day's export was complete).

## Provenance of the HealthKit features

The parser never reads the `source` field of the JSON records and pools every record of a day. On the labeled days, the aggregates attributed to the watch include ring samples merged by HealthKit for heart rate (99.7% of the days), steps (95.6%) and active energy (99.4%); HRV and SpO2 come from the watch only, respiratory rate almost entirely so, and the sleep-stage features come from whichever device wrote a sleep record (the ring alone on 204 of 952 days, both devices summed on 8 days). The manuscript therefore describes the two sources as two data paths (ring scores from the API, HealthKit aggregates from the watch export), and the independence of the label validation from the ring's sensors is restricted to HRV, SpO2 and respiratory rate (Sections 3.1, 3.10, 4.7, Limitation 9 of the revised manuscript; `audit/AUDIT_F4_provenance_shap_ablation_agreement.md`). Lesson for any reuse of Health Auto Export data: filter by `source` at parse time.

## Record of the original submission

`manuscript/submitted/` holds the Editorial Manager PDF of the original submission (KNOSYS-D-26-14863, received from the co-author on 2 November 2026; sha256 in its README). `audit/em_submission_diff.md` (script `scripts/_t11_em_diff.py`) shows that its manuscript block is the v8 base plus a title page with the authors (single-blind review). The folder is excluded from the public release (e-mail addresses and a telephone number; `audit/hygiene_summary.md`).

## Verification notes

- Erratum of Section 4.9 of the submitted manuscript: the stored confidence-threshold table came from an earlier 16-feature XML pipeline; `scripts/_check_t13_safety_layer.py` recomputes the safety layer on the canonical Random Forest and exits 1 by design (the mismatch is the finding); the audit confirmed the erratum by executing the old pipeline's logic.
- The saved Random Forest of the revision (`results/models/rf_revision.joblib`), the frozen model (`../models/rf_frozen_20260331.joblib`) and the deployed model (`../models/rf_model.joblib`) are byte-identical; retraining reproduces them byte for byte.
- `balanced_accuracy_score` is called without fixed labels in some scripts; on this dataset all three classes are present in every evaluated set, so the result is unaffected.
- Line endings: the csv files written by pandas on Windows carry CRLF in the working copy; the repository stores them with LF (`.gitattributes`). Byte-for-byte determinism checks compare two runs of the same script and are unaffected.
- Rounding: every figure printed with 3 decimals is the csv value rounded once (`scripts/_check_table_rounding.py` checks all csv-derived tables of the manuscript and the Supplementary); the 4-decimal values of the registers are never rounded again.
