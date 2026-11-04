# wearable-nutrition-xai: code, results and verification of Revision 1 of the KBS manuscript

This folder holds the scripts, result files, figure registers and verification reports of Revision 1 of the manuscript submitted to Knowledge-Based Systems. The public repository `wearable-nutrition-xai` is a filtered copy of the authors' private working repository (section "What is excluded and why").

- Manuscript: "Explainable Artificial Intelligence for Personalized Nutrition: A Multimodal Framework Integrating Smart Ring and Smartwatch Data", Adrian Lazar and Catalin Buiu (National University of Science and Technology POLITEHNICA Bucharest), Knowledge-Based Systems, Revision 1 under review (2026).
- Public repository: `wearable-nutrition-xai` (GitHub, https://github.com/adrianlazar-ml/wearable-nutrition-xai); release v1.0-R1 is the state of the revised submission.
- Archived release: Zenodo DOI 10.5281/zenodo.XXXXXXX (placeholder, filled at the first release).
- License: MIT for the code and the documentation (`LICENSE` at the repository root); CC BY 4.0 for the daily aggregates published under `data/`, `apple_watch_api/` and `models/dataset_snapshot_20260331.csv` (`LICENSE-DATA`). How to cite: `CITATION.cff`.

The canonical figures of the submitted manuscript are in `../CIFRE_CANONICE.md`; every figure produced in the revision is in `CIFRE_REVIZIE.md` with its script, seed, run date and note. Text decisions are in `TEXT_DECISIONS.md`, the audit of the revision in `audit/`.

## History

The git history of this repository starts on 26 May 2026, with the snapshot made for the dissertation defense. Design choices made before that date (the labeling thresholds 60 / 85 / 75, the temporal split, the feature set) are documented in the dissertation and in the submitted manuscript and cannot be dated from git. The revision work (September to November 2026) is committed task by task; the audit of the revision and its application are in `audit/` (`AUDIT_SUMMARY.md`, `AUDIT_DECISIONS.md`, `apply_summary.md`).

The tags of the private repository are kept: `canonical-8020-20260331` (the frozen pipeline and the canonical figures), `baseline-pre-remediere-2026-05-26`, `sprint-3-complete-2026-05-27`, `sprint-6-complete-2026-05-28`, and `b3-abandoned`, a single commit off the main line that records the abandoned experiment B3 (a 70/10/20 train / validation / test split that exposed the ring-generation confound; it justifies the canonical 80/20 temporal split and Limitation 7 of the manuscript). The history of the public repository was rewritten with git filter-repo (section "What is excluded and why"), so the commit hashes cited in `CIFRE_REVIZIE.md`, in `audit/` and in the task summaries refer to the private repository; the authors keep the old-to-new commit map.

## Environment

Frozen environment of the submitted manuscript (`../models/frozen_meta_20260331.json`, verified at T0.1): Python 3.11.9, scikit-learn 1.8.0, numpy 2.4.3, pandas 2.3.3, imbalanced-learn 0.14.1, xgboost 3.2.0, scipy 1.17.1. Additional packages of the revision, pinned in `requirements.txt`: statsmodels 0.15.0, shap 0.51.0, torch 2.11.0 (CPU), lightgbm 4.6.0, catboost 1.2.10, pytorch-tabnet 4.1.0 (imported from `vendor/`, one-line scipy import patch, see `vendor/pytorch_tabnet/PATCH.md`), rtdl_revisiting_models 0.0.2, python-docx 1.2.0. All runs are CPU, single thread, seed 42.

## Reproduction

Run everything from the repository root with `python -W error` (every script was verified to emit no warning). Order of the phases, as in `KBS_Revision_Plan.md`:

1. T0, reproducibility gate: `python sanity_check_frozen.py` (canonical Random Forest, macro F1 0.5063); `python revision/scripts/save_test_predictions.py` (test predictions and probabilities of the three canonical models, saved models); `python revision/scripts/persistence_baseline.py`; `python revision/scripts/test_t3_labels.py` (parametrized labeling identical to the frozen one; a plain script that prints ALL PASS, not a pytest file).
2. T1, analyses on the saved predictions: `extended_metrics.py` (T1.1, about 20 to 30 s), `statistical_tests.py` (T1.2), `calibration_rf.py --stage A` then `--stage B` and `safety_layer_calibrated.py` (T1.3, a few seconds each stage), `shap_stability.py --mode bootstrap` and `--mode seeds` (T1.4, about one minute per mode), `group_ablation.py` (T1.5, about 20 to 25 s).
3. T3, labels and time: `threshold_sensitivity.py` (T3.1), `label_agreement.py` (T3.2, a few seconds), `expanding_window.py` (T3.3, about 20 to 30 s).
4. T2, additional models: `extra_models.py --mode main` (T2.1, about 3 minutes) and `--mode seeds` (about 11 minutes); `hyperparameter_search.py --mode cv` (T2.2, about 28 minutes), `--mode test` (3 to 4 minutes) and `--mode seeds` (about 6 minutes).
5. F4, manuscript assembly: `assemble_manuscript.py` (needs the submitted v8 docx and the legacy figure folder, see its `--help`), then `_check_t41_numbers.py`, `_check_t41_outputs.py`, `_check_table_rounding.py`, `style_check.py`. The same run builds the sixth docx, `manuscript/KBS_Cover_Letter_R1.docx`, from `F4_cover_letter_R1.md` (T11.5; checked by `_check_t115_cover_letter.py`). This step is not runnable from the public repository: `manuscript/` (base docx, F4 text blocks, outputs) and `F4_cover_letter_R1.md` are not published, and the title-page e-mail addresses and telephone number are read from `authors.json` (gitignored; template `authors.example.json`), so the script stops with a message when only the template is present.

Determinism: every producing script was run twice in a row and the outputs compared byte for byte by its `_check_t*_outputs.py` script (sha256 on csv, md, npy, png, pdf; json compared without the run-date fields). The 1,000-resample bootstrap matrix (`results/bootstrap_indices_1000x192_seed42.npy`, `numpy.random.default_rng(42)`) is shared by all T1, T2 and T3 scripts, so differences between systems are paired. Unit tests: `python -m pytest revision/scripts/ -q`.

## Data

What is in the repository and what is not:

- Raw data are not published: the Apple Health exports (`../data/apple_health/`, including the native XML export and the daily Health Auto Export JSON files) and the Oura exports (`../data/Oura/`, `../data/sleep_detailed.csv`) contain record-level, minute-level and location data of one participant. The public release is made from a filtered copy of the repository that excludes them (`audit/public_release_list.md`; procedure in `audit/hygiene_summary.md`, section Public release).
- Published daily aggregates (one row per day, CC BY 4.0, `LICENSE-DATA`): `../data/activity.csv`, `readiness.csv`, `sleep.csv`, `sleep_metrics.csv`, `stress.csv`, `oura_historical.csv` and `oura_combined.csv` (Oura API daily scores and metrics), `dataset_combinat.csv`, `dataset_combinat_json.csv` and `dataset_combinat_json_extended.csv` (daily feature tables of the dissertation pipelines), `../models/dataset_snapshot_20260331.csv`, `../apple_watch_api/apple_health_daily.csv`.
- `../models/dataset_snapshot_20260331.csv` holds the 958 labeled days with the 19 lag-1 features and the labels. It is the input of `persistence_baseline.py`, `extended_metrics.py` and the `_check_t1x_inputs` scripts, and the reference against which `common.load_canonical()` checks the dataset it rebuilds; it is not the only input of the T1, T2 and T3 scripts.
- `../data/oura_combined.csv` (daily ring scores and metrics, 1,009 days) and `../apple_watch_api/apple_health_daily.csv` (the output of `parse_health_auto_export.py` on the 1,079 daily JSON files; the audit reproduced it on every date except 2026-03-29, csv built before that day's export was complete) are read by `common.load_raw()`, the entry point of the producing scripts (`calibration_rf`, `expanding_window`, `extra_models`, `group_ablation`, `hyperparameter_search`, `label_agreement`, `save_test_predictions`, `shap_stability`, `threshold_sensitivity`, `watch_only_sensitivity`): the dataset is rebuilt from them and checked against the snapshot at every run, and `oura_combined.csv` carries the raw readiness and sleep scores needed to regenerate the labels for other thresholds (T3.1).

## Provenance of the HealthKit features

The parser never reads the `source` field of the JSON records and pools every record of a day. On the labeled days, the aggregates attributed to the watch include ring samples merged by HealthKit for heart rate (99.7% of the days), steps (95.6%) and active energy (99.4%); HRV and SpO2 come from the watch only, respiratory rate almost entirely so, and the sleep-stage features come from whichever device wrote a sleep record (the ring alone on 204 of 952 days, both devices summed on 8 days). The manuscript therefore describes the two sources as two data paths (ring scores from the API, HealthKit aggregates from the watch export), and the independence of the label validation from the ring's sensors is restricted to HRV, SpO2 and respiratory rate (Sections 3.1, 3.10, 4.7, Limitation 9 of the revised manuscript; `audit/AUDIT_F4_provenance_shap_ablation_agreement.md`). Lesson for any reuse of Health Auto Export data: filter by `source` at parse time.

## Running on the published aggregates

Verified on 3 November 2026 from the filtered copy, same frozen environment:

- `python -m pytest revision/scripts/ -q`: 111 passed, 2 failed; the two failures are in `test_assemble_manuscript.py` and read `manuscript/F4_manuscript_changes.md`, which is not published. Every test of the T0, T1, T2 and T3 scripts passes.
- `python -W error revision/scripts/group_ablation.py` (T1.5, about 23 s) and `python -W error revision/scripts/label_agreement.py` (T3.2, about 5 s), end to end from `data/oura_combined.csv` and `apple_watch_api/apple_health_daily.csv`: every csv, png and pdf byte-identical to the committed files (full Random Forest 0.5063 / 0.6406 / 0.1416; Cliff's delta Active minus Rest of aw_hr_mean -0.840 [-0.912, -0.749]); the md tables and the run_meta json differ only in the run date.
- `_check_t41_outputs.py`, `_check_t41_numbers.py`, `_check_table_rounding.py`, `style_check.py`, `_check_t8_parity.py`, `_check_t11*.py`, `_check_t115_cover_letter.py` and `_t11_em_diff.py` need `manuscript/` and cannot run from the public repository.
- The root scripts of the dissertation pipeline that read `data/apple_health/` or `data/Oura/` (`cap5_pipeline.py`, `cap5_compare_algorithms.py`, `parse_health_auto_export.py` on the JSON folder, `oura_collect.py`) cannot run; `sanity_check_frozen.py` and the frozen 19-feature pipeline on `data/dataset_combinat_json_extended.csv` can.

## What is excluded and why

The public repository was produced with `git clone --no-local` of the private repository and `git filter-repo --invert-paths` on the list in `audit/public_release_list.md`, over the whole history (every commit and tag). Excluded:

- `data/Oura/`, `data/apple_health/`, `data/sleep_detailed.csv`, `apple_watch_api/raw_json/`: raw personal health data (per record, per minute, per night, per session, GPS) and the Oura account and payment files of the participant.
- `manuscript/` (the submitted base docx, the F4 text blocks, the six Revision 1 docx outputs, the figure exports and the Editorial Manager PDF of the original submission) and `F4_cover_letter_R1.md`: the manuscript is not part of the public repository; the published article is the reference.
- `TEXT_DECISIONS.md`, `T11_summary.md`, `T11b_summary.md`, `audit/em_submission_diff.md`, `audit/t11_docx_text_diff.md`: process documents that quote the title page and the cover letter (e-mail addresses, telephone number).
- `_referinte_upb/` (dissertation and university documents), `disertatie_audit.zip`, `audit_csv_output.txt`, `sanity_check_output.txt` (outputs with local paths and raw excerpts), `.env` (Oura OAuth2 credentials committed once in May 2026 and untracked the same month; the application and the token were deleted and revoked on 3 November 2026).
- In the remaining history the two e-mail addresses and the telephone number of the title page (old versions of `scripts/assemble_manuscript.py`, one commit message) were replaced by `***REMOVED-EMAIL***` / `***REMOVED-TEL***` with `git filter-repo --replace-text`. The whole history was then searched for e-mail addresses, telephone numbers, credential values and account documents: none remain (`scripts/_t12b_grep_history.py`).

## Record of the original submission

In the private repository, `manuscript/submitted/` holds the Editorial Manager PDF of the original submission (received from the co-author on 2 November 2026; sha256 in its README). `audit/em_submission_diff.md` (script `scripts/_t11_em_diff.py`) shows that its manuscript block is the v8 base plus a title page with the authors (single-blind review). Neither file is in the public repository (section above).

## Verification notes

- Erratum of Section 4.9 of the submitted manuscript: the stored confidence-threshold table came from an earlier 16-feature XML pipeline; `scripts/_check_t13_safety_layer.py` recomputes the safety layer on the canonical Random Forest and exits 1 by design (the mismatch is the finding); the audit confirmed the erratum by executing the old pipeline's logic.
- The saved Random Forest of the revision (`results/models/rf_revision.joblib`), the frozen model (`../models/rf_frozen_20260331.joblib`) and the deployed model (`../models/rf_model.joblib`) are byte-identical; retraining reproduces them byte for byte.
- `balanced_accuracy_score` is called without fixed labels in some scripts; on this dataset all three classes are present in every evaluated set, so the result is unaffected.
- Line endings: the csv files written by pandas on Windows carry CRLF in the working copy; the repository stores them with LF (`.gitattributes`). Byte-for-byte determinism checks compare two runs of the same script and are unaffected.
- Rounding: every figure printed with 3 decimals is the csv value rounded once (`scripts/_check_table_rounding.py` checks all csv-derived tables of the manuscript and the Supplementary); the 4-decimal values of the registers are never rounded again.
