# Explainable Artificial Intelligence for Personalized Nutrition: A Multimodal Framework Integrating Smart Ring and Smartwatch Data

Adrian Lazar, Catalin Buiu

National University of Science and Technology POLITEHNICA Bucharest, Faculty of Automatic Control and Computers, Department of Automatic Control and Systems Engineering.

This repository, `wearable-nutrition-xai`, is the public entry point to the code and the daily aggregates behind the manuscript. The detailed guide to the revision work (environment, order of the scripts, determinism checks, provenance notes) is `revision/README.md`.

## 2. Purpose

The repository holds the code and the daily aggregate data supporting the Revision 1 manuscript submitted to Knowledge-Based Systems (Elsevier). The work originates in the first author's MSc dissertation, "AI System for Personalized Nutrition and Diet Recommendations Based on Wearable and mHealth Data" (National University of Science and Technology POLITEHNICA Bucharest, June 2026): a three-class daily physiological state classifier (Rest, Normal, Active) trained on one participant's Oura ring and Apple Watch data, with SHAP explanations feeding a rule-based nutrition recommender. The dissertation pipeline scripts at the root and the revision scripts under `revision/` are both included, so every figure reported in the manuscript can be regenerated.

## 3. Repository history

- The git history starts at the pre-audit snapshot of 26 May 2026, the state of the code at the dissertation defense. Design choices made before that date (the labeling thresholds 60 / 85 / 75, the temporal 80/20 split, the feature set) are documented in the dissertation and in the manuscript and cannot be dated from git.
- The frozen pipeline of 31 March 2026 (the frozen Random Forest, its imputer and the canonical figures) is tagged `canonical-8020-20260331`.
- The tag `b3-abandoned` marks a single commit off the main line that records the abandoned experiment B3, a 70/10/20 train / validation / test split that exposed the ring-generation confound. It is the evidence for Limitation 7 of the manuscript.
- The public repository is a filtered copy of the authors' private working repository, with the raw data and the manuscript files removed from the whole history. Two commit-maps (old hash to new hash) exist outside the repository, so the private commit hashes cited in `revision/CIFRE_REVIZIE.md` and in `revision/audit/` map to the public history through them.

## 4. What is included

Daily aggregates (one row per day of a single participant, the first author, 2023 to 2026; license CC BY 4.0, see `LICENSE-DATA`):

- `data/activity.csv`, `data/readiness.csv`, `data/sleep.csv`, `data/sleep_metrics.csv`, `data/stress.csv`: Oura API daily scores and metrics.
- `data/oura_historical.csv` (manual export, before the API) and `data/oura_combined.csv` (the merged daily ring table, 1,009 days).
- `data/dataset_combinat.csv`, `data/dataset_combinat_json.csv`, `data/dataset_combinat_json_extended.csv`: the daily feature tables of the three dissertation pipelines (XML, JSON, JSON extended).
- `models/dataset_snapshot_20260331.csv`: the 958 labeled days with the 19 lag-1 features and the labels, the reference against which the revision scripts check the dataset they rebuild.
- `apple_watch_api/apple_health_daily.csv`: the daily HealthKit aggregates (1,083 days) produced by `parse_health_auto_export.py` from the Health Auto Export JSON files.

Models: the frozen Random Forest `models/rf_frozen_20260331.joblib` and its imputer `models/imputer_frozen_20260331.joblib` (byte-identical to `models/rf_model.joblib`, `models/imputer.joblib` and to `revision/results/models/rf_revision.joblib`, `revision/results/models/imputer_revision.joblib`; their sha256 are recorded in `revision/CIFRE_REVIZIE.md`), with `models/frozen_meta_20260331.json`.

Code:

- The dissertation pipeline scripts at the repository root: data collection (`oura_auth.py`, `oura_collect.py`, `oura_merge.py`, `parse_health_auto_export.py`), the classification pipelines (`cap5_pipeline*.py`, `cap5_compare_algorithms*.py`, `t3_labels.py`), the validation experiments (`threshold_sensitivity.py`, `ablation_study.py`, `confidence_threshold.py`, `smote_comparison.py`, `experiments_new.py`, `learning_curves_comparison.py`, `circularity_sanity_check.py`), the SHAP figures (`generate_waterfall_*.py`), the frozen model (`freeze_production_model.py`, `sanity_check_frozen.py`, `prospective_validation.py`), the recommender and the demo (`nutrition_recommender.py`, `demo_end_to_end_v2.py`, `streamlit_app_v2.py`), and `generate_canonical_numbers.py`, which writes `CIFRE_CANONICE.md`. Older versions are under `scripts_archive/`; the dissertation outputs (tables, figures) are under `outputs/`. The dissertation-era documents at the root (audit and remediation reports, `PIPELINE_MAP.md`) are written in Romanian.
- `revision/`: the revision scripts (`revision/scripts/`), the saved test predictions, result tables, bootstrap indices and saved models (`revision/results/`), the figures (`revision/figures/`), the registers (`CIFRE_CANONICE.md` at the root for the submitted manuscript, `revision/CIFRE_REVIZIE.md` for every figure produced in the revision), the plan (`revision/KBS_Revision_Plan.md`) and the audit reports (`revision/audit/`).

## 5. What is excluded and why

- Raw per-record and per-minute data: the Apple Health export (native XML and the daily Health Auto Export JSON files), the Oura raw exports (per record, per night, per session) and GPS data.
- The Oura subscription invoices and account documents.
- The manuscript files (base docx, text blocks, the Revision 1 docx outputs, the cover letter): the manuscript is under review, and the published article will be the reference.
- Credentials (the Oura OAuth2 application and token were deleted and revoked) and the title-page contact details.

The raw biometric data are sensitive personal data of one participant and are not shared. The exclusion list and the procedure are in `revision/audit/public_release_list.md` and `revision/audit/hygiene_summary.md`.

## 6. How to reproduce

Environment: Python 3.11.9 with the frozen package versions recorded in `models/frozen_meta_20260331.json` (scikit-learn 1.8.0, numpy 2.4.3, pandas 2.3.3, imbalanced-learn 0.14.1, xgboost 3.2.0, scipy 1.17.1) and the revision-only packages pinned in `revision/requirements.txt`:

```
python -m pip install scikit-learn==1.8.0 numpy==2.4.3 pandas==2.3.3 imbalanced-learn==0.14.1 xgboost==3.2.0 scipy==1.17.1
python -m pip install -r revision/requirements.txt
```

All runs are CPU, single thread, seed 42. Run everything from the repository root.

Unit tests:

```
python -m pytest revision/scripts/ -q
```

Expected on this clone: 111 passed, 2 failed. The two failures are in `revision/scripts/test_assemble_manuscript.py` and read `manuscript/F4_manuscript_changes.md`, one of the excluded manuscript sources. Every test of the T0, T1, T2 and T3 scripts passes.

Two scripts verified to run end to end on the published aggregates (they rebuild the dataset from `data/oura_combined.csv` and `apple_watch_api/apple_health_daily.csv` and check it against the snapshot):

```
python -W error revision/scripts/group_ablation.py
python -W error revision/scripts/label_agreement.py
```

- `group_ablation.py` (T1.5, about 23 s) writes `revision/results/t15_logo.csv`, `t15_logo_table.md`, `t15_kogi.csv`, `t15_kogi_table.md`, `t15_run_meta.json` and `revision/figures/t15_group_ablation.png` / `.pdf`. Full Random Forest: macro F1 0.5063, balanced accuracy 0.6406, MCC 0.1416.
- `label_agreement.py` (T3.2, about 5 s) writes `revision/results/t32_inventory.csv`, `t32_descriptives.csv`, `t32_descriptives_table.md`, `t32_tests.csv`, `t32_tests_table.md`, `t32_run_meta.json`. Cliff's delta Active minus Rest of `aw_hr_mean`: -0.840 [-0.912, -0.749].

Every csv, png and pdf is byte-identical to the committed file; the md tables and the run_meta json differ only in the run date. The order of the other revision scripts, with their running times, is in `revision/README.md`.

Checks that cannot run from this repository because they need the excluded manuscript docx: `revision/scripts/assemble_manuscript.py` and its checkers `_check_t41_outputs.py`, `_check_t41_numbers.py`, `_check_table_rounding.py`, `style_check.py`, `_check_t8_parity.py`, `_check_t11*.py`, `_check_t115_cover_letter.py`, `_t11_em_diff.py`. The root scripts that read the raw folders (`cap5_pipeline.py`, `cap5_compare_algorithms.py`, `parse_health_auto_export.py` on the JSON folder, `oura_collect.py`) cannot run either; `sanity_check_frozen.py` and the frozen 19-feature pipeline on `data/dataset_combinat_json_extended.csv` can.

## 7. LLM use

Parts of the audit and drafting workflow used an LLM assistant (Claude, Anthropic), as declared in the manuscript. All reported figures were reproduced from the scripts in this repository; the registers `CIFRE_CANONICE.md` and `revision/CIFRE_REVIZIE.md` record script, seed and run date for each value.

## 8. Licenses

- Code and documentation: MIT (`LICENSE`).
- Daily aggregates (the csv files listed in section 4): Creative Commons Attribution 4.0 International (`LICENSE-DATA`).

## 9. Citation

See `CITATION.cff` (software citation and the preferred citation of the article). The archived release v1.0-R1 will carry the Zenodo DOI 10.5281/zenodo.XXXXXXX (placeholder, filled after the release is archived); the research data statement of the Knowledge-Based Systems manuscript (Option C) will cite this DOI.
