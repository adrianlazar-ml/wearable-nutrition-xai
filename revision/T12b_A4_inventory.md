# T12b-A4: inventory of the public clone (report only, nothing deleted)

Date: 3 to 4 November 2026. Repository: `wearable-nutrition-xai` (public clone), branch `main` at ff8d0a1 when the inventory started; 500 tracked files. No file was deleted or renamed in this task; the only edits are three documentation corrections found during the smoke test (section 4) and this report.

## 1. Tracked files not referenced by any tracked .py or .md

Scope: the 277 tracked files that are not `.py`, `.md`, `LICENSE*`, `CITATION.cff` or `.git*`. A file counts as referenced when its repository path or its basename occurs verbatim in at least one of the 216 tracked `.py` or `.md` files. 247 are referenced; 30 are not (4,814,095 bytes), grouped by folder:

| Folder | Files | Bytes | Files | Note |
|---|---|---|---|---|
| `_data_samples/` | 10 | 30,177 | `activity_HEADER.csv`, `dataset_combinat_HEADER.csv`, `dataset_combinat_json_HEADER.csv`, `dataset_combinat_json_extended_HEADER.csv`, `oura_combined_HEADER.csv`, `oura_historical_HEADER.csv`, `readiness_HEADER.csv`, `sleep_HEADER.csv`, `sleep_metrics_HEADER.csv`, `stress_HEADER.csv` | 3-row samples of the published aggregates; no script or document names them; the folder itself is named only in `revision/audit/public_release_dryrun.md` |
| `revision/figures/` | 16 | 4,514,251 | `Figure_1`, `Figure_3`, `Figure_4`, `Figure_5`, `Figure_6`, `Figure_7`, `Figure_S1`, `Figure_S2` (`.png` and `.pdf` of each) | not referenced by full basename, but referenced by stem (`Figure_3` and so on) in `revision/figures/FIGURES_MANIFEST.md` and produced by `revision/scripts/export_figures_kbs.py`; `Figure_2.*` and `Figure_S3.*` are named in full elsewhere |
| `revision/results/` | 3 | 26,146 | `t14_shap_stability_bootstrap.csv`, `t14_shap_stability_seeds.csv`, `t14_run_meta_seeds.json` | written by `shap_stability.py` with f-string names (`t14_shap_stability_{mode}.csv`); referenced by pattern, not verbatim |
| `revision/results/figures/` | 1 | 243,521 | `t14_shap_rank_stability_bootstrap.png` | same f-string pattern in `shap_stability.py`; the `.pdf` twin is named in full in `CIFRE_REVIZIE.md` |

Only the 10 files under `_data_samples/` are unreferenced in substance; the other 20 are outputs whose names are built at run time.

## 2. Identical content at different paths (sha256 over all 500 tracked files)

| Group | Bytes | Paths | Reading |
|---|---|---|---|
| imputer | 1,295 | `models/imputer.joblib`, `models/imputer_frozen_20260331.joblib`, `revision/results/models/imputer_revision.joblib` | the frozen imputer, its deployed copy (read by `demo_end_to_end_v2.py`, `streamlit_app_v2.py`) and the revision retrain; identity is the T0.1 reproducibility claim |
| Random Forest | 1,810,593 | `models/rf_frozen_20260331.joblib`, `models/rf_model.joblib`, `revision/results/models/rf_revision.joblib` | same as above for the model (3.6 MB of duplicates in total) |
| Figure 2 | 241,296 | `revision/figures/Figure_2.png`, `revision/results/figures/t13_reliability_rf.png` | manuscript export of the script figure (`FIGURES_MANIFEST.md`: "PNG identic cu sursa") |
| Figure 3 | 209,159 | `revision/figures/Figure_3.png`, `revision/results/figures/t15_group_ablation.png` | same |
| Figure 4 | 116,818 | `revision/figures/Figure_4.png`, `revision/results/figures/t33_expanding_window.png` | same |
| Figure 5 | 243,521 | `revision/figures/Figure_5.png`, `revision/results/figures/t14_shap_rank_stability_bootstrap.png` | same |
| Figure S1 | 286,262 | `revision/figures/Figure_S1.png`, `revision/results/figures/t31_threshold_grid.png` | same |
| Figure S2 | 237,507 | `revision/figures/Figure_S2.png`, `revision/results/figures/t14_shap_rank_stability_seeds.png` | same |
| Figure S3 | 126,248 | `revision/figures/Figure_S3.png`, `revision/results/figures/t13_safety_layer_coverage.png` | same |
| RF predictions | 14,156 | `revision/results/t31_predictions_rf_R60_A85.csv`, `revision/results/test_predictions_rf.csv` | the T3.1 grid cell at the canonical thresholds (60, 85) reproduces the canonical test predictions byte for byte; expected |

The PDF exports differ from the script PDFs (export settings), so only the PNGs pair up. `Figure_1`, `Figure_6` and `Figure_7` have no twin: they are the files of the submitted manuscript, not generated in the revision.

## 3. Historical folders and the root README

Folders that predate the revision: `scripts_archive/` (15 files, 180,796 bytes), the root `audit/` (8 Romanian sprint and audit reports of May 2026, 88,945 bytes), the root reports `AUDIT_NEW_2026-05-26.md`, `AUDIT_PREDEFENSA_2026-06-11.md`, `PLAN_REMEDIERE_v0.5.md`, `SANITY_CHECK_NUMERIC_19042026.md`, `ghid_actualizare_valori.md`, `audit_prompt.md`, `PIPELINE_MAP.md`, and `outputs/figures/xml_pipeline/` (11 files, 1,388,824 bytes) and `outputs/figures/json_pipeline/` (12 files, 1,574,619 bytes), the figures of the two superseded 16-feature pipelines.

The root README does not say they are pre-revision history kept for the record. Section 4 only says "Older versions are under `scripts_archive/`; the dissertation outputs (tables, figures) are under `outputs/`. The dissertation-era documents at the root (audit and remediation reports, `PIPELINE_MAP.md`) are written in Romanian." Proposed sentence for section 4 of the root README (not applied):

> The folders `scripts_archive/` and `audit/`, the root reports `AUDIT_*.md`, `PLAN_REMEDIERE_v0.5.md`, `SANITY_CHECK_NUMERIC_19042026.md`, `ghid_actualizare_valori.md` and `audit_prompt.md`, and the figure folders `outputs/figures/xml_pipeline/` and `outputs/figures/json_pipeline/` (the two superseded 16-feature pipelines) are pre-revision history from the dissertation (March to June 2026), kept for the record and not used by any revision script; the manuscript figures come from `outputs/figures/json_extended_pipeline/`, `outputs/figures/experiments/` and `revision/`.

## 4. Reproducibility smoke test in a fresh virtual environment

Environment: `python -m venv` with the system Python 3.11.9, then `pip install` of the frozen versions (scikit-learn 1.8.0, numpy 2.4.3, pandas 2.3.3, imbalanced-learn 0.14.1, xgboost 3.2.0, scipy 1.17.1, matplotlib 3.10.8, joblib 1.5.3, pytest 9.1.1) and of `revision/requirements.txt` (statsmodels 0.15.0, shap 0.51.0, torch 2.11.0+cpu, lightgbm 4.6.0, catboost 1.2.10, python-docx 1.2.0; pytorch-tabnet 4.1.0 and rtdl_revisiting_models 0.0.2 with `--no-deps`). `pip freeze` matched every pin. The working tree was clean at ff8d0a1 before the run; every script ran from the repository root with `python -u -W error`; after the comparison the tree was restored with `git checkout -- .` (0 modified files, 0 untracked files left).

Steps in the order of `revision/README.md`, section Reproduction (run started 2026-11-03 23:40, ended 2026-11-04 06:02):

| Step | Command | Exit | Elapsed | Documented |
|---|---|---|---|---|
| tests | `python -m pytest revision/scripts/ -q` | 1 (expected) | 31 s | 111 passed, 2 failed in `test_assemble_manuscript.py` (needs the excluded `manuscript/F4_manuscript_changes.md`): 111 passed, 2 failed, as documented |
| T0 | `sanity_check_frozen.py` | 0 | 2 s | |
| T0 | `save_test_predictions.py` | 0 | 4 s | |
| T0 | `persistence_baseline.py` | 0 | 4 s | |
| T0 | `pytest revision/scripts/test_t3_labels.py` (as written in `revision/README.md`) | 5 | 1 s | "no tests ran": the file is a plain script with a `main()`, not a pytest file. Rerun as `python revision/scripts/test_t3_labels.py`: exit 0, "ALL PASS" (10,201 pairs, 100 rows, 29 threshold configurations). `revision/README.md` corrected in this commit |
| T1.1 | `extended_metrics.py` | 0 | 19 s | 20 to 30 s |
| T1.2 | `statistical_tests.py` | 0 | 9 s | |
| T1.3 | `calibration_rf.py --stage A`, `--stage B`, `safety_layer_calibrated.py` | 0, 0, 0 | 4 s, 10 s, 3 s | a few seconds each |
| T1.4 | `shap_stability.py --mode bootstrap`, `--mode seeds` | 0, 0 | 67 s, 106 s | about one minute per mode |
| T1.5 | `group_ablation.py` | 0 | 34 s | 20 to 25 s |
| T3.1 | `threshold_sensitivity.py` | 0 | 49 s | |
| T3.2 | `label_agreement.py` | 0 | 5 s | a few seconds |
| T3.3 | `expanding_window.py` | 0 | 37 s | 20 to 30 s |
| T2.1 | `extra_models.py --mode main`, `--mode seeds` | 0, 0 | 250 s, 475 s | about 3 and 11 minutes |
| T2.2 | `hyperparameter_search.py --mode cv` | 0 | 21,394 s | about 28 minutes; see the note below |
| T2.2 | `hyperparameter_search.py --mode test`, `--mode seeds` | 0, 0 | 151 s, 260 s | 3 to 4 and about 6 minutes |
| F4 | `assemble_manuscript.py` and the checkers `_check_t41_*`, `_check_table_rounding`, `style_check`, `_check_t8_parity`, `_check_t11*`, `_check_t115_cover_letter`, `_t11_em_diff` | skipped | | documented as needing the excluded `manuscript/` docx and `authors.json` |
| T3.2b | `watch_only_sensitivity.py` | not run | | not in the numbered Reproduction list of `revision/README.md`; its inputs (`revision/audit/p0_parser_vs_watch_only.csv`) are in the repository |

Note on the T2.2 cv run: the script's own `elapsed_seconds` was 21,388.8 s, of which one single fold timing was 19,943.3 s (the committed value for the same fold is 3.5 s); every other fold took the documented few seconds. The run went through the night and the stall is consistent with the machine sleeping, not with the code; every output value is identical (below). The documented 28 minutes stand for an attended run.

Comparison of every file the run modified (39 tracked files; 0 new untracked files) with the committed version:

| Verdict | Files | Detail |
|---|---|---|
| identical content (CRLF in the working copy only, `.gitattributes` stores LF) | 5 csv | `test_predictions_rf.csv`, `test_predictions_xgb.csv`, `test_predictions_svm.csv`, `test_predictions_majority.csv`, `test_predictions_persistence.csv` |
| identical values, `run_date` column differs | 3 csv | `persistence_baseline.csv`, `test_predictions_metrics.csv`, `t11_extended_metrics.csv` (every numeric cell equal as text) |
| identical except the run-date line | 18 md tables | `t11_*`, `t12_*`, `t13_*`, `t15_*`, `t21_*`, `t22_*`, `t31_*`, `t32_*`, `t33_*` tables |
| identical except `run_date` and `elapsed_seconds` | 8 json | `t12_run_meta.json` (`run_date` only); `t13`, `t14` (bootstrap, seeds), `t15`, `t31`, `t32`, `t33` run_meta (`run_date` and `elapsed_seconds`) |
| run date, timings, local paths and the sha256 of the md tables | 5 json | `t21_run_meta.json`, `t21_run_meta_seeds.json`, `t22_run_meta_cv.json`, `t22_run_meta_seeds.json`, `t22_run_meta_test.json`: `versions.pytorch_tabnet_path` and `versions.rtdl_path` record the absolute paths of the running environment; `files_sha256` / `table_files_sha256` of the md tables change because the tables carry the run date; every csv sha256 is unchanged |
| untouched | all csv not listed, all png, all pdf, all npy, all joblib | not modified by the run, so byte-identical, including the 18 figures under `revision/results/figures/` and the saved models |

Result: no value mismatch. Every csv, png, pdf, npy and joblib produced by the 20 steps is byte-identical or value-identical to the committed file; the only differences are run dates, timings and the two environment paths in the T2.1 / T2.2 metadata.

Documentation corrections made in this commit (found by the smoke test): root README section 6 said `group_ablation.py` writes to `revision/figures/`; the script writes to `revision/results/figures/` (section 4 of the README now distinguishes the two figure folders). `revision/README.md` said `pytest revision/scripts/test_t3_labels.py`; the file is a plain script and the line now says `python revision/scripts/test_t3_labels.py`.

## 5. Other observations

- Local Windows paths with the user name are in 16 tracked files: the 5 T2.1 / T2.2 run_meta json files (`versions.pytorch_tabnet_path`, `versions.rtdl_path`), `AUDIT_PREDEFENSA_2026-06-11.md`, `outputs/reports/membership_vs_api_comparison.txt`, `parse_health_auto_export.py`, `revision/F5_format_report.md`, `revision/audit/AUDIT_F4_provenance_shap_ablation_agreement.md`, `revision/audit/_check_f1_empirical_output.txt`, `revision/audit/_check_f4_empirical_output.txt`, `revision/audit/_check_f5_empirical_output.txt`, `revision/audit/public_release_dryrun.md`, `revision/audit/public_release_list.md`, `revision/scripts/assemble_manuscript.py`. They disclose a user name and folder layout, not personal data; `public_release_dryrun.md` already lists most of them.
- No `__pycache__` or other cache file is tracked.

## 6. Proposed delete list (nothing deleted in this task)

| Candidate | Bytes | Reason to delete | Reason to keep |
|---|---|---|---|
| `_data_samples/` (10 files) | 30,177 | unreferenced by any script or document; 3-row samples of files that are published in full under `data/` and `models/`; the eleventh sample (sleep_detailed) was already removed at T12b-A2 as per-record data | none; `public_release_dryrun.md` mentions the folder as "header-only csv" |
| `models/rf_model.joblib`, `models/imputer.joblib` | 1,811,888 | byte-identical to the frozen files; the dedicated names are the "deployed" copies | read by `demo_end_to_end_v2.py` and `streamlit_app_v2.py`; deleting them means editing the demo paths |
| root `audit/`, `AUDIT_*.md`, `PLAN_REMEDIERE_v0.5.md`, `SANITY_CHECK_NUMERIC_19042026.md`, `ghid_actualizare_valori.md`, `audit_prompt.md` | about 190,000 | Romanian pre-revision process documents; `AUDIT_PREDEFENSA_2026-06-11.md` carries a local path | they document the May 2026 audit that the Repository history section relies on; prefer the README sentence of section 3 over deletion |
| `outputs/figures/xml_pipeline/`, `outputs/figures/json_pipeline/` (23 files) | 2,963,443 | figures of the two superseded 16-feature pipelines, not used by the manuscript | the erratum of Section 4.9 (confidence-threshold table from the 16-feature XML pipeline) is explained by these outputs; keep with the README sentence |
| `revision/figures/Figure_{2,3,4,5,S1,S2,S3}.png` | 1,460,811 | byte-identical to the script PNGs under `revision/results/figures/` | they are the files submitted with the manuscript, listed in `FIGURES_MANIFEST.md`; keep |

Recommended action from this list: delete `_data_samples/` only (10 files); scrub the two absolute paths from the 5 run_meta json files by storing paths relative to the repository root in `extra_models.py` and `hyperparameter_search.py` at the next rerun, rather than editing the committed json by hand; keep everything else and add the proposed README sentence.
