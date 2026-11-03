# public_release_dryrun.md - dry-run of the filtered public clone (T12, 2026-11-03; evidence only, nothing published)

Private repository: `C:\Users\Adrian\disertatie_v2`, master b1dd500, no remote. Procedure under test: `hygiene_summary.md`, section Public release (clone `--no-local`, `git filter-repo --invert-paths` on the exclusion list, checks, commit-map). Variant run: "code + aggregates" (base list). Variant "code only": reported, not run. Tool: git-filter-repo 2.47.0 (installed with `pip install git-filter-repo --break-system-packages`; was absent). Grep script: `revision/scripts/_t12_grep.py` (kept as evidence). The private repository was not modified: HEAD, `git status` and `git count-objects` identical before and after (section 6), apart from the commit of this report.

## 1. Exclusion list used (T12.0)

`data/`, `apple_watch_api/raw_json/`, `_referinte_upb/`, `disertatie_audit.zip`, `audit_csv_output.txt`, `sanity_check_output.txt` (hygiene_summary step 2) plus `revision/manuscript/submitted/` (T11, the Editorial Manager PDF with e-mail addresses and a telephone number).

Check against `tracked_data_inventory.md` (180 rows: 77 "da", 11 "agregat zilnic (Option C)", 92 "nu"): all 77 "da" files fall under `data/`; 0 "da" files outside the list. The 11 Option C aggregates: 10 under `data/` (activity, dataset_combinat x3, oura_combined, oura_historical, readiness, sleep, sleep_metrics, stress; removed by the base list) and `models/dataset_snapshot_20260331.csv` (kept by the base list). Tracked files under the list: 110 of 633. No `.env`, `.pem` or `.key` file is tracked at HEAD.

## 2. Sizes before and after (T12.1)

| Repository | Commits | Tracked files | Objects | Size |
|---|---|---|---|---|
| Private (b1dd500, loose objects, no pack) | 158 | 633 | 1,799 loose | 325.90 MiB (`count-objects -vH`, size) |
| Clone before filtering (`--no-local`, packed) | 158 | 633 | 1,782 in pack | 150.59 MiB (size-pack) |
| Clone after `filter-repo` | 158 (0 pruned) | 523 | 1,662 in pack | 27.51 MiB (size-pack); working tree 44.1 MB |

Clone: `C:\Users\Adrian\Documents\Backup\disertatie_v2_public_dryrun\` (kept for inspection; no remote: git-filter-repo removed the `origin` that `git clone` had created; nothing pushed). master rewritten b1dd500 -> 171108a. Commit map (159 old -> new pairs, no commit mapped to zero) saved outside the clone: `C:\Users\Adrian\Documents\Backup\commit-map_dryrun_2026-11-03.txt`. `git ls-files` on `data/` = 0 and on every excluded path = 0. The "263 MiB" of hygiene_summary.md is outdated (the private repository holds 325.90 MiB of loose objects today).

## 3. Sensitive-string grep (T12.0 step 3 on the private tree, T12.2 step 1 on the clone)

Patterns: e-mail regex, `0040`, `tel:`, `C:\Users`, `/Users/`, token, api_key / apikey, Bearer, password, secret, client_id, `.env`, Oura + account / invoice / subscription; tracked `.env` / `.pem` / `.key` names. Classes: (a) under an excluded path, (b) stays after filtering (to be decided), (c) false positive by rule (decimal values such as 0.0040, the word "token" for text tokens or the packages pytokens / tiktoken, "secretariat").

| Tree | Hits | (a) | (b) | (c) |
|---|---|---|---|---|
| Private, HEAD b1dd500 | 4,207 | 3,968 | 154 | 85 |
| Clone after filtering | 239 | 0 | 154 | 85 |

The 154 (b) hits, grouped (identical before and after, since none lies under an excluded path):

| Group | Files | Hits | Assessment |
|---|---|---|---|
| Authors' e-mail addresses and the corresponding author's telephone (T11) | revision/scripts/assemble_manuscript.py (EMAIL_FIRST, EMAIL_SECOND, TEL_SECOND), revision/F4_cover_letter_R1.md, revision/TEXT_DECISIONS.md, revision/T11_summary.md, revision/audit/em_submission_diff.md, revision/audit/t11_docx_text_diff.md, revision/manuscript/assembly_plan.md | 18 e-mail + 9 phone + 9 `tel:` | real; by design of the single-blind title page. The e-mail addresses are on the published title page; the telephone number is the item to decide (see section 7) |
| Same data inside docx (extracted text incl. raw w:t runs, headers, footers, comments) | KBS_Manuscript_R1.docx, KBS_Manuscript_R1_marked.docx (both e-mails + phone), KBS_Cover_Letter_R1.docx (***REMOVED-EMAIL***); 0 in BASE_MANUSCRIPT.docx, BASE_MANUSCRIPT_v8.docx, Highlights, Supplementary, Response | 3 files | real; same decision as above |
| Local Windows paths with the user name | revision/results/t21_run_meta.json, t21_run_meta_seeds.json, t22_run_meta_cv.json, t22_run_meta_seeds.json, t22_run_meta_test.json (`pytorch_tabnet_path`, `rtdl_path`); also `apple_watch_api/api_sync.log` line 3 (`C:\Users\Adrian\disertatie\...`, not matched by the regex because of the single backslash) and AUDIT_PREDEFENSA K6 (mentions `C:\Users\Adrian\iCloudDrive`) | 10 + 2 | harmless (the user name is the author's); cosmetic decision |
| Credential mechanism in code and process documents, no value | oura_auth.py, oura_collect.py (token / client_id / client_secret / Bearer read from environment variables), .gitignore, PIPELINE_MAP.md, audit_prompt.md, AUDIT_PREDEFENSA_2026-06-11.md, audit/AUDIT_INDEPENDENT_2026-05-29.md, revision/audit/AUDIT_F4_provenance_shap_ablation_agreement.md, revision/CIFRE_REVIZIE.md (line 451, "token" of a model), t21_pip_freeze_* (pytokens, tiktoken) | 67 | no secret value in the tree; the two audit documents state that the real Oura secrets are in the git history (see next group) |
| Oura account / invoice / subscription | revision/audit/tracked_data_inventory.md (38 rows naming the invoice pdf and account csv files under data/Oura/Subscriptions/, file names and sizes only), hygiene_summary.md, AUDIT_DECISIONS.md | 41 | file names only; the files themselves are removed |

Finding outside the grep (history, not tree): `.env` with the real Oura credentials was committed in f7cc4ca ("Snapshot pre-audit, 26.05.2026") and untracked in 91d8ee6 ("Sprint 6: ... untrack .env (secret tracked)"); keys OURA_CLIENT_ID (36 characters), OURA_CLIENT_SECRET (43), OURA_REDIRECT_URI (30), OURA_ACCESS_TOKEN (45); values not printed here. `.env` is NOT on the exclusion list, so after filtering the clone still carries it in history (clone commits c912b62 and b7b9716; `git log --all -- .env` in the clone lists both). The audit of 2026-05-29 (S1) and AUDIT_PREDEFENSA (K3) already required rotation of these credentials; their current status is not recorded in the repository.

## 4. Checks in the clone (T12.2 step 2)

- `python revision/scripts/_check_t41_outputs.py`: 76 checks, 0 FAIL. The git-based checks compare the working tree with HEAD and the sha256 of files (BASE_MANUSCRIPT_v8.docx, CIFRE_CANONICE.md), never a commit hash, so the rewritten history does not affect them.
- `python -m pytest revision/scripts/ -q`: 107 passed, 2 failed, 3 errors. All five read the raw inputs through `common.load_raw()` (`data/oura_combined.csv` and `apple_watch_api/apple_health_daily.csv`) and stop at `FileNotFoundError: data/oura_combined.csv`: test_threshold_sensitivity.py::test_canonical_cell_reproduces_canonical_labels and ::test_rows_and_cut_do_not_depend_on_thresholds (direct `load_raw`), test_extra_models.py::test_inner_split_sizes_and_active and ::test_canonical_preprocess_sizes (module fixture `data()` -> `extra_models.load_data()` -> `common.load_canonical()`), test_hyperparameter_search.py::test_fold_rows_and_active (same fixture through `hs.em.load_data()`).
- Consequence beyond the tests: `common.load_canonical()` rebuilds the dataset from the raw daily files, and it is the entry point of the producing scripts calibration_rf, expanding_window, extra_models, group_ablation, hyperparameter_search, label_agreement, save_test_predictions, shap_stability, threshold_sensitivity and watch_only_sensitivity. With the base list none of them can run in the clone; only persistence_baseline, extended_metrics and the `_check_t1x_inputs` scripts read `models/dataset_snapshot_20260331.csv`. The sentence of `revision/README.md` ("`../models/dataset_snapshot_20260331.csv` ... is the only input of the T1, T2 and T3 scripts") is therefore not accurate and must be corrected at release, whatever Option C decides.

## 5. Remaining tracked tree in the clone (T12.2 step 3)

523 files, 44.1 MB. No file above 5 MB (largest: revision/manuscript/BASE_MANUSCRIPT_v8.docx 3.12 MB, KBS_Manuscript_R1_marked.docx 2.06 MB, KBS_Manuscript_R1.docx 2.06 MB, models/rf_frozen_20260331.joblib 1.81 MB, models/rf_model.joblib 1.81 MB).

| Folder (level 1 / 2) | Files | MB |
|---|---|---|
| root: 11 markdown / text files (AUDIT_NEW, AUDIT_PREDEFENSA, CIFRE_CANONICE, the tooling context file, PIPELINE_MAP, PLAN_REMEDIERE_v0.5, README, SANITY_CHECK_NUMERIC, audit_prompt, ghid_actualizare_valori, .gitattributes, .gitignore) | 13 | 0.24 |
| root: 33 Python scripts of the frozen pipeline (ablation_study ... threshold_sensitivity, sanity_check_frozen, t3_labels) | 33 | 0.46 |
| _data_samples/ (header-only csv) | 11 | 0.04 |
| apple_watch_api/ (api_sync.log, apple_health_auto.csv, apple_health_daily.csv) | 3 | 0.19 |
| audit/ | 8 | 0.09 |
| models/ (dataset_snapshot_20260331.csv, frozen_meta, 2 joblib forests, imputer, ...) | 6 | 3.74 |
| outputs/bootstrap_ci, outputs/figures, outputs/reports, outputs/tables | 1 / 62 / 9 / 15 | 0.00 / 9.16 / 0.16 / 0.01 |
| revision/ (root files), revision/audit, revision/f8, revision/figures, revision/manuscript, revision/results, revision/scripts, revision/vendor | 15 / 47 / 7 / 21 / 35 / 143 / 62 / 14 | 0.40 / 1.55 / 0.08 / 4.94 / 13.65 / 8.18 / 0.88 / 0.15 |
| scripts/canonical, scripts_archive/comparison, scripts_archive/exploration, scripts_archive/old_versions | 2 / 5 / 5 / 5 | 0.02 / 0.06 / 0.04 / 0.08 |

## 6. Private repository before and after the dry-run

Before (T12.0): HEAD b1dd500; `git status`: the three untracked items of the tooling context file only; `count-objects`: 1,799 loose objects, 325.90 MiB, 0 packs. After the clone, the filtering and the checks, before the commit of this report: identical (HEAD b1dd500, same three untracked items plus the new script `_t12_grep.py` before it was committed, 1,799 objects, 325.90 MiB). The values after the report commit are in T12.3 of the session summary.

## 7. Variant "code only" (report, not run)

In addition to the base list it would remove `models/dataset_snapshot_20260331.csv` (116,680 bytes; 958 labeled days, 19 lag-1 features, labels) and `apple_watch_api/apple_health_daily.csv` (187,182 bytes; the parser output, 1,083 rows). Two more files of the same folder are candidates: `apple_watch_api/apple_health_auto.csv` (3,101 bytes, 41 rows of daily aggregates of 26 March 2026 with the source column) and `apple_watch_api/api_sync.log` (685 bytes, 7 lines with a local path). The ten Option C aggregates under `data/` are already removed by the base list. Effects: pytest unchanged (107 / 2 / 3: no test reads the snapshot), `_check_t41_outputs.py` unchanged (76 / 76), but persistence_baseline.py, extended_metrics.py, `_check_t11_inputs.py`, `_check_t13_inputs.py` and `_check_t13_safety_layer.py` lose their only input, so no result of the revision can be regenerated from the public clone; the README must then say so for every script.

## 8. Open before a real release (decisions, nothing applied)

1. `.env` in history: add `--path .env` to the filter-repo list (and verify `git log --all -- .env` is empty afterwards), and settle the rotation of the Oura credentials recorded as required since 2026-05-29. Without this the release would publish a client secret and an access token.
2. Category (b) of the grep: the telephone number of the corresponding author in 7 text files and 2 docx (decide: keep, since it is on the submitted title page, or strip from the process documents and keep it only in the docx); the e-mail addresses (same files plus the cover letter docx); the local paths in the 5 run_meta json files, api_sync.log and AUDIT_PREDEFENSA K6.
3. `data/oura_combined.csv` and `apple_watch_api/apple_health_daily.csv`: the producing scripts read them through `common.load_raw()`; either Option C includes `data/oura_combined.csv` explicitly (it is under the excluded `data/`), or the README states which scripts cannot run; in both cases correct the README sentence on the snapshot being the only input (section 4).
4. LICENSE: absent at the root. README placeholders still present (3): "[title and authors to be completed at acceptance]", "[Zenodo DOI to be added]", "[license to be chosen by the authors]"; the README does not cite Tomczak & Tomczak, so the author-order question of F5 does not arise there. Commit-map: keep `commit-map_dryrun_<date>.txt` next to the release and note its path at the top of CIFRE_REVIZIE.md (hygiene_summary step 5).
5. Oura invoice file names in `revision/audit/tracked_data_inventory.md` (names and sizes of removed files): harmless, decide whether the inventory itself ships.
6. hygiene_summary.md: update the size figure (325.90 MiB) and add `.env` to the procedure (done in the same commit as this report, as a correction of the procedure).
