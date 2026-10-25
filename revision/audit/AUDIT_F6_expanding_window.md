# AUDIT F6: expanding-window temporal robustness (T3.3) and the open question of slice F1

Auditor role: external, read-and-report only. Branch `audit-r1`, HEAD `f3b9e6a` (slices F1 to F5 committed). Run date: 2026-10-25.
Environment used for every command below: Python 3.11.9, scikit-learn 1.8.0, numpy 2.4.3, pandas 2.3.3, imbalanced-learn 0.14.1, shap 0.51.0, scipy 1.17.1 (identical to `versions` in `t33_run_meta.json`).

Slices F1 to F5 were read only for their Findings sections; nothing from them is reused as a fact. Finding F1 of slice F1 (two "previous day" conventions) is the subject of question 12 below. Finding F3.2 (two bootstrap skip rules) applies to T3.3 as it did to T3.1: the per-fold CIs use `statistical_tests.paired_bootstrap` (absent-class skip) and the `*_t04` persistence CIs use `persistence_baseline.bootstrap_ci` (no skip, own RNG); the register states both (`CIFRE_REVIZIE.md:423`), so the point is not re-examined.

Time limits respected: the longest execution is the empirical script (26 s, which calls `run_fold` for the four folds) and the recompute script (about 40 s, three RF refits of under 2 s each plus 9,000 bootstrap evaluations). No new experiment; the chronological-order pooled CI of E6 is a diagnostic that sizes a documentation gap and is not a proposed figure.

Files written by this audit (nothing else was modified; no file under `revision/results/`, `revision/scripts/`, `revision/manuscript/` or the repo root was touched; `pytest` on `test_expanding_window.py` wrote nothing under `revision/`):

- `revision/audit/AUDIT_F6_expanding_window.md` (this report)
- `revision/audit/AUDIT_SUMMARY.md` (T5, synthesis of slices F1 to F6)
- `revision/audit/recompute_f6.py` + `recompute_f6_output.txt` (T3; imports numpy, pandas, scikit-learn, imbalanced-learn and the standard library only; nothing from `revision/scripts/` or the repo root; run with `python -I`; 154 checks)
- `revision/audit/_check_f6_empirical.py` + `_check_f6_empirical_output.txt` (T2 evidence E0 to E10; imports `expanding_window` and the modules it imports in memory and calls none of their `main()`; 69 checks, 26 s)

All five files LF, ASCII (`file`: "ASCII text"; both scripts set `sys.stdout.reconfigure(newline="\n")`). Abbreviations: R = `recompute_f6_output.txt`, E = `_check_f6_empirical_output.txt`, `ew` = `revision/scripts/expanding_window.py`, `ts` = `threshold_sensitivity.py`, `st` = `statistical_tests.py`, `pb` = `persistence_baseline.py`, `cm` = `common.py`, `ss` = `shap_stability.py`, `ga` = `group_ablation.py`.

---

## T0. Inventory

`git status --short` at the start: unchanged from slice F5 (one deleted tracked zip, three untracked paths outside `revision/`, the untracked `data/apple_health/New Automation/` folder, the author's untracked `revision/audit/AUDIT_DECISIONS.md`). Nothing under `revision/results/` or `revision/scripts/` is modified or untracked.

sha256 first 16 hex chars of the inputs read (E0):

| file | sha256[:16] | role |
|---|---|---|
| `revision/scripts/expanding_window.py` | 649ebad9f59ff337 | T3.3 (fold_bounds, run_fold, pooled_bootstrap, main) |
| `revision/scripts/test_expanding_window.py` | 7b4e9d631a52611a | unit tests (5 / 5 pass, 4.7 s, re-run by the audit) |
| `revision/scripts/common.py` | fb73b00eb9a650d3 | load_raw, build_dataset, preprocess_split (audited in F1) |
| `revision/scripts/threshold_sensitivity.py` | 74a8c5d02354b479 | smote_k, persistence_predictions, majority_predictions, is_comparable, top_shap (audited in F3) |
| `revision/scripts/persistence_baseline.py` | c77ec25813d6dd6e | evaluate, bootstrap_ci (T0.4, audited in F1) |
| `revision/scripts/statistical_tests.py` | c8506d2cdc5e21fa | paired_bootstrap, INDEX_FILE, LABELS (audited in F2) |
| `revision/scripts/shap_stability.py` | 4ca77cc4e3827e78 | mean_abs_shap, aggregate_multiclass, rank_desc (audited in F4) |
| `revision/scripts/group_ablation.py` | 654d864dd88c8a1f | metrics, f1_macro, f1_active, read_t11_rf_f1_ci |
| `revision/scripts/save_test_predictions.py` | 24c7c7d9cf976360 | build_models (RF 100 / 10 / 5 / seed 42) |
| `revision/t3_labels.py` | 5e8b726fc629c665 | label rule (question 12, E9b) |
| `revision/results/t33_expanding_window.csv` | 22f997ef679cf1c6 | fold table (== `CIFRE_REVIZIE.md` artefact table) |
| `revision/results/t33_shap_top3.csv`, `t33_run_meta.json` | 776aead3ee6e3040 / 2b642dce394290ed | SHAP top-3, metadata |
| `revision/results/t33_predictions_rf_fold{1,2,3,4}.csv` | 56ae01513e4217af / e3aa6c26585bab4b / 02d9e0dc3f377c77 / 85136044365d3984 | per-block predictions |
| `revision/results/test_predictions_rf.csv`, `test_predictions_persistence.csv` | aea39c6dd2989e24 / ab228a11e65bfe17 | canonical references (T0.2, T0.4) |
| `revision/results/bootstrap_indices_1000x192_seed42.npy` | fff3b2834c061962 | shared matrix (== `default_rng(42)`, E5, R4) |
| `models/dataset_snapshot_20260331.csv` | 176649b59fef3adb | input of the independent refits and of the persistence reconstruction |
| `data/oura_combined.csv`, `apple_watch_api/apple_health_daily.csv` | 11d6d4de0eac31f9 / 4692a651003b6696 | raw frame (1,009 rows) for question 12 |
| `revision/manuscript/KBS_Manuscript_R1.docx`, `KBS_Response_R1.docx` | cdd4b55303bb61f5 / b8c65ba7c559c9d7 | Section 3.11, 4.8, Table 11, Figure 4 caption, Discussion; response to R1.10 |

Git: T3.3 was committed once, `3b7f994` (2026-10-19 06:24:54), which adds the script, the unit tests, the csv / json / md outputs, the four prediction files and the figure together; `t33_run_meta.json` records `run_date` 2026-10-19T06:22:11. The script was never modified afterwards (`git log --all -- revision/scripts/expanding_window.py` has one entry). See Q8.

---

## T1. Trace tables (from reading the code)

### (a) `run_fold(k, X, y, dates, idx)` (ew:160-210), one fold

| step | code | subset / object | note |
|---|---|---|---|
| block bounds | ew:84-89 `fold_bounds(k, n, block)`: `te0 = n - block * k`; returns `(te0, te0, te0 + block)`; ew:162 called with `len(X)` = 958 and the module constants `BLOCK = 192` (ew:76) | train rows `[0, te0)`, test rows `[te0, te0 + 192)` | computed from N and the block length; raises for `n - block * k < 1` (fold 5) |
| disjointness | ew:92-99 `all_fold_bounds()` asserts no overlap between the four test ranges; ew:275 called in `main` | 4 folds (`N_FOLDS = 4`, ew:77) | union == rows 190..957 checked by the unit test (test_expanding_window.py:39) and by E1 |
| labels, counts, k | ew:163-165 `y_tr = y.iloc[:tr_end]`, `y_te = y.iloc[te0:te1]`, `label_counts`, `kk = smote_k(min(c_tr))` (ts:133-140: 3 when the smallest train class has >= 4 rows) | train / block | `comparable = is_comparable(c_te[2], c_tr[2])` (ts:143-146: Active >= 5 in test and >= 4 in train), ew:171 |
| persistence | ew:173 `persistence_predictions(y, te0)[:te1 - te0]` (ts:149-156: `y.shift(1)` on the 958 series, sliced from `te0`) | block | row `te0` gets the label of row `te0 - 1` (last train day) |
| persistence point metrics and own-RNG CI | ew:174-177 `t04_evaluate("persistence", y_true, p_pers)` (pb:103-119; `bootstrap_ci` pb:93-100 draws its own `default_rng(42)` resamples, no skip rule) | block | `pers_f1_macro`, `pers_f1_active`, `pers_accuracy`, `pers_mcc`, `*_t04` CI |
| majority | ew:178-179 `majority_predictions(y_tr, 192)` (ts:159-162: mode of train) | train mode on the block | Normal on all folds (R3) |
| preprocessing | ew:183 `preprocess_split(X.iloc[:tr_end], X.iloc[te0:te1], y_tr, y_te, ..., k_neighbors=kk, seed=42)` (cm:98-120): `SimpleImputer(median).fit_transform(X_tr)`, `.transform(X_te)`, `SMOTE(random_state=42, k_neighbors=kk).fit_resample(X_tr_imp, y_tr)` | imputer and SMOTE fit on rows `[0, tr_end)` only; block transformed | same function as the canonical split (cm:90-95 calls it) |
| model | ew:184-188 `build_models()["rf"]` (save_test_predictions.py:48-49: 100 trees, depth 10, min leaf 5, seed 42), `fit(X_train_sm, y_train_sm)`, `predict` / `predict_proba` on `split.X_test` | SMOTE train -> imputed block | `assert list(rf.classes_) == LABELS` (ew:186) |
| point metrics | ew:190 `metrics(y_true, y_pred)` (ga:183-193: macro F1 with fixed labels, accuracy, MCC, balanced accuracy, per-class F1, 3 x 3 confusion `cm_rn` etc.) | block | |
| paired bootstrap | ew:191-199 `paired_bootstrap(y_true, y_pred, p_pers, f1_macro, idx)` and `(..., f1_active, idx)` (st:133-163: every row of `idx` applied to `y_true`, RF and persistence alike; a row with a class absent from the resampled `y_true` is skipped and counted) | the shared 1000 x 192 matrix, indices 0..191 relative to the block | marginal CIs of RF and persistence, delta CI, `prop_diff_gt_0`, `n_used`, `n_skipped` |
| SHAP | ew:201-209 `mean_abs_shap(rf, split.X_test)` (ss:123-124: `TreeExplainer(model).shap_values(X)` then `aggregate_multiclass`, ss:108-120: `np.abs(arr).mean(axis=(0, 2))` over samples and classes), `rank_desc` (ss:127-129), `top_shap` (ts:165-169) | imputed block, fold model | `shap_top1..3`, `aw_hr_mean_rank` |
| outputs | ew:213-218 `save_fold_predictions` (fold, date, y_true, y_pred, p_rest, p_normal, p_active; LF); ew:380 table csv `%.10g`; ew:401 SHAP csv; ew:426 json | | |

### (b) pooled (ew:319-320, 344-351)

In the fold loop (ew:311), after `save_fold_predictions`, `pooled_true.append(arr["y_true"])` and `pooled_pred.append(arr["y_pred"])` in the loop order k = 1, 2, 3, 4 (fold 1 = the most recent block first). ew:344 concatenates the four arrays (768 days); ew:350 `pooled_metrics` (ga `metrics` on the concatenation: macro F1, F1 Active, accuracy, MCC, balanced accuracy, F1 Rest / Normal); ew:351 `pooled_bootstrap(yt, yp, np.random.default_rng(42))` (ew:107-124): 1,000 draws `rng.integers(0, 768, 768)` from a fresh `default_rng(42)`, a resample with a class absent from the resampled `y_true` is skipped and counted, percentile 2.5 / 97.5 of macro F1, F1 Active, accuracy and MCC. The 192 matrix is not used for pooled. The pooled row carries no train size, no persistence and no SHAP.

---

## T2. Targeted questions

### Q1. Block bounds computed from N = 958 and 192; starts 766 / 574 / 382 / 190; disjoint and covering 190..957? PASS
`fold_bounds` computes `te0 = n - block * k` (ew:88) with `n = len(X)` passed at ew:162 and `block = BLOCK = 192` (ew:76); nothing is hard-coded except the two constants. E1: `all_fold_bounds()` == `[(766, 766, 958), (574, 574, 766), (382, 382, 574), (190, 190, 382)]`, `test_start(k) == 958 - 192 k` for all k, fold 1 start == `int(958 * 0.8)` == 766, pairwise disjoint, union == rows 190..957, train == `[0, test_start)` with no gap, fold 5 raises `ValueError`. R1 recomputes the same bounds from the two numbers and finds `t33_run_meta.json` `fold_bounds` identical. The four prediction files hold exactly the snapshot rows 766-957 / 574-765 / 382-573 / 190-381 (dates and labels, R1).

### Q2. Imputer and SMOTE fit only on rows 0..start-1; fold 3 imputer == median of rows 0..381? PASS
`preprocess_split` receives `X.iloc[:tr_end]` as the train part (ew:183) and fits the imputer and SMOTE on it alone (cm:107-111); the block enters only through `imputer.transform` (cm:109). E2 for every fold: `imputer.statistics_` == nan-median of rows 0..start-1 (exact), != median of rows 0..end-1 and != median of all 958; fold 3 printed in full (19 values, e.g. readiness 69.5 on rows 0..381 against 71.0 on rows 0..573 and 72.0 on the block). The statistics differ between consecutive folds on 15 / 16 / 17 of the 19 features (E2), so each fold has its own train statistics. R6 refits the imputer on the snapshot rows 0..189 / 0..381 / 0..573 and obtains the same statistics, and the RF trained on them reproduces the three prediction files (see T3).

### Q3. smote_k gives 3 for all four folds; rows after SMOTE == 3 x majority? PASS
Smallest train class: 38 / 22 / 14 / 9 Active (E2), all >= 4, so `smote_k` returns `min(3, n_min - 1)` = 3 (ts:136-137); `t33_run_meta.json` `smote_k_per_fold` says 3 for all. Rows after SMOTE: 1,710 / 1,269 / 816 / 435 = 3 x 570 / 423 / 272 / 145 (E2, R6), with the first `tr_end` rows equal to the imputed train (synthetic rows appended, E2).

### Q4. Persistence: first block day uses the label of row start-1, the rest the previous row in the block? PASS
`persistence_predictions` shifts the 958 series by one row and slices from `te0` (ts:152-153), so element 0 is `y[te0 - 1]`. E3, first three predictions: fold 2 `[1, 1, 1]` == `y[573..575]` (2025-02-28 -> 2025-03-01 .. 03-03); fold 4 `[2, 0, 1]` == `y[189..191]` (2023-12-23 -> 2023-12-24 .. 12-26); fold 3 `[1, 1, 1]` == `y[381..383]`; all 192 elements of each fold equal `y[te0 - 1 : te1 - 1]`. R3 reconstructs the same vectors from the snapshot labels and obtains macro F1 0.303896 / 0.307017 / 0.411992 / 0.316074.

### Q5. Fold 1 reproduces the canonical predictions byte for byte? PASS
E4: the six columns date, y_true, y_pred, p_rest, p_normal, p_active of `t33_predictions_rf_fold1.csv` are identical as text to `test_predictions_rf.csv`, and the fold file with its leading `fold,` field removed equals the canonical file byte for byte after CRLF -> LF normalisation (the canonical T0.2 file has 193 CR bytes, the T3.3 file none; see Observations). R1: probabilities exact (max abs diff 0.0). E7: `run_fold(1)` at HEAD gives the same `y_pred` and probabilities at 9.8e-17, and the same csv row on every column.

### Q6. Same indices 0..191 on all four folds, applied to y_true and both predictions; skip rule == F2; 0 / 0 / 1 / 8? PASS
`expanding_window.paired_bootstrap` is the same object as `statistical_tests.paired_bootstrap` (E0, `is` check), which applies each row of `idx` to `y_true`, `pred_a` and `pred_b` (st:150-156) and skips rows with a class absent from the resampled `y_true` (st:152-154); the matrix loaded at ew:278 is byte-identical to `default_rng(42).integers(0, 192, (1000, 192))` (E5, R4; range 0..191). E5 counts resamples with an absent class from the `y_true` of the four files: 0 / 0 / 1 / 8 (Active absent on folds 3 and 4, which have 8 and 5 Active days); R4 reproduces every marginal and paired CI at 4 decimals with the same skip counts, `n_used` 1000 / 1000 / 999 / 992 and the csv `prop_gt_0` at 1e-9.

### Q7. Pooled: CI on 768 days with its own default_rng(42); chronological concatenation; no resample skipped? UNCERTAIN (order)
CI and RNG: PASS. `pooled_bootstrap` draws `rng.integers(0, 768, 768)` from `np.random.default_rng(42)` created at ew:351; the 192 matrix is not involved (ew:107-124). R5: pooled 768 days (166 / 565 / 37), macro F1 0.393152, F1 Active 0.138614, accuracy 0.602865, MCC 0.089441, CIs [0.3523, 0.4349] / [0.0522, 0.2316] / [0.5716, 0.6354] / [0.0281, 0.1474], 0 skipped, all identical to the register and to the csv at 1e-9.
Order: the concatenation follows the fold loop k = 1..4 (ew:311, :319-320, :344), so the pooled vectors start with the most recent block (2025-09-19) and end with the oldest (2024-07-22); E6 confirms the concatenated dates are not monotonic. The point metrics are order-invariant, but the CI is not: the same `default_rng(42)` stream applied to the chronological concatenation (fold 4 first) gives macro F1 [0.3513, 0.4320] and F1 Active [0.0488, 0.2286] instead of [0.3523, 0.4349] and [0.0522, 0.2316] (E6, diagnostic). No document (register row 146, note at `CIFRE_REVIZIE.md:423`, Section 4.8, Table 11 note) states the order. Finding F6.3.

### Q8. Number of folds, block length and comparability rule fixed in code and consistent with the plan; any trace of other block lengths? UNCERTAIN
In code: `N_BOOT, BLOCK, N_TOTAL = 1000, 192, 958` and `N_FOLDS = 4` (ew:76-77), comparability `is_comparable` with `MIN_ACTIVE_TEST = 5`, `MIN_ACTIVE_TRAIN = 4` from `threshold_sensitivity` (the T3.1 convention), all asserted by `test_fixed_design` (test_expanding_window.py:81-84) and recorded in `t33_run_meta.json`. Four is the largest number of 192-row blocks that 958 rows allow (`fold_bounds(5)` raises, E1), so once the block length equals the canonical test size the fold count has no free choice.
In the plan: task T3.3 (`KBS_Revision_Plan.md:99`, unchanged since 14 October) says "3-4 folduri temporale pe cele 958 zile (de ex. train pana la sfarsitul lui 2024, test S1 2025; ...)", a calendar-based example, and "Daca timpul preseaza: 2 folduri"; the old section 8 item 11 (sacrifice order, commits 6c9b81b .. 3b104f5) protects "T3.3 cu minimum 2 folduri". The plan therefore fixed a range (2 to 4 folds), not the 192-row block design; the register's "design (fixat inainte de a vedea vreun rezultat, nemodificat dupa)" (`CIFRE_REVIZIE.md:423`) is the author's statement.
In git: the script, the tests, the results and the figure were committed together in `3b7f994` (06:24:54, 2 min 43 s after the recorded `run_date`), the script has no later commit, and no path or diff line in the whole history mentions another block length (`git log --all -p` filtered on `BLOCK` / `N_FOLDS`, and on "bloc|block" with a three-digit number other than 192, returns only the committed values and unrelated T2.2 lines). Nothing contradicts pre-run fixation; git alone cannot show it, as in F4.3 and F5 Q12. Finding F6.2.

### Q9. SHAP per fold: block of the fold, model of the fold, T1.4 aggregation? PASS
ew:201 `mean_abs_shap(rf, split.X_test)` uses the RF fitted at ew:185 and the imputed block of the same fold; `mean_abs_shap` is the `shap_stability` object (E0) whose `aggregate_multiclass` takes `np.abs(arr).mean(axis=(0, 2))` over samples and classes (ss:118-120; E8). E7: `run_fold` at HEAD reproduces `t33_shap_top3.csv` for the four folds at 1e-9 (rank 1: aw_hr_mean 0.0611 / oura_readiness_score 0.0601 / oura_readiness_score 0.0663 / aw_hr_mean 0.0685).

### Q10. Manuscript sentences about T3.3 (Section 4.8, Discussion, response letter) and Table 11 against the csv? FAIL (two Table 11 cells)
Sentences, all PASS (R7, from the recomputed values): "Macro F1 is 0.361, 0.377 and 0.312 on the three earlier blocks against 0.506" and "F1 Active is 0.000, 0.074 and 0.103 against 0.444" (folds 4, 3, 2, 1); "exceeds persistence with an interval separated from zero only on the final block" (delta CI excludes zero on fold 1 only); "on the block tested on July 2024 to February 2025 the model is below persistence (0.377 against 0.412)" and "falls below it on one block" / "is negative on one block" (fold 3 only); "the leading SHAP feature is mean heart rate on the first and last blocks and the readiness score on the two intermediate ones" and "changes on two blocks" (folds 2, 3); "macro F1 is 0.31 to 0.38 and F1 Active 0.00 to 0.10" (0.3124-0.3771, 0.0000-0.1026); "F1 for the Active class 0.000 to 0.103"; pooled "0.393 [0.352, 0.435] and F1 Active 0.139 [0.052, 0.232]"; "does not increase monotonically with training size" (0.3605 / 0.3771 / 0.3124 / 0.5063 for train 190 / 382 / 574 / 766); "the final block is the most favorable of the four"; Discussion "few Active days (9 to 22)", "prevalence of Rest days varies between 15% and 31%" (31.2 / 17.2 / 15.1 / 22.9), "33 of the 46 Active days fall on or after July 2024" (33, E10 and R7). Section 3.11: 766 / 574 / 382 / 190 train days with 38 / 22 / 14 / 9 Active, PASS. The response letter (R1.10) repeats the same figures, PASS.
Table 11 (`KBS_Manuscript_R1.docx`, cells read through zipfile, R7): 17 of the 19 numeric cells with intervals equal the csv values rounded once to 3 decimals, as do the train / test columns, the pooled row and the SHAP column. Two cells of the "Dec 2023 to Jul 2024" row (fold 4) do not: macro F1 is printed "0.361 [0.310, 0.407]" where the csv gives 0.309485 -> 0.309, and persistence "0.316 [0.274, 0.363]" where the csv gives 0.273482 -> 0.273. Both printed values are what the register's 4-decimal figures (0.3095, 0.2735; `CIFRE_REVIZIE.md:145`) give when rounded a second time. Finding F6.1.

### Q11. The caveat that the folds are not replicates and that no formal comparison is made: in the manuscript? PASS
`KBS_Manuscript_R1.docx` Section 3.11 (E0 extract, line 136 of the text dump): "The blocks are not replicates of one experiment, since training size and period differ, so no formal comparison between blocks is made; a pooled estimate over the 768 evaluated days is given for context only." Section 4.8 adds "this pools four models with different training histories and is not an estimate for any one of them", and the Discussion gives the differing prevalences ("the prevalence of Rest days varies between 15% and 31% across blocks"). The response letter says "The result does not demonstrate long-term stability, and we report it in full." The same caveat is in the register (`CIFRE_REVIZIE.md:425`).

### Q12. Gap rows per T3.3 block and per comparable T3.1 cell; where the two "previous day" conventions pick different source days. Answered: the counts must be given per block (see the separate section below).

T2 tally: Q1 PASS, Q2 PASS, Q3 PASS, Q4 PASS, Q5 PASS, Q6 PASS, Q7 UNCERTAIN, Q8 UNCERTAIN, Q9 PASS, Q10 FAIL, Q11 PASS, Q12 answered (8 PASS, 1 FAIL, 2 UNCERTAIN, 1 question answered).

---

## Question 12 (open from slice F1): the two "previous day" conventions on the T3.3 blocks and the T3.1 cells

Definitions used (E9), the same as in slice F1: the persistence prediction for a row uses the label of the previous row of the 958-day labelled series ("previous labelled day"; ts:152); the 19 features of a row are `shift(1)` on the 1,009-row raw frame before the label drop (cm:66-69), so they come from the previous raw record, which is the previous calendar day except where the raw frame itself has a gap (12 consecutive-row pairs span more than one day). A "gap row" is a row whose previous labelled row is more than one calendar day back. E9 verifies on all 958 rows that `readiness_score_lag1` equals the readiness of the previous raw record (0 mismatches). The six comparable T3.1 cells (R55_A80, R55_A85, R60_A80, R60_A85, R65_A80, R65_A85) regenerate the labels on the same 958 days (a row is unlabelled only when readiness or sleep is missing, `t3_labels.py:93`, independent of the thresholds): E9 confirms that each cell's 958 dates and its `t31_predictions_rf_*.csv` test dates equal the canonical ones, so their test block is fold 1 and their counts are those of fold 1.

| block / cell | rows | n rows with gap (previous labelled row > 1 day back) | n rows where the feature source day != the persistence source day | source days differ by |
|---|---|---|---|---|
| T3.3 fold 1 (canonical test; 2025-09-19 .. 2026-03-30) | 766-957 | 1 (2025-10-29) | 0 (both use 2025-10-27; the raw frame has no 2025-10-28) | - |
| T3.3 fold 2 (2025-03-01 .. 2025-09-18) | 574-765 | 10 | 7 (03-07, 03-09, 03-11, 03-23, 03-27, 04-08, 04-13 of 2025) | 1 day on all 7 |
| T3.3 fold 3 (2024-07-23 .. 2025-02-28) | 382-573 | 14 | 14 | 1 to 9 days (2024-11-02: persistence 10-23, features 11-01) |
| T3.3 fold 4 (2023-12-24 .. 2024-07-22) | 190-381 | 13 | 13 | 1 to 5 days (2024-02-20: persistence 02-14, features 02-19) |
| T3.1 R55_A80, R55_A85, R60_A80, R60_A85, R65_A80, R65_A85 (test = rows 766-957) | 766-957 | 1 each | 0 each | - |

Train side (for completeness, E9): rows whose previous row is more than one day back number 41 / 31 / 17 / 4 within the train parts of folds 1 / 2 / 3 / 4.

Because the planned wording (`AUDIT_DECISIONS.md` item 2) says that persistence "coincides with a function of two of the features (the previous day's readiness and sleep scores) on all but one test day", E9b applies the canonical label rule to the lag-1 readiness and sleep features of each block and compares with the persistence prediction: fold 1 equal on 192 of 192 days (the gap day 2025-10-29 included, because its features and its persistence label both come from 2025-10-27); fold 2 equal on 185, not computable on 7 (the lag-1 scores are missing there and are imputed with the train median before the model sees them); fold 3 equal on 178, not computable on 14; fold 4 equal on 179, not computable on 13; never unequal where computable.

Conclusion in two sentences. On the canonical test set and on the six comparable T3.1 cells the two conventions select the same source day on every one of the 192 days (the single gap day resolves to the same record), so the sentence is exact there and "all but one" is conservative by one day; on the three earlier T3.3 blocks the conventions diverge on 7, 14 and 13 days (of 10, 14 and 13 gap rows), so the sentence cannot be generalised to T3.3 and any statement about the relation between persistence and the lag-1 scores must be given per block. The persistence figures of folds 2 to 4 (0.3070 / 0.4120 / 0.3161) are "last labelled day" figures on blocks where that day is up to 9 days back, and the register's description "eticheta randului precedent din seria de 958 zile" is the accurate one.

---

## T3. Independent recomputation (`revision/audit/recompute_f6.py`)

Imports numpy, pandas, scikit-learn, imbalanced-learn and the standard library only; nothing from `revision/scripts/`. Inputs: the four prediction files, `t33_expanding_window.csv` and `t33_shap_top3.csv` (compared, not used as a source), `t33_run_meta.json`, `test_predictions_rf.csv`, the regenerated bootstrap matrix (identical to the npy), `models/dataset_snapshot_20260331.csv` (dates, 19 lag-1 features, labels) and `KBS_Manuscript_R1.docx` through zipfile. 154 checks, output in R.

| block | quantity | claimed (register / brief) | recomputed | verdict |
|---|---|---|---|---|
| bounds | starts from 958 - 192 k; json fold_bounds; disjoint cover 190..957 | 766 / 574 / 382 / 190 | same | PASS |
| folds 1-4 | file dates and y_true == snapshot rows; train / test distributions | 158/570/38 44/140/8; 129/423/22 29/147/16; 96/272/14 33/151/8; 36/145/9 60/127/5 | same | PASS (8) |
| fold 2 | F1 macro / F1 Active / accuracy / MCC | 0.3124 / 0.1026 / 0.5990 / -0.0291 | 0.312389 / 0.102564 / 0.598958 / -0.029114 | PASS |
| fold 2 | confusion rr rn ra nr nn na ar an aa | 2 22 5 20 111 16 1 13 2 | same (23 Active predicted, 2 correct of 16) | PASS |
| fold 3 | F1 macro / F1 Active / accuracy / MCC; confusion | 0.3771 / 0.0741 / 0.5938 / 0.1112; 14 17 2 36 99 16 2 5 1 | 0.377142 / 0.074074 / 0.593750 / 0.111212; same | PASS |
| fold 4 | F1 macro / F1 Active / accuracy / MCC; confusion | 0.3605 / 0.0000 / 0.5781 / 0.1046; 21 34 5 30 90 7 0 5 0 | 0.360501 / 0.000000 / 0.578125 / 0.104609; same | PASS |
| fold 1 | F1 macro / F1 Active / accuracy / MCC | 0.5063 / 0.4444 / 0.6406 / 0.1416 | 0.506324 / 0.444444 / 0.640625 / 0.141560 | PASS |
| folds 1-4 | persistence F1 macro from the snapshot (label of row start-1, then shifted labels) | 0.3039 / 0.3070 / 0.4120 / 0.3161 | 0.303896 / 0.307017 / 0.411992 / 0.316074 (fold 1 vector == `test_predictions_persistence.csv`) | PASS |
| folds 1-4 | majority (train mode) F1 macro | Normal; 0.2811 / 0.2891 / 0.2935 / 0.2654 | same | PASS |
| folds 1-4 | RF F1 macro CI (regenerated matrix, absent-class skip) | [0.3784, 0.6130] / [0.2561, 0.3749] / [0.3127, 0.4456] / [0.3095, 0.4069] | same at 4 decimals | PASS |
| folds 1-4 | RF F1 Active CI | [0.1176, 0.7000] / [0.0000, 0.2424] / [0.0000, 0.2222] / [0.0000, 0.0000] | same | PASS |
| folds 1-4 | persistence F1 macro CI (same matrix) | [0.2612, 0.3501] / [0.2559, 0.3682] / [0.3253, 0.5136] / [0.2735, 0.3627] | same | PASS |
| folds 1-4 | delta RF - persistence [CI] | +0.2024 [+0.0642, +0.3173] / +0.0054 [-0.0768, +0.0930] / -0.0349 [-0.1600, +0.0841] / +0.0444 [-0.0317, +0.1168] | same; prop > 0 0.997 / 0.527 / 0.3123 / 0.8851 == csv | PASS |
| folds 1-4 | resamples skipped / used | 0 / 0 / 1 / 8; 1000 / 1000 / 999 / 992 | same | PASS |
| pooled | n, distribution, F1 macro / F1 Active / accuracy / MCC | 768, 166/565/37, 0.3932 / 0.1386 / 0.6029 / 0.0894 | 0.393152 / 0.138614 / 0.602865 / 0.089441 | PASS |
| pooled | CIs with default_rng(42) on 768 (fold order 1..4), skipped | [0.3523, 0.4349] / [0.0522, 0.2316] / [0.5716, 0.6354] / [0.0281, 0.1474]; 0 | same at 4 decimals (1e-9 against the csv); 0 | PASS |
| refit fold 4 | snapshot rows 0..189 -> median imputer -> SMOTE(42, k 3) -> RF 100/10/5/42 -> rows 190..381 | == `t33_predictions_rf_fold4.csv` | 0 days differ, max abs proba diff 9.9e-17, F1 macro 0.360501 | PASS |
| refit fold 3, fold 2 | same procedure on rows 0..381 and 0..573 | == fold 3 / fold 2 files | 0 days differ, 9.7e-17 / 9.8e-17; 0.377142 / 0.312389 | PASS |
| Table 11 | 19 interval cells, train / test columns, pooled row, SHAP column, 3 decimals rounded once from the csv | as printed | 17 / 19 interval cells identical; fold 4 "0.361 [0.310, 0.407]" vs 0.309 and "0.316 [0.274, 0.363]" vs 0.273 | FAIL (2) |
| Section 4.8, Discussion, response | 13 sentence-level numeric claims (listed under Q10) | as printed | all reproduce | PASS |

T3 verdict: 152 / 154 checks PASS. Every figure of the register rows T3.3 (`CIFRE_REVIZIE.md:141-147`) that the brief names reproduces at 4 decimals from the prediction files, the snapshot and a regenerated seed-42 matrix; folds 2, 3 and 4 refit independently from the snapshot reproduce the three prediction files on all 192 days with probabilities at 1e-16; the pooled CI reproduces the RNG stream exactly (no 3-decimal fallback needed). The two failures are the two Table 11 cells of F6.1, which are third-decimal rounding errors of the manuscript, not of the csv.

---

## Findings (FAIL or UNCERTAIN only)

### F6.1. Two lower confidence bounds of the first row of Table 11 are rounded twice and are one unit of the third decimal too high (Q10: FAIL)
Observed: in `KBS_Manuscript_R1.docx` Table 11, row "Dec 2023 to Jul 2024" (fold 4), the macro F1 cell reads "0.361 [0.310, 0.407]" and the persistence cell "0.316 [0.274, 0.363]"; the full-precision values in `t33_expanding_window.csv` are 0.309485 and 0.273482 (R4 and R7 recompute them from the prediction files and the regenerated matrix), which round to 0.309 and 0.273. The printed values are what the 4-decimal register figures 0.3095 and 0.2735 (`CIFRE_REVIZIE.md:145`) give when rounded again. The other 17 interval cells, the two 4.8 sentences and the Discussion ranges round correctly from the csv (R7). Why it matters: a reader checking Table 11 against the released csv finds two cells that do not match; both bounds are lower limits of intervals that already include the comparison, so no statement or conclusion changes. What the authors should decide: whether to reprint the two cells from the csv (0.309, 0.273) and, for the final assembly, to round every table cell from the 10-significant-digit csv rather than from the 4-decimal register.

### F6.2. The 192-row block design and the four folds are fixed in code but their fixation before the run cannot be shown from the plan or from git (Q8: UNCERTAIN)
Observed: `BLOCK = 192`, `N_FOLDS = 4` and the T3.1 comparability thresholds are module constants (ew:76-77) asserted by a unit test, and four is the maximum number of 192-row blocks that 958 rows allow; the plan written before the run (`KBS_Revision_Plan.md:99-100`, 14 October) specifies "3-4 folduri" with a calendar-based example ("train pana la sfarsitul lui 2024, test S1 2025") and a 2-fold fallback, so the choice of equal 192-row blocks cut from the tail was made when the script was written; the script, the tests, the results and the figure entered git in one commit (`3b7f994`, 2 min 43 s after the recorded `run_date`), and the history contains no other block length or fold count. The register states that the design was "fixat inainte de a vedea vreun rezultat, nemodificat dupa" (`CIFRE_REVIZIE.md:423`). Why it matters: the headline of Section 4.8 (the canonical block is the most favorable of the four) is a comparison whose blocks were defined by the same person who saw the canonical result first; the audit found nothing that contradicts the stated fixation and the design has no free parameter once the block equals the canonical test size, but the statement rests on the author's word, as in F4.3 and the T2.2 search space. What the authors should decide: whether Section 3.11 should say that the block length equals the reported test-set size and the fold count is the maximum the series allows (which makes the design verifiable from the two numbers), instead of or in addition to the register's statement.

### F6.3. The pooled interval depends on an undocumented concatenation order (Q7: UNCERTAIN)
Observed: the pooled vectors are built in fold order 1, 2, 3, 4, i.e. the most recent block first (ew:311-320, :344; E6 shows the concatenated dates are not monotonic), and the pooled CI uses a fresh `default_rng(42)` on positions 0..767 (ew:107-124, :351). The point estimates do not depend on the order; the intervals do: with the blocks concatenated chronologically and the same seed, macro F1 gives [0.3513, 0.4320] and F1 Active [0.0488, 0.2286] instead of the reported [0.3523, 0.4349] and [0.0522, 0.2316] (E6, diagnostic). The register (rows 146, note at `:423`), Section 4.8 and the Table 11 note say "the four blocks concatenated" without the order. Why it matters: the pooled figures are "given for context only" (Section 3.11) and nothing is concluded from them, but a reader who concatenates in calendar order will obtain intervals that differ at the third decimal and cannot tell why. What the authors should decide: whether to add "(blocks concatenated from the most recent to the oldest)" to the register note and the Table 11 note, or to leave the pooled interval unexplained as a context figure.

No other FAIL or UNCERTAIN verdicts were reached.

---

## Observations (no verdict change; reported for completeness)

- The canonical `test_predictions_rf.csv` and `test_predictions_persistence.csv` (T0.2 / T0.4) are CRLF files (193 CR bytes each), while every T3.3 output is LF (E4); the text content is identical. The plan's LF rule applies to revision outputs; these two files predate it and were hashed as they are in F1.
- `pers_f1_macro_ci_*_t04` (persistence CI from `persistence_baseline.evaluate`, own RNG, no skip rule) differs from the shared-matrix persistence CI at the 4th decimal on folds 3 and 4 (0.325307 vs 0.325296; 0.273502 vs 0.273482), as the register states (`CIFRE_REVIZIE.md:423`); the manuscript prints the shared-matrix values.
- The fold rows of `t33_expanding_window.csv` carry no accuracy or MCC interval (columns `accuracy_ci_*`, `mcc_ci_*` are empty for folds 1-4 and filled for pooled only); Table 11 prints none for the folds, so nothing is misreported.
- Section 4.8 says "the first and last blocks" for the blocks on which mean heart rate leads; in the chronological order of Table 11 these are the Dec 2023 and Sep 2025 blocks (folds 4 and 1), consistent with the csv.
- Fold 4 is comparable at the limit (5 Active in the block, 9 in train; `comparable = True`); the register says so (`:145`), the manuscript does not flag it, and the F1 Active interval [0, 0] reflects that none of the 5 Active days is recovered in any of the 992 used resamples.
- The manuscript's source (`F4_manuscript_changes.md`) numbers the table and figure "Table 8" and "Figure [t33]"; the assembled docx uses "Table 11" and "Figure 4"; the response letter refers to "a table and a figure", so no cross-reference is broken.
- `python -m pytest revision/scripts/test_expanding_window.py -q`: 5 passed in 4.7 s, as the register claims (`:148`).

---

## Summary

- T2: Q1 PASS, Q2 PASS, Q3 PASS, Q4 PASS, Q5 PASS, Q6 PASS, Q7 UNCERTAIN, Q8 UNCERTAIN, Q9 PASS, Q10 FAIL, Q11 PASS, Q12 answered (8 PASS, 1 FAIL, 2 UNCERTAIN).
- T3: 152 / 154 checks PASS from an independent implementation (every register figure of the four folds and of pooled at 4 decimals, the pooled RNG stream reproduced exactly, folds 2-4 refit from the snapshot identical to the prediction files on 576 days at 1e-16); the 2 failures are the two double-rounded Table 11 cells of F6.1.
- Question 12: on the canonical test block and on the six comparable T3.1 cells the two "previous day" conventions pick the same source record on all 192 days (the single gap day resolves to the same record), so "all but one test day" is exact there and even conservative by one; on the T3.3 blocks 2, 3 and 4 they diverge on 7, 14 and 13 days (of 10, 14 and 13 gap rows), so the sentence does not generalise and must be given per block.
- Findings: F6.1 (FAIL), two lower bounds in the first row of Table 11 are rounded twice (0.310 and 0.274 printed, 0.309 and 0.273 from the csv); F6.2 (UNCERTAIN), the 192-row, 4-fold design is fixed in code and has no free parameter once the block equals the test size, but its fixation before the run rests on the author's statement because the plan gave only a range and git holds script and results in one commit; F6.3 (UNCERTAIN), the pooled interval depends on the undocumented fold-order concatenation (chronological order gives [0.351, 0.432] instead of [0.352, 0.435]).

Commit command (executed as the last step of the audit; no push):

```
git add revision/audit/AUDIT_F6_expanding_window.md revision/audit/AUDIT_SUMMARY.md revision/audit/recompute_f6.py revision/audit/recompute_f6_output.txt revision/audit/_check_f6_empirical.py revision/audit/_check_f6_empirical_output.txt && git commit -m "audit F6: expanding window; audit summary"
```
