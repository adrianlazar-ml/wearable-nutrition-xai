# AUDIT F3: calibration and safety layer (T1.3), labelling-threshold sensitivity (T3.1)

Auditor role: external, read-and-report only. Branch `audit-r1`, HEAD `0930bff` (slices F1 and F2 committed). Run date: 2026-10-25.
Environment used for every command below: Python 3.11.9, scikit-learn 1.8.0, numpy 2.4.3, pandas 2.3.3, scipy 1.17.1, statsmodels 0.15.0, imbalanced-learn 0.14.1 (identical to `t13_run_meta.json` / `t31_run_meta.json` where listed).

Slices F1 and F2 were read only for their Findings sections; nothing from them is reused as a fact. F1's finding F1 (persistence = previous labelled row, not calendar day) applies unchanged to the per-cell persistence of T3.1 (same `shift(1)` rule on the 958-row series, `threshold_sensitivity.py:152`); it is not re-examined.

Files written by this audit (nothing else was modified; no file under `revision/results/`, `outputs/` or the repo root was touched):

- `revision/audit/AUDIT_F3_calibration_thresholds.md` (this report)
- `revision/audit/recompute_f3.py` + `recompute_f3_output.txt` (T3; imports numpy, pandas, statsmodels, sklearn, imblearn; nothing from `revision/scripts/`; the calibrator joblib files are not loaded, the calibrators are refit from the OOF csv)
- `revision/audit/_check_f3_empirical.py` + `_check_f3_empirical_output.txt` (T2 evidence; imports `calibration_rf`, `common`, `threshold_sensitivity`, `statistical_tests`, `extended_metrics`, `save_test_predictions` from `revision/scripts/` and `confidence_threshold` from the root; refits the 4 OOF folds and 3 grid cells in memory; calls `confidence_threshold.load_and_prepare()` / `train_model()` but never `main()`, so nothing is written under `outputs/`)

All files LF (`file`: ASCII text). Execution time: recompute 8 s; empirical check 33 s, of which 21.5 s is the Q12 rerun of the old 16-feature pipeline (declared exception, under the 5-minute cap).

---

## T0. Inventory

`git status --short` at the start: unchanged from slices F1 / F2 (one deleted tracked zip, three untracked paths outside `revision/`), plus one untracked file `revision/audit/AUDIT_DECISIONS.md` created by the author at 19:42 today, which this audit did not write and does not stage. sha256 first 16 hex chars of the inputs read:

| file | sha256[:16] | role |
|---|---|---|
| `confidence_threshold.py` (root) | c003759e951a8c6c | safety-layer functions; `git diff --quiet f7cc4ca HEAD -- confidence_threshold.py` is empty |
| `threshold_sensitivity.py` (root, Table 7 script) | identical to f7cc4ca (`git diff --quiet`) | SMOTE k convention source |
| `revision/results/test_predictions_rf.csv` | aea39c6dd2989e24 | the only test probabilities used |
| `revision/results/t13_oof_probabilities_rf.csv` | 0f3aa702fada78c9 | 612 OOF rows |
| `revision/results/t13_oof_folds_rf.csv` | 2c0f1ab0e8f675bb | fold table |
| `revision/results/t13_calibration.csv` | 3af8929d0e7e4f6c | claims |
| `revision/results/t13_reliability_bins.csv` | ce4a082f4e4fe960 | claims |
| `revision/results/t13_safety_layer_variants.csv` | 4617cc3a14f1407e | claims |
| `revision/results/models/calibrator_platt_rf.joblib` | 90b067f2bd6788c2 | matches `CIFRE_REVIZIE.md:196` |
| `revision/results/models/calibrator_isotonic_rf.joblib` | 75da9f35ab236e6e | matches `CIFRE_REVIZIE.md:197` |
| `revision/results/t31_threshold_grid.csv` | 95c1c0fcc6494f43 | matches `CIFRE_REVIZIE.md:229` |
| `revision/results/t31_predictions_rf_R55_A85.csv` | 32d1be28ee010cb3 | matches `CIFRE_REVIZIE.md:236` |
| `revision/results/bootstrap_indices_1000x192_seed42.npy` | fff3b2834c061962 | byte-equal to `default_rng(42).integers(0, 192, size=(1000, 192))` (R) |
| `models/dataset_snapshot_20260331.csv` | 176649b59fef3adb | dates and canonical labels only (no raw scores) |
| `data/oura_combined.csv` | 11d6d4de0eac31f9 | raw readiness / sleep scores (label inputs) |

Note on inputs: the snapshot csv holds only the 19 lag-1 columns plus `nutrition_label`, so the raw readiness and sleep scores needed to regenerate labels with R = 55 were taken from `data/oura_combined.csv`; the regenerated 958 dates equal the snapshot dates and the regenerated canonical labels equal `nutrition_label` on all 958 rows (R, section "T3.1 labels").

---

## T1. Trace tables (from reading the code)

Line numbers refer to the files at HEAD `0930bff`.

### (a) T1.3: from OOF probabilities to the safety layer

| # | Operation | File:line | Applied on | Parameters | Output |
|---|---|---|---|---|---|
| 1 | `load_canonical()` -> X, y, dates (958 x 19); keep the first `int(958 * 0.8)` = 766 rows | calibration_rf.py:135-139 | 958 -> 766 train rows | TRAIN_RATIO 0.8 | X_tr, y_tr, d_tr |
| 2 | `TimeSeriesSplit(n_splits=4).split(X_tr)` | :144-147 | 766 rows, positional | no shuffle; first train 154, 4 validation blocks of 153 | (tr, va) per fold |
| 3 | `preprocess_split(X_tr.iloc[tr], X_tr.iloc[va], ...)`: median imputer fit on fold train, transform on fold validation; `SMOTE(42, k=3)` on fold train only | :149-150 -> common.py:107-111 | fold train (154 / 307 / 460 / 613) and fold validation (153) | k 3, seed 42 | imputed validation, SMOTE train (354 / 675 / 1014 / 1359) |
| 4 | `build_models()["rf"].fit` on the fold's SMOTE train | :151-152 -> save_test_predictions.py:48-49 | SMOTE train | 100 trees, depth 10, min leaf 5, seed 42 | fold RF (not saved; `rf_revision.joblib` untouched) |
| 5 | `predict_proba` on the fold validation; check `classes_ == [0,1,2]`; check `max(train date) < min(val date)` | :153-156, :162, :170 | validation 153 rows | | OOF rows (date, fold, y_true, p_rest, p_normal, p_active) |
| 6 | Concatenate; checks: 612 = 766 - 154 rows, rows sum to 1, dates sorted and unique; save `t13_oof_probabilities_rf.csv`, `t13_oof_folds_rf.csv` | :179-195 | 612 rows | | the two csv files |
| 7 | Stage B inputs: OOF csv, fold csv, `test_predictions_rf.csv`, npy matrix; checks 192 / 612 rows, argmax == y_pred, canonical 0.5063 / 0.6406 / 0.1416, manual Brier == sklearn `scale_by_half=False` at 1e-12 | :448-464 | test 192, OOF 612 | | y_te, P_raw_te, y_oof, P_raw_oof |
| 8 | For fit_folds in {1-4, 2-4}: subset OOF rows by fold; `PlattOvR().fit(P_raw_oof, y_oof)`; `IsotonicOvR().fit(...)` | :497-504 | 612 (1-4) or 459 (2-4) OOF rows only | | calibrators |
| 9 | Platt per class: `LogisticRegression(C=np.inf, l1_ratio=0, lbfgs, max_iter 10000, tol 1e-8)` on the single feature logit(clip(p_c, 1e-6, 1 - 1e-6)) vs (y == c) | :288-298, logit :273-275 | OOF rows | | 3 binary models |
| 10 | Isotonic per class: `IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip")` on p_c vs (y == c) | :325-331 | OOF rows | | 3 isotonic models |
| 11 | Monotonicity check on a 1001-point grid, all classes; dump the fit 1-4 calibrators to joblib | :341-343, :505-511 | | | `calibrator_{platt,isotonic}_rf.joblib` |
| 12 | Apply: per-class calibrated p_c, then row normalisation `Q / Q.sum(1)`; rows with sum 0 -> 1/3 each and counted | :301-308, :333-339, normalize_ovr :261-270 | test 192 and OOF (in-sample) | | P_platt, P_iso |
| 13 | Metrics: ECE / MCE top-label (`conf = max p`, `hit = argmax == y`, bins `searchsorted(EDGES[1:-1], p)`, 10 equal-width bins, empty bins dropped, weight n_b / N), ECE per class (OvR), Brier multiclass (mean over rows of sum over classes), Brier per class, log loss (`sklearn log_loss, labels=[0,1,2]`) | :201-258, :363-370 | test and OOF | N_BINS 10 | point values |
| 14 | Bootstrap CI (percentile 2.5 / 97.5) on the shared npy matrix for ECE top, Brier multiclass, log loss, Brier Active on test; nan results skipped and counted | :347-359, :539-548 | test 192 resampled | 1000 rows | ci_low, ci_high, n_skipped |
| 15 | Argmax effect (non-canonical): days changed vs raw, F1 macro / F1 Active / accuracy / MCC of the calibrated argmax; p_active min / median / max, days >= 0.50 | :529-534, :373-377 | test 192 | | informative rows |
| 16 | Reliability tables (top-label and Active) per split / fit / variant -> `t13_reliability_bins.csv`; `t13_calibration.csv`, md, figure, `t13_run_meta.json` | :557-565, :582-588, :590-635 | | | outputs |
| 17 | Stage C: read `test_predictions_rf.csv`, load the two joblib calibrators, build P for raw / platt / isotonic; check row sums | safety_layer_calibrated.py:69-78 | test 192 | | probs dict |
| 18 | For each variant and threshold in {0, 0.40, ..., 0.80}: `yp = argmax(P)`; `ct.apply_threshold(yp.copy(), P, t)`; `ct.caloric_error_analysis(y, ya)`; coverage = (192 - n_over) / 192; F1 macro / accuracy / F1 Active; emitted states | :80-93 | test 192 | THRESHOLDS :52 | rows |
| 19 | `apply_threshold`: for every row, `max_prob = max(y_proba[i])`; `if max_prob < threshold: y_adjusted[i] = 1` | confidence_threshold.py:144-167 | 192 rows | strict `<` | y_adjusted, n_overridden, details |
| 20 | `caloric_error_analysis`: `abs(CALORIC_MAP[pred] - CALORIC_MAP[true])` for every (true, pred) pair; mean and max | confidence_threshold.py:173-184, CALORIC_MAP :58 | all 192 rows (overridden days count as Normal) | Rest -200 / Normal 0 / Active +200 | mean, max, list |
| 21 | Sanity: raw @0.50 == 65.625 percent / 66 / 58.3333 / 200; raw @0 == 0.5063 / 0.6406; coverage non-increasing; states sum to 192; save csv / md / figure | :96-110, :134-160 | | EXPECTED_RAW_050 :55 | `t13_safety_layer_variants.csv` |
| 22 | Erratum checker: reads `test_predictions_rf.csv`, applies the same two functions at 7 thresholds, READS the stored `outputs/tables/confidence_threshold_analysis.csv` (does not rerun the old pipeline), prints both tables, FAILs intentionally on the 0.50 row | _check_t13_safety_layer.py:65-96 | test 192 vs stored table | THRESHOLDS :54 | exit 1 (expected) |

### (b) T3.1: one grid cell (R, A_r, A_s = 75)

| # | Operation | File:line | Applied on | Parameters | Output |
|---|---|---|---|---|---|
| 1 | `load_raw()` once for all cells (Oura + Apple Watch merge, sorted by date) | threshold_sensitivity.py:326 -> common.py:46-58 | 1,009 raw rows | | df |
| 2 | Load npy matrix; check byte-equal to `default_rng(42)` (abort on mismatch) | :327-331 | | seed 42 | idx 1000 x 192 |
| 3 | `fit_eval_labels(R, A_r, 75, df)`: `build_dataset(df, rest_cutoff=R, active_readiness=A_r, active_sleep=75)` = lag-1 shift on 1,009 rows, `t3_labels.assign_labels(df, **thresholds)`, drop rows with NaN label (NaN iff readiness or sleep missing, threshold-independent) | :186 -> common.py:61-71 -> revision/t3_labels.py:79-101 | 1,009 -> 958 rows | R, A_r, A_s | X (958 x 19), y, dates |
| 4 | Guard: 958 rows and the 19 canonical columns; `i = int(958 * 0.8)` must be 766 | :187-191 | | | i = 766 |
| 5 | Label counts on 958 / train 766 / test 192; `smote_k(min(train counts))`: 3 if n_min >= 4, n_min - 1 if 2 <= n_min <= 3, None below | :192-198, smote_k :133-140 | re-labelled y | K_CANON 3 | counts, k, trainable |
| 6 | `preprocess_split(X[:766], X[766:], y_tr, y_te, dates_te, k_neighbors=k, seed=42)`: imputer fit on the 766 train rows (X is label-independent, so identical medians in every cell), SMOTE on the re-labelled train | :202 -> common.py:98-120 | train 766 / test 192 | k per cell | Split |
| 7 | `build_models()["rf"].fit(X_train_sm, y_train_sm)`; `predict` / `predict_proba` on the imputed test | :203-207 | SMOTE train, test 192 | canonical RF hyperparameters | y_pred, proba |
| 8 | Persistence: `shift(1)` of the cell's re-labelled 958-row series, test part (first test row = re-labelled row 765, 2025-09-18); evaluated with the unmodified `persistence_baseline.evaluate()` | :149-156, :224-228 -> persistence_baseline.py:103-119 | re-labelled y | | pers_* columns (own-RNG CI) |
| 9 | Majority: mode of the re-labelled train | :159-162, :230-232 | train 766 | | maj_* |
| 10 | `is_comparable(test Active >= 5 and train Active >= 4)` stored in the row | :143-146, :222 | counts | MIN_ACTIVE_TEST 5, MIN_ACTIVE_TRAIN 4 | `comparable` |
| 11 | `metrics(y_true, y_pred)` (F1 macro with labels fixed, accuracy, MCC, balanced accuracy, per-class F1, confusion matrix) | :238-239 -> group_ablation.py:183-193 | test 192 | | f1_*, cm_* |
| 12 | `paired_bootstrap(y_true, y_pred, p_pers, f1_macro, idx)`: one index row applied to all three arrays; a resample is skipped when a class is absent from the resampled y_true; percentile CI of the difference and marginal CIs | :241-246 -> statistical_tests.py:133-169 | re-labelled test 192 | shared matrix | delta, CIs, n_skipped |
| 13 | Same for F1 Active (marginal CI only) | :247-248 | | | f1_active_ci_* |
| 14 | SHAP: TreeExplainer mean abs SHAP on the imputed test, ranks, top-3, rank of `aw_hr_mean_lag1` | :252-261 | test 192 | | shap_top*, aw_hr_mean_rank |
| 15 | Identity checks: cell (60, 85) vs `test_predictions_rf.csv` (y_pred element-wise, proba 1e-9, dates), persistence csv, T0.4 CI, T1.1 CI, majority, T1.2b delta, T1.4 SHAP; Table 7 T2 / T3 / T5 at 3 decimals (abort) and T4* / T6* (informative) | :336-382, :402-410 | | TABLE7 :116-123 | abort on FAIL |
| 16 | Label identity for A_r = 90: labels with A_s = 75 vs A_s = 80 | :399-401 | 958 rows | | run_meta flag |
| 17 | Save `t31_predictions_rf_R{R}_A{A_r}.csv`, grid csv / md, `t31_shap_top3.csv`, figure (non-comparable cells hatched + asterisk, canonical outlined), `t31_run_meta.json` | :411-414, :460-517, make_figure :266-311 | | | outputs |

No UNCERTAIN cells: every subset could be established from the code.

---

## T2. Targeted questions

Evidence lines marked (R) are in `recompute_f3_output.txt`, (E nn) in `_check_f3_empirical_output.txt`.

**Q1. Per fold: imputer, SMOTE, RF fit on the fold's train rows only; predict_proba on a strictly later validation block?** PASS.
`preprocess_split` is called with the fold's train and validation parts (calibration_rf.py:149-150) and fits the imputer and SMOTE on its first argument only (common.py:107-111); the RF is a fresh `build_models()["rf"]` per fold (:151-152). From `t13_oof_folds_rf.csv`, `train_end < val_start` holds on all 4 folds (2023-11-16 < 2023-11-17, 2024-05-06 < 2024-05-07, 2024-10-13 < 2024-10-14, 2025-04-15 < 2025-04-16; E34-E37). Refitting the 4 folds from scratch reproduces the saved OOF probabilities to 9.7e-17, with SMOTE sizes 354 / 675 / 1014 / 1359 and fold-specific imputer medians (E34-E38). OOF dates are unique and sorted; the last OOF date (2025-09-18) precedes the first test date (2025-09-19), so no validation row is a test row (E39). 612 rows, 127 / 452 / 33 (R).

**Q2. Calibrators fit only on the 612 OOF rows; nothing from the test set?** PASS.
`platt = PlattCls().fit(P_raw_oof, y_oof)` with `sub = oof[oof.fold.isin([1,2,3,4])]` (calibration_rf.py:497-504); the test arrays appear only in `predict_proba` (:514). Refitting `PlattOvR` on the OOF csv reproduces the joblib coefficients exactly (max abs diff 0.0, E07); a fit on OOF + test rows would give a different Active slope (0.2007 vs 0.2095, E09), and the fit on folds 2-4 only gives 0.3830 (E08), so the stored object is the OOF 1-4 fit.

**Q3. Platt = logistic regression on logit(p_c) with p clipped to [1e-6, 1 - 1e-6], no effective regularisation; coefficients as claimed?** PASS.
Joblib: `LogisticRegression` with `C=inf`, `l1_ratio=0`, lbfgs, tol 1e-8, one input feature, `penalty` attribute left at sklearn 1.8's default marker "deprecated" (E03-E05); clip 1e-6 (E02, logit at calibration_rf.py:273-275). Coefficients Rest 0.396962 / -1.059280, Normal 0.221038 / 1.026014, Active 0.209471 / -2.367283 equal the `t13_run_meta.json` values and the register row (E06). An unpenalised maximum-likelihood fit with `statsmodels.Logit` on the same inputs gives the same six numbers to 6 decimals (R), which confirms that `C=inf` is effectively no regularisation.

**Q4. Isotonic monotone on a 1001-point grid for all 3 classes?** PASS.
From the joblib: `y_min=0, y_max=1, out_of_bounds="clip", increasing=True`, 24 / 16 / 12 thresholds (E10-E12); `np.diff(predict(grid)) >= 0` strictly holds on all three classes (E13), and an independent `IsotonicRegression` refit on the OOF csv reproduces the stored predictions on test exactly (E14, R).

**Q5. Row normalisation after OvR: sum 1 on all 192 test rows; any zero-sum rows?** PASS.
Row sums before normalisation lie in [0.9226, 1.0544] (Platt) and [0.6402, 1.1301] (isotonic); no row has sum 0; after normalisation max |sum - 1| = 2.2e-16; fallback counter 0 for both (E15-E16; R). `n_fallback_rows` is 0 in all 12 split / fit / variant combinations (`t13_run_meta.json`).

**Q6. ECE: confidence = max p, 10 equal-width bins on [0, 1], weights n_b / N, empty bins ignored; bin edge convention; comparison with sklearn `calibration_curve(strategy="uniform")`.** PASS.
`ece_mce` keeps bins with `n_b > 0` and weights by `n_b / len(conf)` (calibration_rf.py:223-228). Bins are `searchsorted(EDGES[1:-1], p)` with `side="left"` (:205): a value equal to an interior edge falls in the lower bin, so bin 0 = [0, 0.1] and bin b > 0 = (b/10, (b+1)/10]; a probability of exactly 0.5 goes to bin 4 (0.4, 0.5] and exactly 1.0 to bin 9 (E17-E24, E32). sklearn's rule uses edges `linspace(0, 1 + 1e-8, 11)`, so the two assign different bins only for values within about 1e-9 above an edge (E19, E22); on every actual array (test max p for raw / platt / isotonic, test and OOF p_active) the bin ids are identical (E25-E29), and no test day has max p exactly on an edge (R). ECE built from `calibration_curve` outputs is 0.137701 = the module value; `prob_true` / `prob_pred` equal the csv `acc_b` / `conf_b` (E30). The alternative left-closed [lo, hi) convention gives the same ECE on this data (R).

**Q7. Brier multiclass = mean over rows of the sum over classes of (p_c - y_c)^2; equals sklearn `brier_score_loss(scale_by_half=False)` at 1e-12?** PASS.
Manual 0.509098758360 = sklearn 0.509098758360, difference 0.0 (E33, R). `scale_by_half=True` would give 0.2545 (R), which is why the register row names the convention.

**Q8. Reliability csv: days per bin sum to 192; the OOF Active bins "3% to 17%" reproduce from the OOF csv?** FAIL (text claim).
Every (split, fit, variant, kind) group of `t13_reliability_bins.csv` sums to 192 (test), 612 (fit 1-4) or 459 (fit 2-4) (E31). The csv values themselves reproduce from `t13_oof_probabilities_rf.csv` (R, section "OOF raw p_active reliability"). But the sentence "In every bin of the raw Active probability, including 0.4 to 0.7, the observed fraction of Active days lies between 3% and 17%" is not what the csv says: the non-empty bins have Active fractions 3.3%, 6.8%, 8.3%, 7.4%, 8.8%, 0.0% (bin (0.5, 0.6], n = 8), 16.7% (bin (0.6, 0.7], n = 6) and 0.0% (bin (0.7, 0.8], n = 2); the minimum is 0%, not 3% (R; csv rows 75 and 77). See Finding F3.1.

**Q9. apply_threshold: confidence = max(predict_proba); below threshold -> Normal; above -> argmax; strict or non-strict; days with max p exactly 0.50?** PASS.
`if max_prob < threshold: y_adjusted[i] = 1` (confidence_threshold.py:155-159), i.e. strict "< 0.50"; the kept prediction is the caller's `y_pred_raw` = argmax (safety_layer_calibrated.py:83, 85). No test day has max p equal to 0.50, so "<" and "<=" give the same 66 overrides (E40, R).

**Q10. caloric_error_analysis: |offset[pred] - offset[true]| with -200 / 0 / +200 over all 192 days; mean over 192?** PASS.
The loop runs over every (true, pred) pair, including the overridden days (confidence_threshold.py:180-184); `len(errors)` = 192, mean 58.3333, max 200 (E41). The mean over covered days only would be 66.67 kcal (R, diagnostic); the reported 58.3 is the all-192 convention, the same one under which "always Normal" gives 54.17 (R).

**Q11. The functions used by safety_layer_calibrated.py are the root ones, unmodified since f7cc4ca?** PASS.
`git diff --quiet f7cc4ca HEAD -- confidence_threshold.py` returns 0 (no change; the file's only commit is f7cc4ca); the module resolved at run time is `confidence_threshold.py` at the root (E01), imported after `sys.path.insert(0, ROOT)` (safety_layer_calibrated.py:43-45), and no file of that name exists under `revision/`. `main()` is never called.

**Q12. Does `_check_t13_safety_layer.py` run the old 16-feature pipeline and reproduce 80.2% / 68.8 kcal?** PASS on the erratum, with a correction to the premise.
The checker does not run the old pipeline: it reads the stored `outputs/tables/confidence_threshold_analysis.csv` (line 81) and compares it with the canonical RF through the same two functions (:72-96); its docstring says so (:20-27). Since the rerun fits the 5-minute budget, this audit executed the old pipeline's logic itself (`ct.load_and_prepare()` + `ct.train_model()` + the 7 thresholds, no writes): 21.5 s, 16 features, test 44 / 140 / 8, and the result equals the stored table on all 7 thresholds x 7 columns (E52-E54, both tables printed at the end of the E file). At 0.50: coverage 80.2083%, 38 overridden, 68.75 kcal; at 0: F1 0.5201, accuracy 0.6198, 3 Active predicted. The canonical RF gives 65.625% / 66 / 58.3333 (R). The old and canonical raw predictions agree on 147 / 192 days (E55). The erratum attribution in `CIFRE_REVIZIE.md:367` and `KBS_Revision_Plan.md` section 8 point 2 is therefore confirmed by execution, not only by reading.

**Q13. Labels regenerated per cell on the 958 days with the same rule; the 958-day set and the 766 / 192 cut independent of the thresholds; test dates identical to `test_predictions_rf.csv` for 3 cells?** PASS.
`assign_labels` returns NaN iff readiness or sleep is missing (t3_labels.py:93, :100), independently of the thresholds; `build_dataset` drops only NaN labels (common.py:69). Regenerating the labels for all 9 cells from the raw scores gives 958 labelled rows in every cell, the same dates as the snapshot, and the 958 / train / test distributions of `t31_threshold_grid.csv` cell by cell (R). The test dates of cells R55_A85, R60_A90 and R60_A85 refit through `fit_eval_labels` equal `test_predictions_rf.csv` (E43-E45), and the csv `y_true` of R55_A85 equals the regenerated R = 55 test labels (R).

**Q14. Imputer and SMOTE refit per cell on the re-labelled train; k = 3 when Active in train >= 4, n_min - 1 for 2-3; effective k for A_r = 90?** PASS.
`fit_eval_labels` calls `preprocess_split` for every cell (threshold_sensitivity.py:202), which fits a new imputer and SMOTE (common.py:107-111). `smote_k(2) = 1, smote_k(3) = 2, smote_k(4) = 3, smote_k(38) = 3` (E47), the same branches as the root Table 7 script (`threshold_sensitivity.py` root :173-184). The three A_r = 90 cells have 2 Active in train, k = 1 in the csv and in the refit, and 1818 = 3 x 606 rows after SMOTE (E44; `t31_run_meta.json` `cells_per_smote_k` {"1": 3, "3": 6}). Observation: because X does not depend on the labels, the per-cell imputer refit yields identical medians in every cell (E43-E45 feature 0 = 71.00); only SMOTE changes.

**Q15. Persistence per cell = previous-row label of the RE-labelled series?** PASS.
`persistence_predictions(cell["y"], i_split)` shifts the cell's own `y` (threshold_sensitivity.py:152, :224); on R55_A85, R60_A90 and R60_A85 it equals `[re-labelled label of 2025-09-18] + y_true[:-1]` (E43-E45). For R55_A85 the first test day uses the R = 55 label of 2025-09-18 (readiness 70, sleep 64 -> Normal, same as canonical), and persistence F1 macro recomputed from the csv `y_true` is 0.321145 = 0.3211 (R). For R = 65 that day becomes Rest (sleep 64 < 65), so the first persistence prediction does depend on R there (diagnostic, consistent with the rule).

**Q16. Comparability rule applied automatically; the 3 non-comparable cells marked in csv and figure?** PASS.
`is_comparable` (threshold_sensitivity.py:143-146) is evaluated inside `evaluate_cell` (:222) from the counts, never by hand; the csv column `comparable` is [T, T, F, T, T, F, T, T, F] in grid order (E46); `make_figure` hatches cells with `comparable == False` and appends an asterisk to their value (:280-285); the csv `comparable` flags equal the recomputed `is_comparable` on the 3 refit cells (E43-E45).

**Q17. CIs on the shared matrix with the per-cell y_true; resamples with an absent class skipped and counted (340 for A_r = 90); same skip rule as extended_metrics?** UNCERTAIN.
The CIs come from `statistical_tests.paired_bootstrap` with the shared matrix and the cell's `y_true` (threshold_sensitivity.py:241-248; R reproduces R55_A85's delta +0.1842 [+0.0281, +0.2919] and both marginal CIs). The skip rule is "a class absent from the resampled y_true" (statistical_tests.py:152-154), applied to every metric; 340 / 1000 skipped on R60_A90, expected (191/192)^192 = 0.367 (E49-E50). This is the same function as T1.2b, but it is NOT the rule of `extended_metrics.bootstrap`, which skips only when the metric returns nan (G-mean, the two AUCs) and computes F1 macro, accuracy, MCC and kappa on all 1000 resamples (extended_metrics.py:85-130, :195-213). On the canonical test set the two rules coincide (0 resamples lack a class, E51). On the three A_r = 90 cells they do not: for R60_A90 the RF macro-F1 CI is [0.6427, 0.7347] on 660 resamples under the T3.1 rule and would be [0.3119, 0.7299] on 1000 under the T1.1 rule (E49). See Finding F3.2.

**Q18. Any selection on the test set in T3.1; grid thresholds fixed in code before any fit?** PASS.
`REST_GRID = (55, 60, 65)`, `ACTIVE_R_GRID = (80, 85, 90)`, `ACTIVE_SLEEP = 75`, `MIN_ACTIVE_TEST = 5`, `MIN_ACTIVE_TRAIN = 4` are module constants (threshold_sensitivity.py:101-106); the only command-line option is `--results` (:527-529; E42). No threshold, hyperparameter or model is chosen from a test result; the Table 7 values in `TABLE7` (:116-123) are identity targets that abort the run, not choices. The RF hyperparameters come from `build_models()` unchanged.

**Q19. Cells (60, 90) and (65, 90) have the same labels as Table 7 T4 / T6 (90 / 80); every readiness >= 90 day has sleep >= 80?** PASS.
Among the 958 labelled days, 3 have readiness >= 90 and none of them has sleep < 80; the label vectors (R, 90, 75) and (R, 90, 80) are element-wise identical for R = 55, 60 and 65 (R), matching `labels_identical_to_table7_90_80` in `t31_run_meta.json`.

---

## T3. Independent recomputation (`revision/audit/recompute_f3.py`)

Imports: numpy, pandas, statsmodels, sklearn, imblearn only; nothing from `revision/scripts/`. Metrics are implemented from their definitions (ECE with the sklearn-uniform bin rule written out, Brier as the row-wise sum of squared errors, log loss as minus the mean log of the true-class probability, the safety layer as a boolean mask on max p, the caloric error from the offset dictionary, persistence as a shifted copy of the csv `y_true`, the paired bootstrap on the matrix regenerated with `default_rng(42)`), the Platt calibrators are refit with `statsmodels.Logit` and the isotonic ones with `IsotonicRegression`, and the labels are regenerated from the raw readiness / sleep scores. Output in `recompute_f3_output.txt`.

| Item | Recomputed | Claimed | Match (4 dp) |
|---|---|---|---|
| ECE top-label raw, test | 0.137701 [0.081513, 0.207356] | 0.1377 [0.0815, 0.2074] | yes |
| MCE top-label raw | 0.883183 | 0.8832 | yes |
| Brier multiclass raw | 0.509099 [0.474740, 0.548922] | 0.5091 [0.4747, 0.5489] | yes |
| Log loss raw | 0.852175 [0.770082, 0.964825] | 0.8522 [0.7701, 0.9648] | yes |
| ECE Active raw / Brier Active raw | 0.093601 / 0.050199 | 0.0936 / 0.0502 | yes |
| Platt slope / intercept Rest, Normal, Active (unpenalised Logit) | 0.396962 / -1.059280; 0.221038 / 1.026014; 0.209471 / -2.367283 | 0.3970 / -1.0593; 0.2210 / 1.0260; 0.2095 / -2.3673 | yes (6 of 6) |
| ECE Platt / isotonic, test | 0.036063 [0.012344, 0.099635] / 0.040481 [0.018928, 0.107674] | 0.0361 [0.0123, 0.0996] / 0.0405 [0.0189, 0.1077] | yes |
| Brier multiclass Platt / isotonic | 0.412354 / 0.409517 | 0.4124 / 0.4095 | yes |
| Log loss Platt / isotonic | 0.696883 / 0.711098 | 0.6969 / 0.7111 | yes |
| ECE Active Platt / isotonic; p_active max | 0.020622 / 0.034033; 0.102910 / 0.189588 | 0.0206 / 0.0340; 0.1029 / 0.1896 | yes |
| Argmax changed vs raw; calibrated argmax F1 macro | 53 / 53; 0.281124 | 53; 0.2811 | yes |
| Safety layer raw @0.50: coverage, covered, overridden | 65.625%, 126, 66 | 65.6% (126/192), 66 | yes |
| Safety layer raw @0.50: mean / max caloric error; emitted R/N/A | 58.3333 / 200; 28 / 162 / 2 | 58.3 / 200; 28 / 162 / 2 | yes |
| Safety layer raw @0.50: F1 macro / accuracy / F1 Active | 0.506708 / 0.708333 / 0.400000 | 0.5067 / 0.7083 / 0.4000 | yes |
| Mean caloric error @0; always Normal | 71.875; 54.166667 | 71.9; 54.2 | yes |
| R55_A85 labels on 958 | 97 / 815 / 46 | 97 / 815 / 46 | yes |
| All 9 cells: 958 / train / test distributions | == `t31_threshold_grid.csv` | | yes |
| R55_A85 F1 macro / F1 Active / accuracy / MCC | 0.505319 / 0.615385 / 0.713542 / 0.037877 | 0.5053 / 0.6154 / 0.7135 / 0.0379 | yes |
| R55_A85 persistence F1 macro (first day = R55 label of 2025-09-18 = Normal) | 0.321145 | 0.3211 | yes |
| R55_A85 delta RF minus persistence | +0.184174 [+0.028099, +0.291884], prop > 0 0.988 | +0.1842 [+0.0281, +0.2919], 0.988 | yes |
| R55_A85 marginal CIs RF / persistence | [0.357385, 0.608411] / [0.277085, 0.367875] | [0.3574, 0.6084] / [0.2771, 0.3679] | yes |
| Scratch retraining of R55_A85 (sklearn / imblearn only) | y_pred identical on 192 days, max proba diff 9.9e-17 | | yes |
| OOF Active fraction in every non-empty raw p_active bin within [3%, 17%] | min 0.0000, max 0.1667 | "between 3% and 17%" | NO |

T3 verdict: PASS. 74 / 75 comparisons identical at 4 decimals; every numeric target listed in the brief reproduces. The single failure is the text claim of Finding F3.1, which this audit added as a check and which is not one of the brief's numeric targets. Deviation from the brief: the labels were regenerated from `data/oura_combined.csv` rather than from the snapshot, because the snapshot does not contain the raw readiness and sleep scores; the regenerated dates and canonical labels equal the snapshot on all 958 rows.

---

## Findings (FAIL or UNCERTAIN only)

### F3.1. The reliability sentence "between 3% and 17%" misstates the lower bound; two OOF bins have 0% Active (Q8: FAIL)
Observed: in `t13_reliability_bins.csv` (split oof, fit 1-4, raw, kind active) the non-empty bins have Active fractions 0.0327, 0.0678, 0.0833, 0.0741, 0.0882, 0.0000 (bin (0.5, 0.6], 8 days), 0.1667 (bin (0.6, 0.7], 6 days) and 0.0000 (bin (0.7, 0.8], 2 days); recomputed from `t13_oof_probabilities_rf.csv` with the same result (R). The sentence "In every bin of the raw Active probability, including 0.4 to 0.7, the observed fraction of Active days lies between 3% and 17%" appears in `KBS_Manuscript_R1.docx` (calibration subsection; source `F4_manuscript_changes.md:115`), in `KBS_Response_R1.docx` (reply to R1.9; source `F4_response_letter.md:81`, "is between 3% and 17%") and in `CIFRE_REVIZIE.md:363` ("intre 3% si 17%"). Why it matters: the number is wrong as stated (the range is 0% to 16.7%), and the two 0% bins hold 10 of the 16 OOF days with raw p_active above 0.5, which is the strongest part of the argument the sentence is making. The direction of the conclusion (raw p_active does not separate Active days) is unchanged and if anything stronger. What the authors should decide: whether to rewrite as "never exceeds 17% (0% to 17%)" or to give the per-bin counts, in the manuscript, the response letter and the register note.

### F3.2. The bootstrap skip rule of T3.1 (and T1.2) differs from the one of T1.1 for the same metric, with a large effect on the three non-comparable cells (Q17: UNCERTAIN)
Observed: `statistical_tests.paired_bootstrap` skips a resample whenever a class is absent from the resampled `y_true`, for every metric (statistical_tests.py:152-154); `extended_metrics.bootstrap` skips only when the metric returns nan, and F1 macro with fixed labels never does (extended_metrics.py:107-110, :195-213). Both scripts document their own rule, no document says the rules differ, and on the canonical test set they coincide (0 resamples lack a class, E51). On R60_A90 (1 Active in test) the RF macro-F1 CI is [0.6427, 0.7347] on 660 resamples under the T3.1 rule and would be [0.3119, 0.7299] under the T1.1 rule (E49); the same applies to R55_A90 and R65_A90 and to their persistence and delta CIs (`CIFRE_REVIZIE.md` rows 117, 120, 123; `t31_threshold_grid.csv`). Why it matters: those cells are flagged non-comparable and excluded from the three answers of T3.1, so no conclusion changes, but their CIs are printed in the register and in the grid table, and a reader applying the T1.1 procedure would obtain a much wider interval for the same cell. What the authors should decide: whether to state in the T3.1 table note (and in the scripts README) that the shared-matrix CIs of T1.2 / T3.1 drop resamples with an absent class while the T1.1 CIs do not, or to blank the CIs of the three non-comparable cells.

No other FAIL or UNCERTAIN verdicts were reached.

---

## Observations (no verdict change; reported for completeness)

- Q12: `_check_t13_safety_layer.py` is a comparison against the stored table, not a rerun of the old pipeline; the provenance statement in `CIFRE_REVIZIE.md:367` rested on reading `confidence_threshold.main()`. The rerun performed here (21.5 s, E52-E54) is the first execution-based confirmation: the 16-feature XML pipeline reproduces the stored table exactly, so the erratum (80.2% / 38 / 68.8 kcal replaced by 65.6% / 66 / 58.3 kcal) is correctly attributed.
- Q6: the script's bin edges are exact multiples of 0.1 (floating-point `linspace`), sklearn's are 1e-9 higher; a probability within 1e-9 above an edge would be binned differently (E19, E22). No such value occurs in any array used (E25-E29), and the left-closed alternative gives the same ECE (R).
- Q14: the per-cell imputer refit is numerically a no-op (same X, same medians in all cells); the register's "imputer refit per cell" is true but only SMOTE differs between cells.
- Q5: the isotonic OvR row sums before normalisation range from 0.64 to 1.13 (E16), so the normalisation step moves isotonic probabilities by up to a third; for Platt the range is 0.92 to 1.05 (E15). The register describes the normalisation but not its size.
- Q10: the mean caloric error over covered days only would be 66.7 kcal at 0.50 (R); the manuscript's 58.3 kcal is the all-192 convention, the same under which "always Normal" scores 54.2 kcal (register note `CIFRE_REVIZIE.md:365` already says the two must be read together).
- Q15: the first persistence prediction of a cell depends on R through the label of 2025-09-18 (readiness 70, sleep 64): Normal for R = 55 and 60, Rest for R = 65.
- T0: `revision/audit/AUDIT_DECISIONS.md` is an untracked file written by the author during this session; it is not part of this audit and was left unstaged.

---

## Summary

- T2: Q1 PASS, Q2 PASS, Q3 PASS, Q4 PASS, Q5 PASS, Q6 PASS, Q7 PASS, Q8 FAIL, Q9 PASS, Q10 PASS, Q11 PASS, Q12 PASS, Q13 PASS, Q14 PASS, Q15 PASS, Q16 PASS, Q17 UNCERTAIN, Q18 PASS, Q19 PASS (17 PASS, 1 FAIL, 1 UNCERTAIN).
- T3: PASS, 74 / 75 values identical at 4 decimals from an independent implementation (calibrators refit with statsmodels / sklearn, labels regenerated from the raw scores, cell R55_A85 retrained from scratch); the one failure is the text claim of F3.1.
- Findings: F3.1, the manuscript, response letter and register say the OOF Active fraction lies "between 3% and 17%" in every raw p_active bin, but two bins (0.5-0.6 and 0.7-0.8) have 0%; F3.2, the shared-matrix bootstrap of T3.1 / T1.2 drops resamples with an absent class while T1.1 does not, which widens the macro-F1 CI of R60_A90 from [0.64, 0.73] to [0.31, 0.73] under the other rule and is stated nowhere as a difference.
