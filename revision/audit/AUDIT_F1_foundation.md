# AUDIT F1: pipeline foundation (split, preprocessing, labels, baselines, bootstrap)

Auditor role: external, read-and-report only. Branch `audit-r1` from `e46ddaaec5ce1b52583412266e05d9ad5a05c5d1` (master). Run date: 2026-10-25.
Environment used for every command below: Python 3.11.9, scikit-learn 1.8.0, numpy 2.4.3, pandas 2.3.3, imbalanced-learn 0.14.1 (output of `python -c "import sys,sklearn,numpy,pandas,imblearn; ..."`), identical to `models/frozen_meta_20260331.json` `environment`.

Files written by this audit (nothing else was modified):

- `revision/audit/AUDIT_F1_foundation.md` (this report)
- `revision/audit/_check_f1_empirical.py` + `_check_f1_empirical_output.txt` (T2/T4 evidence; imports `revision/scripts/common.py`, no retraining, no writes)
- `revision/audit/recompute_f1.py` + `recompute_f1_output.txt` (T3; imports only numpy, pandas, sklearn)

Scratch retraining outputs (T4) were written to the session scratchpad, outside the repository.

---

## T0. Inventory and branch

`git checkout -b audit-r1` from HEAD `e46ddaa`. `git status --short` immediately after:

```
 D disertatie_audit.zip
?? _referinte_upb/Lazar_Adrian_Lucrare_Disertatie_2026.pdf.docx
?? eswa_architecture.py
?? outputs/figures/eswa_architecture/
```

The tree was not clean at the start: one tracked file deleted in the working tree (`disertatie_audit.zip`) and three untracked paths. None of them is an input or output of this slice. The audit did not touch them. T0 pass criterion "arbore curat": FAIL (pre-existing state, see Findings F0).

Inventory (size in bytes, sha256 first 16 hex chars, from `stat -c %s` and `sha256sum`):

| File | Size | sha256[:16] |
|---|---:|---|
| revision/scripts/common.py | 6220 | fb73b00eb9a650d3 |
| revision/t3_labels.py | 3814 | 5e8b726fc629c665 |
| revision/scripts/save_test_predictions.py | 4286 | 24c7c7d9cf976360 |
| revision/scripts/persistence_baseline.py | 7945 | c77ec25813d6dd6e |
| revision/scripts/extended_metrics.py | 16434 | 02c78dcf65c845fd |
| revision/results/test_predictions_rf.csv | 14349 | aea39c6dd2989e24 |
| revision/results/test_predictions_xgb.csv | 9537 | e0afa80d26440534 |
| revision/results/test_predictions_svm.csv | 14764 | c48625b27debfcd2 |
| revision/results/test_predictions_persistence.csv | 5421 | ab228a11e65bfe17 |
| revision/results/test_predictions_majority.csv | 5421 | eaf03527561e10fa |
| revision/results/test_predictions_metrics.csv | 232 | e8e22cc813096876 |
| revision/results/persistence_baseline.csv | 582 | 7e521c5d7579c22f |
| revision/results/bootstrap_indices_1000x192_seed42.npy | 1536128 | fff3b2834c061962 |
| revision/results/models/rf_revision.joblib | 1810593 | b8113662207ab548 |
| revision/results/models/imputer_revision.joblib | 1295 | 9b5e16ba6cf9a70a |
| revision/results/models/scaler_revision.joblib | 1663 | 981b7001eaf30521 |
| models/dataset_snapshot_20260331.csv | 116680 | 176649b59fef3adb |
| models/frozen_meta_20260331.json | 3036 | fd874fe5ef871cc3 |
| models/rf_frozen_20260331.joblib | 1810593 | b8113662207ab548 |
| models/rf_model.joblib | 1810593 | b8113662207ab548 |
| models/imputer.joblib | 1295 | 9b5e16ba6cf9a70a |
| revision/CIFRE_REVIZIE.md | 188016 | 510ba2479c80d682 |
| CIFRE_CANONICE.md | 25648 | a80e9d6c5986c98b |
| sanity_check_frozen.py | 3870 | 3950d7140f7d3a56 |

Name mapping versus the brief: all names exist as given. The canonical comparison script referred to by `common.py:4-5` is `cap5_compare_algorithms_json_ext.py` at repo root. The frozen imputer is `models/imputer_frozen_20260331.joblib` (same sha256 as `models/imputer.joblib`, verified in T2 Q6 by `statistics_` equality).

Observation: `rf_revision.joblib`, `rf_frozen_20260331.joblib` and `rf_model.joblib` are byte-identical (sha256 b8113662207ab548). T4 shows that retraining with `save_test_predictions.py` reproduces the same bytes, so this is byte-level reproducibility, not a copy.

---

## T1. Data-flow table (from reading the code; no execution needed for this table)

Line numbers refer to the files at HEAD `e46ddaa`.

| # | Operation | File:line | Applied on | Parameters | Output |
|---|---|---|---|---|---|
| 1 | Read Oura daily CSV, parse `date` | common.py:49-50 | all raw Oura rows (1,009; verified T2 Q4) | `data/oura_combined.csv` | DataFrame `oura` |
| 2 | Read Apple Watch daily CSV, parse `date` | common.py:51-52 | all AW rows (1,083 lines minus header from `wc -l`) | `apple_watch_api/apple_health_daily.csv` | DataFrame `aw` |
| 3 | Derive `aw_active_calories = aw_active_energy_kj / 4.184`, rounded 1 dp | common.py:53 | all AW rows | 4.184 | new AW column |
| 4 | Left-merge 8 AW columns onto Oura frame, one column at a time, dropping AW NaN per column before merging | common.py:54-57 | Oura rows (left side), AW rows (right side) | `on="date", how="left"` | 1,009-row frame, Oura dates only |
| 5 | Sort by date, reset index | common.py:58 | 1,009 rows | | sorted raw frame |
| 6 | Lag-1 features: `df[f].shift(1)` for the 19 base columns | common.py:66-67 (same in cap5_compare_algorithms_json_ext.py:107-108; freeze_production_model.py:70) | 1,009 rows (before any drop) | row shift of 1 | 19 `*_lag1` columns |
| 7 | Labels: `t3_labels.assign_labels(df)` | common.py:68 -> t3_labels.py:79-101 | 1,009 rows | rest 60, active readiness 85, active sleep 75 | float label 0/1/2 or NaN |
| 8 | Keep `date` + 19 features + `label`; drop rows with NaN label | common.py:69 | 1,009 -> 958 | `dropna(subset=["label"])` only | 958 rows, NaN features kept |
| 9 | Sort by date, reset index (again) | common.py:70 | 958 | | sorted 958 |
| 10 | Return `X` (958x19), `y` as int, `dates` | common.py:71 | 958 | | X, y, dates |
| 11 | Split index `i = int(len(X) * 0.8)` = 766; positional `iloc[:i]` / `iloc[i:]` | common.py:93-94 (TRAIN_RATIO common.py:33) | 958 -> 766 / 192 | 0.8 | X_tr, X_te, y_tr, y_te, dates_te |
| 12 | `SimpleImputer(strategy="median")` fit_transform | common.py:107-108 | train 766 only | median | imputed train 766x19 |
| 13 | Imputer transform | common.py:109 | test 192 | fitted on 766 | imputed test 192x19 |
| 14 | `SMOTE(random_state=42, k_neighbors=3).fit_resample` | common.py:111 | imputed train 766 (unscaled) | seed 42, k 3 | train 1,710 (570 per class, verified T2 Q7) |
| 15 | `StandardScaler().fit_transform` | common.py:113-114 | imputed train 766, pre-SMOTE | | scaled train 766 |
| 16 | Scaler transform | common.py:115 | imputed test 192 | fitted on 766 | scaled test 192 |
| 17 | Second `SMOTE(random_state=42, k_neighbors=3)` | common.py:116 | scaled train 766 | seed 42, k 3 | scaled train 1,710 (SVM) |
| 18 | Pack `Split` (y_train/y_test index reset) | common.py:118-120 | | | `Split` dataclass |
| 19 | Size check against CANON (958/766/192, test 44/140/8) | common.py:123-129, CANON at :43 | | | exit 1 on mismatch |
| 20 | RF `RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=5)` fit | save_test_predictions.py:48-49, :68 | train after SMOTE (1,710) | | fitted RF |
| 21 | XGB fit | save_test_predictions.py:50-51, :68 | train after SMOTE (1,710) | 100 est, depth 6, lr 0.1 | fitted XGB |
| 22 | SVM fit | save_test_predictions.py:52, :65 | scaled train after SMOTE (1,710) | RBF, C 1, gamma scale, probability | fitted SVC |
| 23 | `predict` / `predict_proba` | save_test_predictions.py:70-72 | imputed test 192 (RF/XGB), scaled test 192 (SVM) | asserts `classes_ == [0,1,2]` | y_pred, proba |
| 24 | Save `test_predictions_{key}.csv` (date, y_true, y_pred, p_rest, p_normal, p_active) and model joblib | save_test_predictions.py:74-77 | test 192 | | csv + joblib |
| 25 | Metrics `f1_score(average="macro", zero_division=0)`, `accuracy_score`, `matthews_corrcoef`, rounded 4 dp, compared to EXPECTED with tolerance 5e-5 | save_test_predictions.py:79-83, EXPECTED :39-43 | test 192 | | PASS/FAIL per model, exit 1 on any FAIL |
| 26 | Save imputer / scaler joblib, `test_predictions_metrics.csv` | save_test_predictions.py:92-97 | | | artefacts |
| 27 | Persistence: read snapshot, sort by date, `label = nutrition_label`, `y_prev = y.shift(1)`, drop NaN labels | persistence_baseline.py:56-59, :64-69, :86-88 | snapshot 958 rows (labels already non-NaN) | | date, y, y_prev |
| 28 | Persistence split `int(len(d) * 0.8)` positional | persistence_baseline.py:133-134 | 958 -> 766 / 192 | 0.8 | train, test |
| 29 | Majority = `train["y"].mode()[0]` | persistence_baseline.py:140 | train 766 | | int 1 |
| 30 | Persistence metrics + bootstrap CI (1000 resamples, fresh `default_rng(42)` per metric call) | persistence_baseline.py:93-119, :157-158 | test 192 | seed 42, n_boot 1000 | `persistence_baseline.csv` |
| 31 | Save `test_predictions_persistence.csv`, `test_predictions_majority.csv` with one-hot pseudo-probabilities | persistence_baseline.py:163-167 | test 192 | | csv |
| 32 | extended_metrics: last train label from snapshot (`lab.iloc[765]`) | extended_metrics.py:134-142 | snapshot 958 | split 766 | int |
| 33 | extended_metrics: read rf/xgb/svm csv, check columns, 192 rows, identical y_true, proba rows sum to 1, support (44,140,8) | extended_metrics.py:145-172 | test 192 | | y_true, preds, probas |
| 34 | extended_metrics: persistence reconstructed as `[last_train_label] + y_true[:-1]`, compared to saved csv | extended_metrics.py:174-178 | test 192 | | preds["persistence"] |
| 35 | Bootstrap index matrix `default_rng(42).integers(0, 192, size=(1000, 192))`, compared to npy if present else saved | extended_metrics.py:183-192 | | seed 42 | 1000x192 int64 |
| 36 | Bootstrap loop: same row `r` applied to `y_true[r]` and `yp[r]`; NaN metric values skipped and counted; percentiles 2.5 / 97.5 | extended_metrics.py:195-213 (call site :269) | test 192 resampled | | ci_low, ci_high, n_used, n_skipped |

No UNCERTAIN cells: every subset could be established from the code.

---

## T2. Targeted questions

Evidence lines prefixed `Q<n>` are from `revision/audit/_check_f1_empirical_output.txt`.

### Q1. Lag direction. PASS
Code: `df[f"{f}_lag1"] = df[f].shift(1)` on the date-sorted raw frame (common.py:66-67 after common.py:58). pandas `shift(1)` moves values forward by one row, so row t receives row t-1.
Empirical (three dates, output lines `Q1 ...`):

| date t | readiness_score_lag1 at t | previous raw row (date, readiness) | readiness at t |
|---|---|---|---|
| 2024-01-15 | 59.0 | 2024-01-14, 59.0 | 52.0 |
| 2025-09-19 | 70.0 | 2025-09-18, 70.0 | 64.0 |
| 2026-03-30 | 68.0 | 2026-03-29, 68.0 | 71.0 |

Identity `lag1(row t) == raw(row t-1)` holds for all 19 features over all 1,009 rows (NaN-aware): `mismatching features: []`. The same three values are present in the built 958-row X.

### Q2. Any feature computed from day-t values (same row as the label)? UNCERTAIN
Row level: no. All 19 features are `shift(1)` of a base column (common.py:66-67); the label uses `readiness_score` and `sleep_score` of the same row (t3_labels.py:91-92, called at common.py:68). The 19 names and sources (common.py:34-41, load_raw common.py:49-57):

| # | Feature | Base column | Source file | Note |
|---|---|---|---|---|
| 1 | readiness_score_lag1 | readiness_score | data/oura_combined.csv | also a label input (at t) |
| 2 | sleep_score_lag1 | sleep_score | data/oura_combined.csv | also a label input (at t) |
| 3 | hrv_mean_lag1 | hrv_mean | data/oura_combined.csv | |
| 4 | hrv_balance_lag1 | hrv_balance | data/oura_combined.csv | |
| 5 | recovery_index_lag1 | recovery_index | data/oura_combined.csv | |
| 6 | rhr_lowest_lag1 | rhr_lowest | data/oura_combined.csv | |
| 7 | rhr_avg_lag1 | rhr_avg | data/oura_combined.csv | |
| 8 | activity_score_lag1 | activity_score | data/oura_combined.csv | |
| 9 | steps_oura_lag1 | steps_oura | data/oura_combined.csv | |
| 10 | respiratory_rate_lag1 | respiratory_rate | data/oura_combined.csv | |
| 11 | temp_deviation_lag1 | temp_deviation | data/oura_combined.csv | |
| 12 | aw_hr_mean_lag1 | aw_hr_mean | apple_watch_api/apple_health_daily.csv | |
| 13 | aw_hrv_mean_lag1 | aw_hrv_mean | apple_watch_api/apple_health_daily.csv | |
| 14 | aw_steps_lag1 | aw_steps | apple_watch_api/apple_health_daily.csv | |
| 15 | aw_active_calories_lag1 | aw_active_calories | derived: aw_active_energy_kj / 4.184 (common.py:53) | |
| 16 | aw_spo2_lag1 | aw_spo2 | apple_watch_api/apple_health_daily.csv | |
| 17 | aw_sleep_deep_hours_lag1 | aw_sleep_deep_hours | apple_watch_api/apple_health_daily.csv | sleep-night metric |
| 18 | aw_sleep_rem_hours_lag1 | aw_sleep_rem_hours | apple_watch_api/apple_health_daily.csv | sleep-night metric |
| 19 | aw_resp_rate_lag1 | aw_resp_rate | apple_watch_api/apple_health_daily.csv | |

Why UNCERTAIN rather than PASS: "row t-1" is only equivalent to "a different night than the label's" if both sources attach a sleep record to the same calendar date. Oura rows take their date from the API `day` field (oura_merge.py:85, :92; oura_collect.py:56). Apple Watch rows take their date from the Health Auto Export entry `date` field (parse_health_auto_export.py:186-189, :215). Which night a `date` refers to (the night ending on that morning, or the night starting on that evening) is a data-source convention, not visible in the code. Exact question for the authors: for a given calendar date d, is `aw_sleep_deep_hours[d]` / `aw_sleep_rem_hours[d]` the night that ends on the morning of d (same convention as Oura `sleep_score[d]`)? If instead it is the night that starts on the evening of d, then `aw_sleep_*_lag1(t)` and the label's `sleep_score(t)` describe the same night.

### Q3. Previous row vs previous calendar day; gap counts; same convention for features and persistence? FAIL (convention differs)
Lag is on the previous row (common.py:67, `shift(1)`), computed on the 1,009-row raw frame before the label drop (common.py:66-69). Persistence is `y.shift(1)` on the 958-row labelled snapshot (persistence_baseline.py:69, after `dropna` at :87) and, equivalently, `[last_train_label] + y_true[:-1]` (extended_metrics.py:174).
Counts (output lines `Q3 ...`):

| Series | consecutive-row pairs with gap > 1 day |
|---|---|
| raw 1,009 rows (2023-06-09 .. 2026-03-30, 1,026 calendar days) | 12 (max gap 5 days) |
| 958 labelled rows | 42 (gap 2: 31, 3: 5, 4: 4, 6: 1, 10: 1) |
| of which inside train (rows 1..765) | 41 |
| of which inside test (rows 766..957) | 1 (2025-10-29; previous labelled row 2025-10-27) |
| train end -> test start | 1 day (2025-09-18 -> 2025-09-19) |

Consequence: for 38 of the 42 gap rows in the 958 frame, the 19 features come from the previous calendar day (a raw row that exists but has no label), while the persistence prediction for the same row comes from the last labelled day, 2 to 10 days earlier. For the remaining 4 gap rows the calendar-previous day is absent from the raw file too, so the features also come from an older row (`rows whose calendar prev day absent from raw: 4`; `lag1 vs calendar-previous-day readiness: mismatches 0`).
In the test window the single gap row, 2025-10-29, has no raw row for 2025-10-28 (`raw rows around 2025-10-29` in the output), so both conventions use the 2025-10-27 row there. The test-set figures are therefore unaffected; the difference is between how the two are described and how train rows are built.

### Q4. Why 1,009 -> 958? PASS
Only one drop criterion exists: `dropna(subset=["label"])` (common.py:69; cap5_compare_algorithms_json_ext.py:122; freeze_production_model.py:74). Label is NaN when `readiness_score` or `sleep_score` is NaN (t3_labels.py:93, :100). Empirically: label NaN 51, readiness NaN 51, sleep NaN 51, either 51, so the 51 dropped rows have both scores missing. Rows with NaN features are kept (0 dropped rows have all 19 lag features NaN). The first raw row 2023-06-09 is among the 51 (readiness and sleep NaN). Dropped dates are listed in the output (`dropped rows, dates:`).

### Q5. Sorting before the cut; positional vs date cut; boundary dates. PASS
Sort at common.py:58 (raw) and common.py:70 (958 frame); cut by position `int(958 * 0.8) = 766` with `iloc` (common.py:93-94). Output: `split idx 766 | train end 2025-09-18 | test start 2025-09-19 | test end 2026-03-30`; `raw sorted: True`, built frame `sorted: True`. Train 158/570/38, test 44/140/8.

### Q6. Imputer fit on 766 only; any statistic on 958 or test? PASS
`imputer.fit_transform(X_tr)` then `imputer.transform(X_te)` (common.py:107-109). Output: `imputer.statistics_ == median(train 766): True | == median(all 958): False`; `imputer_revision == recomputed: True | imputer_frozen == recomputed: True`. NaN cells before imputation: train 461, test 49. No other fit, min/max, median or scaler on the 958 or on test was found in common.py, save_test_predictions.py, persistence_baseline.py or extended_metrics.py (the only `.fit*` calls are common.py:108, :111, :114, :116 and the model fits at save_test_predictions.py:65, :68).

### Q7. SMOTE train only, after imputation, k=3, seed 42; 1,710 rows; test stays 192 real rows. PASS
common.py:111 (`SMOTE(random_state=seed, k_neighbors=k_neighbors)` on `X_tr_imp`, defaults `k_neighbors=3`, `seed=42` at common.py:91, :99, :32). Output: `rows after SMOTE (trees): 1710 {0: 570, 1: 570, 2: 570}`, `X_test rows: 192`, and the first 766 rows of the SMOTE output equal the imputed train (synthetic rows are appended).

### Q8. SVM scaler fit on real 766 before SMOTE; SMOTE on the scaled matrix; no scaling for RF/XGB. PASS
common.py:113-116: scaler fit on `X_tr_imp` (766), then a second SMOTE on `X_tr_scaled`. Output: `scaler.mean_ == mean(imputed train 766): True | scaler.n_samples_seen_: 766.0`; `rows after SMOTE (scaled, SVM): 1710`; `scaler_revision == recomputed: True`. RF/XGB are fitted on `s.X_train_sm` (unscaled) and predict on `s.X_test` (save_test_predictions.py:68-69); SVM on `s.X_train_scaled_sm` / `s.X_test_scaled` (:65-66). Matches cap5_compare_algorithms_json_ext.py:160-174.

### Q9. Label rule equals the claimed one; 202/710/46. PASS
t3_labels.py:94-99: `is_rest = (r < 60) | (s < 60)`, `is_active = (r >= 85) & (s >= 75)`, Rest assigned after Active so Rest has priority (:98-99); defaults at :27-29. Same comparisons as the canonical row function (cap5_compare_algorithms_json_ext.py:110-117). Output: vectorised rule equals the row rule over all 1,009 rows; distribution `{0: 202, 1: 710, 2: 46}`; no row satisfies both conditions. Boundary cases exist in the data (readiness == 60 on 28 rows, sleep == 60 on 13, readiness == 85 on 19, sleep == 75 on 45), so the strict/non-strict choice is material and the code matches the claim (`<` for Rest, `>=` for Active).

### Q10. Persistence = true label of row t-1; first test day uses the last train day. PASS
persistence_baseline.py:69 (`y.shift(1)` on the full 958 series before the split at :133-134), check at :137-138; extended_metrics.py:174 reconstructs it from `last_train_label()` (:134-142, `lab.iloc[765]`) and `y_ref[:-1]`, and compares to the csv (:176-178). Output: `persistence csv y_pred == [last_train_label] + y_true[:-1]: True`, last train label (2025-09-18) = 1. First 5 test rows of `test_predictions_persistence.csv`: y_true `[1, 1, 1, 0, 2]`, y_pred `[1, 1, 1, 1, 0]` (row 0 = train label, rows 1-4 = y_true of the previous row).

### Q11. Majority = mode of train. PASS
persistence_baseline.py:140 `train["y"].mode()[0]`; cap5_compare_algorithms_json_ext.py:178-180 `DummyClassifier(most_frequent)` fitted on pre-SMOTE `y_train`; extended_metrics.py:175 hard-codes class 1. Output: train mode 1; the mode of the 958 and of the test is also 1, so the distinction is not observable on this data but holds in the code.

### Q12. Metric functions; differences between canonical and revision. PASS
Canonical: `f1_score(y_test, y_pred, average="macro", zero_division=0)`, `matthews_corrcoef`, `accuracy_score` (cap5_compare_algorithms_json_ext.py:227-237; sanity_check_frozen.py:77-79). save_test_predictions.py:80-82 and persistence_baseline.py:104-105, :114 use the same calls, without `labels=`. extended_metrics.py:110-112 adds `labels=[0, 1, 2]`. On the full 192-row test all three classes are present, so the two forms coincide (T3 output: RF 0.506324 / 0.640625 / 0.141560). In the bootstrap the two forms also coincide on this matrix because no resample lacks a class (Q13). Note: the canonical RF CI [0.373, 0.610] in CIFRE_CANONICE.md comes from `bootstrap_ci_frozen.py` with 2,000 resamples drawn sequentially (bootstrap_ci_frozen.py:49, :177-189), a different RNG stream from the 1,000x192 matrix; CIFRE_REVIZIE.md line 43 states this.

### Q13. Bootstrap matrix; same index vector for y_true and y_pred; resamples with a missing class. PASS
Output: `regenerated == npy: True | dtypes int64 int64 | bytes equal: True | shape (1000, 192)`. extended_metrics.py:269 applies the same row `r` to `y_true[r]` and `yp[r]`; persistence_baseline.py:98-99 applies the same `idx` to both. persistence_baseline.py draws indices sequentially from a fresh `default_rng(42)` per `bootstrap_ci` call (:94-98); empirically these 1000 sequential draws equal the matrix (`sequential per-resample draws ... == matrix: True`), which is why T0.4 and T1.1 CIs coincide.
Missing-class handling: extended_metrics.py skips a resample only when the metric returns NaN (:209-210), which happens for G-mean (:98-99) and the AUC/PR metrics (:120-121, :125-126); for F1/MCC/accuracy/kappa with `labels` fixed the resample is kept and an absent class contributes F1 = 0 via `zero_division=0`. persistence_baseline.py has no handling (f1 without `labels=`, no skip). On this matrix the question is moot: `resamples with a class absent from y_true: 0` (minimum Active count in a resample: 1).

### Q14. Hash 63bc3a2915bae864. PASS
Computed on the normalised snapshot DataFrame, not on file bytes: columns sorted alphabetically, `to_csv(index=False, lineterminator="\n")`, UTF-8, sha256 first 16 hex (freeze_production_model.py:90-96; replica sanity_check_frozen.py:46-48). Reproduced on the current `models/dataset_snapshot_20260331.csv`: `63bc3a2915bae864`. Also reproduced on the frame rebuilt from the raw CSVs through `common.load_raw` + `build_dataset` with the same normalisation: `63bc3a2915bae864` (snapshot and rebuilt frame are equal in dates, labels and all 19 features, NaN-aware). The sha256 of the raw file bytes is different (`176649b59fef3adb`), as expected.

### Q15. RF parameters in rf_revision.joblib. PASS
`get_params()`: n_estimators 100, max_depth 10, min_samples_leaf 5, random_state 42, class_weight None, max_features 'sqrt', criterion 'gini', bootstrap True; `n_features_in_` 19; `classes_` [0, 1, 2]. `rf_frozen_20260331.joblib` has identical params. `rf_revision.predict(recomputed X_test)` equals the csv `y_pred` (no retraining).

T2 tally: PASS 13, FAIL 1 (Q3), UNCERTAIN 1 (Q2).

---

## T3. Independent recomputation (`revision/audit/recompute_f1.py`)

Imports: numpy, pandas, sklearn.metrics only. Inputs: `test_predictions_rf.csv` (y_true, y_pred), `models/dataset_snapshot_20260331.csv` (last train label and train distribution), regenerated bootstrap matrix (compared to the npy, identical). Output in `revision/audit/recompute_f1_output.txt`.

| key | claimed | recomputed | verdict |
|---|---|---|---|
| RF F1 macro | 0.5063 | 0.5063 (0.506324) | PASS |
| RF accuracy | 0.6406 | 0.6406 (0.640625) | PASS |
| RF MCC | 0.1416 | 0.1416 (0.141560) | PASS |
| Persistence F1 macro | 0.3039 | 0.3039 (0.303896) | PASS |
| Persistence F1 Active | 0 | 0.000000 | PASS |
| Persistence MCC | -0.0566 | -0.0566 (-0.056604) | PASS |
| Persistence accuracy | 0.5625 | 0.5625 | PASS |
| Majority F1 macro | 0.2811 | 0.2811 (0.281124) | PASS |
| Majority accuracy | 0.7292 | 0.7292 (0.729167) | PASS |
| Persistence F1 CI 95% | [0.2612, 0.3501] | [0.261179, 0.350085] | PASS |
| RF F1 CI 95% (this matrix) | [0.3784, 0.6130] | [0.378446, 0.613000] | PASS |

CIs were computed both with `labels=[0,1,2]` fixed and without; identical to 6 decimals (no resample lacks a class). Majority class from the snapshot train distribution {0: 158, 1: 570, 2: 38} is 1. T3: identical to the claims at 4 decimals.

---

## T4. Split reproduction and scratch retraining

Split and labelling only (`_check_f1_empirical.py`, via `common.load_raw`, `build_dataset`, `canonical_split`): 1,009 raw rows -> 958 labelled (2023-06-10 .. 2026-03-30) -> 766 / 192; distributions 202/710/46, train 158/570/38, test 44/140/8; train end 2025-09-18, test start 2025-09-19, test end 2026-03-30; 1,710 rows after SMOTE (570 per class) for both the tree and the SVM branch; test 192 rows. All identical to the claims.

Retraining: `save_test_predictions.main(<scratchpad>/t4_out)` ran in 4 seconds (under the 2-minute limit) and printed ALL PASS. `persistence_baseline.py --out <scratchpad>/t4_pers` printed PASS canonical sizes. sha256[:16] scratch vs committed:

| File | committed | scratch |
|---|---|---|
| test_predictions_rf.csv | aea39c6dd2989e24 | aea39c6dd2989e24 |
| test_predictions_xgb.csv | e0afa80d26440534 | e0afa80d26440534 |
| test_predictions_svm.csv | c48625b27debfcd2 | c48625b27debfcd2 |
| test_predictions_persistence.csv | ab228a11e65bfe17 | ab228a11e65bfe17 |
| test_predictions_majority.csv | eaf03527561e10fa | eaf03527561e10fa |
| models/rf_revision.joblib | b8113662207ab548 | b8113662207ab548 |
| models/imputer_revision.joblib | 9b5e16ba6cf9a70a | 9b5e16ba6cf9a70a |
| models/scaler_revision.joblib | 981b7001eaf30521 | 981b7001eaf30521 |

`test_predictions_metrics.csv` and `persistence_baseline.csv` differ only by the `run_date` column (2026-10-25 vs 2026-09-14); the metric values are identical. T4: all figures identical. `git status` after T4 shows only `?? revision/audit/` in addition to the pre-existing entries.

---

## Findings (FAIL or UNCERTAIN only)

### F0. Working tree not clean at audit start (T0: FAIL)
Observed: `git status --short` at HEAD `e46ddaa` lists ` D disertatie_audit.zip` and three untracked paths (`_referinte_upb/Lazar_Adrian_Lucrare_Disertatie_2026.pdf.docx`, `eswa_architecture.py`, `outputs/figures/eswa_architecture/`). Why it matters: the pass criterion of T0 is a clean tree; none of these paths is read by the scripts in this slice, so no figure depends on them. What the authors should decide: whether to commit, restore or ignore these paths before the next audit slice, so that "clean tree" can be asserted.

### F1. Lag-1 features and the persistence baseline use different "previous day" conventions (Q3: FAIL)
Observed: features are `shift(1)` on the 1,009-row raw frame before dropping unlabelled days (common.py:66-69), so for 38 of the 42 gap rows in the 958 frame the features come from the previous calendar day while persistence (`shift(1)` on the 958 labelled series, persistence_baseline.py:69, :87; extended_metrics.py:174) comes from the last labelled day 2 to 10 days earlier; for the other 4 gap rows the features also come from an older row. In the raw frame 12 consecutive-row pairs span more than one day (max 5), so "lag-1" is not always the previous calendar day either. Why it matters: the brief and the manuscript describe both as "day t-1"; in the test window only one row (2025-10-29) is a gap row and there both conventions resolve to 2025-10-27, so the reported test figures are unchanged, but the train rows used to fit the models mix 1-day and multi-day lags, and the persistence baseline's definition differs from the feature lag on 41 train rows. What the authors should decide: whether the text should state the row-based definition (and the 12 / 42 gap counts), and whether the persistence baseline should be described as "last labelled day" rather than "day t-1".

### F2. Calendar attribution of the Apple Watch sleep metrics relative to the label's night (Q2: UNCERTAIN)
Observed: Oura rows are dated by the API `day` field (oura_merge.py:85, :92); Apple Watch rows are dated by the Health Auto Export `date` field (parse_health_auto_export.py:186-189, :215). The code contains nothing that states which night a date refers to in each source. Why it matters: `sleep_score(t)` defines the label; if `aw_sleep_deep_hours[d]` and `aw_sleep_rem_hours[d]` are attributed to the night that starts on the evening of d, then `aw_sleep_deep_hours_lag1(t)` and `aw_sleep_rem_hours_lag1(t)` measure the same night as the label of day t, which would make two of the 19 features same-night with the target despite the row shift. If both sources attribute sleep to the wake-up date, there is no overlap. What the authors should decide: state the date convention of each source (from the export settings or the API documentation) and confirm whether the two sleep-architecture features refer to the night before the label's night.

No other FAIL or UNCERTAIN verdicts were reached.
