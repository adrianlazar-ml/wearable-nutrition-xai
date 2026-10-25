# AUDIT F4: provenance of the Watch metrics (P0), SHAP stability (T1.4), feature-group ablation (T1.5), label agreement (T3.2)

Auditor role: external, read-and-report only. Branch `audit-r1`, HEAD `aaf334b` (slices F1, F2 and F3 committed). Run date: 2026-10-25.
Environment used for every command below: Python 3.11.9, scikit-learn 1.8.0, numpy 2.4.3, pandas 2.3.3, scipy 1.17.1, shap 0.51.0, imbalanced-learn 0.14.1 (identical to `t14_run_meta_*.json`, `t15_run_meta.json`, `t32_run_meta.json`).

Slices F1 to F3 were read only for their Findings sections; nothing from them is reused as a fact. F1's finding F1 (row-based lag) applies to the lag-1 features used by T1.4 and T1.5 unchanged and is not re-examined. F1's finding F2 (calendar attribution of the Watch sleep metrics) receives new evidence in P0 (section "Sleep date convention") and is not closed here.

Files written by this audit (nothing else was modified; no file under `revision/results/`, `revision/scripts/`, `data/`, `apple_watch_api/`, `models/` or the repo root was touched; the folder `data/apple_health/New Automation/` was read only and is not staged, see T0):

- `revision/audit/AUDIT_F4_provenance_shap_ablation_agreement.md` (this report)
- `revision/audit/_check_f4_empirical.py` + `_check_f4_empirical_output.txt` (P0 pass over the raw exports and T2 evidence; two sub-commands `p0` and `t2`, outputs concatenated; imports the revision scripts in memory and calls none of their `main()`; P0.4 calls `parse_health_auto_export.load_json_file()` and `aggregate_daily()` on three files, never `main()`)
- `revision/audit/p0_source_mix.csv` (8,557 rows: date x metric, n_total, n_oura, n_watch_only, plus n_oura_pure and n_iphone; written by the P0 pass)
- `revision/audit/p0_parser_vs_watch_only.csv` (9,634 rows: date x metric, parser aggregate over all records, aggregate over the records whose source does not contain "Oura", record counts; written by the P0 pass; added to the commit beyond the brief's list because P0.4 is reported on all days, not only three)
- `revision/audit/recompute_f4.py` + `recompute_f4_output.txt` (T3; imports numpy, pandas, scipy, scikit-learn, joblib and shap only; nothing from `revision/scripts/` or the repo root; the daily frame is rebuilt from the two csv files, the labels are taken from the snapshot)

All files LF, UTF-8 (`file`: ASCII / UTF-8 text, 0 CR bytes). Execution time: P0 pass 8 s for 1,079 files (the declared 10-minute exception was not needed), T2 evidence 9 s (two bootstrap retrains, two seed retrains, four `fit_eval` fits), recompute 8 s.

---

## T0. Inventory

`git status --short` at the start: unchanged from slice F3 (one deleted tracked zip, three untracked paths outside `revision/`, the author's untracked `revision/audit/AUDIT_DECISIONS.md`), plus the untracked folder `data/apple_health/New Automation/` copied by the author today. The folder is NOT covered by `.gitignore` (`git check-ignore` returns 1; the ignore file lists `.env`, caches, venvs, `backups/`, logs, OS files only). It is not staged by this audit and the commit command at the end names files explicitly. The author should decide whether to add `data/apple_health/` to `.gitignore` (787 MB).

The copied folder is identical to the parser's source folder `C:\Users\Adrian\iCloudDrive\iCloud~com~ifunography~HealthExport\New Automation` (`parse_health_auto_export.py:20-23`): same 1,079 file names, same sizes on all 1,079, sha256 identical on the 4 sampled files (2023-06-09, 2024-09-15, 2026-03-15, 2026-03-30). It is nevertheless treated as the independent source, as instructed.

sha256 first 16 hex chars of the inputs read:

| file | sha256[:16] | role |
|---|---|---|
| `parse_health_auto_export.py` (root) | d7bd273d1319690e | the Watch parser |
| `apple_watch_api/apple_health_daily.csv` | 4692a651003b6696 | parser output, 1,083 rows, 2023-04-11..2026-03-29 |
| `data/oura_combined.csv` | 11d6d4de0eac31f9 | Oura daily frame |
| `models/dataset_snapshot_20260331.csv` | 176649b59fef3adb | 958 dates, 19 lag-1 features, labels |
| `revision/scripts/common.py` | fb73b00eb9a650d3 | shared loader / split |
| `revision/scripts/shap_stability.py` | 4ca77cc4e3827e78 | T1.4 |
| `revision/scripts/_check_t14_inputs.py` | 726fdebfa71cb746 | T1.4 reference procedure |
| `revision/scripts/group_ablation.py` | 654d864dd88c8a1f | T1.5 |
| `revision/scripts/label_agreement.py` | e486f710d5de0dba | T3.2 |
| `revision/results/models/rf_revision.joblib` | b8113662207ab548 | canonical RF |
| `revision/results/models/imputer_revision.joblib` | 9b5e16ba6cf9a70a | canonical imputer |
| `revision/results/t14_shap_reference.csv` | bce417dde117fc7f | claims |
| `revision/results/t14_shap_values_bootstrap.csv` | 17caacd3a4099c8c | 100 x 19 bootstrap values |
| `revision/results/t14_shap_values_seeds.csv` | 0ca565e3b250533a | 100 x 19 seed values |
| `revision/results/bootstrap_train_indices_100x766_seed42.npy` | f4052b964e758aca | matches `CIFRE_REVIZIE.md:204` |
| `revision/results/t15_logo.csv` | 4709ccc43e0a853d | matches `CIFRE_REVIZIE.md:223` |
| `revision/results/t15_predictions_logo_G5_activity_energy.csv` | 3949e289f864e939 | matches `CIFRE_REVIZIE.md:214` |
| `revision/results/test_predictions_rf.csv` | aea39c6dd2989e24 | full-RF test predictions |
| `revision/results/bootstrap_indices_1000x192_seed42.npy` | fff3b2834c061962 | shared test matrix |
| `revision/results/t32_tests.csv` | 41f0d1c55125a5f5 | matches `CIFRE_REVIZIE.md:252` |
| `revision/results/t32_inventory.csv` | 55ccdcf8ffd11128 | matches `CIFRE_REVIZIE.md:249` |
| `revision/results/t32_run_meta.json` | 3678a9ccee6a2d19 | matches `CIFRE_REVIZIE.md:254` |
| `revision/audit/p0_source_mix.csv` (output) | 85aeb3598be365d0 | |
| `revision/audit/p0_parser_vs_watch_only.csv` (output) | 26476d299951007b | |

Raw export folder (P0.1): 1,079 files, all `HealthAutoExport-YYYY-MM-DD.json` (no CSV), file-name dates 2023-04-11..2026-03-30, 787 MB. Study window 2023-06-09..2026-03-30 = 1,026 days: 1,020 files present, 6 missing (2025-05-29, 2025-07-18, 2025-07-19, 2025-07-22, 2025-07-26, 2025-09-03); 59 files precede the window. Of the 958 labelled dates (2023-06-10..2026-03-30), 952 have a file; the 6 without are exactly the 6 missing dates. `apple_health_daily.csv` has a row for 5 of them (not 2025-07-18) with NaN in all 8 columns used by the pipeline (only `aw_physical_effort` / `aw_wrist_temp` are set), and no row for 2026-03-30. So the coverage is complete for every labelled day that carries any Watch value in the pipeline, and the "partial coverage" branch (P0.3, rolling Spearman) was not needed; the 7 labelled days without Watch data (6 missing + 2026-03-30) are the 7 NaN of `aw_hr_mean` in `t32_inventory.csv`.

---

## P0. Provenance of the Watch metrics

### P0.1 What the parser does with `source` (reading `parse_health_auto_export.py`)

| Question | Answer | File:line |
|---|---|---|
| Does it filter on `source`? | No. `load_json_file()` reads only `date`, `qty`, `Avg` / `Min` / `Max` (heart_rate) and `totalSleep` / `deep` / `rem` / `core` / `awake` (sleep_analysis). The string "source" does not occur in the file. Every record of a known metric is kept regardless of device. | `parse_health_auto_export.py:185-228` (record loop), `:191-201` (heart_rate), `:203-219` (sleep_analysis), `:221-228` (qty metrics) |
| Days with several sleep_analysis records (different sources)? | They are summed: `aggregate_daily()` groups the raw rows by `date` and applies `sum` to `sleep_deep` / `sleep_rem` / `sleep_total` (METRIC_CONFIG). A day with a Watch entry and an Oura entry gets deep = Watch deep + Oura deep. No dedup, no preference by source, no multiplicity check. | `:89-103` (config "sum"), `:237-256` (per-day loop), `:249-250` (sum branch) |
| How is `aw_hr_mean` computed? | Unweighted mean of the `Avg` field over all heart_rate records of the day; each record is a one-minute bucket whose `source` lists the devices that contributed samples to that minute ("Apple Watch - Adrian", "Oura", "Apple Watch - Adrian\|Oura"). No weighting by sample count, no exclusion by source. Rounded to 2 decimals. | `:31-33` (config "mean"), `:192-200` (Avg extraction), `:247-248` (mean branch) |
| `aw_steps`, `aw_active_energy_kj` (-> `aw_active_calories` = kJ / 4.184, `common.py:53`) | Sum of `qty` over all per-minute records of the day, all sources pooled. | `:66-71`, `:249-250` |
| `aw_hrv_mean`, `aw_resp_rate`, `aw_spo2` | Mean of `qty` over all records of the day, all sources pooled. | `:40-42`, `:56-61`, `:247-248` |
| Duplicate records across files? | The script concatenates all files and groups by the record date (`:283-302`). Each file holds only its own day for the 8 metrics audited, except 47 sleep_analysis records (naps, `date` = next day 00:00) that sit in the previous day's file; they are attributed to their own `date`, so 47 days receive a second sleep entry which is summed (E: output `[P0.2] entries whose date differs`). Not an Oura issue; it is a nap-summing issue (44 labelled days). | `:234-238` |

The re-implemented aggregation (same rules, no `source` filter) reproduces `apple_health_daily.csv` on every common date for all 9 columns compared, with one exception: 2026-03-29 (csv built before that day's export was complete; `aw_hr_mean` 85.66 vs 86.48, `aw_steps` 1,814 vs 2,425, kJ 1,415.6 vs 1,466.5). So the csv is the output of exactly this parser on exactly these files (E: `[P0.3]` block).

### P0.2 Source mix per metric (from the 1,079 raw files; `p0_source_mix.csv`)

Distinct `source` strings per metric (record counts over all files): heart_rate "Apple Watch - Adrian|Oura" 755,694, "Apple Watch - Adrian" 29,898, "Oura" 13,014; active_energy "Watch|Oura" 979,782, "Watch" 120,866, "Watch|Zepp Life|Oura" 3,018, "Oura" 629; step_count "Watch|iPhone|Oura" 320,832, "Watch|Oura" 34,506, "Watch|iPhone" 27,438, "Watch" 4,131, "Oura" 1,507; sleep_analysis "Watch" 887, "Oura" 218, "iPhone" 19; respiratory_rate "Watch" 46,578, "Oura" 840, "Oura | Watch" 88; heart_rate_variability "Watch" 6,228 only; resting_heart_rate "Watch" 1,064 only; blood_oxygen_saturation "Watch" 15,346 (+ 6 records from a localized app source).

Per metric, on the 952 labelled dates with a file (denominator = days having at least one record of the metric):

| metric (feature) | first date with "Oura" in source | labelled days with >= 1 Oura record: 2023 / 2024 / 2025 / 2026 / all | share of records with Oura, all | labelled days with NO record free of Oura |
|---|---|---|---|---|
| heart_rate (`aw_hr_mean`) | 2023-06-09 (first study day) | 198/198 / 322/325 / 340/340 / 89/89 / **949/952 (0.997)** | 0.997 | 632 / 952 |
| active_energy (`aw_active_calories`) | 2023-06-09 | 198/198 / 323/325 / 339/340 / 86/89 / **946/952 (0.994)** | 0.958 | 214 / 952 |
| step_count (`aw_steps`) | 2023-07-19 | 159/198 / 325/325 / 340/340 / 86/89 / **910/952 (0.956)** | 0.961 | 415 / 952 |
| sleep_analysis (`aw_sleep_deep_hours`, `aw_sleep_rem_hours`) | 2023-07-05 | 18/198 / 69/325 / 93/340 / 32/89 / **212/952 (0.223)** | 0.218 | 204 / 952 (Oura-only days) |
| respiratory_rate (`aw_resp_rate`) | 2023-06-10 | 196/198 / 298/325 / 337/340 / 86/89 / **917/952 (0.963)** | 0.022 (one Oura record per day) | 12 / 952 |
| heart_rate_variability (`aw_hrv_mean`) | never | 0 / 946 | 0 | 0 |
| resting_heart_rate (not a feature) | never | 0 / 945 | 0 | 0 |
| blood_oxygen_saturation (`aw_spo2`) | never | 0 / 944 | 0 | 0 |

Over the whole folder (including 2023-04-11..2023-06-08 and unlabelled days) the per-year fractions are 0.79 / 0.95 / 1.00 / 1.00 for heart_rate and 0.64 / 0.98 / 1.00 / 0.97 for step_count (E: `[P0.2] fraction of days`). The mixing is therefore present from the first day of the dataset, not a regime that starts later.

Sleep date convention (evidence for F1's finding F2): the sleep_analysis `date` equals the `sleepEnd` date on 1,078 of 1,124 entries (Watch 865/906, Oura 213/218) and the sleep starts on the previous evening in 462 entries; the exceptions are naps. So the Health Auto Export sleep entry of day d is the night that ENDS on d (wake-up date). Sleep multiplicity on labelled dates: 44 days with 2 entries, 8 of them Watch + Oura (summed by the parser), 204 days with Oura as the only sleep source.

### P0.4 Effect of the Oura records on the aggregated values

Three days through the repository parser functions (`load_json_file` + `aggregate_daily`), all records versus the records whose `source` does not contain "Oura" (E: `[P0.4]` block; the parser values equal the csv on all three days):

| day | `aw_hr_mean` all / excl. Oura | `aw_steps` all / excl. Oura | `aw_active_energy_kj` all / excl. Oura | `aw_sleep_deep_hours` | `aw_resp_rate` |
|---|---|---|---|---|---|
| 2023-06-09 | 92.92 / 80.85 (diff 12.07, 13.0%) | 5,250 / 5,250 (0) | 3,906.35 / 782.69 (80.0%) | 0.60 / 0.60 | 16.18 / 16.18 |
| 2024-09-15 | 92.77 / undefined (every minute bucket carries Oura) | 1,794.65 / 4.8 (99.7%) | 2,035.76 / undefined | 0.87 / 0.87 | 14.71 / 14.72 |
| 2026-03-15 | 89.56 / undefined | 4,664.44 / 9.0 (99.8%) | 2,727.63 / 185.49 (93.2%) | 0.84 / 0.84 | 16.37 / 16.39 |

On all 952 labelled dates (`p0_parser_vs_watch_only.csv`, E: `[P0.4-ext]`): heart_rate, excluding Oura leaves no value on 632 days and changes it on 317 of the remaining 320 (median +1.23 bpm, 5th-95th percentile -10.6 to +10.0, max 27.4; Spearman between the two variants 0.68); active_energy, no value on 214 days, median relative change 99.1% (Spearman 0.03); step_count, no value on 415 days, median relative change 99.6% (Spearman 0.01); respiratory_rate, change on 765 days of median -0.01 breaths/min (max 2.61; median relative 0.1%); sleep deep / REM, change on 5 days (max 2.81 h deep, 1.97 h REM); HRV, SpO2, resting HR unchanged.

Reading of these numbers: a record tagged "Apple Watch - Adrian|Oura" is one minute bucket in which Apple Health merged samples from both devices; the export does not say which samples came from which device. Excluding those buckets removes the Watch samples that fell in them too, so "excl. Oura" is not "Watch only", it is a lower bound of what the Watch alone would give. The Watch's own contribution to `aw_hr_mean`, `aw_steps` and `aw_active_calories` cannot be reconstructed from these exports.

P0 pass criteria: table 1 complete (PASS); item 2 computed on the full window (PASS); item 3 not required; item 4 both values on three days (PASS). The content of the result is Findings F4.1 and F4.2.

---

## T1. Trace tables (from reading the code)

Line numbers refer to the files at HEAD `aaf334b`.

### (a) T1.4: reference, bootstrap, seeds

| # | Operation | File:line | Applied on | Output |
|---|---|---|---|---|
| 1 | `load_raw()` merges `oura_combined.csv` with 8 `aw_*` columns of `apple_health_daily.csv` on date (`aw_active_calories` = kJ / 4.184 rounded 1); `build_dataset()` shifts the 19 base columns by one row, assigns labels, drops unlabelled rows | common.py:46-71 | 1,009 raw rows -> 958 x 19 with NaN | X, y, dates |
| 2 | `canonical_split()` -> `preprocess_split()`: median imputer fit on rows 0..765, transform rows 766..957; `SMOTE(42, k=3)` on the imputed train | common.py:90-120 | 766 train (NaN 461), 192 test (NaN 49) | `Split` (X_test imputed, unscaled; X_train_sm 1,710 rows) |
| 3 | `rf_ref = joblib.load(rf_revision.joblib)`; `ref_mean = mean_abs_shap(rf_ref, canon.X_test)` = `TreeExplainer(model).shap_values(X)` -> `aggregate_multiclass` = `np.abs(arr).mean(axis=(0, 2))` on (192, 19, 3) | shap_stability.py:327-328, :123-124, :108-120 | imputed test, 192 x 19 | 19 mean abs SHAP values |
| 4 | `ref_rank = rankdata(-ref_mean, method="min")`; `top_k` = 5 largest by stable argsort | :127-134 | 19 values | ranks 1..19, top-5 set |
| 5 | Identity test: bootstrap mode `bootstrap_split(idx = arange(766))` + RF seed 42; seeds mode `run_one(canon, 42)`; checks ranking equal, max diff < 1e-12, F1 macro == 0.5063 | :333-339 | original train | abort on FAIL |
| 6 | Bootstrap: `idx = default_rng(42).integers(0, 766, (100, 766))`, saved to npy; per run `bootstrap_split()` takes `X.iloc[:766].iloc[idx[r]]` (raw rows, duplicates kept), test = `X.iloc[766:]` unchanged; `preprocess_split()` refits imputer and SMOTE (seed 42) on the bootstrap rows; `fit_rf(split, 42)`; `run_one()` predicts on the run's imputed test, computes F1 and mean abs SHAP | :356-371, :172-176, :154-169 | 100 runs | 100 x 19 values, F1 per run |
| 7 | Seeds: `run_one(canon, r)`, r = 0..99: same `canon` split (SMOTE seed 42 fixed), `rf.set_params(random_state=r)`; check run 42 == reference (ranking and 1e-12) | :374-385, :154-158 | 100 runs | 100 x 19 values |
| 8 | `summarize()`: ranks per run, Jaccard of top-5 sets vs reference, `kendalltau` vs reference and over the 4,950 run pairs, per-feature rank median / IQR / top-5 frequency, percentiles of the informative F1 | :179-231 | 100 x 19 | table, extra |
| 9 | Sanity checks (permutation of 1..19 per run, bounds), outputs csv / md / json / figure | :392-415 | | `t14_*` files |

### (b) T1.5: `fit_eval` on a subset

| # | Operation | File:line | Applied on | Output |
|---|---|---|---|---|
| 1 | `build_groups("G1")`: `_BASE_GROUPS` + `sleep_score` appended to G1, names sorted in canonical order, `_lag1` suffix; `check_partition()` (no overlap, union == 19); sizes must be [2, 4, 3, 2, 4, 4] (abort) | group_ablation.py:97-135, :289-294 | 19 names | 6 groups |
| 2 | `build_dataset(load_raw())`; `i = 766`; test dates | :296-300 | 958 x 19 | X, y, dates_te |
| 3 | `fit_eval(X, y, features)`: `subset_in_canonical_order`; `X.iloc[:766][feats]` / `X.iloc[766:][feats]`; `SimpleImputer(median)` fit on train; `SMOTE(42, k=3)`; `build_models()["rf"]` (100 / 10 / 5 / seed 42) fit; `predict` / `predict_proba` on the imputed test | :143-165; save_test_predictions.py:46-49 | any subset | y_true, y_pred, proba (192) |
| 4 | T1.5.0: `fit_eval(FEATURES)`; `metrics()`; checks vs 0.5063 / 0.6406 / 0.1416, y_true, y_pred, proba (1e-9) and dates of `test_predictions_rf.csv` (abort) | :304-315 | 19 | m_full, p_full |
| 5 | T1.5.0b: Oura-only (11 names without `aw_`) / Watch-only (8 `aw_`) through `fit_eval`; F1 macro and F1 Active vs 0.428 / 0.286 and 0.324 / 0.100 at 3 dp (abort) | :318-328 | 11 / 8 | device rows |
| 6 | `idx = np.load(bootstrap_indices_1000x192_seed42.npy)` == `default_rng(42).integers(0, 192, (1000, 192))` (abort); T1.1 CI read from `CIFRE_REVIZIE.md` by regex | :331-336, :201-208 | | idx, t11_ci |
| 7 | LOGO per group: `kept = FEATURES minus group`; `fit_eval(kept)`; `metrics()`; `paired_bootstrap(y_true, p_full, p, metric, idx)` for F1 macro and F1 Active: each row of idx applied to y_true, p_full and p alike, rows lacking a class skipped, diff = metric(full) - metric(LOGO), percentile 2.5 / 97.5, proportion of diffs > 0; check `diff_point == full - LOGO` at 1e-12; predictions saved | :343-370; statistical_tests.py:133-170 | 17 / 15 / 16 / 17 / 15 / 15 features | `t15_logo.csv` rows, 6 prediction files |
| 8 | Sanity: marginal CI of the full model on F1 macro == T1.1 at 4 dp (abort); 0 resamples skipped | :372-379 | | |
| 9 | KOGI per group: `fit_eval(group)`; metrics, no CI | :395-403 | 2 / 4 / 3 / 2 / 4 / 4 | `t15_kogi.csv` |
| 10 | Outputs: csv (%.10g), md tables, figure, `t15_run_meta.json` (groups, SLEEP_SCORE_GROUP, sha256 of predictions) | :411-460 | | |

### (c) T3.2: from the daily frame to Cliff's delta

| # | Operation | File:line | Applied on | Output |
|---|---|---|---|---|
| 1 | `df = load_raw()` (1,009 rows, Oura + Watch merged on date, day-t values) | common.py:46-58 | | df |
| 2 | `build_frame(df)`: `X, y, dates = build_dataset(df)` (958); `daily = df.set_index("date")`; `frame[v] = daily.loc[dates, v]` for the 10 variables (day t, no imputation); the 3 lag variables are the canonical lag-1 columns of X | label_agreement.py:203-214 | 958 dates | frame |
| 3 | Identity flags per variable: `lag_identity(df[v], df[v].shift(1))` AND `df[v].shift(1)[sel] == X[v_lag1]` (NaN coded as -1e30) | :215-222 | 1,009 / 958 rows | ident (abort if any False) |
| 4 | Inventory: n per class = non-NaN days per class | :278-287 | | `t32_inventory.csv` |
| 5 | Descriptives per class: n, mean, median, p25, p75, IQR on the non-NaN values | :292-299 | | `t32_descriptives.csv` |
| 6 | Tests per variable: `groups` = per-class non-NaN arrays; `stats.kruskal(*groups)` (tie-corrected H); `eps2 = H / (n - 1)`; `dunn_test()` (mean ranks of the pooled ranking, tie term sum(t^3 - t) / (12 (N - 1)), two-sided normal p); `holm()` over the 3 pairs; `cliffs_delta(groups[a], groups[b])` = P(x > y) - P(x < y) by sorted searchsorted; `cliffs_delta_ci()` with ONE `default_rng(42)` created once and consumed sequentially over variables x pairs, 1,000 resamples, x indices drawn then y indices per resample | :306-335, :118-182 | per variable | `t32_tests.csv` |
| 7 | `concordance()` from the sign of Cliff Active-Rest vs the `expected` constant | :185-191, :334 | | agreement column |
| 8 | Sanity (p in [0, 1], eps2 in [0, 1], CI contains point, n == inventory, Holm >= raw), outputs, figures, `t32_run_meta.json` with the expected directions and the caveat | :341-415 | | |

---

## T2. Targeted questions

E = `_check_f4_empirical_output.txt` (block in brackets), R = `recompute_f4_output.txt`.

**SHAP**

1. PASS. `ref_mean` is computed on `canon.X_test` (shap_stability.py:328), the imputed, unscaled test matrix (common.py:109), equal to `imputer_revision.joblib.transform(raw X_test)` at 0.0 (E [Q1]); the raw test has 49 NaN. Only this matrix reproduces `t14_shap_reference.csv` (max diff 4.95e-12, the csv is written with `%.10g`); the scaled test gives a different top-5 (max diff 3.6e-2) and the train gives another. `rf_revision.joblib` is loaded (shap_stability.py:327) and never refit (`random_state` 42, 100 trees).
2. PASS. shap 0.51.0 returns an ndarray of shape (192, 19, 3) (E [Q1], [Q2]); the aggregation is `mean(axis=(0, 2))`: mean over samples and the 3 classes. The sum over classes would give 0.1833 for aw_hr_mean (not 0.0611) and the Active-only mean 0.0816 with a different rank 5 (E [Q2]). `aggregate_multiclass` handles the list form (np.asarray -> (3, 192, 19), axes (0, 1)) with max difference 6e-17 against the array form; the dispatch on `shape[0] == n_samples` cannot misfire because 192 != 3. The verbatim canonical branch in `_check_t14_inputs.py:77-91` does the same.
3. PASS. max |sum_f SHAP + expected_value - predict_proba| = 8.3e-16 on the 192 x 3 matrix (E [Q3], R section 1), below 1e-12.
4. PASS. `bootstrap_train_indices_100x766_seed42.npy` equals `default_rng(42).integers(0, 766, size=(100, 766))` element-wise, int64, 485.2 unique rows per run on average (E [Q4]). `bootstrap_split()` takes `X.iloc[:766].iloc[idx]` from the raw X (461 NaN in the train rows, so before imputation), with replacement, 766 of 766; `preprocess_split()` refits the imputer and SMOTE (seed 42) on the sample; the test is `X.iloc[766:]`, the 192 original days (y and dates verified equal). Runs 0 and 1 re-executed reproduce the stored values at 4.9e-12 and the stored ranks exactly. Note: the test matrix is re-imputed with the run's imputer (medians differ from the canonical ones by up to 56.6 on one column), which is what the docstring says ("imputer fit on the run's train part").
5. PASS. `fit_rf()` sets only `random_state` on the canonical RF and fits `split.X_train_sm`, the same 1,710 SMOTE rows for every seed (E [Q5]); run 42 equals the reference in the csv at 0.0 and re-executed at 0.0 (F1 0.5063); run 7 re-executed matches the csv at 4.6e-12.
6. PASS. `rank_desc` = `rankdata(-values, method="min")` (shap_stability.py:129); `jaccard` on sets (:137-141); `kendall` = `scipy.stats.kendalltau` on the 19 ranks (tau-b by default, no ties in any run so equal to tau-a). Recomputed from `t14_shap_values_bootstrap.csv` / `_seeds.csv`: mean Jaccard top-5 0.4904 / 0.5276, mean tau 0.3909 / 0.7506, identical to the claims (E [Q6]).

**Ablation**

7. PASS. `build_groups("G1")` executed: 19 names, 19 unique, set equal to FEATURES, sizes [2, 4, 3, 2, 4, 4]; the six sets equal the listing in the brief and in `CIFRE_REVIZIE.md`; `t15_run_meta.json` stores the same groups (E [Q7]).
8. PASS. `fit_eval` keeps the canonical column order (`subset_in_canonical_order`), refits the median imputer and SMOTE on the subset and uses `build_models()["rf"]` (group_ablation.py:152-165). Executed: full 19 reproduces `test_predictions_rf.csv` (y_pred identical, probabilities 9.8e-17); Oura-only 0.428134 / Active 0.285714; Watch-only 0.324330 / 0.100000; LOGO G5 reproduces `t15_predictions_logo_G5_activity_energy.csv` (y_pred identical, probabilities 9.9e-17, F1 0.377973) (E [Q8]). The per-subset imputer refit is numerically the column-wise median of the full train, so only SMOTE differs in reduced space.
9. PASS. `paired_bootstrap` applies each row of `idx` to `y_true`, `pred_a` and `pred_b` (statistical_tests.py:147-155); the call is `paired_bootstrap(y_true, p_full, p, ...)` (group_ablation.py:350), so diff = F1(full) - F1(LOGO) and a positive delta means removal lowers the score, as the table note says. Executed on the G5 predictions: +0.128351 [0.0164, 0.2388], 1000 used, 0 skipped, equal to `t15_logo.csv` (E [Q9]; R section 2 from an independent implementation).
10. UNCERTAIN. No alternative run exists in git: script, tests and all `t15_*` results were added in one commit (4b38b18, 2026-10-19 01:13:30) and the only later change is the figure-hygiene diff of 8d45dce (markers, titles, y label; no number). `SLEEP_SCORE_GROUP` is "G1" in the script, in the run meta and in the register. But the group definition did change between the plan and the run: the plan of 2026-10-14 (commit 2e2172d, `KBS_Revision_Plan.md:80`) defined "HR (Watch), HRV (Oura), somn (Oura), activitate/pasi (ambele), temperatura (Oura), alte", while the six groups actually used (composite scores, heart rate, HRV, sleep architecture, activity/energy, respiration/thermal/SpO2) first appear in the plan in commit 6c9b81b at 01:13:55, 25 seconds after the results commit. Whether the final grouping was fixed before the first execution cannot be established from the repository.

**Agreement**

11. PASS. `load_raw()` has 1,009 rows; `frame` has 958 rows whose dates equal the snapshot dates; the 10 day-t columns come from `daily.loc[dates, v]` with no imputation (NaN counts 7 / 13 / 7 / 7 / 7 / 15 / 2 / 2 / 2 / 1 equal `t32_inventory.csv` n_missing; n per class equal `t32_tests.csv`). Independent identity `daily[v].shift(1)` at the 958 dates == canonical lag-1 feature: 10 / 10 (E [Q11]). Observation: the first clause of the script's test, `lag_identity(df[v], df[v].shift(1))` (label_agreement.py:216-219), compares a series with its own shift and is true by construction; the effective check is the second clause against `build_dataset` X.
12. PASS, with a limit. The directions are the `expected` field of the `VARIABLES` / `LAG_VARIABLES` constants (label_agreement.py:87-108), copied into `t32_run_meta.json` (equal, E [Q12]) and frozen by `test_fixed_design` (test_label_agreement.py:113-120). Git shows no change after the first run: the script, the test and the results were added together in 8b09bd4 (2026-10-19 06:09) and `git log -S'expected="'` returns that commit only. Limit: pre-run fixation is documented in the repository only for two of the ten (the plan of 2026-10-14 states "Active days have higher HRV and lower resting HR than Rest", `KBS_Revision_Plan.md:95`); for the others the only evidence is that the "+" directions on the activity variables were kept although they came out discordant.
13. PASS. `cliffs_delta` = (count of y strictly below x minus count strictly above) / (n_x n_y) by sorted `searchsorted` = P(x > y) - P(x < y); `cliffs_delta_ci` resamples the days within each class independently, 1,000 times, from the generator passed in; `main()` creates one `default_rng(42)` (label_agreement.py:306) used for every variable and pair in order (E [Q13/Q14]). Recomputed with a pairwise-comparison implementation: aw_hr_mean -0.840374 [-0.912437, -0.749244], aw_active_calories -0.420462 [-0.573991, -0.246183], identical to the stored values at 6 decimals (R section 3). The CI is exact only because the recompute mirrors the draw order (first variable, first pair for aw_hr_mean; the six preceding CIs of aw_hr_mean and aw_hrv_mean are consumed before aw_active_calories); an unrelated generator gives [-0.914, -0.750] for aw_hr_mean.
14. PASS. `stats.kruskal` is tie-corrected; `eps2 = H / (n - 1)`; Dunn uses the tie term sum(t^3 - t) / (12 (N - 1)) in the variance and two-sided normal p; `holm()` over the 3 pairs (label_agreement.py:147-182, :311-315). Recomputed for aw_hr_mean: H 182.9865 (scipy and a hand-written tie-corrected formula agree), eps2 0.192617, Dunn-Holm p 1.959e-26 / 6.448e-8 / 4.246e-31 equal to `t32_tests.csv` (R section 3).
15. FAIL. By `source` (P0.2, E [Q15]): `aw_hrv_mean` and `aw_spo2` are Watch-only (0 Oura records on 946 / 944 labelled days). `aw_hr_mean` is a mixed aggregate on 949 of 952 labelled days (99.7% of its records carry Oura; on 632 days no record is free of Oura); `aw_active_calories` on 946 / 952 (95.8% of records; 214 days with no Oura-free record); `aw_steps` on 910 / 952 (96.1%; 415 days). `aw_resp_rate` contains one Oura record per day on 917 / 952 days (2.2% of records; effect on the daily mean median 0.01, max 2.61 breaths/min). So of the six level-A variables, two are "independent device", three are not, and one is nearly so. See Finding F4.1.
16. PASS. The caveat is in the manuscript, not only in the register: Section 3.10, last sentence of the paragraph ("Consecutive days are autocorrelated, so the p-values are approximate and optimistic; inference rests on the effect sizes", `KBS_Manuscript_R1.docx` paragraph 132) and the Table 10 caption in Section 4.7 ("p-values are approximate because consecutive days are autocorrelated", paragraph 471); also in the Supplementary table note.

---

## T3. Independent recomputation (`revision/audit/recompute_f4.py`)

27 / 27 checks PASS (R). Inputs: snapshot csv (X and labels), `imputer_revision.joblib`, `rf_revision.joblib`, the two prediction csv files, the two raw daily csv files; no revision script imported.

| quantity | recomputed | target | verdict |
|---|---|---|---|
| mean abs SHAP top-5 (order and value) | aw_hr_mean 0.061087, hrv_balance 0.027305, aw_active_calories 0.020501, readiness_score 0.019161, activity_score 0.019134 | 0.0611 / 0.0273 / 0.0205 / 0.0192 / 0.0191 | PASS (4 dp, order) |
| additivity | 8.3e-16 | < 1e-12 | PASS |
| F1 macro full / LOGO G5 | 0.506324 / 0.377973 | 0.5063 / 0.3780 | PASS |
| delta full - G5, CI (regenerated 1000 x 192 matrix, equal to the npy; 0 skipped) | +0.128351 [0.016355, 0.238835] | +0.1284 [0.0164, 0.2388] | PASS |
| aw_hr_mean medians R / N / A | 91.95 / 86.02 / 81.32 | same | PASS |
| aw_hr_mean H, epsilon2 | 182.9865, 0.192617 | 182.99, 0.1926 | PASS |
| aw_hr_mean Cliff A-R [CI] | -0.840374 [-0.912437, -0.749244] | -0.840 [-0.912, -0.749] | PASS (3 dp; exact at 6) |
| aw_active_calories Cliff A-R [CI] | -0.420462 [-0.573991, -0.246183] | -0.420 [-0.574, -0.246] | PASS |

The Cliff CIs are exact because the recompute reproduces the script's single-generator draw order (declared in the script header); this is a property of the specification (seed 42, sequential draws), not of the implementation, which is independent.

---

## Findings (FAIL or UNCERTAIN only)

### F4.1. The three "Apple Watch" activity and heart-rate features are Apple Health aggregates that contain Oura ring samples on 96% to 100% of the labelled days; the parser ignores `source` (P0, Q15: FAIL)
Observed: `parse_health_auto_export.py` never reads the `source` field and pools every record of a day (`:185-228`, `:241-254`). In the raw exports, 99.7% of the heart_rate records, 95.8% of the active_energy records and 96.1% of the step_count records on the 952 labelled days with a file carry "Oura" in `source`, from the first dataset day (2023-06-09 for heart rate and energy, 2023-07-19 for steps); on 632 / 214 / 415 labelled days not a single record of the metric is free of Oura. Removing the Oura-tagged minute buckets changes `aw_hr_mean` by a median of 1.2 bpm where a value remains (up to 27 bpm) and removes 99% of `aw_steps` and `aw_active_calories`; the Watch-only series cannot be reconstructed because a mixed bucket merges both devices' samples. Only `aw_hrv_mean`, `aw_spo2` (and, for the resting heart rate, a metric not used) are pure Watch; `aw_resp_rate` has one Oura record per day with a negligible effect. Where it is described otherwise: Table 1 of the manuscript lists features 12 to 15 with Source "Watch"; the Figure 6 caption says "Feature names prefixed with aw_ originate from the Apple Watch"; Section 3.10 says the Watch indicators "are independent of the ring scores that define the labels and are the only genuine validation"; Section 4.7 and Table 10 call them "watch indicators ... independent of the ring scores"; `label_agreement.py:13` and `t32_descriptives_table.md` call level A "device-independent (Apple Watch)". Why it matters: (a) the headline label-validation result (aw_hr_mean, Cliff -0.84) is computed on a 24-hour heart-rate series that includes the Oura ring's own heart-rate samples, the sensor stream from which `rhr_lowest` and the readiness score are derived, so the "independent device" argument of T3.2 level A holds for aw_hrv_mean and aw_spo2 only (and those two show no Active-vs-Rest effect); (b) the device ablation (Oura-only 0.428 versus Watch-only 0.324 versus both 0.506) and the "two devices are complementary" claim compare Oura API scores with Apple Health aggregates that mix both devices, not two devices; (c) the SHAP rank-1 feature is a mixed-source series. The reported numbers are unchanged; their description is not accurate. What the authors should decide: whether to rename the source of these features (for example "Apple Health, Watch and ring samples merged") in Table 1, Figure 6, Sections 3.1, 3.10, 4.6, 4.7 and the Supplementary, to restrict the level-A validation claim to aw_hrv_mean and aw_spo2 and present aw_hr_mean as "label-rule-independent but not device-independent", and whether a per-source re-export (Health Auto Export can filter by source) is feasible to rebuild the three features from Watch samples only.

### F4.2. The sleep-architecture features come from the Oura ring alone on 21% of the labelled days, and on 8 days the parser adds the Watch and Oura sleep entries together (P0: FAIL)
Observed: sleep_analysis records on labelled dates come from "Oura" alone on 204 of 952 days (2023: 9%, 2024: 21%, 2025: 27%, 2026: 36%) and from "Apple Watch" on the rest; on 8 labelled days both devices wrote an entry and `aggregate_daily()` sums `deep` and `rem` over all entries of the day (`parse_health_auto_export.py:92-97`, `:249-250`), which changes deep sleep on 5 days by up to 2.81 h and REM by up to 1.97 h; 44 labelled days additionally include a nap entry in the sum. `aw_sleep_deep_hours` and `aw_sleep_rem_hours` (features 17 and 18, Source "Watch" in Table 1, group G4 "sleep architecture") are therefore a Watch series with Oura days interleaved, and the Oura sleep staging is the same session data that produces `sleep_score`, one of the two label inputs (the lag relation of F1's F2 finding then applies to those days with a same-source twist). Why it matters: G4 is reported as a group whose removal does not change F1 (delta -0.008), so no conclusion rests on it, but the provenance statement is wrong on a fifth of the days and the summing rule is undocumented. What the authors should decide: whether to state the mixed provenance and the summing rule, or to restrict the two features to Watch entries (which would change the canonical dataset and is out of scope for the revision).

### F4.3. The six-group partition of T1.5 differs from the plan written before the run, and its final form is recorded after the results (Q10: UNCERTAIN)
Observed: `KBS_Revision_Plan.md` at commit 2e2172d (2026-10-14) defines the T1.5 groups as "HR (Watch), HRV (Oura), somn (Oura), activitate/pasi (ambele), temperatura (Oura), alte"; the grouping actually run (composite scores; heart rate = aw_hr_mean + rhr_lowest + rhr_avg + recovery_index; HRV; sleep architecture; activity/energy; respiration/thermal/SpO2) first appears in the plan in commit 6c9b81b at 01:13:55 on 2026-10-19, 25 seconds after the T1.5 results commit (4b38b18, 01:13:30). The script offers one documented alternative (`SLEEP_SCORE_GROUP = "G4"`), no run with it exists in the repository, and nothing else in git points to another configuration. Why it matters: the only CI that excludes zero (G5, +0.128 [0.016, 0.239]) depends on the grouping; with the 14 October grouping the activity group would have been "activity/steps (both devices)" without `aw_active_calories` in a mixed group, and the register does not say that the grouping was changed. What the authors should decide: whether to state in Section 3.9 or the register that the physiological grouping was defined when the script was written (before any result), and whether the G4 alternative should be run and reported as a sensitivity.

### F4.4. "JSON export with step deduplication" (Figure 1 caption, Section 3.3, Section 4.10) is not verifiable from the repository, and the JSON step records are multi-source (UNCERTAIN)
Observed: the step_count records on labelled days are tagged "Apple Watch - Adrian|iPhone - Adrian|Oura" (320,832 records over all files), "Apple Watch - Adrian|Oura" (34,506) or "Apple Watch - Adrian|iPhone - Adrian" (27,438); the JSON-path parser contains no deduplication code (the only `dedup` occurrences are in the XML-versus-JSON comparison scripts and the Streamlit app), so whatever deduplication exists is done inside the export application before the file is written. Section 4.10 compares XML (about 15,634 steps/day) with JSON (about 6,902) and attributes the difference to deduplication in JSON. Why it matters: the manuscript presents the JSON source as "deduplicated"; the audit can confirm that the JSON daily sums are the per-minute bucket sums written by the exporter, but cannot confirm from the repository that a minute bucket tagged with three sources holds a deduplicated count rather than a sum. What the authors should decide: whether the export application's aggregation rule (documented by the vendor) can be cited, or whether the caption should say "JSON export (per-minute buckets aggregated by the export application)".

No other FAIL or UNCERTAIN verdicts were reached.

---

## Observations (no verdict change; reported for completeness)

- P0 / F1-F2: the Health Auto Export sleep entry of date d is the night ending on the morning of d (date == sleepEnd date on 1,078 / 1,124 entries); this is the Apple-side convention that F1's finding F2 asked for. If the Oura `day` field follows the same wake-up convention, there is no same-night overlap between the sleep-architecture lag-1 features and the label of day t.
- Section 3.1 says "a custom parser processed 1,083 daily JSON files"; the folder holds 1,079 files and the csv has 1,083 rows because 5 dates that now have no file still have a row (with NaN in all 8 columns used) and 2026-03-30 has no row. The 5 rows carry only `aw_physical_effort` / `aw_wrist_temp`, so the figures are unaffected; the file count in the text is the row count of the csv at freeze time.
- Q1: `t14_shap_reference.csv` is written with `%.10g`, so the on-disk reference reproduces at 5e-12, not 0; the identity test of the script compares in memory and reaches 0.
- Q4: in bootstrap mode the test matrix is re-imputed with the run's imputer, so the explained test features differ slightly between runs (documented in `SHAP_PROCEDURE["data"]`); the seeds mode uses the canonical test matrix.
- Q11: the first clause of the identity test is tautological (see Q11); the claim "identity PASS 10 / 10" is nevertheless true by the second clause and by the independent check.
- Q12: the expected directions are frozen in code, run meta and unit test; the register row `CIFRE_REVIZIE.md:139` says "directions fixed before" without pointing to a pre-run record other than the plan sentence of 2026-10-14 for two of them.
- P0.2: `blood_oxygen_saturation` has 6 records from a localized third-party source name and `active_energy` / `step_count` have about 4,000 records tagged "Zepp Life"; negligible.
- T0: `revision/audit/AUDIT_DECISIONS.md` is the author's untracked file and was left unstaged, as in F3.

---

## Summary

- T2: Q1 PASS, Q2 PASS, Q3 PASS, Q4 PASS, Q5 PASS, Q6 PASS, Q7 PASS, Q8 PASS, Q9 PASS, Q10 UNCERTAIN, Q11 PASS, Q12 PASS, Q13 PASS, Q14 PASS, Q15 FAIL, Q16 PASS (14 PASS, 1 FAIL, 1 UNCERTAIN).
- T3: PASS, 27 / 27 values identical at the required precision from an independent implementation (SHAP top-5 at 4 decimals and additivity at 8e-16; LOGO G5 F1 and delta CI at 4 decimals on a regenerated matrix; aw_hr_mean medians, H, epsilon-squared, Cliff's delta with CI and aw_active_calories Cliff's delta with CI, exact at 6 decimals).
- P0 in three sentences: the parser never reads the `source` field, pools all records of a day (unweighted mean of per-minute `Avg` for heart rate, sum for steps, energy and sleep stages) and so merges Apple Watch and Oura ring samples. On the labelled days, heart rate is mixed on 99.7% (2023 100%, 2024 99.1%, 2025 100%, 2026 100%), active energy on 99.4% (100 / 99.4 / 99.7 / 96.6), steps on 95.6% (80.3 / 100 / 100 / 96.6), sleep stages on 22.3% (9.1 / 21.2 / 27.4 / 36.0, Oura as the only sleep source on 204 days), respiratory rate on 96.3% with one Oura record a day, while HRV, resting heart rate and SpO2 are Watch only. The first Oura record is on 2023-06-09 for heart rate and energy (the first dataset day), 2023-06-10 for respiratory rate, 2023-07-05 for sleep and 2023-07-19 for steps.
- Findings: F4.1, the "Apple Watch" features aw_hr_mean, aw_active_calories and aw_steps are Apple Health aggregates containing Oura samples on 96% to 100% of days, so the device-independence claim of T3.2 level A (manuscript 3.10, 4.7, Table 1, Figure 6) holds only for aw_hrv_mean and aw_spo2 and the device ablation compares mixed aggregates, not devices; F4.2, the sleep-architecture features come from Oura alone on 21% of days and are summed across devices on 8 days; F4.3, the T1.5 grouping differs from the 14 October plan and its final form is recorded 25 seconds after the results, so pre-run fixation cannot be shown; F4.4, the "step deduplication" of the JSON source is not verifiable from the repository.
