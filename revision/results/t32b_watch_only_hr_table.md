# T3.2b - watch-only heart-rate sensitivity of the T3.2 aw_hr_mean row (register only)

Subset: 320 labelled days (R / N / A 66 / 240 / 14) with at least one non-Oura heart-rate record on day t; label of day t from models/dataset_snapshot_20260331.csv; values from revision/audit/p0_parser_vs_watch_only.csv (metric heart_rate). parser_all = mean over all HealthKit records of the day (== feature aw_hr_mean of day t within 0.011 on 319 / 320 days; the exception ['2026-03-29'] is the documented incomplete-export day, csv 85.66 vs 86.48); watch_only = mean over the records whose source does not contain "Oura" (non-Oura records per day: median 1 [1, 2] of 809 [736, 860] in total; 297 days with <= 5). Cliff's delta = first class minus second, 95 % percentile CI from 1000 resamples within each class (default_rng(42), fresh per series and pair). Spearman(parser_all, watch_only) = 0.6768 on the same 320 days. Seed 42.

| series | n | n_rest | median_rest | p25_rest | p75_rest | n_normal | median_normal | p25_normal | p75_normal | n_active | median_active | p25_active | p75_active | cliff_active_rest | cliff_active_rest_ci_low | cliff_active_rest_ci_high | cliff_active_normal | cliff_active_normal_ci_low | cliff_active_normal_ci_high | cliff_normal_rest | cliff_normal_rest_ci_low | cliff_normal_rest_ci_high | spearman_parser_vs_watch | n_days_differing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| parser_all | 320 | 66 | 92.1000 | 89.9725 | 94.1950 | 240 | 85.5350 | 82.4600 | 88.6500 | 14 | 80.3100 | 76.8250 | 83.8100 | -0.8355 | -0.9827 | -0.6016 | -0.4821 | -0.7750 | -0.1512 | -0.6597 | -0.7730 | -0.5429 | 0.6768 | 317 |
| watch_only | 320 | 66 | 94.0000 | 86.6150 | 99.0000 | 240 | 84.0000 | 79.0000 | 90.0000 | 14 | 82.5000 | 80.0825 | 84.3750 | -0.7035 | -0.8854 | -0.4718 | -0.1348 | -0.3727 | 0.1117 | -0.5436 | -0.6655 | -0.4081 | 0.6768 | 317 |

Per-year distribution of the subset (n subset / n labelled days of the year, share; R / N / A on the subset):

- 2023: 72 / 198 (0.364); R / N / A 15 / 51 / 6; span 2023-06-12 .. 2023-12-31
- 2024: 109 / 325 (0.335); R / N / A 26 / 82 / 1; span 2024-01-02 .. 2024-12-30
- 2025: 114 / 346 (0.329); R / N / A 20 / 88 / 6; span 2025-01-04 .. 2025-12-30
- 2026: 25 / 89 (0.281); R / N / A 5 / 19 / 1; span 2026-01-01 .. 2026-03-29

Reference (T3.2, all 958 days, parser_all): Cliff's delta Active-Rest -0.840.
The subset is defined by the presence of a non-Oura heart-rate record, not at random; this is a sensitivity analysis, not a validation.
