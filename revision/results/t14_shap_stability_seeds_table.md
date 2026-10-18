# T1.4 SHAP ranking stability, mode `seeds` (100 runs, seed 42)

Reference = canonical RF (rf_revision.joblib), mean |SHAP| on the 192 test days.

| Metric | Value |
|---|---|
| aw_hr_mean rank 1 | 100 / 100 (1.00) |
| aw_hr_mean top-3 | 100 / 100 (1.00) |
| Jaccard top-5 vs reference: mean / median / min / max | 0.5276 / 0.4286 / 0.2500 / 1.0000 |
| Jaccard distribution (value: n runs) | 1.00: 4, 0.67: 38, 0.43: 50, 0.25: 8, 0.11: 0, 0.00: 0 |
| Kendall tau vs reference: mean [p2.5, p97.5], min / max | 0.7506 [0.6257, 0.8541], 0.6140 / 1.0000 |
| Kendall tau between runs (4950 pairs): mean [p2.5, p97.5], min / max | 0.7660 [0.6257, 0.8830], 0.4620 / 0.9532 |
| F1 macro on test, retrained models (informative, NON-canonical): mean [p2.5, p97.5], min / max | 0.5142 [0.4682, 0.5677], 0.4465 / 0.5893 (canonical 0.5063) |
| F1 Active on test, retrained models (informative, NON-canonical): mean [p2.5, p97.5], min / max | 0.5157 [0.4000, 0.6423], 0.3529 / 0.6667 |

## Per feature (ordered by reference rank)

| Ref. rank | Feature | Ref. mean abs SHAP | Median rank | IQR (q25-q75) | Min-max rank | Top-5 freq. | Rank-1 freq. |
|---|---|---|---|---|---|---|---|
| 1 | aw_hr_mean | 0.0611 | 1 | 0 (1-1) | 1-1 | 1.00 | 1.00 |
| 2 | oura_hrv_balance | 0.0273 | 2 | 0 (2-2) | 2-2 | 1.00 | 0.00 |
| 3 | aw_active_calories | 0.0205 | 7 | 3 (5-8) | 3-12 | 0.29 | 0.00 |
| 4 | oura_readiness_score | 0.0192 | 7 | 4 (5-9) | 3-15 | 0.27 | 0.00 |
| 5 | oura_activity_score | 0.0191 | 4 | 2 (3-5) | 3-9 | 0.82 | 0.00 |
| 6 | oura_rhr_lowest | 0.0186 | 6 | 4 (4-8) | 3-12 | 0.45 | 0.00 |
| 7 | aw_sleep_deep_hours | 0.0180 | 4 | 2 (3-5) | 3-8 | 0.86 | 0.00 |
| 8 | oura_temp_deviation | 0.0173 | 8 | 3 (7-10) | 4-12 | 0.09 | 0.00 |
| 9 | oura_hrv_mean | 0.0161 | 13 | 2 (12-14) | 6-16 | 0.00 | 0.00 |
| 10 | oura_respiratory_rate | 0.0157 | 10.5 | 3 (9-12) | 4-16 | 0.03 | 0.00 |
| 11 | oura_sleep_score | 0.0156 | 9 | 3 (7-10) | 5-15 | 0.09 | 0.00 |
| 12 | aw_steps | 0.0132 | 9 | 4 (8-12) | 3-15 | 0.08 | 0.00 |
| 13 | oura_steps | 0.0129 | 13 | 2 (12-14) | 7-16 | 0.00 | 0.00 |
| 14 | oura_recovery_index | 0.0126 | 12 | 3 (10-13) | 5-15 | 0.02 | 0.00 |
| 15 | oura_rhr_avg | 0.0119 | 16 | 1 (15-16) | 9-18 | 0.00 | 0.00 |
| 16 | aw_hrv_mean | 0.0103 | 15 | 1 (15-16) | 11-17 | 0.00 | 0.00 |
| 17 | aw_sleep_rem_hours | 0.0094 | 17 | 0 (17-17) | 15-18 | 0.00 | 0.00 |
| 18 | aw_resp_rate | 0.0085 | 18 | 0 (18-18) | 16-19 | 0.00 | 0.00 |
| 19 | aw_spo2 | 0.0060 | 19 | 0 (19-19) | 18-19 | 0.00 | 0.00 |
