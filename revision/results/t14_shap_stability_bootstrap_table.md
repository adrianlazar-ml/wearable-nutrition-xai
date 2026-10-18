# T1.4 SHAP ranking stability, mode `bootstrap` (100 runs, seed 42)

Reference = canonical RF (rf_revision.joblib), mean |SHAP| on the 192 test days.

| Metric | Value |
|---|---|
| aw_hr_mean rank 1 | 88 / 100 (0.88) |
| aw_hr_mean top-3 | 99 / 100 (0.99) |
| Jaccard top-5 vs reference: mean / median / min / max | 0.4904 / 0.4286 / 0.2500 / 1.0000 |
| Jaccard distribution (value: n runs) | 1.00: 3, 0.67: 27, 0.43: 59, 0.25: 11, 0.11: 0, 0.00: 0 |
| Kendall tau vs reference: mean [p2.5, p97.5], min / max | 0.3909 [0.1813, 0.5912], 0.1228 / 0.6140 |
| Kendall tau between runs (4950 pairs): mean [p2.5, p97.5], min / max | 0.2986 [0.0058, 0.5673], -0.1813 / 0.7544 |
| F1 macro on test, retrained models (informative, NON-canonical): mean [p2.5, p97.5], min / max | 0.4103 [0.3097, 0.5293], 0.2999 / 0.5483 (canonical 0.5063) |
| F1 Active on test, retrained models (informative, NON-canonical): mean [p2.5, p97.5], min / max | 0.2606 [0.0000, 0.5591], 0.0000 / 0.6154 |

## Per feature (ordered by reference rank)

| Ref. rank | Feature | Ref. mean abs SHAP | Median rank | IQR (q25-q75) | Min-max rank | Top-5 freq. | Rank-1 freq. |
|---|---|---|---|---|---|---|---|
| 1 | aw_hr_mean | 0.0611 | 1 | 0 (1-1) | 1-4 | 1.00 | 0.88 |
| 2 | oura_hrv_balance | 0.0273 | 2 | 2 (2-4) | 1-14 | 0.89 | 0.08 |
| 3 | aw_active_calories | 0.0205 | 8 | 7 (5-12) | 2-18 | 0.33 | 0.00 |
| 4 | oura_readiness_score | 0.0192 | 5 | 6 (3-9) | 1-19 | 0.54 | 0.02 |
| 5 | oura_activity_score | 0.0191 | 6 | 4 (4-8) | 2-14 | 0.46 | 0.00 |
| 6 | oura_rhr_lowest | 0.0186 | 10 | 9 (6-15) | 3-19 | 0.24 | 0.00 |
| 7 | aw_sleep_deep_hours | 0.0180 | 9 | 6 (6-12) | 2-19 | 0.19 | 0.00 |
| 8 | oura_temp_deviation | 0.0173 | 10 | 11 (5-16) | 1-19 | 0.26 | 0.01 |
| 9 | oura_hrv_mean | 0.0161 | 14 | 7 (10-17) | 3-19 | 0.06 | 0.00 |
| 10 | oura_respiratory_rate | 0.0157 | 12 | 8 (8-16) | 1-19 | 0.11 | 0.01 |
| 11 | oura_sleep_score | 0.0156 | 11 | 7 (8-15) | 3-19 | 0.09 | 0.00 |
| 12 | aw_steps | 0.0132 | 11 | 6 (8-14) | 2-19 | 0.12 | 0.00 |
| 13 | oura_steps | 0.0129 | 10 | 5 (7-12) | 3-17 | 0.11 | 0.00 |
| 14 | oura_recovery_index | 0.0126 | 15 | 7 (11-18) | 2-19 | 0.04 | 0.00 |
| 15 | oura_rhr_avg | 0.0119 | 8.5 | 7 (5-12) | 2-19 | 0.31 | 0.00 |
| 16 | aw_hrv_mean | 0.0103 | 14 | 7.25 (9.75-17) | 3-19 | 0.08 | 0.00 |
| 17 | aw_sleep_rem_hours | 0.0094 | 14 | 6 (11-17) | 5-19 | 0.01 | 0.00 |
| 18 | aw_resp_rate | 0.0085 | 15 | 8 (10-18) | 2-19 | 0.12 | 0.00 |
| 19 | aw_spo2 | 0.0060 | 15 | 6 (11-17) | 4-19 | 0.04 | 0.00 |
