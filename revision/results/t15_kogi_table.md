| config | group | n_features | features_kept | f1_macro | accuracy | mcc | balanced_accuracy | f1_rest | f1_normal | f1_active | cm_rr | cm_rn | cm_ra | cm_nr | cm_nn | cm_na | cm_ar | cm_an | cm_aa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kogi | G1_composite_scores | 2 | oura_readiness_score;oura_sleep_score | 0.2900 | 0.4479 | -0.0267 | 0.2960 | 0.2083 | 0.6173 | 0.0444 | 10 | 25 | 9 | 38 | 75 | 27 | 4 | 3 | 1 |
| kogi | G2_heart_rate | 4 | oura_recovery_index;oura_rhr_lowest;oura_rhr_avg;aw_hr_mean | 0.3589 | 0.5573 | 0.0430 | 0.3668 | 0.2772 | 0.6996 | 0.1000 | 14 | 27 | 3 | 40 | 92 | 8 | 3 | 4 | 1 |
| kogi | G3_hrv | 3 | oura_hrv_mean;oura_hrv_balance;aw_hrv_mean | 0.3541 | 0.5729 | 0.0648 | 0.3583 | 0.2821 | 0.7232 | 0.0571 | 11 | 28 | 5 | 21 | 98 | 21 | 2 | 5 | 1 |
| kogi | G4_sleep_architecture | 2 | aw_sleep_deep_hours;aw_sleep_rem_hours | 0.2751 | 0.3646 | -0.0294 | 0.3417 | 0.2268 | 0.5091 | 0.0896 | 11 | 21 | 12 | 40 | 56 | 44 | 2 | 3 | 3 |
| kogi | G5_activity_energy | 4 | oura_activity_score;oura_steps;aw_steps;aw_active_calories | 0.3238 | 0.4896 | 0.0424 | 0.4462 | 0.1408 | 0.6667 | 0.1639 | 5 | 25 | 14 | 22 | 84 | 34 | 0 | 3 | 5 |
| kogi | G6_resp_thermal_spo2 | 4 | oura_respiratory_rate;oura_temp_deviation;aw_spo2;aw_resp_rate | 0.3147 | 0.5469 | 0.0029 | 0.3123 | 0.2449 | 0.6992 | 0.0000 | 12 | 26 | 6 | 41 | 93 | 6 | 1 | 7 | 0 |

KOGI = RF trained on one group only (same procedure as LOGO, descriptive, no CI). Reference: full RF macro F1 0.5063, majority baseline 0.2811. Informative, NON-canonical. Script: revision/scripts/group_ablation.py, run 2026-10-19.
