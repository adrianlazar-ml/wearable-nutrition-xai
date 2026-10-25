# Table rounding check (round(csv, d) == printed cell)

audit-apply, AUDIT_DECISIONS #17 (process). Script: revision/scripts/_check_table_rounding.py. Every numeric cell of the
tables below is compared with the full-precision value of its source csv in revision/results/, rounded once to the printed
number of decimals. Not checked: Tables 1, 2, 3, 5, 7, 12, 13, 14 (submitted manuscript, not derived from revision/results/); text columns of Table A.1 and the SHAP column of Table 11.

| Document | Table | Source csv | Cells checked | Mismatches |
|---|---|---|---|---|
| manuscript | Table 4 | t11_extended_metrics.csv, t21_extra_models.csv (per-class F1) | 56 | 0 |
| manuscript | Table 6 | t21_extra_models.csv, t21_comparisons.csv, t12_paired_bootstrap.csv | 34 | 0 |
| manuscript | Table 8 | t15_logo.csv | 27 | 0 |
| manuscript | Table 9 | t31_threshold_grid.csv (rows T2-T6; T1 and T7 are base rows) | 20 | 0 |
| manuscript | Table 10 | t32_descriptives.csv, t32_tests.csv | 30 | 0 |
| manuscript | Table 11 | t33_expanding_window.csv | 27 | 0 |
| manuscript | Table A.1 | t22_test_results.csv, t22_comparisons.csv, t22_seeds.csv, t22_cv_results.csv (reference rank) | 42 | 0 |
| supplementary | Table S1. (a) | t12_mcnemar.csv | 23 | 0 |
| supplementary | Table S1. (b) | t12_paired_bootstrap.csv | 24 | 0 |
| supplementary | Table S2. (a) | t13_calibration.csv | 24 | 0 |
| supplementary | Table S2. (b) | t13_calibration.csv | 36 | 0 |
| supplementary | Table S2. (c) | t13_safety_layer_variants.csv | 24 | 0 |
| supplementary | Table S3 | t32_descriptives.csv, t32_tests.csv | 35 | 0 |

Total: 402 cells checked, 0 mismatches.
