| pair | metric | diff_point | ci_low | ci_high | prop_diff_gt_0 | n_skipped | ci_A_low | ci_A_high | ci_B_low | ci_B_high | n_used |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF-XGB | f1_macro | 0.0303 | -0.0796 | 0.1561 | 0.6840 | 0 | 0.3784 | 0.6130 | 0.3406 | 0.5933 | 1000 |
| RF-XGB | balanced_accuracy | 0.0727 | -0.0322 | 0.1998 | 0.8910 | 0 | 0.3851 | 0.6634 | 0.3439 | 0.5763 | 1000 |
| RF-SVM | f1_macro | 0.0716 | -0.0337 | 0.1775 | 0.9020 | 0 | 0.3784 | 0.6130 | 0.3279 | 0.5395 | 1000 |
| RF-SVM | balanced_accuracy | 0.0885 | -0.0232 | 0.2159 | 0.9290 | 0 | 0.3851 | 0.6634 | 0.3288 | 0.5683 | 1000 |
| XGB-SVM | f1_macro | 0.0413 | -0.1021 | 0.1647 | 0.7300 | 0 | 0.3406 | 0.5933 | 0.3279 | 0.5395 | 1000 |
| XGB-SVM | balanced_accuracy | 0.0158 | -0.1195 | 0.1508 | 0.6060 | 0 | 0.3439 | 0.5763 | 0.3288 | 0.5683 | 1000 |
| RF-persistence | f1_macro | 0.2024 | 0.0642 | 0.3173 | 0.9970 | 0 | 0.3784 | 0.6130 | 0.2612 | 0.3501 | 1000 |
| RF-persistence | balanced_accuracy | 0.2188 | 0.0728 | 0.3630 | 0.9970 | 0 | 0.3851 | 0.6634 | 0.2602 | 0.3520 | 1000 |

diff = metric(A) - metric(B) for pair A-B; diff_point on the full test set; [ci_low, ci_high] = percentile 2.5 / 97.5 of the paired bootstrap differences over 1000 resamples (shared index matrix bootstrap_indices_1000x192_seed42.npy, seed 42, same indices applied to y_true, y_pred_A, y_pred_B). prop_diff_gt_0 = proportion of resamples with diff > 0 (descriptive, not a p-value). ci_A / ci_B = marginal CIs of each model on the same resamples (identical to T1.1). n_skipped = resamples with a class absent from y_true. RF-persistence (T1.2b) is a baseline comparison, outside the Holm family of 3 model pairs; no multiplicity adjustment applies to CIs. Script: revision/scripts/statistical_tests.py, run 2026-10-18.
