| model | pair | n00 | n01 | n10 | n11 | chi2_cc | p_cc | p_exact | p_cc_holm | p_exact_holm | recommended | discordant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rf | RF tuned - RF canonical | 54 | 15 | 7 | 116 | 2.2273 | 0.1356 | 0.1338 | 0.4464 | 0.4407 | exact | 22 |
| xgb | XGB tuned - RF canonical | 43 | 26 | 13 | 110 | 3.6923 | 0.0547 | 0.0533 | 0.2733 | 0.2663 | cc | 39 |
| svm | SVM tuned - RF canonical | 48 | 21 | 22 | 101 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | cc | 43 |
| lgbm | LightGBM tuned - RF canonical | 48 | 21 | 11 | 112 | 2.5312 | 0.1116 | 0.1102 | 0.4464 | 0.4407 | cc | 32 |
| catboost | CatBoost tuned - RF canonical | 55 | 14 | 13 | 110 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | cc | 27 |
| tabnet | TabNet tuned - RF canonical | 47 | 22 | 44 | 79 | 6.6818 | 0.0097 | 0.0092 | 0.0682 | 0.0645 | cc | 66 |
| ftt | FT-Transformer tuned - RF canonical | 53 | 16 | 35 | 88 | 6.3529 | 0.0117 | 0.0110 | 0.0703 | 0.0658 | cc | 51 |

McNemar on correct / incorrect per day, A = tuned model, B = RF canonical (n01 = tuned correct and RF wrong, n10 = tuned wrong and RF correct); chi2_cc / p_cc with continuity correction, p_exact binomial; recommended = exact when n01 + n10 < 25; Holm over the family of 7 comparisons (p_cc and p_exact separately). A pair with 0 discordant days (identical predictions) is defined as statistic 0, p 1. McNemar compares per-day accuracy, not macro F1 (T1.2 note). Script: revision/scripts/hyperparameter_search.py --mode test, run 2026-10-19.
