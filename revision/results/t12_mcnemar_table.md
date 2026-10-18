| pair | n00 | n01 | n10 | n11 | chi2_cc | p_cc | p_exact | p_cc_holm | p_exact_holm | recommended | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RF-XGB | 45 | 15 | 24 | 108 | 1.6410 | 0.2002 | 0.1996 | 0.4004 | 0.3992 | cc | Holm family of 3 |
| RF-SVM | 49 | 22 | 20 | 101 | 0.0238 | 0.8774 | 0.8776 | 0.8774 | 0.8776 | cc | Holm family of 3 |
| XGB-SVM | 44 | 27 | 16 | 105 | 2.3256 | 0.1273 | 0.1263 | 0.3818 | 0.3789 | cc | Holm family of 3 |
| RF-persistence | 34 | 50 | 35 | 73 | 2.3059 | 0.1289 | 0.1284 | n/a | n/a | cc | baseline comparison, outside the Holm family |

Convention: n01 = A correct and B wrong, n10 = A wrong and B correct, n11 = both correct, n00 = both wrong (pair = A-B). chi2_cc / p_cc: McNemar chi-square with continuity correction (statsmodels mcnemar(exact=False, correction=True)); p_exact: exact binomial (exact=True); Holm adjustment over the family of 3 model pairs only (RF-persistence is reported outside the family, unadjusted). recommended = exact when n01 + n10 < 25. Test set n = 192. Script: revision/scripts/statistical_tests.py, statsmodels 0.15.0, run 2026-10-18.
