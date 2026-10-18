# T1.3 RF calibration (R1.9)

Canonical RF test probabilities (192 days, Rest/Normal/Active 44/140/8) calibrated with one-vs-rest Platt / isotonic calibrators fit on out-of-fold probabilities (TimeSeriesSplit 4 on the 766 train days; fit_folds 1-4 = 612 rows / 33 Active, primary; fit_folds 2-4 = 459 rows / 27 Active, sensitivity). raw does not depend on fit_folds (identical rows on test). Point [bootstrap 95 percent CI], 1000 resamples, seed 42, shared matrix bootstrap_indices_1000x192_seed42.npy; CI only for ECE top-label, Brier multiclass, log loss, Brier Active on test. Brier multiclass = mean over samples of the sum over classes of (p_c - y_c)^2, range [0, 2]. ECE = top-label, 10 equal-width bins. Script revision/scripts/calibration_rf.py, run 2026-10-19.

### Test (192 days)

| fit_folds | variant | ECE top-label | MCE top-label | Brier multiclass [0, 2] | Log loss | Rows with p(true class) = 0 |
|---|---|---|---|---|---|---|
| 1-4 | raw | 0.1377 [0.0815, 0.2074] | 0.8832 | 0.5091 [0.4747, 0.5489] | 0.8522 [0.7701, 0.9648] | 0 |
| 1-4 | platt | 0.0361 [0.0123, 0.0996] | 0.8398 | 0.4124 [0.3459, 0.4783] | 0.6969 [0.6018, 0.7971] | 0 |
| 1-4 | isotonic | 0.0405 [0.0189, 0.1077] | 0.5721 | 0.4095 [0.3417, 0.4759] | 0.7111 [0.5994, 0.8328] | 0 |
| 2-4 | raw | 0.1377 [0.0815, 0.2074] | 0.8832 | 0.5091 [0.4747, 0.5489] | 0.8522 [0.7701, 0.9648] | 0 |
| 2-4 | platt | 0.0281 [0.0113, 0.0890] | 0.3168 | 0.4130 [0.3435, 0.4808] | 0.7017 [0.5975, 0.8136] | 0 |
| 2-4 | isotonic | 0.0547 [0.0243, 0.1175] | 0.5785 | 0.4111 [0.3396, 0.4816] | 1.1987 [0.6554, 1.8966] | 3 |

### Test, class Active (8 positives; no CI for ECE Active)

| fit_folds | variant | ECE Active | Brier Active | ECE Rest | Brier Rest | ECE Normal | Brier Normal |
|---|---|---|---|---|---|---|---|
| 1-4 | raw | 0.0936 | 0.0502 [0.0322, 0.0721] | 0.1430 | 0.1999 | 0.2565 | 0.2590 |
| 1-4 | platt | 0.0206 | 0.0395 [0.0168, 0.0673] | 0.0156 | 0.1751 | 0.0361 | 0.1978 |
| 1-4 | isotonic | 0.0340 | 0.0392 [0.0172, 0.0667] | 0.0327 | 0.1749 | 0.0405 | 0.1954 |
| 2-4 | raw | 0.0936 | 0.0502 [0.0322, 0.0721] | 0.1430 | 0.1999 | 0.2565 | 0.2590 |
| 2-4 | platt | 0.0408 | 0.0390 [0.0169, 0.0664] | 0.0297 | 0.1758 | 0.0281 | 0.1982 |
| 2-4 | isotonic | 0.0372 | 0.0395 [0.0176, 0.0668] | 0.0393 | 0.1757 | 0.0547 | 0.1959 |

### Test, effect on argmax (informative, NON-canonical; canonical RF figures unchanged)

| fit_folds | variant | Days with argmax changed vs raw | F1 macro (argmax, non-canonical) | F1 Active (argmax, non-canonical) | Accuracy (argmax, non-canonical) | MCC (argmax, non-canonical) |
|---|---|---|---|---|---|---|
| 1-4 | raw | 0 | 0.5063 | 0.4444 | 0.6406 | 0.1416 |
| 1-4 | platt | 53 | 0.2811 | 0.0000 | 0.7292 | 0.0000 |
| 1-4 | isotonic | 53 | 0.2811 | 0.0000 | 0.7292 | 0.0000 |
| 2-4 | raw | 0 | 0.5063 | 0.4444 | 0.6406 | 0.1416 |
| 2-4 | platt | 53 | 0.2811 | 0.0000 | 0.7292 | 0.0000 |
| 2-4 | isotonic | 53 | 0.2811 | 0.0000 | 0.7292 | 0.0000 |

raw argmax reproduces the canonical RF: F1 macro 0.5063, accuracy 0.6406, MCC 0.1416.

### Test, distribution of p_active

| fit_folds | variant | p_active min | p_active median | p_active max | Days with p_active >= 0.50 |
|---|---|---|---|---|---|
| 1-4 | raw | 0.0009 | 0.0716 | 0.6312 | 2 |
| 1-4 | platt | 0.0218 | 0.0513 | 0.1029 | 0 |
| 1-4 | isotonic | 0.0063 | 0.0676 | 0.1896 | 0 |
| 2-4 | raw | 0.0009 | 0.0716 | 0.6312 | 2 |
| 2-4 | platt | 0.0096 | 0.0478 | 0.1434 | 0 |
| 2-4 | isotonic | 0.0000 | 0.0802 | 0.1992 | 0 |

### OOF, in-sample reference for the calibrators (rows used for the fit)

| fit_folds | variant | Rows | Active rows | ECE top-label | MCE top-label | Brier multiclass [0, 2] | Log loss | ECE Active | Brier Active | Rows with all calibrated p = 0 (fallback 1/3) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-4 | raw | 612 | 33 | 0.0739 | 0.1316 | 0.5183 | 0.8938 | 0.0977 | 0.0769 | 0 |
| 1-4 | platt | 612 | 33 | 0.0123 | 0.1168 | 0.4043 | 0.6973 | 0.0008 | 0.0505 | 0 |
| 1-4 | isotonic | 612 | 33 | 0.0188 | 0.3510 | 0.3954 | 0.6778 | 0.0029 | 0.0500 | 0 |
| 2-4 | raw | 459 | 27 | 0.0807 | 0.1738 | 0.5266 | 0.8498 | 0.1033 | 0.0836 | 0 |
| 2-4 | platt | 459 | 27 | 0.0135 | 0.0867 | 0.3921 | 0.6809 | 0.0033 | 0.0546 | 0 |
| 2-4 | isotonic | 459 | 27 | 0.0209 | 0.3586 | 0.3811 | 0.6577 | 0.0059 | 0.0538 | 0 |

In-sample: the calibrators were fit on these same rows, so these values are optimistic and only show what the calibrators do where they were fit.

