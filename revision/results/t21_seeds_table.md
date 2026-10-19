| model | metric | n_seeds | seed42 | mean | min | max | p2_5 | p97_5 |
|---|---|---|---|---|---|---|---|---|
| lgbm | f1_macro | 5 | 0.4089 | 0.4089 | 0.4089 | 0.4089 | 0.4089 | 0.4089 |
| lgbm | f1_active | 5 | 0.2222 | 0.2222 | 0.2222 | 0.2222 | 0.2222 | 0.2222 |
| lgbm | accuracy | 5 | 0.6979 | 0.6979 | 0.6979 | 0.6979 | 0.6979 | 0.6979 |
| lgbm | mcc | 5 | 0.0782 | 0.0782 | 0.0782 | 0.0782 | 0.0782 | 0.0782 |
| catboost | f1_macro | 5 | 0.4980 | 0.5199 | 0.4980 | 0.5388 | 0.5002 | 0.5372 |
| catboost | f1_active | 5 | 0.4615 | 0.4846 | 0.4615 | 0.5000 | 0.4615 | 0.5000 |
| catboost | accuracy | 5 | 0.6875 | 0.7094 | 0.6875 | 0.7188 | 0.6896 | 0.7188 |
| catboost | mcc | 5 | 0.1296 | 0.1731 | 0.1296 | 0.2046 | 0.1328 | 0.2039 |
| tabnet | f1_macro | 5 | 0.2911 | 0.3237 | 0.2687 | 0.3599 | 0.2710 | 0.3597 |
| tabnet | f1_active | 5 | 0.0000 | 0.1461 | 0.0000 | 0.2222 | 0.0116 | 0.2211 |
| tabnet | accuracy | 5 | 0.4583 | 0.4385 | 0.4010 | 0.4688 | 0.4042 | 0.4677 |
| tabnet | mcc | 5 | -0.0204 | 0.0364 | -0.0292 | 0.1293 | -0.0283 | 0.1218 |
| tabnet | best_epoch | 5 | 5.0000 | 22.6000 | 5.0000 | 53.0000 | 5.3000 | 51.5000 |
| ftt | f1_macro | 5 | 0.4115 | 0.3788 | 0.3364 | 0.4194 | 0.3391 | 0.4186 |
| ftt | f1_active | 5 | 0.1739 | 0.2056 | 0.1739 | 0.2500 | 0.1747 | 0.2472 |
| ftt | accuracy | 5 | 0.6302 | 0.5437 | 0.3802 | 0.6302 | 0.3937 | 0.6292 |
| ftt | mcc | 5 | 0.1218 | 0.0521 | -0.0171 | 0.1218 | -0.0147 | 0.1184 |
| ftt | best_epoch | 5 | 11.0000 | 10.2000 | 3.0000 | 17.0000 | 3.3000 | 16.7000 |

Informative, NON-canonical: the 4 new models refit with model seeds [42, 0, 1, 2, 3] (SMOTE stays at seed 42, as in T1.4 mode seeds; only the model varies; networks: early stop + refit per seed), each evaluated on the 192 test days. mean / min / max / p2_5 / p97_5 are descriptive statistics of the 5 values (not a CI). The reported figure is the seed-42 run (== --mode main, identity checked). best_epoch rows only for the networks. Script: revision/scripts/extra_models.py --mode seeds, run 2026-10-19.
