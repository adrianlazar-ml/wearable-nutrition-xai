# T1.3 stage C: safety layer with raw vs calibrated RF probabilities (test, 192 days)

Safety layer = confidence_threshold.apply_threshold / caloric_error_analysis, unchanged: if max(p) < threshold the prediction becomes Normal; caloric error = |offset[pred] - offset[true]| with Rest -200 / Normal 0 / Active +200 kcal over all 192 days; coverage = (192 - overridden) / 192. raw = canonical RF (19 features, test_predictions_rf.csv). platt / isotonic = one-vs-rest calibrators fit on OOF folds 1-4 (calibration_rf.py); prediction = argmax of the calibrated probabilities. Canonical threshold 0.50 (not re-chosen here). NOTE: CIFRE_CANONICE.md section 7 (80.2 percent, 68.8 kcal) comes from the 16-feature XML pipeline, not from the canonical RF; the canonical RF gives 65.6 percent / 58.3 kcal at 0.50 (see _check_t13_safety_layer.py). Script revision/scripts/safety_layer_calibrated.py.

## Canonical threshold 0.50

| variant | threshold | coverage_pct | n_overridden | mean_cal_error | max_cal_error | f1_macro | accuracy | f1_active | n_pred_rest | n_pred_normal | n_pred_active |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.5000 | 65.6250 | 66 | 58.3333 | 200 | 0.5067 | 0.7083 | 0.4000 | 28 | 162 | 2 |
| platt | 0.5000 | 100.0000 | 0 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| isotonic | 0.5000 | 100.0000 | 0 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |

## Full threshold grid

| variant | threshold | coverage_pct | n_overridden | mean_cal_error | max_cal_error | f1_macro | accuracy | f1_active | n_pred_rest | n_pred_normal | n_pred_active |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.0000 | 100.0000 | 0 | 71.8750 | 200 | 0.5063 | 0.6406 | 0.4444 | 43 | 139 | 10 |
| raw | 0.4000 | 96.8750 | 6 | 68.7500 | 200 | 0.5305 | 0.6562 | 0.5000 | 42 | 142 | 8 |
| raw | 0.4500 | 85.4167 | 28 | 63.5417 | 200 | 0.5306 | 0.6823 | 0.4615 | 38 | 149 | 5 |
| raw | 0.5000 | 65.6250 | 66 | 58.3333 | 200 | 0.5067 | 0.7083 | 0.4000 | 28 | 162 | 2 |
| raw | 0.5500 | 44.2708 | 107 | 57.2917 | 200 | 0.4650 | 0.7135 | 0.4000 | 15 | 175 | 2 |
| raw | 0.6000 | 24.4792 | 145 | 58.3333 | 200 | 0.3629 | 0.7083 | 0.2222 | 7 | 184 | 1 |
| raw | 0.7000 | 6.7708 | 179 | 55.2083 | 200 | 0.2800 | 0.7240 | 0.0000 | 1 | 191 | 0 |
| raw | 0.7500 | 1.5625 | 189 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| raw | 0.8000 | 0.5208 | 191 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| platt | 0.0000 | 100.0000 | 0 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| platt | 0.4000 | 100.0000 | 0 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| platt | 0.4500 | 100.0000 | 0 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| platt | 0.5000 | 100.0000 | 0 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| platt | 0.5500 | 100.0000 | 0 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| platt | 0.6000 | 100.0000 | 0 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| platt | 0.7000 | 88.0208 | 23 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| platt | 0.7500 | 22.9167 | 148 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| platt | 0.8000 | 0.5208 | 191 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| isotonic | 0.0000 | 100.0000 | 0 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| isotonic | 0.4000 | 100.0000 | 0 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| isotonic | 0.4500 | 100.0000 | 0 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| isotonic | 0.5000 | 100.0000 | 0 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| isotonic | 0.5500 | 100.0000 | 0 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| isotonic | 0.6000 | 98.9583 | 2 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| isotonic | 0.7000 | 82.2917 | 34 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| isotonic | 0.7500 | 40.6250 | 114 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
| isotonic | 0.8000 | 9.3750 | 174 | 54.1667 | 200 | 0.2811 | 0.7292 | 0.0000 | 0 | 192 | 0 |
