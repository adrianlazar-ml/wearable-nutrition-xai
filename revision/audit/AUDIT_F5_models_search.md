# AUDIT F5: additional models with default parameters (T2.1) and hyperparameter search (T2.2)

Auditor role: external, read-and-report only. Branch `audit-r1`, HEAD `27bd4e6` (slices F1 to F4 committed). Run date: 2026-10-25.
Environment used for every command below: Python 3.11.9, scikit-learn 1.8.0, numpy 2.4.3, pandas 2.3.3, scipy 1.17.1, statsmodels 0.15.0, imbalanced-learn 0.14.1, lightgbm 4.6.0, catboost 1.2.10, torch 2.11.0+cpu, pytorch-tabnet 4.1.0 (vendored copy), rtdl_revisiting_models 0.0.2 (identical to `versions` in `t21_run_meta.json`, `t22_run_meta_cv.json`, `t22_run_meta_test.json`).

Slices F1 to F4 were read only for their Findings sections; nothing from them is reused as a fact. F3's finding F3.2 (two bootstrap skip rules) applies to T2.1 and T2.2 unchanged: the marginal CIs use `extended_metrics.bootstrap` (nan skip), the paired CIs use `statistical_tests.paired_bootstrap` (absent-class skip); on the canonical test set 0 of the 1,000 resamples lack a class (R, `INFO resamples with an absent class: 0`), so the two rules coincide here and the finding is not re-examined.

Time limits respected: no CV re-run (27.6 min), no FT-Transformer fit, no TabNet fit. The only refits executed are CatBoost default (two fits of 18 s, E6 / E7), LightGBM default (0.3 s, R6 / E7) and the two LightGBM tuned fits of the E8 diagnostic (under 1 s).

Files written by this audit (nothing else was modified; no file under `revision/results/`, `revision/scripts/`, `revision/vendor/`, `revision/manuscript/` or the repo root was touched):

- `revision/audit/AUDIT_F5_models_search.md` (this report)
- `revision/audit/recompute_f5.py` + `recompute_f5_output.txt` (T3; imports numpy, pandas, scipy, scikit-learn, statsmodels, imbalanced-learn, lightgbm, joblib only; nothing from `revision/scripts/` or the repo root; no torch; run with `python -I`; 121 checks, 1 m 52 s)
- `revision/audit/_check_f5_empirical.py` + `_check_f5_empirical_output.txt` (T2 evidence E1 to E8; imports `extra_models`, `hyperparameter_search` and the vendored `pytorch_tabnet` in memory and calls none of their `main()` / `run_*()`; the regenerated search space is written to a temporary directory; 22 checks, about 45 s)

All four files LF, ASCII (`file`: "ASCII text"; both scripts set `sys.stdout.reconfigure(newline="\n")` because Python on Windows otherwise writes CRLF through a redirect). Abbreviations below: R = `recompute_f5_output.txt`, E = `_check_f5_empirical_output.txt`, `em` = `revision/scripts/extra_models.py`, `hs` = `revision/scripts/hyperparameter_search.py`.

---

## T0. Inventory

`git status --short` at the start: unchanged from slice F4 (one deleted tracked zip, three untracked paths outside `revision/`, the untracked `data/apple_health/New Automation/` folder, the author's untracked `revision/audit/AUDIT_DECISIONS.md`). Nothing under `revision/results/` or `revision/scripts/` is modified or untracked. The commit command at the end names files explicitly.

sha256 first 16 hex chars of the inputs read:

| file | sha256[:16] | role |
|---|---|---|
| `revision/scripts/extra_models.py` | 5e456459d495a4cd | T2.1 (fit_eval_model, EarlyStopper, tabnet_fit, ftt_build, ftt_train) |
| `revision/scripts/hyperparameter_search.py` | 26365a62ef7ce4c7 | T2.2 (search spaces, make_folds, fit_predict, select_winner, final_fit) |
| `revision/scripts/common.py` | fb73b00eb9a650d3 | preprocess_split |
| `revision/scripts/save_test_predictions.py` | 24c7c7d9cf976360 | build_models |
| `revision/scripts/statistical_tests.py` | c8506d2cdc5e21fa | paired_bootstrap, mcnemar_counts, mcnemar_tests (audited in F2) |
| `revision/scripts/extended_metrics.py` | 02c78dcf65c845fd | bootstrap, load_inputs, label_metrics |
| `revision/scripts/group_ablation.py` | 654d864dd88c8a1f | f1_macro, metrics, save_predictions |
| `revision/vendor/pytorch_tabnet/{multiclass_utils, callbacks, abstract_model}.py` | f27edafce2362540 / 4ba594e8d9f5c60c / 690aca040992a8e5 | vendored TabNet (patch, EarlyStopping, fit loop) |
| `revision/requirements.txt` | 8dc6d0dd1ac7d4bc | pins |
| `revision/results/t21_predictions_{lgbm, catboost, tabnet, ftt}.csv` | 1743cae643f09edf / e610fd508cdb352f / 7b193890426ed978 / 5d30a2755a59d53d | T2.1 predictions (== `CIFRE_REVIZIE.md` artefact table) |
| `revision/results/t21_extra_models.csv`, `t21_comparisons.csv`, `t21_seeds.csv`, `t21_run_meta.json` | 642704ca9ad13bd0 / dee1cbfe60533d49 / cee63f2147fb689c / 059513c413f69003 | T2.1 tables and metadata |
| `revision/results/t22_search_space.json` | 79c1aed12f0151e7 | search space (== the sha256 claimed) |
| `revision/results/t22_cv_results.csv`, `t22_run_meta_cv.json` | 7dbf66e5fd1ae163 / 99517e1a9fb534fd | CV results and metadata |
| `revision/results/t22_predictions_{rf, xgb, svm, lgbm, catboost, tabnet, ftt}.csv` | 87eae70e6082253e / 0e4f58f4a03fae81 / 0a2edf1cb180730b / 5726ad181f41c6bd / 222bcbf98147bf33 / da1c55ccd74e8e6e / 8b2699776d189f10 | T2.2 tuned predictions |
| `revision/results/t22_test_results.csv`, `t22_test_metrics.csv`, `t22_comparisons.csv`, `t22_mcnemar.csv`, `t22_run_meta_test.json` | 0ae17cc0595cd1b6 / e49e23403c20d740 / 9f828319430656a7 / 5c4b2024f880ca88 / 8edd0b0b85718249 | T2.2 test tables and metadata |
| `revision/results/test_predictions_{rf, xgb, svm, persistence}.csv` | aea39c6dd2989e24 / e0afa80d26440534 / c48625b27debfcd2 / ab228a11e65bfe17 | canonical references |
| `revision/results/bootstrap_indices_1000x192_seed42.npy` | fff3b2834c061962 | shared matrix (== `default_rng(42)`, R) |
| `models/dataset_snapshot_20260331.csv`, `revision/results/models/imputer_revision.joblib` | 176649b59fef3adb / 9b5e16ba6cf9a70a | inputs of the independent LightGBM refit |

Git: T2.1 was committed in 98f3186 (2026-10-19 20:07) and f7c70d7 (20:34, "optional config args for T2.2 (T2.1 identity re-run)"); T2.2 in a single commit 0ce52a8 (21:48:11) that contains the search space, the CV results, the test results and the script together, so git alone cannot order the search space against the fits (see Q12 for the evidence that does).

---

## T1. Trace tables (from reading the code)

### (a) `fit_eval_model(name, seed, data)` (em:373-454), one tree and one network

| step | tree (lgbm / catboost; rf identical) | network (tabnet; ftt in brackets) | rows |
|---|---|---|---|
| data | `load_data()` em:228-239: `build_dataset` on the raw frame, canonical thresholds, asserts 958 x 19, cut 766, label counts (202, 710, 46) / train (158, 570, 38) | same | 958 |
| canonical split | `canonical_preprocess` em:242-249 -> `common.preprocess_split(X[:766], X[766:], ...)` common.py:98-120: imputer fit on train (:108), SMOTE(42, k 3) on imputed train (:111), scaler fit on imputed train pre-SMOTE (:114), second SMOTE on scaled train (:116); test only transformed (:109, :115) | same call | train 766 -> 1,710 (570 x 3); test 192 |
| input flow | `SCALED[name]` False -> `X_train_sm` / `X_test` em:383-385 | True -> `X_train_scaled_sm` / `X_test_scaled` em:383-385 | |
| stage A (networks only) | none (em:387-396: one fit, no validation) | `inner_preprocess` em:252-270: `preprocess_split(X[:612], X[612:766], ...)`, k 3, asserts Active (25, 13), validation 154 real rows; `X_in` = scaled SMOTE inner-train (em:399), `X_val` = scaled validation (em:400) | inner-train 612 -> 1,356; validation 154 |
| early stopping | n/a | TabNet: `tabnet_fit(X_in, y_in, seed, 200, 16, X_val, y_val)` em:406 -> `TabNetClassifier.fit(eval_set, eval_metric=[F1MacroMetric], max_epochs 200, patience 16)` em:294-302; package `EarlyStopping` callbacks.py:131-148 (strict `>`, tol 0); `best_epoch = m_a.best_epoch + 1` em:407 (package counter is 0-based). [FT-T: `ftt_build(19, seed)` em:421 -> `ftt_train(..., 200, 64, X_val, y_val, 16)` em:422, own loop em:355-366 with `EarlyStopper` em:165-187, `best_epoch` 1-based] | 21 epochs run, best 5 [27, best 11] |
| refit | n/a | `tabnet_fit(X_tr, y_tr, seed, best_epoch, 0)` em:409: no eval_set, max_epochs = best_epoch, patience 0; asserts `len(history["loss"]) == best_epoch` em:410-411. [FT-T: fresh `ftt_build` em:425, `ftt_train(..., best_epoch, 64)` em:426 without validation; asserts epochs_run == best_epoch em:427-428] | 1,710 rows, 5 epochs [11] |
| prediction | `predict_proba(X_te)` em:391, `predict(X_te)` em:392 | refit model `m.predict_proba` / `m.predict` em:412-414 [ftt_proba em:429, argmax em:430] | 192 |
| checks | classes_ == [0, 1, 2] em:442, shape em:444, `y_pred = argmax(proba)` em:446, argmax == predict() em:447-448, rows sum to 1 em:449-451 | same | |
| output | `info` with y_true (= `split.y_test`, em:380), y_pred, proba, dates_test, params em:452 | plus best_epoch, val_f1_macro_at_best, stage_a_epochs_run, val_history em:401-404, :408, :424, :438 | |

Seeding: `seed_everything(seed)` em:156-162 before every model construction (em:388, :293, :320); SMOTE seed is always 42 (em:248, :262). Metrics, CIs and comparisons are computed afterwards in `run_main` from the returned vectors (em:625-699); the test vectors enter no decision.

### (b) one CV iteration of `hyperparameter_search.py --mode cv` (hs:444-532)

| step | code | subset |
|---|---|---|
| search space | `build_search_space()` hs:172-184 -> `em.write_json(res / "t22_search_space.json")` hs:449-450, BEFORE `load_data` hs:452 and `make_folds` hs:453 | 5 x 20 sampled (`ParameterSampler(random_state 42)` hs:167-169) + 2 x 5 grid + 7 references |
| folds | `make_folds` hs:228-249: `fold_bounds()` hs:191-198 = `TimeSeriesSplit(3, test_size 154)` on 766 rows, asserted == `EXPECT_FOLDS` hs:102, :232; per fold `preprocess_split(X[a:b], X[b:c], ...)` hs:240 with k from `smote_k(min(label_counts(y_tr)))` hs:237; asserts Active (10, 7) / (17, 8) / (25, 13) hs:241-243 and validation size 154 hs:244-245 | train 0-303 / 0-457 / 0-611 -> 669 / 1,008 / 1,356 after SMOTE; validation 304-457 / 458-611 / 612-765 |
| configuration loop | `for name in MODELS` hs:461; `todo` = eligible configs + `(REF_ID, reference, False)` hs:465; LightGBM eligible configs get `LGBM_FIXED` (subsample_freq 1) hs:468; `score_config_on_folds` hs:470 | 117 configurations x 3 folds |
| per fold | `fold_matrices(name, split)` hs:335-339 picks scaled (svm, networks) or unscaled matrices; tabular: `fit_predict(name, cfg, seed, X_tr, y_tr, X_va)` hs:353 -> `build_tabular` hs:253-266 (canonical `build_models()` object with `set_params`, or `em.make_tree(**cfg)`), `fit`, `predict_proba` / `predict` hs:289-291, y_pred = predict() for svm else argmax hs:294; score = `f1_macro(y_va, y_pred)` hs:354. Networks: `fit_predict(..., early_stop=True)` hs:348 -> `em.tabnet_fit(..., 200, 16, Xev, y_ev, fit_overrides, **ctor)` hs:301 or `em.ftt_build(arch)` + `em.ftt_train(..., 200, 64, X_ev, y_ev, 16)` hs:316-319; score = `best_val_f1` hs:349 (validation macro F1 at best_epoch) | |
| aggregation | `f1_mean = arr.mean()`, `f1_sd = arr.std(ddof=1)` hs:358-360 | 3 folds |
| record | row with model, config_id, eligible, params (`config_key`, sorted json), 3 fold scores, mean, sd, best_epoch per fold hs:476-480; failed fits recorded, never replaced hs:471-474 | `t22_cv_results.csv` (117 rows) |

### (c) selection and final evaluation (`--mode test`, hs:550-730)

| step | code | subset |
|---|---|---|
| winner | `load_winners` hs:535-543 -> `select_winner` hs:201-209: eligible rows of the model, sort by (f1_mean desc, config_id asc), first row; params from the csv row | one per model |
| reference rank | `reference_rank` hs:212-216: 1 + number of eligible rows with f1_mean strictly greater than the reference row | |
| data | `em.load_data()`, `em.canonical_preprocess` (766 / 192), `em.inner_preprocess` (612 / 154) hs:555-558; `y_true = split.y_test` checked against the saved files hs:561-562 | |
| final fit | `final_fit(m, cfg, 42, split, inner)` hs:363-377: trees -> `fit_predict` on `X_train_sm` (or scaled for svm) and predict `X_test` hs:366, :374; networks -> stage A `fit_predict(..., X_in, y_in, X_val, y_val, early_stop=True)` hs:369-370 on the inner split with the winning configuration, `cfg["epochs"] = best_epoch` hs:373, refit `fit_predict` on the 766-day matrix without validation hs:374 (hs:308-313 TabNet, hs:323-327 FT-T) | once per model |
| checks | proba sums hs:576-577, argmax == y_pred except svm hs:578-581, identity with the reference file when the winner equals the reference configuration hs:582-587 | |
| metrics | `em.metrics_with_ci` on the 7 tuned vectors hs:595-596; RF canonical marginal CI == T1.1 hs:602-604 | shared matrix |
| comparisons | (a) RF canonical, (b) own reference file, (c) RF tuned, (d) persistence hs:630-631; `paired_bootstrap(y_true, yp, op)` = tuned minus other hs:636; diff_point == difference of the point values hs:637-638 | 7 x 4 x 2 metrics minus rf-(c) = 54 rows |
| McNemar | `mcnemar_counts(y_true, tuned, preds["rf"])` (A = tuned) hs:650, `mcnemar_safe` hs:219-224, Holm over the 7 rows for p_cc and p_exact separately hs:658-659 | |
| outputs | `t22_predictions_*.csv` hs:573-574, `t22_test_results.csv` hs:675, `t22_comparisons.csv` hs:690, `t22_mcnemar.csv` hs:700, `t22_run_meta_test.json` hs:709-725 (winners, best_epoch, hierarchy) | |

---

## T2. Targeted questions

Network protocol (T2.1)

1. **Inner split temporal, validation strictly later?** Yes. `inner_bounds()` em:220-224 returns (612, 154); `inner_preprocess` em:257, :261 slices `X.iloc[:612]` / `X.iloc[612:766]` of the date-sorted frame (`build_dataset` sorts by date, common.py:70). Validation dates 2025-04-15 to 2025-09-18, last inner-train date 2025-04-14; dates monotonic (E5, `validation dates strictly after the last inner-train date`).

2. **Imputer, scaler, SMOTE fit on the 612 only; validation 154 real rows?** Yes. `preprocess_split` fits the imputer on `X_tr` (common.py:108), the scaler on the imputed train (:114) and SMOTE on the train only (:111, :116); the validation part is transformed (:109, :115) and never resampled. `inner_preprocess` asserts 154 rows in `y_test`, `X_test`, `X_test_scaled` (em:266-267). E5: the inner validation matrix equals the raw rows 612-765 imputed with an imputer refit on rows 0-611, the imputer and scaler statistics equal a refit on those rows, and the inner split equals fold 3 of T2.2 exactly.

3. **Early stopping on validation macro F1, strict, patience 16, consistent counters?** Yes. TabNet: `eval_metric=[F1MacroMetric]` em:297-298, the metric is `f1_macro(y_true, argmax(y_score))` em:197-198 (labels [0, 1, 2], zero_division 0); the package callback improves only when `loss_change > tol` with tol 0 (callbacks.py:136-139), sets `wait = 1` after an improvement and stops when `wait >= patience` on a non-improving epoch (:142-148); `best_epoch` is the 0-based `epoch_idx` of `fit` (abstract_model.py:253-270), converted with + 1 at em:407. FT-T: `EarlyStopper.update` em:178-187 replicates the same rule 1-based; `ftt_train` computes `f1_macro` on the validation rows after every epoch (em:363). E1: fed with the two recorded validation histories of `t21_run_meta.json`, the vendored callback and `EarlyStopper` both give (best_epoch, stopped epoch, epochs run) = (5, 21, 21) for TabNet and (11, 27, 27) for FT-T, equal to the recorded values; on 2,000 random sequences with many ties x patience {1, 2, 3, 5, 16} the two rules agree on all three quantities; a constant sequence [0.3 x 4] with patience 2 stops at epoch 3 with best_epoch 1 under both (strict improvement). Both stop on the 16th consecutive non-improving epoch (best + 16 = 21 and 27).

4. **Refit on 766 (1,710 after SMOTE) for exactly best_epoch epochs, no validation, no early stopping; test weights from the refit?** Yes. TabNet: `tabnet_fit(X_tr, y_tr, seed, best_epoch, 0)` em:409 passes no `eval_set` (em:296-298 adds it only when `X_val is not None`), `max_epochs = best_epoch`, `patience 0`; the package then adds no `EarlyStopping` (abstract_model.py:676-687, the branch that raises the "No early stopping" warning) and runs `range(max_epochs)` (:253); em:410-411 asserts `len(history["loss"]) == best_epoch`. The evaluated object `m` is the refit (em:412-414); `m_a` is used only for `best_epoch` / `best_cost` (em:407-408). FT-T: a fresh model from `ftt_build` em:425, `ftt_train` without `X_val` (stopper None, em:352) for `max_epochs = best_epoch` em:426, asserted em:427-428; `ftt_proba(m, X_te)` on that model em:429. The training matrix is `X_tr` = `split.X_train_scaled_sm` (1,710 rows, em:383; `n_train_rows_after_smote` 1710 in `t21_run_meta.json`).

5. **Refit reseeded identically to stage A?** Yes for the initialisation and the RNG state, no for the trajectory, and the documents say "refit from scratch", not "the same training stopped in time". `tabnet_fit` calls `seed_everything(seed)` em:293 before constructing `TabNetClassifier(seed=seed)`, whose `__post_init__` and `_set_network` call `torch.manual_seed(self.seed)` again (abstract_model.py:78, :601); the DataLoader shuffles with the global torch RNG from that state. `ftt_build` calls `seed_everything(seed)` em:320 before construction; E3: two constructions give bit-identical state_dicts (53 tensors, 899,523 parameters). The DataLoader generator is `torch.Generator().manual_seed(seed)` in both stages (em:350-351). The data differ, however: 1,356 rows (22 batches of 64) in stage A against 1,710 rows (27 batches) in the refit, so `randperm` draws different permutations and FT-T takes 242 optimizer steps in stage A at epoch 11 but 297 in the refit (E3); TabNet takes 5 steps of one 1,024-row batch in both stages, on different rows. `best_epoch` is therefore a transferred epoch count. This is what the script docstring (em:35-36, "REFIT from scratch ... for exactly best_epoch epochs"), `t21_run_meta.json` (`protocol.networks`) and the manuscript (Section 3.4: "followed by a refit on the full 766-day training set for that number of epochs") state; none of them claims continuity of the stage-A run. The step-count difference (242 versus 297) is stated nowhere; it does not change any number and is listed under Observations.

6. **Test touched exactly once per model; any decision on X_test / y_test?** Yes, once per model and mode; no decision. In `extra_models.py` the test matrix appears at em:385 (selection of the scaled / unscaled matrix), :391-392, :412-414, :429 (prediction only) and `y_true` at em:380 (returned for the metrics). The early-stopping data are the inner validation rows (em:399-400, rows 612-765 of the train). In `run_main` the test vectors are used only after all fits, for metrics, CIs, comparisons and the sanity checks (em:616-699); none of the checks changes a prediction or a parameter. In `--mode seeds` every seed is evaluated on the test set (em:798); this is 5 evaluations per model, declared "informative, NON-canonical" (em:58-62, table note em:823-827), with no selection on them (the reported figure is the seed-42 run fixed beforehand).

7. **TabNet default: batch 1024, drop_last True; rows per epoch; "practically untrained"?** Confirmed. `t21_run_meta.json` `models.tabnet.params`: batch_size 1024, virtual_batch_size 128, drop_last true, refit_max_epochs 5, n_train_batches_per_epoch 1, best_epoch 5, stage_a_epochs_run 21. E2 through the vendored `create_dataloaders`: 1,356 rows -> 1 batch, 1,024 rows seen per epoch (75.5% of the inner-train); 1,710 rows -> 1 batch, 1,024 rows (59.9% of the train); 669 and 1,008 rows (CV folds 1 and 2) -> 0 batches. With best_epoch 5 the refit performs 5 optimizer steps (Adam, lr 0.02) over 5 x 1,024 = 5,120 row visits; stage A performed 21 steps and its best validation F1 (0.3703) was at step 5. The manuscript sentence (Section 4.4) "TabNet saw a single batch per epoch and stopped after five epochs ... a mismatch between its defaults and the dataset size" is supported by these numbers, with one wording caveat (Observations: the early stopping ran 21 epochs on 1,356 rows and selected epoch 5; the refit ran 5 epochs on 1,710 rows).

8. **Determinism: CatBoost default through `fit_eval_model`.** Reproduced. E6: `em.fit_eval_model("catboost", 42, data)` gives y_pred identical on 192 / 192 days to `t21_predictions_catboost.csv`, max abs probability difference 1.1e-16, F1 macro 0.4980, effective parameters 1,000 iterations, depth 6, lr 0.0813, tree_count 1,000 (18.0 s). The two byte-identical consecutive runs are documented in `CIFRE_REVIZIE.md` (T2.1 determinism row, `_check_t21_outputs.py`) and were not repeated.

Comparisons (T2.1)

9. **Same functions as F2, same matrix, same sign conventions; Holm on the family of 4, separately?** Yes. `paired_bootstrap(y_true, preds["rf"], yp, mfn, idx)` em:659 is the F2 function with A = RF, B = model, so `diff_point` = RF minus model (statistical_tests.py:164), matching the column name `delta_rf_minus_model_*`; the persistence comparison passes A = model (em:671). `idx` is loaded and verified byte-identical to `default_rng(42)` (em:554-559; R confirms). `mcnemar_counts(y_true, yp, preds["rf"])` em:678 gives A = model, so n01 = model correct and RF wrong (statistical_tests.py:109-115), as the csv note says (em:729-731). Holm: `multipletests(comp["p_cc"], method="holm")` and the same on `p_exact`, each over the 4 rows (em:688-689). R reproduces all four p_cc / p_exact and all eight Holm values (manual Holm == statsmodels).

10. **Networks y_pred = argmax; LightGBM / CatBoost argmax == predict on all 192?** Yes. `y_pred = proba.argmax(1)` for every model (em:446) and em:447-448 aborts if it differs from `predict()`; for FT-T `pred_api` is the argmax itself (em:430). E7: LightGBM and CatBoost refit through `make_tree` on the canonical split give `predict()` == argmax on 192 / 192 days and equal to the saved files. R: argmax(proba) == y_pred in all four `t21_predictions_*.csv` and in six of the seven `t22_predictions_*.csv`; `t22_predictions_svm.csv` uses `predict()` with argmax agreement 0.9062, as declared (hs:294, :581, `CIFRE_REVIZIE.md` T2.2 test row).

Search (T2.2)

11. **Folds exactly as claimed; validation without SMOTE; imputer / scaler / SMOTE per fold on the train part only?** Yes. R: `TimeSeriesSplit(3, test_size=154)` on 766 rows gives train 0-303 / 0-457 / 0-611 and validation 304-457 / 458-611 / 612-765; Active counts from the snapshot labels 10 / 17 / 25 in train and 7 / 8 / 13 in validation. `make_folds` asserts the same (hs:232, :241-245). E5 through `hs.make_folds`: each fold's `X_test` equals the raw rows b..c-1 imputed with an imputer refit on rows a..b-1 (154 real rows), `X_test_scaled` equals a scaler refit on the fold's imputed train applied to them, the stored imputer and scaler statistics equal those refits, and the SMOTE train sizes are 669 / 1,008 / 1,356.

12. **Search space written before any fit?** Yes, on three independent pieces of evidence. (i) Order in the script: `write_json` at hs:450 precedes `load_data` (hs:452), `make_folds` (hs:453) and the first fit inside the loop (hs:461-470). (ii) File times on disk: `t22_search_space.json` 2026-10-19 21:10:05, `t22_cv_results.csv` 21:38:04; `t22_run_meta_cv.json` records run_date 21:38:04 (taken after the loop, hs:499) and elapsed 1,679.3 s, so the CV that produced the committed files started at 21:10:05, the second the search space was written. (The files on disk are those of the second, determinism run: `_check_t22_outputs.py:4-9` copies the first-run outputs to a scratch directory, re-runs both modes into `revision/results/` and restores the first-run files only if something differs; `CIFRE_REVIZIE.md` gives 1,679 s for the second CV run, the elapsed value in the committed json. The json and the csv were written by the same run either way.) (iii) Regeneration: R rebuilds the 100 sampled configurations with `ParameterSampler(random_state 42)` from an independent copy of the five spaces and finds them equal to the json (floats included); E4 writes `hs.build_search_space()` with the script's own writer to a temporary file and obtains sha256 79c1aed12f0151e7, byte-identical to the committed file. Git itself cannot order the json against the csv (one commit, T0).

13. **Selection = simple mean over 3 folds; chosen row = maximum; references excluded?** Yes. `f1_mean = arr.mean()` hs:359 (R: equals the mean of the three fold columns on all 117 rows to 1e-9; `f1_sd` is the sample sd). `select_winner` filters `cv.eligible` before the argmax (hs:203); references carry `eligible False` (hs:465). R recomputes the winner of every model from the csv (ties: none at the maximum) and finds it equal to `t22_run_meta_test.json` `winners` and to `best_config_id` / `best_params` in `t22_test_results.csv`.

14. **Network CV optimism declared in the manuscript?** Yes. `KBS_Manuscript_R1.docx` Section 4.4: "Their cross-validation scores are optimistic relative to the tree models, since their number of epochs was selected on the same validation block, which partly explains the reversal"; Appendix Table B1 note: "Network CV scores use the number of epochs selected on each fold's validation block and are therefore optimistic relative to the tree models"; `KBS_Response_R1.docx` (reply to R1.4) repeats it. The CV score of a network is `best_val_f1` (hs:349), the maximum over epochs of the fold's validation F1.

15. **Winner trained once (trees on 766; networks by the T2.1 protocol); no parameter changed after the test?** Yes. `final_fit` hs:363-377 (trace c); the networks' epoch count comes from the inner split with the winning configuration (hs:369-373), never from the test. `run_test` takes `cfg = dict(w["params"])` from the csv winner (hs:569-571) with no modification; R: the `winners` block of `t22_run_meta_test.json` (config_id, params, cv_f1_mean) equals the argmax rows of `t22_cv_results.csv` for all seven models. One reproducibility gap in the manuscript's description of the LightGBM winner is Finding F5.1.

16. **Signs and references of (a) to (d) match the csv labels; CIs declared unadjusted?** Yes. `paired_bootstrap(y_true, yp, op)` hs:636 is tuned minus other; R rebuilds every one of the 54 rows of `t22_comparisons.csv` from the vector named in the `other` column (`rf_canonical` = `test_predictions_rf.csv`, `{m}_reference` = `test_predictions_{m}.csv` or `t21_predictions_{m}.csv`, `rf_tuned` = `t22_predictions_rf.csv`, `persistence` = last train label + shifted y_true) and reproduces value_tuned, value_other, diff_point, both CI bounds and prop_gt_0 to 1e-9. Unadjusted CIs are declared in `t22_comparisons_table.md` (hs:697-698), in `CIFRE_REVIZIE.md` (comparison (a) row: "CI-uri neajustate pentru 7 comparatii") and in the manuscript (Section 3.6 "Intervals are not adjusted for multiple comparisons unless stated"; Section 4.4 "before any adjustment for the seven comparisons").

17. **"Third of 21" exact; 0.0085 correct?** Yes. R: RF eligible configurations above the reference 0.3595: config 8 (0.3680) and config 17 (0.3640), so rank 1 + 2 = 3 of 20 + 1; winner 0.36798 minus reference 0.35945 = 0.00853 -> 0.0085. The other reference ranks (13 / 4 / 5 / 13 / 6 / 2) also reproduce. The manuscript rounds to "0.009 below the selected one", consistent.

18. **Versions pinned; one-line patch; PATCH.md exact?** Yes with one wording point. `revision/requirements.txt` pins `torch==2.11.0`, `lightgbm==4.6.0`, `catboost==1.2.10`, `pytorch-tabnet==4.1.0`, `rtdl_revisiting_models==0.0.2`; the installed versions are 2.11.0+cpu / 4.6.0 / 1.2.10 / 4.1.0 / 0.0.2 (`importlib.metadata`, `t21_run_meta.json`). The "+cpu" local label cannot be expressed in a PyPI pin; the comment in the file names the CPU wheel. `diff -rq` between `site-packages/pytorch_tabnet` (installed wheel) and `revision/vendor/pytorch_tabnet`: the only `.py` difference is `multiclass_utils.py`, one line (`from scipy.sparse.base import spmatrix` -> `from scipy.sparse import spmatrix`), identical to the diff printed in `PATCH.md`; the vendor folder additionally holds `LICENSE` and `PATCH.md`; `__pycache__` differs but is not tracked (`git ls-files` lists 0 cache files). `extra_models.py` asserts at run time that the import comes from the vendor folder (em:581); E confirms.

Tally: 18 / 18 answered yes on the question asked; Q5 (trajectory not continuous, declared as a refit), Q7 (wording) and Q18 (+cpu label) carry observations; Q15 carries Finding F5.1.

---

## T3. Independent recomputation (`revision/audit/recompute_f5.py`)

Pass criterion: identical at 4 decimals to the claims; LightGBM y_pred identity. Result: ALL PASS, 121 / 121 checks (R).

| claim | recomputed | source |
|---|---|---|
| F1 macro, 9 systems | RF 0.5063 > CatBoost 0.4980 > XGB 0.4760 > SVM 0.4347 > FT-T 0.4115 > LightGBM 0.4089 > persistence 0.3039 > TabNet 0.2911 > majority 0.2811; each equal to `t21_extra_models.csv` to 1e-9 | R1 |
| balanced accuracy, RF + 4 models | 0.5227 / 0.3895 / 0.4737 / 0.2823 / 0.4290 == csv | R1 |
| delta RF minus model, F1 macro [CI] | LightGBM +0.0974 [-0.0627, +0.2335]; CatBoost +0.0083 [-0.0841, +0.1131]; TabNet +0.2152 [+0.0772, +0.3345]; FT-T +0.0948 [-0.0085, +0.1912]; 0 skipped; balanced-accuracy and persistence deltas == csv to 1e-9 | R1 |
| McNemar RF-TabNet | n01 / n10 = 32 / 67, chi2_cc 11.6768, p_cc 0.0006; all four pairs == csv | R2 |
| Holm over 4 | TabNet 0.0025 (cc) / 0.0022 (exact); LightGBM 0.3279; CatBoost 0.3279; FT-T 0.8897; manual == statsmodels == csv | R2 |
| T2.2 McNemar + Holm over 7 | all seven pairs == `t22_mcnemar.csv`; none below 0.05 after Holm | R2 |
| CV mean and sd | f1_mean == mean of the 3 folds, f1_sd == sample sd on 117 rows | R3 |
| winners | RF cfg 8 (0.3680), XGB 10, SVM 1, LightGBM 13, CatBoost 1, TabNet 1, FT-T 2 == run_meta_test == test_results | R3 |
| RF reference | rank 3 / 21 (above it: cfg 8 0.3680, cfg 17 0.3640); 0.3680 - 0.3595 = 0.0085 | R3 |
| tuned F1 macro | LightGBM 0.5739 > RF 0.5683 > CatBoost 0.5368 > XGB 0.5043 > SVM 0.4386 > FT-T 0.3698 > TabNet 0.3412 | R4 |
| LightGBM tuned minus RF canonical | +0.0676 [+0.0001, +0.1483] (ci_low 8.5e-05 before rounding) | R4 |
| LightGBM tuned minus RF tuned | +0.0056 [-0.0305, +0.0450] | R4 |
| RF tuned minus RF canonical | +0.0619 [+0.0028, +0.1398] | R4 |
| all 54 rows of `t22_comparisons.csv` | reproduced to 1e-9 from the labelled vectors | R4 |
| search space | 100 sampled configurations regenerated == json; sha256 79c1aed12f0151e7 | R5 |
| folds | bounds and Active counts as claimed; SMOTE sizes 1,356 / 1,710 | R5 |
| LightGBM default refit | snapshot train rows + `imputer_revision.joblib` (statistics == a refit on the snapshot) + SMOTE(42, k 3) -> 1,710 rows (570 x 3) -> `LGBMClassifier(random_state 42, n_jobs 1, verbose -1)`: y_pred identical on 192 / 192 days to `t21_predictions_lgbm.csv`, max abs probability difference 1.1e-16, F1 macro 0.4089; argmax == predict() | R6 |

Verdict T3: PASS on every item, including the LightGBM identity.

---

## Findings (FAIL or UNCERTAIN only)

### F5.1. The tuned LightGBM configuration is not reproducible from the manuscript: Appendix Table B1 and the register omit the fixed `subsample_freq = 1`, without which the stated configuration gives 0.4965, not 0.5739 (Q15: FAIL)
Observed: every sampled LightGBM configuration is fitted with `LGBM_FIXED = {"subsample_freq": 1}` (hs:124, :468), recorded in `t22_search_space.json` (`models.lgbm.fixed`), in the `params` column of `t22_cv_results.csv`, `t22_cv_summary.md`, `t22_test_results.csv` and in `t22_run_meta_test.json`. In LightGBM `subsample` (bagging fraction) is inactive unless `subsample_freq` (bagging frequency) is positive; the library default is 0. The manuscript's Appendix Table B1 describes the LightGBM search space as "estimators {100, 200, 500}; leaves {7, 15, 31}; depth {none, 4, 6}; lr log-uniform [0.01, 0.3]; min child samples {5, 10, 20, 40}; subsample, colsample {0.6, 0.8, 1}; lambda {0, 1, 10}" and the selected configuration as "200, 7 leaves, depth 4, lr 0.079, mcs 20, subsample 0.6, colsample 0.6, lambda 10"; `subsample_freq`, "bagging" and "bagging frequency" appear nowhere in `KBS_Manuscript_R1.docx`, `KBS_Supplementary_R1.docx` or `KBS_Response_R1.docx`, and the T2.2 rows of `CIFRE_REVIZIE.md` (winner description in the CV row; comparison rows) do not name it either (the seeds row says only "LightGBM tuned are bagging"). E8 (diagnostic, two fits under 1 s, not a proposed figure): the winner refit as run reproduces `t22_predictions_lgbm.csv` and 0.5739; the same configuration without `subsample_freq` (library default 0, bagging off) gives test macro F1 0.4965 and different predictions. Why it matters: LightGBM tuned is the first model of the tuned hierarchy and the headline of Section 4.4 ("tuning raised LightGBM from 0.409 to 0.574 [0.417, 0.680]"); a reader who reproduces the Table B1 configuration with the library defaults obtains a model that is below the reported Random Forest instead of above it. The reported numbers are correct for the configuration that was run; the description of that configuration is incomplete in the manuscript and in the register. What the authors should decide: whether to add "bagging frequency 1 (fixed) for every sampled LightGBM configuration" to the LightGBM search-space cell of Table B1 and to the selected-configuration cell, and the same words to the T2.2 CV row of `CIFRE_REVIZIE.md`.

### F5.2. The TabNet reference row of the CV table scores a network that received no gradient step on folds 1 and 2; the manuscript reports its rank ("6 of 6") without that caveat (Q13, Q17: UNCERTAIN)
Observed: the reference TabNet configuration (batch 1,024, drop_last True) yields 0 batches per epoch on 669 and 1,008 SMOTE rows (E2; `t22_run_meta_cv.json` `timing.tabnet.configs.ref.extra`: n_batches_per_epoch 0, 0, 1), so its fold-1 and fold-2 scores (0.0683, 0.1560; `t22_cv_results.csv` row `tabnet, ref`) are the validation F1 of the randomly initialised network, and its mean (0.1982) and rank "6 of 6" are degenerate. `CIFRE_REVIZIE.md` (T2.2 CV row) states this ("referinta TabNet ... are 0 batch-uri pe fold 1 si 2 ... de aici 0.0683 / 0.1560"); Appendix Table B1 of `KBS_Manuscript_R1.docx` prints "6 of 6" in the "Reference rank" column with the general note "reference rank = position of the reported configuration among the 20 sampled (or 5 fixed) plus itself" and no caveat. Why it matters: no selection or test figure depends on the reference row (it is not eligible, and the test-set gain of tuning for TabNet, comparison (b), is measured against the T2.1 default that did train, with one batch per epoch), but the printed rank invites the reading that the default TabNet is the worst of six trained configurations on CV, whereas on two of three folds it was not trained. What the authors should decide: whether to add to the Table B1 note that the TabNet reference did not train on folds 1 and 2 (training part smaller than its batch size with drop_last), or to blank that rank.

No other FAIL or UNCERTAIN verdicts were reached.

---

## Observations (no verdict change; reported for completeness)

- **O1 (Q5).** The refit is a new run: same initial weights and same seeded generator as stage A (E3), but a different permutation (1,710 rows against 1,356) and, for FT-Transformer, 27 instead of 22 batches per epoch, so 297 optimizer steps at the transferred best_epoch 11 against 242 in stage A. All documents describe a refit "for that number of epochs"; none states that the step count differs. Nothing to change in the numbers.
- **O2 (Q7).** Section 4.4: "With its default batch size of 1,024 on 1,710 oversampled training rows, TabNet saw a single batch per epoch and stopped after five epochs." The refit did run five epochs on 1,710 rows; the early stopping ran 21 epochs on the 1,356-row inner-train and selected epoch 5 (`t21_run_meta.json`: stage_a_epochs_run 21, best_epoch 5). "Stopped after five epochs" reads as early stopping at epoch 5. A wording such as "the epoch count selected on the validation block was five, so the refit performed five optimizer steps" would match the record.
- **O3 (Q9).** Section 4.4 reports for TabNet "McNemar p = 0.0006, Holm-adjusted 0.0025" inside a paragraph and a table (Table A) that present seven comparisons with Random Forest; the Holm family is the 4 new models (em:688-689, `t21_comparisons_table.md`). With a family of 7 the adjusted value would be 0.0044 (7 x 0.000633), still below 0.05; the family size is not stated in the manuscript text.
- **O4 (Q12).** `hs` docstring line 16 says the TabNet grid has "drop_last False explicit"; the grid dictionaries (hs:126-130) and the five TabNet entries of `t22_search_space.json` carry no `drop_last` key (R: keys `batch_size, lr, n_d, n_steps, virtual_batch_size`), while the reference entry carries `drop_last: true`. The value False is applied by `tabnet_kwargs` (hs:274, `cfg.get("drop_last", False)`), so the behaviour is deterministic and the tuned TabNet trains on all 1,710 rows (14 batches of 128, E2), but a reader of the json cannot see it.
- **O5 (Q18).** `torch==2.11.0` in `requirements.txt` against the installed `2.11.0+cpu`; the local label cannot be pinned from PyPI and the file's comment names the CPU wheel. No action needed beyond the comment that is already there.
- **O6.** `em.make_tree("rf", seed)` ignores `seed` and returns the canonical `build_models()["rf"]` (random_state 42) (em:276-277). RF is never called with another seed in `extra_models.py` (not in `NEW_MODELS`), and `hs.build_tabular` sets `random_state=seed` explicitly (hs:257), so no output is affected.
- **O7.** The paired CIs of T2.1 / T2.2 and the marginal CIs come from the two different skip rules of F3.2; with 0 absent-class resamples on this test set they coincide (R). The 0-skip check is asserted at run time (em:642, :665; hs:599, :639).
- **O8.** `t22_comparisons.csv` holds 54 rows: comparison (b) for RF duplicates comparison (a) (own reference of RF is the canonical RF), and (c) is skipped for RF (hs:633-634). All 54 reproduce (R4).
- **O9.** In `--mode seeds` of both scripts the test set is evaluated 5 times per model (20 runs in T2.1, 35 in T2.2); the outputs are declared informative and non-canonical and no selection is made on them. The manuscript reports the seed ranges as such.

---

## Summary

T2 tally: 18 / 18 yes; Finding F5.1 (FAIL, documentation of the LightGBM winner) attached to Q15; Finding F5.2 (UNCERTAIN, TabNet reference rank) attached to Q13 / Q17; observations on Q5, Q7, Q9, Q12, Q18.

T3 verdict: PASS, 121 / 121 checks; every T2.1 and T2.2 figure named in the brief reproduces at 4 decimals from the prediction files and the regenerated bootstrap matrix; the independent LightGBM default refit from the snapshot reproduces `t21_predictions_lgbm.csv` on all 192 days; the CatBoost default refit through `fit_eval_model` reproduces `t21_predictions_catboost.csv` (E6).

Findings in one sentence each:

- F5.1 (FAIL): the manuscript and the register describe the tuned LightGBM configuration without the fixed `subsample_freq = 1` that every sampled configuration used, and the configuration as described gives test macro F1 0.4965 instead of the reported 0.5739.
- F5.2 (UNCERTAIN): the TabNet reference row of the CV table, printed as "6 of 6" in Appendix Table B1, was not trained on folds 1 and 2 (0 batches per epoch with batch 1,024 and drop_last on 669 / 1,008 rows), a caveat present in the register but not in the manuscript.

Commit command (executed as the last step of the audit; no push):

```
git add revision/audit/AUDIT_F5_models_search.md revision/audit/recompute_f5.py revision/audit/recompute_f5_output.txt revision/audit/_check_f5_empirical.py revision/audit/_check_f5_empirical_output.txt && git commit -m "audit F5: additional models and hyperparameter search"
```
