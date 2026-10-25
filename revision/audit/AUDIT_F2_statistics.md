# AUDIT F2: extended metrics (T1.1) and statistical comparisons (T1.2, T1.2b)

Auditor role: external, read-and-report only. Branch `audit-r1`, HEAD `3ba0c41` (slice F1 committed). Run date: 2026-10-25.
Environment used for every command below: Python 3.11.9, scikit-learn 1.8.0, numpy 2.4.3, pandas 2.3.3, scipy 1.17.1, statsmodels 0.15.0 (the versions recorded in `revision/results/t12_run_meta.json` for the original run are identical where listed).

Slice F1 was read only for its Findings section; none of its conclusions is reused here as a fact. Its finding F1 (persistence = last labelled row, not calendar day t-1) does not change any figure in this slice because the test window has one gap row where both conventions coincide; it is not re-examined.

Files written by this audit (nothing else was modified):

- `revision/audit/AUDIT_F2_statistics.md` (this report)
- `revision/audit/recompute_f2.py` + `recompute_f2_output.txt` (T3; imports only numpy, pandas, scipy, sklearn, statsmodels; nothing from `revision/scripts/`)
- `revision/audit/_check_f2_empirical.py` + `_check_f2_empirical_output.txt` (T2 evidence; imports `extended_metrics.py` and `statistical_tests.py` to call their functions on synthetic inputs; no writes)

All files LF. Total execution time of the two scripts: under one minute.

---

## T0. Inventory

`git status --short` at the start is unchanged from slice F1 (one deleted tracked zip, three untracked paths; none is an input of this slice). Size in bytes, sha256 first 16 hex chars:

| File | Size | sha256[:16] |
|---|---:|---|
| revision/scripts/extended_metrics.py | 16434 | 02c78dcf65c845fd |
| revision/scripts/statistical_tests.py | 18470 | c8506d2cdc5e21fa |
| revision/scripts/test_extended_metrics.py | 2268 | 1a23c61954fde881 |
| revision/results/t11_extended_metrics.csv | 3999 | ba6d3a4f61092517 |
| revision/results/t11_extended_metrics_table.md | 1873 | 64d914804ca552e5 |
| revision/results/t12_mcnemar.csv | 514 | ef5a9d7791335e45 |
| revision/results/t12_mcnemar_table.md | 1124 | b5efe526dc0274dc |
| revision/results/t12_paired_bootstrap.csv | 1136 | 796e50fc37085517 |
| revision/results/t12_paired_bootstrap_table.md | 1802 | c070fdf98e375c4c |
| revision/results/t12_cochran_q.csv | 64 | 13c8c0f2cf66c904 |
| revision/results/t12_run_meta.json | 961 | d0c9beba6fb054f0 |
| revision/results/test_predictions_rf.csv | 14349 | aea39c6dd2989e24 |
| revision/results/test_predictions_xgb.csv | 9537 | e0afa80d26440534 |
| revision/results/test_predictions_svm.csv | 14764 | c48625b27debfcd2 |
| revision/results/test_predictions_persistence.csv | 5421 | ab228a11e65bfe17 |
| revision/results/test_predictions_majority.csv | 5421 | eaf03527561e10fa |
| revision/results/bootstrap_indices_1000x192_seed42.npy | 1536128 | fff3b2834c061962 |
| revision/results/models/rf_revision.joblib | 1810593 | b8113662207ab548 |
| revision/results/models/xgb_revision.joblib | 751753 | 268760acaf1055c5 |
| revision/results/models/svm_revision.joblib | 213939 | 811a86c96f08a586 |

The four prediction csv files and the index matrix have the sha256 recorded in `t12_run_meta.json` `inputs_sha256` (full hashes compared: identical), so the inputs audited here are the inputs of the original T1.2 run. The three joblib models were loaded only to read `classes_` (Q4); no prediction was recomputed from them.

---

## T1. Trace table (from reading the code)

Line numbers refer to the files at HEAD `3ba0c41`. "Hard" = `y_pred` column; "proba" = the three columns `p_rest, p_normal, p_active` read in that order (`extended_metrics.py:63`, `:163`). `y_true` is the `y_true` column, checked identical across files (`extended_metrics.py:158-161`, `statistical_tests.py:196-200`). All metrics are on the 192 test rows.

| # | Metric / test | Exact definition in code | Inputs | Classes absent from a resample | What is reported |
|---|---|---|---|---|---|
| 1 | Balanced accuracy | `extended_metrics.py:107` `balanced_accuracy_score(y, p)`; no `labels` argument (sklearn has none) | hard (`:162`) vs `y_true` | sklearn drops a class with 0 true samples (warning) instead of fixing labels; never triggered (0 of 1000 resamples lack a class, `:259`) | point, percentile CI [2.5, 97.5] (`:212`), `n_resamples_used`, note |
| 2 | Cohen's kappa | `:108` `cohen_kappa_score(y, p, labels=[0,1,2])`, `weights=None` | hard | labels fixed; an absent class gives an all-zero row | same |
| 3 | G-mean | `:85-101` manual: `confusion_matrix(labels=[0,1,2])`, `recall_c = diag/row sum`, `prod(recalls) ** (1/3)` | hard | returns `nan` when any class has support 0 (`:98-99`); `bootstrap()` skips `nan` and counts it (`:209-213`) | same; skip count in note (`:272-273`) |
| 4 | ROC-AUC OvR macro | `:119-122` `roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=[0,1,2])` | `y_true` + proba (3 columns, row sums checked at 1e-6, `:164-167`) | `nan` when `len(unique(y)) < 3` (`:120-121`), skipped by `bootstrap()` | same; RF/XGB/SVM only; persistence/majority rows `n/a: hard labels only` (`:289-291`) |
| 5 | PR-AUC macro | `:124-128` `average_precision_score(label_binarize(y, classes=[0,1,2]), proba, average="macro")`, i.e. macro average precision, not the area under an interpolated PR curve | `y_true` + proba | `nan` when a class is absent (`:125-126`), skipped | same; RF/XGB/SVM only |
| 6 | F1 macro / accuracy / MCC / QWK (sanity) | `:110-113` `f1_score(labels=[0,1,2], average="macro", zero_division=0)`, `accuracy_score`, `matthews_corrcoef`, `cohen_kappa_score(labels, weights="quadratic")` | hard | labels fixed for F1 and QWK; not for accuracy / MCC (not needed) | same, note `canonical, sanity check` |
| 7 | Bootstrap matrix | `:183-192` `default_rng(42).integers(0, 192, size=(1000, 192))`; if the npy exists it must be identical or the script exits | | | saved once; reused by every metric and system (`:258`, `:269`, `:282`) |
| 8 | Persistence / majority predictions | `:174` `[last_train_label()] + y_true[:-1]`, checked equal to `test_predictions_persistence.csv` (`:176-178`); majority `np.full(192, 1)` (`:175`) | `y_true`, last train label from the snapshot (`:134-142`) | | hard-label metrics only |
| 9 | McNemar counts | `statistical_tests.py:104-117`: `a = pred_a == y`, `b = pred_b == y`; `n11 = a&b`, `n01 = a&~b`, `n10 = ~a&b`, `n00 = ~a&~b` | hard (`:201`) for A and B | n/a (no resampling) | `n00, n01, n10, n11` per pair (`:263`) |
| 10 | McNemar tests | `:120-130` `mcnemar([[n11, n01], [n10, n00]], exact=False, correction=True)` and `mcnemar(..., exact=True)` (statsmodels 0.15.0) | counts from row 9 | | `chi2_cc, p_cc, p_exact`; `recommended = "exact"` iff `n01 + n10 < 25` (`:79`, `:261-266`) |
| 11 | Holm | `:270-272` `multipletests(method="holm")` on the first 3 rows (RF-XGB, RF-SVM, XGB-SVM), separately for `p_cc` and `p_exact`; RF-persistence left `nan` (`:265`) | p-values of row 10 | | `p_cc_holm, p_exact_holm`; `n/a` in the md table (`:229`) |
| 12 | Paired bootstrap | `:133-169`: for each row of the matrix, `yt = y[row]`, `ma = metric(yt, pred_a[row])`, `mb = metric(yt, pred_b[row])`, `diff = ma - mb` (`:150-159`); metrics `:83-86` (F1 macro with `labels=[0,1,2], zero_division=0`; `balanced_accuracy_score`) | hard for A and B; matrix loaded and checked equal to the seed-42 stream (`:205-211`) | row skipped and counted when `len(unique(yt)) < 3` (`:152-154`) | `diff_point` on the full set, CI [2.5, 97.5] of `diff`, `prop_diff_gt_0`, `n_skipped`, marginal CIs of A and B on the same resamples, `n_used` (`:164-169`, `:288-289`) |
| 13 | Cochran's Q | `:172-177` `cochrans_q(X)` with `X = column_stack([pred == y for RF, XGB, SVM]).astype(int)`, shape 192 x 3 | hard | n/a | `statistic, df, p, n_models, n_days` |
| 14 | Built-in sanity checks | `:292-301` marginal CIs and point differences compared at 4 decimals with hard-coded `CANON_POINT` / `T11_CI` (`:91-100`); `:268` cell sum = 192; `:273-275` p in [0, 1]; `:286` zero warnings in the loop; `:290` zero skips | | | script exits 1 on any failure (`:367-369`); `t12_run_meta.json` lists `sanity_checks_failed: []` |

No UNCERTAIN cells.

---

## T2. Targeted questions

Evidence lines marked (R) are in `recompute_f2_output.txt`, (E) in `_check_f2_empirical_output.txt`.

**Q1. Balanced accuracy = mean of the 3 class recalls with fixed labels; equals sklearn?** PASS.
`extended_metrics.py:107` calls `sklearn.metrics.balanced_accuracy_score`, which has no `labels` argument. On the full test set and on every one of the 1000 resamples the value equals the mean of `recall_score(labels=[0,1,2], average=None)` exactly (max absolute difference 0.0e+00, E line 1; manual mean recall 0.5227272727 = sklearn, R "definition cross-checks"). The equality holds because no resample lacks a class (R: "resamples with a class absent from y_true: 0"). Observation: on a synthetic input where Active is absent from `y_true` but predicted, sklearn returns 0.5 with a warning (it drops the class), whereas `gmean()` returns `nan` and is skipped (E line 2); the two conventions differ only in a case that does not occur. See Observations.

**Q2. Kappa is unweighted and never confused with QWK 0.2551?** PASS.
`extended_metrics.py:108` uses `weights=None` (default); QWK is a separate entry at `:113` with `weights="quadratic"`. `label_metrics()["cohen_kappa"]` on RF gives 0.141524 and `["qwk"]` gives 0.255061, equal to the two sklearn calls (E line 3). The csv has distinct rows `rf,cohen_kappa,0.1415241058` and `rf,qwk,0.2550607287`; the register row T1.1 (`CIFRE_REVIZIE.md:22`) says "QWK canonic 0.2551 ramane"; the manuscript Section 3.6 lists "Cohen's kappa" among the added metrics and QWK among the canonical ones.

**Q3. G-mean = (recall_R x recall_N x recall_A)^(1/3); behaviour with an absent class; what the bootstrap does.** PASS.
Definition at `extended_metrics.py:96-101`, labels fixed by `confusion_matrix(labels=[0,1,2])`. With a class absent from `y_true` the function returns `nan` (`:98-99`; E line 4). `bootstrap()` (`:204-213`) drops `nan` values and counts them: on 3 real rows plus one synthetic row without Active it reports `used 3, skipped 1, warnings 0` (E line 5). On the real matrix 0 resamples are skipped (csv `n_resamples_used = 1000` for every row). Recomputed RF G-mean 0.492306 [0.296015, 0.622836] (R).

**Q4. ROC-AUC one-vs-rest macro; `label_binarize(classes=[0,1,2])` in the same order as the proba columns; column order = `classes_`?** PASS.
`extended_metrics.py:122` `roc_auc_score(multi_class="ovr", average="macro", labels=LABELS)`; `:127` `label_binarize(y, classes=LABELS)`; `LABELS = [0, 1, 2]` (`:61`) and `PROBA_COLS = ["p_rest", "p_normal", "p_active"]` (`:63`). Empirically: `classes_` of the three saved models (`revision/results/models/{rf,xgb,svm}_revision.joblib`) are `[0 1 2]`; `save_test_predictions.py:71-75` asserts `classes_ == [0, 1, 2]` and writes `proba[:, 0/1/2]` to `p_rest/p_normal/p_active`; `argmax(proba) == y_pred` on 100% of days for RF and XGB (E line 7), which ties the column order to the hard labels; the manual per-class OvR mean equals the sklearn value 0.5580748399 (R); swapping `p_normal` and `p_active` changes the macro ROC-AUC to 0.500737 (R), so the claimed 0.5581 is obtained only with the csv order.

**Q5. PR-AUC: average precision or interpolated area; named correctly?** UNCERTAIN (naming; value correct).
`extended_metrics.py:128` is `average_precision_score(..., average="macro")`, i.e. the macro mean of per-class average precision (AP). The script docstring (`:14`) says so; the csv column is `pr_auc_macro`, the markdown table header is "PR-AUC (macro)" (`:76`), the register (`CIFRE_REVIZIE.md:25`) says "RF macro PR-AUC", and `KBS_Manuscript_R1.docx` Section 3.6 says "macro PR-AUC"; the strings "average precision" / "average_precision" do not occur in the manuscript, supplementary or response docx. AP and the trapezoidal area under the interpolated PR curve are different estimators: on the same inputs the trapezoidal macro area is 0.491295 (RF), 0.446418 (XGB), 0.451757 (SVM) versus AP 0.498304, 0.459804, 0.460286 (R, diagnostic line). A reader who reproduces "PR-AUC" with `precision_recall_curve` + `auc` will get values that differ at the second decimal. See Findings F2.1.

**Q6. SVM: hard metrics from `predict()`, proba metrics from `p_*`; McNemar on `y_pred`; what changes with argmax(proba)?** PASS.
Hard-label metrics use `df["y_pred"]` (`extended_metrics.py:162`), probability metrics use the three `p_*` columns (`:163`); McNemar and the paired bootstrap use `df["y_pred"]` (`statistical_tests.py:201`, `:256`, `:283`). The choice (predict, as in the canonical script) is declared in `extended_metrics.py:17-18` and in the T0.2 note (`CIFRE_REVIZIE.md:327`). Agreement argmax(proba) vs predict: RF 1.0000, XGB 1.0000, SVM 0.9115 (175 of 192 days; E line 7), matching the claimed 0.9115. Both conventions computed (E lines 8-10):

| Pair | n01 / n10 with `y_pred` (predict) | p_cc | n01 / n10 with argmax(proba) for SVM | p_cc |
|---|---|---|---|---|
| RF-SVM | 22 / 20 | 0.8774 | 17 / 23 | 0.4292 |
| XGB-SVM | 27 / 16 | 0.1273 | 19 / 16 | 0.7353 |

SVM per-day accuracy would be 0.6719 under argmax(proba) versus 0.6302 under predict (17 days differ). Neither convention changes the conclusion (no pair significant at 0.05), but the counts and p-values are convention-dependent; the reported ones are the declared convention.

**Q7. n01 = A correct and B wrong, n10 the reverse; statsmodels table positions; direct count for RF-XGB.** PASS.
`statistical_tests.py:114-115`: `n01 = a & ~b`, `n10 = ~a & b` with `a` = A correct. Direct count from the two csv files: RF correct and XGB wrong on 15 days, RF wrong and XGB correct on 24, both correct 108, both wrong 45 (R, E line 11), identical to `t12_mcnemar.csv` row RF-XGB (`n00 45, n01 15, n10 24, n11 108`). The table passed to statsmodels is `[[n11, n01], [n10, n00]]` (`:126`), so `table[0,1] = n01` and `table[1,0] = n10`, which are the two cells statsmodels uses; transposing the two discordant cells gives the same statistic and p (R: "with n01/n10 swapped 0.200185"), so the column labels in the csv are what carries the direction, and they match the direct count.

**Q8. Exact variant = two-sided binomial on min(n01, n10) out of n01 + n10; "exact if discordant < 25" applied correctly?** PASS.
`:128` `mcnemar(table, exact=True)` (statsmodels: `2 * binom.cdf(min(n01, n10), n01 + n10, 0.5)`, capped at 1). Recomputed with `scipy.stats.binomtest(min, n, 0.5, alternative="two-sided")`: 0.199591 (RF-XGB), 0.877614 (RF-SVM), 0.126289 (XGB-SVM), 0.128396 (RF-persistence), all equal to `p_exact` at 4 decimals (R). The rule is `DISCORDANT_MIN_FOR_CC = 25` (`:79`) with `recommended = "exact" if n01 + n10 < 25 else "cc"` (`:266`); the discordant counts are 39, 42, 43, 85, so all four rows carry `cc`, as recomputed (R "recommended = cc").

**Q9. Holm recomputed by hand on the 3 p_cc.** PASS.
Sorted p_cc: 0.127263 (XGB-SVM) x 3 = 0.381788; 0.200185 (RF-XGB) x 2 = 0.400370; 0.877371 (RF-SVM) x 1 = 0.877371; already monotone, none above 1. Manual values equal `p_cc_holm` in the csv (0.3818, 0.4004, 0.8774) and `multipletests(method="holm")`; same for `p_exact_holm` (0.3789, 0.3992, 0.8776) (R "Holm (manual)").

**Q10. Cochran's Q on the 192 x 3 correct/incorrect matrix, df 2, direct formula.** PASS.
`statistical_tests.py:174` builds the 192 x 3 matrix of `pred == y_true` for RF, XGB, SVM (column totals 123 / 132 / 121 correct days). Direct formula Q = (k-1)[k sum(C_j^2) - N^2] / [k N - sum(R_i^2)] with k = 3 gives 3.322581, p = chi2.sf(Q, 2) = 0.189894, df = 2; csv: 3.322580645, 2, 0.1898937977 (R). statsmodels `cochrans_q` agrees.

**Q11. Same index row applied to y_true, y_pred_A and y_pred_B; sign = A minus B; marginals = T1.1 at 4 decimals.** PASS.
`statistical_tests.py:150-156`: one `row` indexes `y_true`, `pred_a` and `pred_b`; `:159` `diffs.append(ma - mb)`; the csv label is `f"{LABEL[a]}-{LABEL[b]}"` (`:284`) with `a` the first element of the pair, so "RF-XGB" means RF minus XGB. Swapping the arguments negates the point difference and mirrors the CI exactly (E line 12). Recomputed marginal CIs on the same matrix equal the T1.1 rows at 4 decimals for RF, XGB and persistence on F1 macro and balanced accuracy (R, 16 "marginal" lines); in the csv they are identical to 10 significant digits.

**Q12. Are the paired CIs declared unadjusted for multiple comparisons, and where?** PASS (with a wording observation).
Manuscript `KBS_Manuscript_R1.docx` Section 3.6, last sentence: "Intervals are not adjusted for multiple comparisons unless stated." (source `F4_manuscript_changes.md:39`). `t12_paired_bootstrap_table.md:12` (written at `statistical_tests.py:336-337`): "RF-persistence (T1.2b) is a baseline comparison, outside the Holm family of 3 model pairs; no multiplicity adjustment applies to CIs." Supplementary S1 states that the persistence comparison "is unadjusted". There is no column in `t12_paired_bootstrap.csv`, and the register rows for the three model pairs (`CIFRE_REVIZIE.md:53-58`) do not carry the word; the T2.1 / T2.2 / T1.5 rows do ("CI-uri neajustate"). The phrase "no multiplicity adjustment applies to CIs" can be read as "none is needed" rather than "none was applied"; the manuscript sentence is unambiguous.

**Q13. Anything computed on something other than the 192 test days?** PASS.
`extended_metrics.py` reads the snapshot only in `last_train_label()` (`:134-142`) to obtain the label of train row 766 (= 1, Normal), used as `pred[0]` of persistence (`:174`); 958 and 766 appear only as size assertions (`:140`). `statistical_tests.py` reads the four prediction csv files and the index matrix (`:190-211`) and nothing else (E last line). No threshold, probability or statistic is fitted or computed on train rows in either script. (Probabilities themselves come from `save_test_predictions.py`, audited in slice F1.)

**Q14. Persistence equalities (balanced accuracy == F1 macro; kappa == MCC) explained by prediction multiset == truth multiset, counted directly.** PASS.
Predicted counts of persistence are [44, 140, 8] = true support [44, 140, 8] (E line 13; R last line). The last train label is 1 and the last test label is 1, so the shift drops and adds the same label (E line 14). Per-class precision [0.204545, 0.707143, 0.0] equals per-class recall (E line 15), hence F1 per class = recall and macro F1 = balanced accuracy = 0.303896 (R). Kappa -0.0566037736 = MCC -0.0566037736 (E line 16): with equal marginals the two formulas reduce to the same expression. The identity holds in only 1 of the 1000 resamples (E line 17), which is why the two CIs differ ([0.2602, 0.3520] vs [0.2612, 0.3501]), as the register note says.

Summary: 13 PASS, 0 FAIL, 1 UNCERTAIN (Q5).

---

## T3. Independent recomputation

`revision/audit/recompute_f2.py` (output in `recompute_f2_output.txt`), imports only numpy, pandas, scipy, sklearn, statsmodels. The bootstrap matrix is regenerated with `np.random.default_rng(42).integers(0, 192, size=(1000, 192))` and found byte-equal to the saved npy (int64, 1000 x 192). The metrics are implemented from their definitions (mean recall from a fixed-label confusion matrix; kappa from observed and expected agreement; G-mean as the cube root of the recall product; ROC-AUC and AP as the mean of three binary one-vs-rest scores), McNemar from the direct formula `(|n01 - n10| - 1)^2 / (n01 + n10)` with `scipy.stats.chi2.sf` and `scipy.stats.binomtest`, Holm by sorting and step-down by hand, Cochran's Q from the column and row totals, and the paired bootstrap with one index row per resample.

| Item | Recomputed | Claimed | Match (4 dp) |
|---|---|---|---|
| RF balanced accuracy | 0.522727 [0.385100, 0.663362] | 0.5227 [0.3851, 0.6634] | yes |
| RF Cohen's kappa | 0.141524 [-0.004983, 0.292407] | 0.1415 [-0.0050, 0.2924] | yes |
| RF G-mean | 0.492306 [0.296015, 0.622836] | 0.4923 [0.2960, 0.6228] | yes |
| RF ROC-AUC OvR macro | 0.558075 [0.419871, 0.692052] | 0.5581 [0.4199, 0.6921] | yes |
| RF PR-AUC macro (AP) | 0.498304 [0.371572, 0.635876] | 0.4983 [0.3716, 0.6359] | yes |
| McNemar RF-XGB n01 / n10 / chi2_cc / p_cc / p_exact | 15 / 24 / 1.641026 / 0.200185 / 0.199591 | 15 / 24 / 1.6410 / 0.2002 / 0.1996 | yes |
| McNemar RF-SVM | 22 / 20 / 0.023810 / 0.877371 / 0.877614 | 22 / 20 / 0.0238 / 0.8774 / 0.8776 | yes |
| McNemar XGB-SVM | 27 / 16 / 2.325581 / 0.127263 / 0.126289 | 27 / 16 / 2.3256 / 0.1273 / 0.1263 | yes |
| McNemar RF-persistence | 50 / 35 / 2.305882 / 0.128885 / 0.128396 | 50 / 35 / 2.3059 / 0.1289 / 0.1284 | yes |
| Holm p_cc (RF-XGB, RF-SVM, XGB-SVM) | 0.400370, 0.877371, 0.381788 | 0.4004, 0.8774, 0.3818 | yes |
| Holm p_exact | 0.399182, 0.877614, 0.378868 | 0.3992, 0.8776, 0.3789 | yes |
| Cochran's Q / df / p | 3.322581 / 2 / 0.189894 | 3.3226 / 2 / 0.1899 | yes |
| Paired RF-XGB F1 macro | +0.030319 [-0.079601, 0.156117], prop > 0 0.684 | +0.0303 [-0.0796, 0.1561], 0.684 | yes |
| Paired RF-persistence F1 macro | +0.202428 [0.064213, 0.317328], 0.997 | +0.2024 [0.0642, 0.3173], 0.997 | yes |
| Paired RF-persistence balanced accuracy | +0.218831 [0.072825, 0.362959], 0.997 | +0.2188 [0.0728, 0.3630], 0.997 | yes |
| Marginal CIs inside the paired runs (RF, XGB, persistence; both metrics) | equal to T1.1 rows | | yes (16 of 16) |
| Persistence bal. acc / F1 / kappa / G-mean | 0.303896 / 0.303896 / -0.056604 / 0 | 0.3039 / 0.3039 / -0.0566 / 0 | yes |
| Majority bal. acc / kappa / G-mean | 0.333333 / 0 / 0 | 0.3333 / 0 / 0 | yes |

76 of 76 comparisons identical at 4 decimals; 0 resamples skipped for any metric. **T3 verdict: PASS.**

---

## Findings (FAIL or UNCERTAIN only)

### F2.1. "PR-AUC" is macro average precision, and no document names the estimator (Q5: UNCERTAIN)
Observed: `extended_metrics.py:128` computes `average_precision_score(..., average="macro")`. The script docstring (`:14`) says so; the csv column, the markdown table (`:76`), the register (`CIFRE_REVIZIE.md:25`) and the manuscript Section 3.6 (`KBS_Manuscript_R1.docx`, source `F4_manuscript_changes.md:39`) all say only "macro PR-AUC"; "average precision" appears nowhere in the three R1 docx files. The trapezoidal area under the interpolated PR curve, which is what `precision_recall_curve` + `auc` would give, is 0.4913 (RF), 0.4464 (XGB), 0.4518 (SVM) against the reported 0.4983, 0.4598, 0.4603 (diagnostic computed only to size the gap; not a proposed figure). Why it matters: the two estimators are both commonly called PR-AUC; a reader reproducing the number with the other one will find a discrepancy at the second decimal and may read it as an error. The values themselves are correct for the estimator used. What the authors should decide: whether to add "(average precision)" after "macro PR-AUC" in Section 3.6 and in the Supplementary table note, and the sklearn function name in the register row.

No other FAIL or UNCERTAIN verdicts were reached.

---

## Observations (no verdict change; reported for completeness)

- Q1: `balanced_accuracy_score` (`extended_metrics.py:107`) does not fix labels and silently drops a class absent from `y_true` (with a warning, which `bootstrap()` would count), while G-mean and the two AUCs return `nan` and are skipped. The docstring (`:25-26`) describes skipping only for AUC and G-mean, so this is documented behaviour; it has no effect because no resample lacks a class. If the matrix or test set ever changed so that a resample lacked Active, balanced accuracy, accuracy, MCC and kappa would be computed on that resample while G-mean and AUC would skip it, and the CIs would no longer be on the same resample set across metrics.
- Q6: the predict-versus-argmax convention for SVM changes the McNemar counts (RF-SVM 22/20 vs 17/23; XGB-SVM 27/16 vs 19/16) and SVM per-day accuracy (0.6302 vs 0.6719), not the conclusions. The convention is declared in the script and the register; it is not stated in the manuscript text that the SVM hard labels are `predict()` rather than the argmax of the Platt probabilities.
- Q12: the note in `t12_paired_bootstrap_table.md:12` ("no multiplicity adjustment applies to CIs") is ambiguous in isolation; the manuscript sentence in Section 3.6 is the clear statement.
- `statistical_tests.py:295-297` checks `round(diff_point, 4) == round(canon_a - canon_b, 4)` with canonical values already rounded to 4 decimals; this can fail by one unit in the fourth decimal without any error in the data. It passed in this run (all 8 differences), so it is a fragility of the check, not of the figures.

---

## Summary

- T2: Q1 PASS, Q2 PASS, Q3 PASS, Q4 PASS, Q5 UNCERTAIN, Q6 PASS, Q7 PASS, Q8 PASS, Q9 PASS, Q10 PASS, Q11 PASS, Q12 PASS, Q13 PASS, Q14 PASS.
- T3: PASS, 76 / 76 values identical at 4 decimals from an independent implementation on the csv files and the regenerated seed-42 matrix.
- Findings: F2.1, the reported "macro PR-AUC" is macro average precision and no document says so; the value is correct for that estimator, the trapezoidal alternative would differ at the second decimal.
