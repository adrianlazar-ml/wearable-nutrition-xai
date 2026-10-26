# rounding_pairs_check.md: the 19 cells corrected at #17 (commit 36647f5), read-only check (2026-10-26)

Criterion per pair: the old and new printed values differ by exactly one unit of the last printed decimal (ulp); the csv value at full precision lies within half an ulp of the new value (so the new value is the single rounding, half away from zero on the decimal value) and within one ulp of the old value (so the old value is the neighbour reached by rounding twice, through the 4-decimal register value that ends in 5). A pair that jumps more than one ulp is not a double rounding and is marked FAIL. Distances are |csv - old| / |csv - new|.

| Table | Row | Column | Old printed | New printed | csv (full precision) | distance old / new | old-new | tie at the rounding digit | Status | csv source | register / canonical value (4 dec.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Table 4 | QWK | XGBoost | 0.181 | 0.180 | 0.1804878049 | 0.0005121951 / 0.0004878049 | 1 ulp | no | OK | t11_extended_metrics.csv (xgb, qwk, value) | CIFRE_CANONICE sect. 10: XGB QWK 0,1805; v8 Table 4 printed 0.1805 |
| Table 4 | Cohen's kappa | SVM (RBF), CI low | -0.040 | -0.039 | -0.03949872469 | 0.00050127531 / 0.00049872469 | 1 ulp | no | OK | t11_extended_metrics.csv (svm, cohen_kappa, ci_low) | CIFRE_CANONICE:371 (Revision 1 block, source T1.1): -0.0395 |
| Table 4 | Macro ROC-AUC (OvR) | SVM (RBF) | 0.560 | 0.559 | 0.5594624823 | 0.0005375177 / 0.0004624823 | 1 ulp | no | OK | t11_extended_metrics.csv (svm, roc_auc_ovr_macro, value) | CIFRE_CANONICE:371 (T1.1): 0.5595 |
| Table 4 | Macro PR-AUC | SVM (RBF), CI low | 0.357 | 0.356 | 0.3564876078 | 0.0005123922 / 0.0004876078 | 1 ulp | no | OK | t11_extended_metrics.csv (svm, pr_auc_macro, ci_low) | CIFRE_CANONICE:371 (T1.1): 0.3565 |
| Table 6 | FT-Transformer | Macro F1 | 0.412 | 0.411 | 0.4114983514 | 0.0005016486 / 0.0004983514 | 1 ulp | no | OK | t21_extra_models.csv (ftt, f1_macro, value) | CIFRE_REVIZIE T2.1: 0.4115 |
| Table 6 | FT-Transformer | RF minus model, CI low | -0.009 | -0.008 | -0.008450188949 | 0.000549811051 / 0.000450188949 | 1 ulp | no | OK | t21_comparisons.csv (ftt, delta ci_low) | CIFRE_REVIZIE T2.1: -0.0085 |
| Table 6 | LightGBM | RF minus model, CI high | +0.234 | +0.233 | 0.2334955665 | 0.0005044335 / 0.0004955665 | 1 ulp | no | OK | t21_comparisons.csv (lgbm, delta ci_high) | CIFRE_REVIZIE T2.1: +0.2335 |
| Table 6 | TabNet | RF minus model, CI high | +0.335 | +0.334 | 0.3344904622 | 0.0005095378 / 0.0004904622 | 1 ulp | no | OK | t21_comparisons.csv (tabnet, delta ci_high) | CIFRE_REVIZIE T2.1: +0.3345; also 4.4 text |
| Table 8 | HRV | Difference, CI low | -0.042 | -0.041 | -0.04147742164 | 0.00052257836 / 0.00047742164 | 1 ulp | no | OK | t15_logo.csv (G3_hrv, delta_f1_macro_ci_low) | CIFRE_REVIZIE T1.5 (4 dec.) |
| Table 8 | Sleep architecture | Difference, CI low | -0.080 | -0.079 | -0.07946375204 | 0.00053624796 / 0.00046375204 | 1 ulp | no | OK | t15_logo.csv (G4, delta_f1_macro_ci_low) | CIFRE_REVIZIE T1.5 (4 dec.) |
| Table 10 | Mean 24-h heart rate | Rest, median | 91.9 | 92.0 | 91.95 | 0.05 / 0.05 | 1 ulp | tie | OK | t32_descriptives.csv (aw_hr_mean, Rest, median) | CIFRE_REVIZIE T3.2: 91.95; also 4.7 text |
| Table 10 | SpO2 | Cliff's delta, CI low | -0.06 | -0.05 | -0.05475247525 | 0.00524752475 / 0.00475247525 | 1 ulp | no | OK | t32_tests.csv (aw_spo2, cliff_active_rest_ci_low) | CIFRE_REVIZIE T3.2: -0.055 |
| Table 11 | Dec 2023 to Jul 2024 | Macro F1, CI low | 0.310 | 0.309 | 0.3094854232 | 0.0005145768 / 0.0004854232 | 1 ulp | no | OK | t33_expanding_window.csv (fold 4, f1_macro_ci_low) | audit F6.1; CIFRE_REVIZIE T3.3: 0.3095 |
| Table 11 | Dec 2023 to Jul 2024 | Persistence, CI low | 0.274 | 0.273 | 0.27348167 | 0.00051833 / 0.00048167 | 1 ulp | no | OK | t33_expanding_window.csv (fold 4, pers_f1_macro_ci_low) | audit F6.1; CIFRE_REVIZIE T3.3: 0.2735 |
| Table A.1 | Random Forest | Seed range, min | 0.509 | 0.508 | 0.5084580174 | 0.0005419826 / 0.0004580174 | 1 ulp | no | OK | t22_seeds.csv (rf, min f1_macro over seeds) | CIFRE_REVIZIE T2.2: 0.5085; also 4.4 text |
| Table A.1 | SVM (RBF) | CV macro F1, sd | 0.018 | 0.017 | 0.01747081337 | 0.00052918663 / 0.00047081337 | 1 ulp | no | OK | t22_test_results.csv (svm, cv_f1_sd) | CIFRE_REVIZIE T2.2: 0.0175 |
| Table A.1 | LightGBM | CV macro F1, sd | 0.046 | 0.045 | 0.04549755067 | 0.00050244933 / 0.00049755067 | 1 ulp | no | OK | t22_test_results.csv (lgbm, cv_f1_sd) | CIFRE_REVIZIE T2.2: 0.0455 |
| Table A.1 | LightGBM | Test macro F1, CI low | 0.417 | 0.416 | 0.4164888528 | 0.0005111472 / 0.0004888528 | 1 ulp | no | OK | t22_test_results.csv (lgbm, f1_macro_ci_low) | CIFRE_REVIZIE T2.2: 0.4165; also 4.4 text |
| Table A.1 | FT-Transformer | Tuned minus reported RF | -0.137 | -0.136 | -0.136489392 | 0.000510608 / 0.000489392 | 1 ulp | no | OK | t22_comparisons.csv (ftt, a, diff_point) | CIFRE_REVIZIE T2.2: -0.1365 |

Pairs: 19 OK, 0 FAIL out of 19.

## Ties at the rounding digit (point 4)

Exactly one of the 19 csv values is a tie at the printed precision: Table 10 Mean 24-h heart rate / Rest, median: csv 91.95, half-up 92.0, half-even 92.0 (both give the new value; the digit before the tie is 9, odd, so half-even also rounds up). The other 18 csv values are not ties; their 4-decimal register renderings (0.1805, 0.4115, 0.3095, ...) are ties at the third decimal, which is why rounding the register value again gave the old printed values. The checker rounds the csv value, not the register value. In the Supplementary, three exact .5 ties (S3: 682.5, 3,220.5, 2,642.5) are printed half-even by the producer's format() (682, 3,220, 2,642) and were accepted, not changed.

## Table 4 pairs versus CIFRE_CANONICE.md (point 3)

- QWK XGBoost: CIFRE_CANONICE section 10 gives 0,1805 and the submitted v8 Table 4 printed 0.1805 (four decimals). The F4 rewrite of Table 4 (T4.1) printed 0.181, the half-up rounding of the canonical four-decimal figure; the csv value 0.180488 rounds once to 0.180. The old value does not coincide with the canonical figure (which has four decimals) but is derived from it. This is the one pair where "cite the canonical figure" and "round the csv once" give different third decimals; listed separately below as "de revenit (decizie)".
- Cohen's kappa SVM CI low (-0.040), macro ROC-AUC SVM (0.560), macro PR-AUC SVM CI low (0.357): T1.1 metrics added in revision; they appear in CIFRE_CANONICE.md only in the appended Revision 1 block (line 371, source T1.1) at four decimals (-0.0395, 0.5595, 0.3565), the same values as the CIFRE_REVIZIE T1.1 rows. The old printed values are half-up roundings of those four-decimal figures; the csv values round once to the new values. No pre-revision canonical figure is involved.
- RF macro F1 CI [0.373, 0.610] and MCC CI [-0.0045, 0.2843] (Table 5, canonical, independent resamples): Table 5 is a base table outside the rounding checker and was not touched. Current Table 5 rows: Macro F1 | 0.5063 | [0.373, 0.610] // Accuracy | 0.641 | [0.573, 0.708] // MCC | 0.1416 | [-0.0045, 0.2843] // F1 Active | 0.444 | [0.111, 0.700]. The T1.1 shared-matrix interval [0.3784, 0.6130] is printed 0.378 / 0.613 in Tables 6, 9 and 11, as before the audit application; the rule of the T1.1 note (canonical CIs for the canonical figures in Table 5, shared-matrix CIs for the revision tables) is respected.

### De revenit (decizie, nu corectie automata)

| Table | Cell | Printed now | Alternative | Reason |
|---|---|---|---|---|
| Table 4 | QWK, XGBoost | 0.180 (csv 0.180488 rounded once) | 0.181 (canonical 0,1805 rounded half-up) or 0.1805 (four decimals, as in v8 Table 4) | the cell cites a canonical figure of CIFRE_CANONICE section 10; the rest of Table 4 prints three decimals; _check_t41_numbers.py accepts 0.180 and 0.181 against 0.1805 (tolerance half a unit) |

## Point 5: goal labels in 3.12

3.12 reads "The declared goal (fat loss, maintenance or muscle gain) is recorded in the user profile and does not modify this target." streamlit_app_v2.py lines 217-220:

```
        p["goal"] = st.selectbox(
            "Obiectiv", ["fat_loss", "maintenance", "muscle_gain"],
            index=["fat_loss", "maintenance", "muscle_gain"].index(p["goal"]),
        )
```

The three labels are the selectbox options of line 218 (default profile value "fat_loss", nutrition_recommender.py line 27, USER_PROFILE). Confirmed.

## Summary

Pairs OK: 19; FAIL: 0; de revenit (decision): 1 (Table 4 QWK XGBoost: 0.180 from the csv versus 0.181 from the canonical 0,1805). All 19 corrections are one-ulp double roundings; none jumps further than one unit. Verdict: merge possible; the QWK cell is a wording-of-canonical-figure decision, revertable by one F4 edit and a rerun, before or after the merge.
