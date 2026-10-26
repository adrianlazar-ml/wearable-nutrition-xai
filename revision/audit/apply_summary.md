# apply_summary.md: application of AUDIT_DECISIONS #1-#27 (branch apply-audit, 2026-10-26)

## Parameters

| Parameter | Value | Basis |
|---|---|---|
| P1 | A (text repositioning only) | no row "Cliff aw_hr_mean on the minutes without ring samples" exists in CIFRE_REVIZIE; by the brief's own rule P1 = A; confirmed by the first author at the start of the task |
| P2 | DA, code-faithful wording | decision of the first author at the start of the task. The code shows that `goal` is a profile field (nutrition_recommender.py) and an interface selector (streamlit_app_v2.py 217-219) that changes neither the caloric target nor the displayed TDEE (calculate_tdee uses activity_level); 3.12 now says the declared goal is recorded in the profile and does not modify the target, not the brief's "changes only the displayed TDEE" |
| P3 | 3 deterministic rules (one per predicted state) | read from 3.12 after GOAL_FIX: the per-state caloric offset and macronutrient split (Rest / Normal / Active), plus the confidence rule stated separately |
| P4 | the AI4Food reference intake of 25 kcal/kg/day (Lacruz-Pleguezuelos et al., 2025) and the macronutrient guidelines cited in 3.12 | read from 3.12; the Introduction sentence cites the AI4Food reference by name and refers to the other guidelines as "cited there" (Table 2 is not cited in the Introduction, to keep the order of first citation) |
| P5 | DA | confirmed by the first author; the canonical Random Forest (100 / 10 / 5) stays the reported model, no canonical figure changed |

## Entries

| # | Status | Commit(s) | Note |
|---|---|---|---|
| 1 | aplicat | T3 a9c0fb3; R2.b T5 b47169d | 3.3 lag convention (12 gap pairs), 3.5 persistence = most recent labeled day, 191 / 192 |
| 2 | aplicat | T3 a9c0fb3; R2.b T5 b47169d | "coincides on every day of the test set with the label rule applied to two of the nineteen features" (3.5, 3.9, 4.6, R2.b) |
| 3 | aplicat | T3 a9c0fb3; register T6 84b50f2 | "macro PR-AUC (average precision)" in 3.6; estimator named in the T1.1 row; no PR-AUC in the Supplementary |
| 4 | aplicat | T3 a9c0fb3 | SVM hard labels from predict(), 3.6 |
| 5 | aplicat | T3 a9c0fb3 | Supplementary S1(b) caption; results md note and statistical_tests.py string reworded identically |
| 6 | inchis | - | not applied (closed by the audit) |
| 7 | aplicat, option A | T4 a9789cd; R1.5 T5 b47169d | 3.1 provenance paragraph (split in four sentences for the style rule), Table 1 Source column (three labels), 3.10 / 4.7 / Table 10 level A redefined, Figure 6 legend, 4.6 and Discussion "two data paths", new Limitation (9), Supplementary S3, Response R1.5; no sensitivity line (P1 = A) |
| 8 | aplicat | T2 36647f5; T5 b47169d | "never exceeds 17% in any bin of the raw Active probability (0% to 16.7%)" in 4.5, R1.9, register note T1.3; General changes bullet |
| 9 | aplicat | T3 a9c0fb3 | Figure S1 legend: 660 / 1,000 resamples used, 340 dropped, cells not interpreted |
| 10 | aplicat | T3 a9c0fb3; T4 a9789cd | 3.3 sleep provenance and summing rule (204 / 952, 8, 44); Table 1 rows 17-18 |
| 11 | aplicat | T3 a9c0fb3; register T6 | "fixed as constants in the analysis script before its first execution" (3.9; T1.5 note) |
| 12 | aplicat | T3 a9c0fb3 | Figure 1 legend, 3.3, 4.10, contribution bullet 5: JSON export = export application's merged aggregates; ring named among the pooled sources |
| 13 | aplicat | T2 36647f5 | 1,083 -> 1,079 |
| 14 | aplicat | T2 36647f5 | bagging frequency 1 (fixed) in the Table A.1 note and the T2.2 row |
| 15 | aplicat | T3 a9c0fb3 | TabNet reference untrained on folds 1-2, rank "6 of 6" not informative |
| 16 | aplicat | T3 a9c0fb3 | O2 (five optimizer steps), O3 (Holm family of the four added models) |
| 17 | aplicat, extended | T2 36647f5 | Table 11 0.310 / 0.274 -> 0.309 / 0.273; the new rounding check found 17 further double-rounded cells (Table 4 x4, Table 6 x4, Table 8 x2, Table 10 x2, Table A.1 x5), corrected with their echoes in 4.4, 4.7, the Table A.1 note and R1.2; 402 cells, 0 mismatches after |
| 18 | aplicat | T3 a9c0fb3 | 3.11 block length and four-block maximum (with #24) |
| 19 | aplicat | T3 a9c0fb3; register T6 | Table 11 note and T3.3 pooled row: blocks concatenated from the most recent to the oldest |
| 20 | verificat, nimic de corectat | T2 36647f5 | revision/audit/t20_provenance.md: 8,734 (canonical csv), 15,634 / 6,902, 63.9% / 61.8%, r 0.99 / 0.98 / 0.88 all sourced; 0.88 is the all-sources row of the 30 March 2026 report; 9,106 absent |
| 21 | aplicat | T3 a9c0fb3; R1.5 T5 b47169d | sensitivity versus validation (3.10), result sentence in 4.7 with -0.732 [-0.817, -0.640], +0.129 [-0.063, +0.308], SpO2 no effect; closest variant for the HRV clause (interval is Active versus Rest) |
| 22 | aplicat (after GOAL_FIX) | T4 a9789cd; R2.c T5 b47169d | three rules, AI4Food reference, confidence rule at 0.50, traceability; "declared goal" dropped; sentence split in two |
| 23 | aplicat | T3 a9c0fb3; R1.9 T5 b47169d | 4.5, Limitation (5), R1.9: class-specific threshold not tuned (612 OOF days, 33 Active) |
| 24 | aplicat | T3 a9c0fb3; R1.10 T5 b47169d | expanding versus rolling rationale (3.11), rolling window not run (Discussion, R1.10) |
| 25 | aplicat | T3 a9c0fb3; R1.4 T5 b47169d | budget sentence in 3.5, consequence attached to the 4.4 sentence on the tuned networks; Table A.1 note; R1.4 |
| 26 | aplicat | T3 a9c0fb3 | 3.8 SHAP on the same 192 test days, variation reflects the model |
| 27 | aplicat | T3 a9c0fb3; R1.7 T5 b47169d | per-variable attribution via SHAP, group ablation tests necessity (4.6, R1.7) |
| R1.8 | verified, completed | T3 a9c0fb3 | 3.6: one-versus-rest ROC-AUC was present; "G-mean, with the classes fixed so that an unpredicted class gives zero" added |
| GOAL_FIX | aplicat | T1 39807f5 | 3.12, BASE_XREFS tuple; TEXT_DECISIONS line |

Entries "de clarificat": none. Commits on apply-audit, in order: ad24004 / c512efa (T0), 39807f5 (T1), 36647f5 (T2), a9c0fb3 (T3), a9789cd (T4), 6ad9131 / 4894215 / a6f2f2c / b47169d (T5; the first three carried a stale or style-failing response docx, corrected in the next), 84b50f2 (T6), then the T7 commit "audit-apply: summary".

## Reference figures after the application

| Check | Baseline (master e46ddaa / eea3c68) | After |
|---|---|---|
| Manuscript: numbers extracted / FOUND / NEGASIT | 1,531 / 1,068 / 0 | 1,581 / 1,099 / 0 |
| Supplementary: numbers / FOUND / NEGASIT | 372 / 308 / 0 | 377 / 311 / 0 |
| _check_t41_outputs.py | 73 / 73 | 73 / 73 |
| _check_table_rounding.py (new) | 402 cells, 19 mismatches | 402 cells, 0 mismatches |
| style_check manuscript / response / Supplementary | 8/8 / 5/5 / 2/2 | 8/8 (26.5 words/sentence) / 5/5 (26.8) / 2/2 |
| Marked docx: paragraphs / cells / removed items | 108 / 408 / 26 | 111 / 416 / 26 |
| Word comments | 9 + 1 | 9 + 1 |
| General changes bullets (response letter) | 7 | 19 |
| Limitations | 8 | 9 |

## What changed in the registers

CIFRE_REVIZIE.md (no canonical figure changed; the 4-decimal result rows are untouched):
- row T1.1 "RF macro PR-AUC": note names the estimator (average_precision_score, macro) (#3);
- row T2.2 "Spatii de cautare": note states subsample_freq = 1 fixed for every sampled LightGBM configuration (#14);
- row T3.3 "Pooled": note states the concatenation order (#19);
- note T1.3 "ce arata cifrele": "intre 3% si 17%" corrected to "nu depaseste 17% (0% la 16.7%)" with the audit reference (#8);
- note T1.5 "definitia grupurilor": last sentence reworded to the 3.9 statement "fixed as constants in the analysis script before its first execution" (#11);
- new section "Status aplicare audit (2026-10-26)": parameters P1-P5, the applied entries with commits, and 10 rows (ID AUDIT) registering the audit figures that now appear in the text with their sources: 1,079 files; 61.3% / 71.1% / 36.8% / 66.6% ring share of the XML records; 949 / 910 / 946 of 952 labeled days with ring samples (99.7% / 95.6% / 99.4%); sleep 204 / 952, 8, 44; gap pairs 12 / 42, 1 test row; persistence versus label rule 192 / 192; OOF Active fraction 0.0000 to 0.1667; the 19 double-rounded cells; LightGBM without subsample_freq 0.4965; TabNet reference 0 / 0 / 1 batches. These are not new experiments: they are the audit's own figures (revision/audit/, 25 Oct 2026) or the March 2026 reports, registered so that the rule "no figure in the text without a register row" stays checkable. This is the one judgment call of the task worth a second look: the brief asked for no new figure in the register, and the audit decisions themselves put these figures into the manuscript text (#7 FORMULARE, #8, #10, #13).

TEXT_DECISIONS.md: new section "AUDIT (26 oct 2026)" with the GOAL_FIX decision (T1) and one line each for the two previous-day conventions (#1), the restricted persistence statement (#2), "two data paths" (#7), sensitivity versus validation (#21), the safety layer without a per-class threshold (#23), expanding versus rolling (#18, #24), the asymmetric budget (#25), fixed test days for SHAP (#26), the rounding rule (#17) and #22 after GOAL_FIX; each says what is stated and what is not.

AUDIT_DECISIONS.md: header "Aplicare: INCHISA (26 oct 2026)"; Status column "aplicat (commit)" on #1-#5 and #7-#27, "verificat, nimic de corectat" on #20, #6 unchanged.

## Not done, and why

- No sensitivity of aw_hr_mean without ring minutes (P1 = A; would be a new experiment).
- "two devices" and "device-independent" remain in the Abstract, in Introduction contribution bullet 1 and in the Conclusions: rule 7 of the brief (Abstract, Highlights, Conclusions untouched; layer C decides). The Highlights are unchanged.
- No Health Auto Export documentation reference was added for #12 (it would be a new reference entry; none exists in References).
- The 12 audit-r1 output .txt files are CRLF in the working copy only (index LF; .gitattributes has no .txt rule); not touched.
- The response letter still quotes the removed phrases ("best performing model", "optimal thresholds", "three years") inside the General changes bullet and the reviewer quote, as since T8.

## Merge

HEAD of apply-audit after the summary commit: see `git log --oneline -1 apply-audit`. Merge command (to be run by the author after reading this summary; fast-forward is possible because master has not moved since eea3c68):

```
git checkout master && git merge --ff-only apply-audit
```
