# #20: provenance of every r and MAE figure of the XML versus JSON comparison (Sections 3.3, 4.10, Table 13, contribution list)

audit-apply, 2026-10-26 (AUDIT_DECISIONS #20, "de verificat in task-ul de aplicare"). Question: which correlation appears for mean heart rate, from which row of the 30 March 2026 report (all sources or Watch-only), and whether the step MAE is 8,734 (sourced) and not 9,106 (no saved output). Result: every figure has a saved source; nothing was corrected.

| Figure in the manuscript | Where | Printed | Source | Source value | Row of the source | Verdict |
|---|---|---|---|---|---|---|
| mean absolute error of the daily steps, XML versus JSON | 3.3 (base), 4.10 (base), contribution list (F4) | 8,734 steps per day | outputs/tables/steps_xml_vs_json_mae.csv (scripts/canonical/xml_vs_json_accuracy.py, 2 June 2026) | mae_abs 8733.88 on 1,069 days (xml_mean 15633.61, json_mean 6902.06, corr 0.9602) | "Steps (XML all sources vs JSON)" | sourced; 9,106 (outputs/reports/apple_xml_exploration_report.txt:187, no saved output) does not appear anywhere in the three R1 docx |
| mean daily steps XML / JSON | 4.10, Table 13 | about 15,634 / about 6,902 | same csv | 15633.61 / 6902.06 | same row | sourced |
| minimum heart rate r | 3.3 | 0.99 | outputs/reports/apple_xml_vs_json_comparison.txt (30 March 2026) | Corr 0.992363 (N 1,066) | "HR Min (XML all vs JSON)" | sourced, all-sources row (Watch-only row: 0.968922) |
| HRV r | 3.3 | 0.98 | same report | Corr 0.980379 (N 1,060) | "HRV Mean (XML all vs JSON)" (identical to the Watch-only row, HRV is 100% Watch) | sourced |
| mean heart rate r | 3.3 | 0.88 | same report | Corr 0.876473 (N 1,066) | "HR Mean (XML all vs JSON)" | sourced, all-sources row; the Watch-only row gives 0.79187 (verdict CHECK in the report) and is NOT the printed value |
| accuracy XML / JSON, 16 features, aligned subset | Table 13, 4.10 | 63.9% / 61.8% | outputs/tables/xml_vs_json_accuracy.csv (same script, 2 June 2026) | rf_accuracy 0.6387 / 0.6178 on 953 rows (762 / 191) | rows "XML (16f nativ)" / "JSON (16f nativ)" | sourced |
| aligned subset size | 4.10 | 953 days | same csv | n_rows 953 | same rows | sourced |

Notes.
- The r values of 3.3 ("metrics unaffected by double-counting agreed closely") come from the all-sources aggregation of the XML export against the JSON aggregates; both sides pool every source that wrote to HealthKit (audit P1, section 3), so they measure agreement between two pooled aggregates, which #12 now states in 3.3 and 4.10 ("raw multi-source records versus the export application's merged aggregates"). The mean-heart-rate correlation of the Watch-only comparison (0.79) is not reported in the manuscript and is not added (P1 = A: no new figure).
- The dissertation's "r >= 0.99" for heart rate and HRV (paragraph 1238 of the final docx, audit P1) is not in the manuscript; the manuscript's 0.88 for mean heart rate is the correct all-sources value.
- The numbers 8,734, 15,634, 6,902, 63.9%, 61.8% and 953 are base-manuscript figures without a row in CIFRE_* and are listed in EXPLAINED of revision/scripts/_check_t41_numbers.py with these sources (unchanged by this task).
