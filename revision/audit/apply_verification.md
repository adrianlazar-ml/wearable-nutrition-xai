# apply_verification.md: T7 final verification of the audit application (branch apply-audit, 2026-10-26)

Run after the T6 commit (84b50f2) on a full regeneration of the five docx (assemble_manuscript.py, RUN_DATE 2026-10-26). Raw outputs: the commands and their last lines are quoted below; the complete number-check tables are revision/manuscript/t41_number_check_R1.md and t41_number_check_S1.md, the outputs check is revision/manuscript/t41_outputs_check.md, the rounding check is revision/audit/table_rounding_check.md.

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | KBS_Manuscript_R1_marked.docx regenerated against BASE_MANUSCRIPT_v8.docx | PASS | 111 paragraphs with highlighted text, 416 highlighted table cells, 26 removed items listed at the end (baseline 108 / 408 / 26 before the audit application; T4 added the 3.1 paragraph and the Table 1 Source cells) |
| 2 | Number check (_check_t41_numbers.py --phrases) | PASS | manuscript 1,581 numbers, FOUND 1,099, NEGASIT 0, SKIPPED 472 (baseline 1,531 / 1,068 / 0); Supplementary 377 / 311 / 0 (baseline 372 / 308 / 0). Every new figure has a row in CIFRE_REVIZIE (T1.3, T2.2, T3.2, T3.3 rows for #21, #23, #24, #25; section "Status aplicare audit" for the audit figures of #1, #2, #7, #8, #10, #13, #17) |
| 3 | _check_t41_outputs.py | PASS | 73 checks, 0 FAIL (same 73 checks as the baseline; no check added) |
| 4 | _check_table_rounding.py (new, #17 process) | PASS | 13 csv-derived tables, 402 cells, 0 mismatches (19 mismatches before T2, all double roundings, corrected) |
| 5 | style_check.py and forbidden-term greps | PASS | manuscript 8/8 (different-from-v8: 26.5 words/sentence, p90 43, 1 sentence > 60 words = the 3.9 group enumeration, 78 words); response 5/5 (26.8 words/sentence, 1 sentence > 60 words, the pre-existing R1.3 sentence); Supplementary 2/2. Greps on the body text of all four docx: em dash (U+2014) 0; "significant" 0; "random forest" (lower case) 0; "distinguish" 0; "stable over time" 0; "[N]" 0; "[guideline" 0; "best performing" / "optimal threshold" / "three years" 0 in the manuscript, Supplementary and Highlights (the response letter holds them only inside the quoted reviewer comment R1.1 and the General changes bullet that lists the removed phrases, unchanged since T8). "two devices": 0 in Results and Discussion; 2 in 3.1 (hardware sentence and the provenance paragraph, where the phrase is factual); 3 left intentionally in the Abstract, in Introduction contribution bullet 1 and in the Conclusions (rule 7 of the brief: Abstract, Highlights and Conclusions are not touched by this task; layer C decides) |
| 6 | Cross-references | PASS | all 25 distinct "Section x.y" references resolve to headings (outputs check); every Table 1-14 / A.1 and Figure 1-7 cited, first citations in numeric order, captions consecutive; Supplementary Table S1-S3 and Figure S1-S3 cited and present; Limitations: "Nine limitations" and items (1)-(9); references "Limitation 3" (3.12) and, in the response letter, Limitation (1), (3), (4), (5), (6), (9) all exist |
| 7 | Word comments | PASS | manuscript 9 (6 "verify in Zotero" on the new references, 3 "T6.2 draft" on the declarations), response letter 1 ("T7 draft" on the Authors line), 0 elsewhere; none added |
| 8 | American English | PASS | 0 British spellings in the body text of all docx (f5_us_spelling pairs; Susnjak & Griffin title excepted), permanent check of _check_t41_outputs.py |
| 9 | LF line endings | PASS | every text file written or modified on apply-audit is LF in the index and in the working copy (37 files with eol=lf plus apply_baseline.txt). Observation, not a change of this task: 12 .txt outputs of the audit-r1 slices (revision/audit/*_output.txt, committed LF on 25 Oct) are checked out CRLF in this working copy because .gitattributes has no rule for .txt; their index content is LF |

Commands (repo root, Git Bash):

```
python revision/scripts/assemble_manuscript.py
python revision/scripts/_check_t41_numbers.py --source revision/manuscript/KBS_Manuscript_R1.docx --out revision/manuscript/t41_number_check_R1.md --phrases
python revision/scripts/_check_t41_numbers.py --source revision/manuscript/KBS_Supplementary_R1.docx --out revision/manuscript/t41_number_check_S1.md
python revision/scripts/_check_t41_outputs.py
python revision/scripts/_check_table_rounding.py
python revision/scripts/style_check.py revision/manuscript/KBS_Manuscript_R1.docx --mode manuscript
python revision/scripts/style_check.py revision/manuscript/KBS_Response_R1.docx --mode response
python revision/scripts/style_check.py revision/manuscript/KBS_Supplementary_R1.docx --mode other
```

Last lines:

```
OK: KBS_Manuscript_R1.docx, KBS_Manuscript_R1_marked.docx, KBS_Highlights_R1.docx, KBS_Supplementary_R1.docx, KBS_Response_R1.docx, 20 figure files, 102 plan rows; marked {'paragraphs_marked': 111, 'cells_marked': 416, 'removed_items': 26}
KBS_Manuscript_R1.docx: 1581 numbers, FOUND 1099, NEGASIT 0, SKIPPED 472
KBS_Supplementary_R1.docx: 377 numbers, FOUND 311, NEGASIT 0, SKIPPED 66
73 checks, 0 FAIL -> revision/manuscript/t41_outputs_check.md
402 cells, 0 mismatches -> revision/audit/table_rounding_check.md
8/8 thresholds met (manuscript); 5/5 thresholds met (response); 2/2 thresholds met (Supplementary)
```
