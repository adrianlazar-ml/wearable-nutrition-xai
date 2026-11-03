# public_release_list.md: exclusion list of the filtered public clone (T12b.1, 2026-11-03)

Final list for `git filter-repo --invert-paths` on the clone `C:\Users\Adrian\Documents\Backup\wearable-nutrition-xai` (clone of branch t12b-public-clone, never of this repository in place). Basis: `tracked_data_inventory.md` (77 files marked da, 11 daily aggregates marked Option C), the dry-run `public_release_dryrun.md` (corrections i to iv), the decisions of the co-author of 3 Nov 2026 (Option C = code plus the 11 daily aggregates; the manuscript docx files are not published) and the T11 files that carry e-mail addresses or the telephone number (dry-run section 3, extended to the files committed after the dry-run).

Tracked files at the branch tip: 638. Files under the list: 140 (1913.0 MB in the working copy). Remaining: 498.

## Paths (one `--path` argument each; directories with the trailing slash, prefix match)

| Path | Tracked files | Bytes (working copy) | Why |
|---|---|---|---|
| `data/Oura/` | 66 | 103,320,962 | raw Oura exports (per night, per session, GPS) and the Oura account, subscription and payment files; every file is marked da |
| `data/apple_health/` | 10 | 1,790,042,815 | Apple Health XML export and the csv files derived from it (per record); every file is marked da |
| `apple_watch_api/raw_json/` | 2 | 15,710 | raw Health Auto Export JSON (per record); the daily folder itself is gitignored, these two files were tracked |
| `_referinte_upb/` | 16 | 2,867,627 | dissertation and university documents (not part of the manuscript code) |
| `revision/manuscript/` | 37 | 15,451,970 | the manuscript docx files (base v2 / v8, the six R1 outputs), the F4 text blocks, the assembly plan with the title page, the Editorial Manager PDF of the submission (submitted/) and the figure exports: the manuscript is not part of the public repository (decision of 3 Nov 2026) |
| `data/sleep_detailed.csv` | 1 | 1,157,919 | Oura sleep sessions (per night, per session): da |
| `disertatie_audit.zip` | 0 | 0 | removed from tracking on 2026-11-03; its blob stays recoverable from f7cc4ca (hygiene_summary.md) |
| `audit_csv_output.txt` | 1 | 44,392 | UTF-16 audit output with local paths and raw csv excerpts (hygiene_summary.md step 2) |
| `sanity_check_output.txt` | 1 | 3,318 | UTF-16 output with local paths (hygiene_summary.md step 2) |
| `.env` | 0 | 0 | Oura OAuth2 credentials committed in f7cc4ca and untracked in 91d8ee6 (dry-run correction i); the app and the token were deleted / revoked on 3 Nov 2026 |
| `revision/F4_cover_letter_R1.md` | 1 | 3,062 | cover letter source with the signatory's e-mail address (T11) |
| `revision/TEXT_DECISIONS.md` | 1 | 65,447 | text decisions register; quotes the title page and the cover letter signature block (e-mail addresses, T11) |
| `revision/T11_summary.md` | 1 | 14,370 | T11 session summary with the title-page strings (e-mail addresses and telephone number) |
| `revision/T11b_summary.md` | 1 | 10,960 | T11b session summary with the cover letter e-mail addresses |
| `revision/audit/em_submission_diff.md` | 1 | 16,306 | diff of the Editorial Manager PDF against v8; reproduces the title page and the cover letter (e-mail addresses, telephone number) |
| `revision/audit/t11_docx_text_diff.md` | 1 | 2,267 | text diff of the T11 docx outputs; reproduces the title page (e-mail addresses, telephone number) |

`.env` and `disertatie_audit.zip` are not in the working copy or the index at the branch tip (0 tracked files); they are listed for the history.

## Checks (T12b.1 PASS)

- Files marked da in the inventory: 77; outside the list: 0.
- The 11 Option C aggregates and `apple_watch_api/apple_health_daily.csv`: 12 files, 12 outside the list and tracked:
  - `data/activity.csv`
  - `data/dataset_combinat.csv`
  - `data/dataset_combinat_json.csv`
  - `data/dataset_combinat_json_extended.csv`
  - `data/oura_combined.csv`
  - `data/oura_historical.csv`
  - `data/readiness.csv`
  - `data/sleep.csv`
  - `data/sleep_metrics.csv`
  - `data/stress.csv`
  - `models/dataset_snapshot_20260331.csv`
  - `apple_watch_api/apple_health_daily.csv`
- `data/` is excluded by subfolder, not as a whole: `data/Oura/` and `data/apple_health/` go, the 10 daily csv files at the top of `data/` stay, except `data/sleep_detailed.csv` (da).
- `apple_watch_api/apple_health_auto.csv` (41 daily rows of 26 March 2026 with the source column) and `apple_watch_api/api_sync.log` (7 lines, one local path) are not marked da and stay (dry-run section 7 lists them as candidates for the code-only variant, which was not chosen).
- `revision/scripts/assemble_manuscript.py` stays: its EMAIL_FIRST, EMAIL_SECOND and TEL_SECOND constants are read at runtime from `revision/authors.json` (gitignored) since T12b.1; `revision/authors.example.json` (placeholders) is committed. `revision/scripts/_check_t11b_inventory.py` imports the two addresses from the same module instead of carrying them.
- The old versions of `assemble_manuscript.py` in the history (T11 to T11c), one commit message (T11b.2) and two files that stay (`revision/audit/public_release_dryrun.md`, one address in a table; `revision/scripts/_check_t11b_inventory.py` before T12b.1) still carry the two addresses and the telephone number: they are scrubbed at filtering with `--replace-text` and `--replace-message` (expressions file kept outside both repositories, replacement strings `***REMOVED-EMAIL***` / `***REMOVED-TEL***`; the four `.env` values of f7cc4ca are in the same file as a belt-and-braces measure).

## What the public clone can and cannot run (consequence of the list)

- Can run: every T1, T2, T3 script (`common.load_raw()` reads `data/oura_combined.csv` and `apple_watch_api/apple_health_daily.csv`, both kept; `models/dataset_snapshot_20260331.csv` kept), the T0 scripts, the unit tests.
- Cannot run: `assemble_manuscript.py` and the checkers that read `revision/manuscript/` (`_check_t41_outputs.py`, `_check_t41_numbers.py`, `_check_table_rounding.py`, `style_check.py`, `_check_t8_parity.py`, `_check_t11*`, `_check_t115_cover_letter.py`, `_t11_em_diff.py`, `export_figures_kbs.py` for the manuscript copies): the base docx and the F4 text blocks are not published.
- The dissertation-era root scripts that read `data/apple_health/` or `data/Oura/` (`cap5_pipeline.py`, `cap5_compare_algorithms.py`, `parse_health_auto_export.py` on the JSON folder, `oura_collect.py`) cannot run; the frozen pipeline (`sanity_check_frozen.py`, `cap5_pipeline_json_extended.py` on `data/dataset_combinat_json_extended.csv`) can.

