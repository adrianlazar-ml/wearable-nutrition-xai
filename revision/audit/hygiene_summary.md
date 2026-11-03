# hygiene_summary.md: repository hygiene before publication (branch repo-hygiene, 2026-10-26)

Branch `repo-hygiene` from master 24919c4 ("audit-apply: rounding pairs check (19 OK, QWK XGB kept at 0.180)", the fast-forward merge of apply-audit). Scope: the category C observations of `AUDIT_DECISIONS.md`. Nothing under `revision/manuscript/`, no F4 source, no register (CIFRE_CANONICE.md, CIFRE_REVIZIE.md, TEXT_DECISIONS.md, STYLE_GUIDE.md) and no result-producing code was changed; no file was deleted; no experiment was run.

## What changed per task

| Task | Commit | Files | Change |
|---|---|---|---|
| T1 | 34dd887 | .gitignore, audit/untracked_inventory.md | rule `data/apple_health/New Automation/` (raw JSON, 1,079 files, 821.7 MB, untracked); inventory of the 1,085 untracked non-ignored files in 4 folders, nothing added or deleted |
| T1b | d883268 | audit/tracked_data_inventory.md | the 180 tracked files under data/, models/ and outputs/ with size and the column "date personale brute?" (see Public release) |
| T2 | e4b64fb | .gitattributes | `*.txt text eol=lf`; `-text` on the two UTF-16 root reports (audit_csv_output.txt, sanity_check_output.txt); 16 .txt working copies under revision/ restored with LF (content unchanged, index unchanged) |
| T3 | 635dc4f | scripts/_check_t13_safety_layer.py, label_agreement.py, extra_models.py, statistical_tests.py | docstring paragraph and three comments; 17 insertions, 0 deletions, no code line changed; pytest 111 passed |
| T4 | e557d64 | README.md | README of the revision folder for the public release (placeholders in [ ]) |
| T5 | this commit | audit/hygiene_summary.md | this summary |

## T1.1: STOP, tracked raw data

`git ls-files data/apple_health/` returns 10 tracked files: export.xml (1,562,338,195 bytes), heart_rate.csv (90.6 MB), active_calories.csv (67.7 MB), steps.csv (27.1 MB), basal_calories.csv (23.5 MB), sleep.csv (11.8 MB), respiratory_rate.csv, spo2.csv, hrv.csv, weight.csv. They are raw personal health data and they are in the git history since the 26 May 2026 snapshot. Per the brief, the step stopped: `data/apple_health/` was not added to .gitignore (it would not untrack them), no `git rm`, no history rewrite. Only the untracked subfolder of raw JSON files was ignored.

Top 10 tracked files by size (bytes): data/apple_health/export.xml 1,562,338,195; data/apple_health/heart_rate.csv 90,615,631; data/apple_health/active_calories.csv 67,682,466; data/Oura/raw/temperature.csv 44,993,767; data/Oura/raw/heartrate.csv 37,410,309; data/apple_health/steps.csv 27,103,810; data/apple_health/basal_calories.csv 23,521,036; data/apple_health/sleep.csv 11,836,556; data/Oura/raw/dailyactivity.csv 7,988,365; data/apple_health/respiratory_rate.csv 5,044,749. Nine of the ten are raw personal data; none of them is read by the revision scripts (they read models/dataset_snapshot_20260331.csv, data/oura_combined.csv and apple_watch_api/apple_health_daily.csv).

Untracked inventory: 1,085 files in 4 folders (details in `untracked_inventory.md`, not repeated here); plus the unstaged deletion of the tracked `disertatie_audit.zip`, left as is at the time (removed from tracking on 3 November 2026, see Public release).

## T2: branch chosen and why

Facts found: the index holds LF for every tracked file (`git ls-files --eol`: 0 `i/crlf`), so the brief's expectation "index CRLF" for the two prediction csv files does not hold; CRLF exists only in working copies (139 files; `core.autocrlf = true`), produced either by pandas `to_csv` (default `lineterminator = os.linesep`, CRLF on Windows: `save_test_predictions.py:76`, `persistence_baseline.py:167`, no explicit line terminator) or by checkouts made before `.gitattributes` existed. Branch 1 of T2.2 applies (the producer writes CRLF), adapted: the working copies of the producer csv files were left as produced, nothing was normalized, and the `-text` rule was NOT added. Reason: `-text` would make git store the platform-dependent CRLF bytes and reverse the LF normalization that `*.csv text eol=lf` already applies; no sha256 of these csv files is registered in CIFRE_REVIZIE (the identity tests compare values; `_check_t12_inputs.py` prints their sha for information only); and the byte-for-byte determinism checks compare two consecutive runs of the same script, both CRLF. `_check_t11_inputs.py` and `_check_t12_inputs.py`: ALL PASS. The two UTF-16 root reports that git already treated as binary got an explicit `-text`. Renormalization (`git add --renormalize` over the tracked files, excluding the deleted zip) staged nothing besides `.gitattributes`: the index content was already normalized. Remaining CRLF working copies under revision/ after T2: the 9 producer files (test_predictions_*.csv x6, persistence_baseline.csv, t11_extended_metrics.csv and its md), all LF in the index; a fresh checkout gives LF. The `.gitattributes` working copy itself was rewritten to LF (its index content was already LF).

## Public release

Conclusion: this private repository stays untouched. Every hash cited in CIFRE_REVIZIE.md (Artefacte, AUDIT section), in AUDIT_DECISIONS.md (the audit commits 3ba0c41 .. eea3c68, the application commits 36647f5 .. 84b50f2) and in the Status sections refers to it, and a history rewrite here would invalidate them. The public release (code only, or code plus aggregated features, Option C of the co-author) is produced from a separate filtered clone, never from this repository. Procedure, to be executed by the author when Option C is decided (described only, not run):

1. `git clone --no-local <path to disertatie_v2> disertatie_v2_public` (a fresh clone; this repository has no remote and was never pushed: `git remote -v` is empty).
2. In the clone, `git filter-repo --invert-paths --path data/ --path apple_watch_api/raw_json/ --path _referinte_upb/ --path disertatie_audit.zip --path audit_csv_output.txt --path sanity_check_output.txt --path revision/manuscript/submitted/` (the list follows `tracked_data_inventory.md`: everything marked "da" is removed; under Option C "code only" add `--path models/dataset_snapshot_20260331.csv --path apple_watch_api/apple_health_daily.csv`; the Oura subscription invoices and account / payment csv files under data/Oura/Subscriptions/ are removed in either case). `git filter-repo` writes `.git/filter-repo/commit-map` (old hash to new hash); keep that file next to the release so that the hashes cited in the registers can be translated.
3. Verify the clone: `git ls-files | grep -c "^data/"` must be 0; `python revision/scripts/_check_t41_outputs.py` and `python -m pytest revision/scripts/ -q` must pass if the snapshot is included, or the README must state which scripts cannot run without it; `git count-objects -vH` should drop from the current 263 MiB (1,478 loose objects) to a few tens of MiB.
4. Add the chosen LICENSE and complete the three README placeholders, then push the clone to the public remote and archive it on Zenodo; the DOI goes into the README and into the Data availability statement at acceptance.
5. Keep this repository as the private record; add a note at the top of CIFRE_REVIZIE.md with the path of the commit-map, so that any hash in the registers can be looked up in the public history.

`disertatie_audit.zip` (3 November 2026): removed from tracking (`git rm`; the file had been missing from the working copy since before the revision work). Its 60 entries are duplicated elsewhere: the root pipeline scripts of the frozen March 2026 version (tracked at the repository root), `scripts_archive/` comparison, exploration and old-version scripts, `_data_samples/` CSV header rows only, and CIFRE_CANONICE.md as of 19 April 2026. It stays in the exclusion list because the blob remains recoverable from commit f7cc4ca ("Snapshot pre-audit, 26.05.2026"), the only commit that ever contained it.

T11.0 (2026-11-03): `revision/manuscript/submitted/` (the Editorial Manager PDF of the original submission, KNOSYS-D-26-14863.pdf, and its README) is added to the exclusion list above: the PDF carries the authors' e-mail addresses and a telephone number on the title page and in the cover letter, and the Editorial Manager metadata page.

The tracked-data inventory classes: 39 files raw per record / minute / night / event (Apple Health export and csv files, Oura raw exports including GPS location), 16 raw per day from the Oura API, 38 account and payment files (Oura subscription invoices), 11 daily aggregates or feature tables (Option C), 92 derived artifacts (models, figures, tables, reports).

## T5: final verification

| # | Check | Result |
|---|---|---|
| 1 | `git diff master..repo-hygiene --stat`: nothing under revision/manuscript/, no F4_*.md, no CIFRE_CANONICE.md, no CIFRE_REVIZIE.md (T2 took branch 1, no register change), no TEXT_DECISIONS.md, no STYLE_GUIDE.md | PASS (9 files: .gitattributes, .gitignore, revision/README.md, two audit inventories, four scripts with comments only, 306 insertions, 0 deletions) |
| 2 | `python revision/scripts/_check_t41_outputs.py` | PASS, 73 checks, 0 FAIL |
| 3 | `python -m pytest revision/scripts/ -q` | PASS, 111 passed (baseline before T3: 111 passed) |
| 4 | LF on every file changed on the branch | PASS (all 9 files LF in the working copy and in the index) |

## Points stopped

- T1.1 (tracked raw data under data/apple_health/): stopped, author's decision; procedure in Public release.
- T2.2 `-text` rule: not added, with the reason above; a decision of the author if the rule is still wanted.

## README figures (T4) and their rows

0.5063 (CIFRE_CANONICE; CIFRE_REVIZIE T0.1); versions 3.11.9 / 1.8.0 / 2.4.3 / 2.3.3 (T0.1), 0.14.1 / 3.2.0 / 1.17.1 / 0.15.0 / 0.51.0 / 2.11.0 / 4.6.0 / 1.2.10 / 4.1.0 / 0.0.2 (T2.1, requirements.txt), 1.2.0 (requirements.txt); timings "20 to 30 s" (T1.1 note: 26.5 / 18.0 s), "about one minute per mode" (T1.4: 54-71 s, 62-71 s), "20 to 25 s" (T1.5: 19.0-24.8 s), "20 to 30 s" (T3.3: 21-31 s), "about 3 minutes" and "about 11 minutes" (T2.1: 168-178 s, 647 s), "about 28 minutes", "3 to 4 minutes", "about 6 minutes" (T2.2: 1,659 s, 150-218 s, 366 s); 1,000 resamples (T1.1 note); 958 days, 19 features (CANONICE); 1,079 files (AUDIT row); 99.7% / 95.6% / 99.4%, 204 of 952, 8 days (AUDIT rows); thresholds 60 / 85 / 75 (CANONICE); 16 features (CANONICE, Table 13 pipeline); dates 26 May 2026 and 2026-03-29 (AUDIT_DECISIONS, audit F4 P0.3).

HEAD of repo-hygiene after this commit: see `git log --oneline -1`. Not merged.
