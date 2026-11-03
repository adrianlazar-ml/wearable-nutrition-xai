# T12b-A2: public root README, file checks and the LLM-use note (public clone)

Date: 3 November 2026. Repository: `wearable-nutrition-xai` (public clone), branch `main`, starting HEAD 47f2d12. Ordinary commit only: no history rewrite, no tag or release changes.

## A2.1 Root README.md

The May 2026 dissertation README (Romanian, pre-remediation figures, references to `data/apple_health/` and `apple_watch_api/raw_json/`, which are not in this repository) was replaced by an English README with the nine required sections in order: title and authors (1), Purpose (2), Repository history (3), What is included (4), What is excluded and why (5), How to reproduce (6), LLM use (7), Licenses (8), Citation (9). `revision/README.md` is kept and linked from the root README as the detailed guide. The LLM-use statement is given once, in section 7, with the register paths corrected (`CIFRE_CANONICE.md` at the root, `revision/CIFRE_REVIZIE.md`).

Checks on the file: 9 headings present; 0 e-mail addresses; 0 telephone numbers (the only hit of a digit-run pattern is the tag name `canonical-8020-20260331`); 0 local Windows paths; 0 Romanian diacritics; 0 em dashes; LF line endings.

Values stated in section 6 were verified on this clone on 3 November 2026: `python -m pytest revision/scripts/ -q` gives 111 passed, 2 failed in 33.65 s (the two failures are in `test_assemble_manuscript.py`, which reads the excluded `manuscript/F4_manuscript_changes.md`); the `group_ablation.py` and `label_agreement.py` end-to-end runs and their figures are those recorded in `revision/README.md` (section "Running on the published aggregates").

## A2.2 File checks

| File | Size (bytes) | Rows | Columns | Granularity | Date range | Read by a tracked script | Decision |
|---|---|---|---|---|---|---|---|
| `apple_watch_api/apple_health_auto.csv` | 3,060 | 40 data rows (41 lines) | 7: date, metric_name, value, min, max, unit, source | per record: 18 hourly heart_rate rows with min and max, 1 workout row, 21 daily rows of 7 other metrics | 2026-03-26 to 2026-03-28 | no (named only in `LICENSE-DATA`, two audit reports and the exclusion list) | `git rm`: not a daily aggregate and not read by any script |
| `apple_watch_api/api_sync.log` | 678 | 7 lines | log lines | log of the Health Auto Export REST API trial | 2026-03-28 16:22 to 16:43 | no | `git rm`: two local Windows paths (`raw_json` sync files), unused |

Other tracked csv, json and log files outside the 11 aggregates, `apple_watch_api/apple_health_daily.csv` and `revision/results/` (report only, removal only for per-record personal data):

| File(s) | Count | Granularity | Read by a tracked script | Classification |
|---|---|---|---|---|
| `_data_samples/sleep_detailed_HEADER.csv` | 1 (3 sample rows, 7,454 bytes) | per night, with 5-minute heart rate and HRV series and bedtimes: a sample of the excluded `data/sleep_detailed.csv` | no | per-record personal data: `git rm` |
| `_data_samples/*_HEADER.csv` (activity, dataset_combinat, dataset_combinat_json, dataset_combinat_json_extended, oura_combined, oura_historical, readiness, sleep, sleep_metrics, stress) | 10 (3 sample rows each) | daily, samples of the published aggregates | no | kept (redundant with the published files, no new data) |
| `models/frozen_meta_20260331.json` | 1 | metadata of the frozen pipeline | `patch_frozen_meta.py`, `revision/scripts/_check_t14_inputs.py`, `scripts/canonical/compare_16f_vs_19f.py` | kept |
| `outputs/tables/*.csv`, `outputs/bootstrap_ci/bootstrap_ci_frozen_test.csv` | 17 | result tables of the dissertation experiments | root scripts, `generate_canonical_numbers.py`, `extract_all_results.py`, `revision/audit/recompute_f3.py`, `revision/scripts/_check_t13_safety_layer.py` | kept |
| `revision/audit/p0_parser_vs_watch_only.csv`, `revision/audit/p0_source_mix.csv` | 2 (9,634 and 8,557 rows) | per day and metric: daily record counts and daily means by source, no record values | `revision/audit/_check_f4_empirical.py`, `revision/scripts/watch_only_sensitivity.py` | kept |
| `revision/authors.example.json` | 1 | placeholder template | `revision/scripts/assemble_manuscript.py` | kept |

The three removed files remain in the old commits of the public history (no rewrite in this task): `apple_health_auto.csv` and `api_sync.log` were kept by the T12b.1 exclusion list and are visible in every commit before this one, as is `sleep_detailed_HEADER.csv`. A later filter would be needed to purge them; this task does not do it.

Tracked files: 502 before, 499 after.

## A2.3 Consistency

- `revision/README.md`: the sentence listing the published aggregates no longer names `apple_health_auto.csv`; the DOI placeholder (10.5281/zenodo.XXXXXXX), the release name (v1.0-R1) and the file lists agree with the root README.
- `CITATION.cff`: authors (Adrian Lazar, Catalin Buiu), affiliation and the article title in `preferred-citation` are identical to the root README; the DOI placeholder is the same.
- Open point, not changed by this task (rule: do not edit `LICENSE-DATA`): the scope paragraph of `LICENSE-DATA` still names `apple_watch_api/apple_health_auto.csv`, a file that no longer exists in the working tree. To be removed from that paragraph in a later task.
- `revision/audit/public_release_list.md` and `revision/audit/public_release_dryrun.md` describe the file as kept; they are historical reports and were left as written.

## A2.4 Commit

Commit "Public README and file cleanup (R1)" on `main`, pushed to `origin`. Changes: `README.md` replaced, `revision/README.md` (one sentence), `revision/T12b_A2_summary.md` (this file), three files removed.
