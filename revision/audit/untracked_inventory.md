# untracked_inventory.md: untracked, non-ignored files in the working copy (repo-hygiene, 2026-10-26)

Read-only inventory for the author's decision (KBS_Revision_Plan section 8, point 8: what to commit, what to ignore, what to delete before the public repository). Nothing in this list was added, deleted or ignored by the hygiene task, except the rule `data/apple_health/New Automation/` added to .gitignore (raw Health Auto Export JSON files, never meant for the repository). Command: `git status --porcelain --ignored=no -uall`.

Total: 1085 files in 4 folders.

| Folder | Files | Size | Content | Note |
|---|---|---|---|---|
| data/apple_health/New Automation | 1079 | 821.7 MB | HealthAutoExport-2023-04-11.json .. HealthAutoExport-2026-03-30.json | raw Health Auto Export daily JSON files (HealthAutoExport-YYYY-MM-DD.json), byte-identical to the parser's iCloud folder (audit F4 P0.1); sensitive personal data, not for publication; now ignored by .gitignore |
| outputs/figures/eswa_architecture | 4 | 3.6 MB | Figure_1_architecture_EN.pdf; Figure_1_architecture_EN.png; Figure_1_architecture_RO.pdf; Figure_1_architecture_RO.png | architecture figure exports (EN / RO, pdf + png) produced by eswa_architecture.py; the manuscript's Figure 1 comes from revision/manuscript/figures/; author's decision (plan 8.8) |
| _referinte_upb | 1 | 0.8 MB | Lazar_Adrian_Lucrare_Disertatie_2026.pdf.docx | final dissertation docx (reference copy); personal document, not for the public repository; author's decision |
| (repo root) | 1 | 0.0 MB | eswa_architecture.py | eswa_architecture.py, figure-drawing script; author's decision (plan 8.8) |

## Tracked file deleted in the working copy (not untracked, listed for completeness)

- disertatie_audit.zip: tracked at HEAD, deleted in the working copy since before the revision work (unstaged deletion, `D`). Author's decision whether to commit the deletion.

## Temporary files created by the assistant in earlier sessions

None inside the repository. The scratch files of the assistant live outside the repository (session scratchpad under %LOCALAPPDATA%\Temp\<assistant scratchpad>). The byte-identical duplicate `revision/CIFRE_CANONICE.md` that appeared after T6.6 was moved to the scratchpad on 2026-10-22 (TEXT_DECISIONS / memory of T7) and is not in the working copy.

## Tracked files under data/apple_health/ (T1.1, STOP)

`git ls-files data/apple_health/` returns 10 tracked files (weight.csv, hrv.csv, spo2.csv, respiratory_rate.csv, sleep.csv, basal_calories.csv, steps.csv, active_calories.csv, heart_rate.csv, export.xml), the largest being export.xml (1,562,338,195 bytes, the native Apple Health XML export) and heart_rate.csv (90,615,631 bytes). They are raw personal health data and they are in the git history. The hygiene task did not add `data/apple_health/` to .gitignore (it would not untrack them) and did not remove them: taking them out of the history (filter-repo / fresh public repository without data/) is the author's decision. Only the untracked subfolder `data/apple_health/New Automation/` (raw JSON) was added to .gitignore.
