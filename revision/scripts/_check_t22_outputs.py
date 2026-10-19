"""
_check_t22_outputs.py - T2.2 Task 4: determinism and hygiene checks for hyperparameter_search.py.

1. Copies the current T2.2 outputs (cv + test modes) to a scratch directory, then re-runs
   `python -W error hyperparameter_search.py --mode cv` and `--mode test` (any warning -> error);
   stderr printed. (--skip-cv re-runs only the test mode, for a quick check; the full check runs both.)
2. sha256 identical between the two consecutive runs for every csv / md (search space json included);
   run_meta json identical without run_date / elapsed_seconds / timing / fit_seconds. If anything
   differs, the first-run files are restored from the scratch copies (the committed figure is the
   first run) and the difference is reported per file.
3. LF only on all text outputs and on CIFRE_REVIZIE.md.
4. md == csv: table lines rebuilt from the csv with the producer's own functions (cv_md_lines,
   test_md_lines, comparisons_md_lines, mcnemar_md_lines, seeds_md_lines) and compared verbatim;
   prediction files: 192 rows, schema, dates / y_true == test_predictions_rf.csv, probabilities sum
   to 1, argmax == y_pred (SVM: predict(), agreement reported), Active count == confusion counts,
   sha256 == meta; winners in the test table == argmax of the cv csv; deltas == differences of the
   point values; identity with the reference file when same_as_reference; run_meta sane;
   seeds outputs (if present): 35 rows, md == csv, seed-42 identity flags.
5. git diff empty on CIFRE_CANONICE.md, results/test_predictions_*.csv, every tracked result file
   of T0.x-T2.1 (tracked in revision/results, not t22_*) and extra_models.py (T2.2 only added
   optional arguments; the T2.1 identity was re-run separately, see CIFRE_REVIZIE.md).

Exits 1 on any FAIL. Evidence script, kept.
Run from repo root (after one full cv + test run):  python revision/scripts/_check_t22_outputs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hyperparameter_search as hs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
SCRIPT = ROOT / "revision" / "scripts" / "hyperparameter_search.py"
CV_FILES = ["t22_search_space.json", "t22_cv_results.csv", "t22_cv_summary.md", "t22_run_meta_cv.json"]
PRED = {m: f"t22_predictions_{m}.csv" for m in hs.MODELS}
TEST_FILES = list(PRED.values()) + ["t22_test_results.csv", "t22_test_results_table.md", "t22_test_metrics.csv",
                                    "t22_comparisons.csv", "t22_comparisons_table.md", "t22_mcnemar.csv", "t22_mcnemar_table.md",
                                    "t22_run_meta_test.json"]
SEED_FILES = ["t22_seeds.csv", "t22_seeds_table.md", "t22_run_meta_seeds.json"]
VOLATILE = ("run_date", "elapsed_seconds", "timing", "fit_seconds")
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_without_volatile(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    for k in VOLATILE:
        d.pop(k, None)
    if "models" in d:                                  # per-model volatile fit times inside run_meta_test
        for v in d["models"].values():
            if isinstance(v, dict):
                v.pop("stage_a_seconds", None)
    return d


def md_has_lines(md: Path, lines: list[str]) -> bool:
    text = md.read_text(encoding="utf-8").splitlines()
    return all(ln in text for ln in lines) and [ln for ln in text if ln in set(lines)] == lines


def run_mode(mode: str) -> None:
    proc = subprocess.run([sys.executable, "-W", "error", str(SCRIPT), "--mode", mode], cwd=ROOT, capture_output=True, text=True)
    check(proc.returncode == 0, f"second run --mode {mode}: exit code {proc.returncode}")
    check(proc.stderr.strip() == "", f"second run --mode {mode}: stderr empty ({len(proc.stderr)} chars)")
    if proc.stderr.strip():
        print(proc.stderr[-3000:])
    last = [ln for ln in proc.stdout.splitlines() if ln.startswith("elapsed")]
    print("  " + (last[-1] if last else "(no elapsed line)"))
    check("ALL CHECKS PASS" in proc.stdout, f"second run --mode {mode}: ALL CHECKS PASS")


def main(scratch: Path | None, skip_cv: bool) -> None:
    scratch = scratch or Path(tempfile.mkdtemp(prefix="t22_prev_"))
    files = ([] if skip_cv else CV_FILES) + TEST_FILES
    print(f"scratch dir for first-run copies: {scratch}")
    for f in files:
        check((RES / f).exists(), f"first-run output exists: {f}")
        shutil.copy2(RES / f, scratch / f)
    if fails:
        sys.exit(1)

    print("\n=== determinism: second run(s) under -W error ===")
    if not skip_cv:
        run_mode("cv")
    run_mode("test")
    differs: list[str] = []
    for f in files:
        if f.endswith(".json") and "run_meta" in f:
            same = json_without_volatile(RES / f) == json_without_volatile(scratch / f)
            label = f"json identical without {VOLATILE}: {f}"
        else:
            same = sha256(RES / f) == sha256(scratch / f)
            label = f"sha256 identical: {f} ({sha256(RES / f)[:16]})"
        check(same, label)
        if not same:
            differs.append(f)
            if f.startswith("t22_predictions_"):
                a, b = pd.read_csv(scratch / f), pd.read_csv(RES / f)
                print(f"   y_pred differs on {int((a.y_pred != b.y_pred).sum())} / 192 days, max abs proba diff "
                      f"{float(np.abs(a[hs.PROBA_COLS].to_numpy() - b[hs.PROBA_COLS].to_numpy()).max()):.2e}")
    if differs:
        for f in files:
            shutil.copy2(scratch / f, RES / f)
        print(f"RESULT first-run outputs restored from {scratch} ({len(files)} files; differing: {differs})")

    print("\n=== LF, md == csv, winners, predictions, comparisons, run_meta ===")
    present_seeds = all((RES / s).exists() for s in SEED_FILES)
    for f in CV_FILES + TEST_FILES + (SEED_FILES if present_seeds else []):
        check(b"\r\n" not in (RES / f).read_bytes(), f"LF only: {f}")
    check(b"\r\n" not in (ROOT / "revision" / "CIFRE_REVIZIE.md").read_bytes(), "CIFRE_REVIZIE.md LF only")
    cv, winners = hs.load_winners(RES)
    space = json.loads((RES / "t22_search_space.json").read_text(encoding="utf-8"))
    check(space == json.loads(json.dumps(hs.em.jsonable(hs.build_search_space(space["reduced"])))),
          f"t22_search_space.json == ParameterSampler(random_state 42) regenerated (reduced = {space['reduced']})")
    n_expected = sum(len(space["models"][m]["configs"]) + 1 for m in hs.MODELS)
    check(len(cv) == n_expected and int((~cv.eligible).sum()) == 7, f"cv csv: {n_expected} rows ({len(cv)}), 7 reference rows")
    check(md_has_lines(RES / "t22_cv_summary.md", hs.cv_md_lines(cv)), "t22_cv_summary.md == t22_cv_results.csv (summary rebuilt)")
    for m in hs.MODELS:
        sub = cv[cv.model == m]
        check(bool((np.abs(sub[["f1_fold1", "f1_fold2", "f1_fold3"]].astype(float).mean(axis=1) - sub.f1_mean.astype(float)) < 1e-9).all()),
              f"{m}: f1_mean == mean of the 3 folds")
    # only empty cells are NaN (-> "n/a" like the producer); the literal "n/a" of best_epoch stays a string
    tt = pd.read_csv(RES / "t22_test_results.csv", keep_default_na=False, na_values=[""])
    check(md_has_lines(RES / "t22_test_results_table.md", hs.test_md_lines(tt)), "t22_test_results_table.md == t22_test_results.csv")
    comp = pd.read_csv(RES / "t22_comparisons.csv")
    check(md_has_lines(RES / "t22_comparisons_table.md", hs.comparisons_md_lines(comp)), "t22_comparisons_table.md == t22_comparisons.csv")
    mc = pd.read_csv(RES / "t22_mcnemar.csv")
    check(md_has_lines(RES / "t22_mcnemar_table.md", hs.mcnemar_md_lines(mc)), "t22_mcnemar_table.md == t22_mcnemar.csv")
    check(list(tt.model) == hs.MODELS and len(mc) == 7 and len(comp) == (7 * 4 - 1) * 2, "test table 7 rows, McNemar 7 rows, comparisons 54 rows")
    for r in tt.itertuples():
        check(int(r.best_config_id) == winners[r.model]["config_id"] and abs(float(r.cv_f1_mean) - winners[r.model]["cv_f1_mean"]) < 1e-9,
              f"{r.model}: test-table winner == argmax of the cv csv (config {r.best_config_id})")
    ref = pd.read_csv(RES / "test_predictions_rf.csv")
    meta = json.loads((RES / "t22_run_meta_test.json").read_text(encoding="utf-8"))
    tm = tt.set_index("model")
    for m, f in PRED.items():
        df = pd.read_csv(RES / f)
        r = tm.loc[m]
        cm = meta["models"][m]["point"]
        sums = df[hs.PROBA_COLS].sum(axis=1)
        ok = (len(df) == 192 and list(df.columns) == ["date", "y_true", "y_pred"] + hs.PROBA_COLS
              and list(df.date) == list(ref.date) and np.array_equal(df.y_true.to_numpy(), ref.y_true.to_numpy())
              and float(np.abs(sums - 1).max()) < 1e-6
              and (m == "svm" or np.array_equal(df[hs.PROBA_COLS].to_numpy().argmax(1), df.y_pred.to_numpy()))
              and int((df.y_pred == 2).sum()) == cm["cm_ra"] + cm["cm_na"] + cm["cm_aa"] == int(r.n_active_pred)
              and abs(float(r.f1_macro) - cm["f1_macro"]) < 1e-9 and meta["prediction_files_sha256"][f] == sha256(RES / f))
        check(ok, f"{f}: 192 rows, schema, dates / y_true == rf csv, proba sum 1, argmax == y_pred, Active count == cm, sha256 == meta")
        if bool(r.same_as_reference):
            rf_ = hs.reference_predictions(RES, m)
            check(np.array_equal(rf_.y_pred.to_numpy(), df.y_pred.to_numpy()), f"{m}: same_as_reference -> predictions == reference file")
    for r in comp.itertuples():
        check(abs(float(r.diff_point) - (float(r.value_tuned) - float(r.value_other))) < 1e-9
              and str(r.include_zero) == str(float(r.ci_low) <= 0 <= float(r.ci_high)),
              f"comparison {r.model} ({r.comparison}, {r.metric}): diff == difference, include_zero flag") if r.metric == "f1_macro" else None
    ca = comp[(comp.comparison == "a") & (comp.metric == "f1_macro")].set_index("model")
    check(abs(float(ca.loc["rf", "value_other"]) - 0.5063) < 5e-5 and all(abs(float(ca.loc[m, "value_tuned"]) - float(tm.loc[m, "f1_macro"])) < 1e-9 for m in hs.MODELS),
          "comparison (a): RF canonical 0.5063, tuned values == test table")
    check(bool((mc.n00 + mc.n01 + mc.n10 + mc.n11 == 192).all()) and bool((mc.p_cc_holm >= mc.p_cc - 1e-12).all()), "McNemar: cells sum to 192, Holm >= raw")
    for m in hs.NETWORKS:
        mm = meta["models"][m]
        check(1 <= mm["best_epoch"] <= hs.MAX_EPOCHS and mm["stage_a_epochs_run"] >= mm["best_epoch"] and len(mm["val_history"]) == mm["stage_a_epochs_run"],
              f"{m}: test-stage best_epoch {mm['best_epoch']} <= stage A epochs {mm['stage_a_epochs_run']} <= 200")
    check(meta["sanity_checks_failed"] == [] and meta["seed"] == 42 and meta["smote_seed"] == 42 and meta["smote_k"] == 3
          and meta["n_resamples"] == 1000 and meta["torch_threads"] == 1 and meta["versions"]["torch_cuda_available"] is False
          and "vendor" in meta["versions"]["pytorch_tabnet_path"]
          and [round(v, 4) for v in meta["rf_canonical_marginal_ci"]] == meta["t11_rf_f1_macro_ci_from_cifre_revizie"]
          and meta["cv_results_sha256"] == sha256(RES / "t22_cv_results.csv"),
          f"run_meta_test: 0 failed checks, seed 42, SMOTE 42 / k 3, 1000 resamples, 1 thread, CPU, vendored TabNet, RF CI == T1.1, cv sha256 (elapsed {meta['elapsed_seconds']} s)")
    mcv = json.loads((RES / "t22_run_meta_cv.json").read_text(encoding="utf-8"))
    check(mcv["sanity_checks_failed"] == [] and mcv["failed_fits"] == [] and mcv["search_space_sha256"] == sha256(RES / "t22_search_space.json")
          and [f["active_train"] for f in mcv["folds"]] == [10, 17, 25] and [f["active_val"] for f in mcv["folds"]] == [7, 8, 13],
          f"run_meta_cv: 0 failed checks / fits, search space sha256, fold Active counts (elapsed {mcv['elapsed_seconds']} s, reduced {mcv['reduced']})")
    if present_seeds:
        seeds = pd.read_csv(RES / "t22_seeds.csv")
        check(len(seeds) == 35 and sorted(set(seeds.seed)) == [0, 1, 2, 3, 42] and list(dict.fromkeys(seeds.model)) == hs.MODELS, "t22_seeds.csv: 35 rows")
        check(md_has_lines(RES / "t22_seeds_table.md", hs.seeds_md_lines(seeds)), "t22_seeds_table.md == t22_seeds.csv (summary rebuilt)")
        ms = json.loads((RES / "t22_run_meta_seeds.json").read_text(encoding="utf-8"))
        check(all(v["y_pred_identical"] and v["max_abs_proba_diff"] < 1e-9 for v in ms["identity_seed42_vs_test"].values()) and ms["sanity_checks_failed"] == [],
              f"seeds run_meta: seed-42 identity with --mode test for 7 / 7, 0 failed checks ({ms['elapsed_seconds']} s)")
        for m in hs.MODELS:
            s42 = seeds[(seeds.model == m) & (seeds.seed == 42)].iloc[0]
            check(abs(float(s42.f1_macro) - float(tm.loc[m, "f1_macro"])) < 1e-9, f"{m}: seeds csv seed 42 == test table")
    else:
        print("INFO seeds outputs not present; seeds checks skipped")

    print("\n=== git diff on protected files ===")
    tracked = subprocess.run(["git", "ls-files", "revision/results", "CIFRE_CANONICE.md", "revision/scripts/extra_models.py"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    protected = [p for p in tracked if "/t22_" not in p]
    diff = subprocess.run(["git", "status", "--porcelain", "--"] + protected, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    check(diff == "", f"git diff empty on CIFRE_CANONICE.md, extra_models.py and {len(protected) - 2} tracked T0.x-T2.1 result files" + (f"\n{diff}" if diff else ""))
    check(any("test_predictions_rf.csv" in p for p in protected) and any("extra_models.py" in p for p in protected), "test_predictions_*.csv and extra_models.py in the protected set")

    print(f"\n{'ALL PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--skip-cv", action="store_true")
    a = ap.parse_args()
    main(a.scratch, a.skip_cv)
