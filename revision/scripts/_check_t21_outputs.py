"""
_check_t21_outputs.py - T2.1 Task 4: determinism and hygiene checks for extra_models.py.

1. Copies the current T2.1 main-mode outputs to a scratch directory, then re-runs
   `python -W error extra_models.py --mode main` (any warning -> error -> FAIL); stderr printed.
2. sha256 identical between the two consecutive runs for the 4 prediction csv, the 2 table csv
   and the 2 table md; json identical without run_date / elapsed_seconds / fit_seconds.
   Determinism verdict PER MODEL: a tree model that differs is a FAIL; a network that differs is
   a RESULT (reported: which csv, on how many days y_pred differs, max abs probability
   difference), not a failure of the task. If anything differs, the FIRST-run files are restored
   from the scratch copies so that the committed figure stays the first run.
3. LF only on all text outputs and on CIFRE_REVIZIE.md.
4. md == csv: the markdown table lines are rebuilt from the csv with the producer's own functions
   (metrics_md_lines, comparisons_md_lines, seeds_md_lines(seeds_summary)) and compared verbatim;
   prediction files: 192 rows, schema, dates and y_true == test_predictions_rf.csv, probabilities
   sum to 1, argmax == y_pred, predicted Active count == run_meta confusion matrix, sha256 == meta;
   run_meta sane (0 failed checks, vendored TabNet, CPU, one thread, RF CI == T1.1, elapsed budgets);
   seeds outputs (if present): LF, md == csv, seed-42 identity flags, 20 rows.
5. git diff empty on CIFRE_CANONICE.md, results/test_predictions_*.csv and every tracked result file
   of T0.x-T3.3 (everything tracked in revision/results except t21_*).

Exits 1 on any FAIL (a non-deterministic network alone does not fail). Evidence script, kept.
Run from repo root (after one full main run):  python revision/scripts/_check_t21_outputs.py
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
import extra_models as em  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
SCRIPT = ROOT / "revision" / "scripts" / "extra_models.py"
PRED = {n: f"t21_predictions_{n}.csv" for n in em.NEW_MODELS}
TABLES = ["t21_extra_models.csv", "t21_extra_models_table.md", "t21_comparisons.csv", "t21_comparisons_table.md"]
META = "t21_run_meta.json"
FILES = list(PRED.values()) + TABLES + [META]
SEED_FILES = ["t21_seeds.csv", "t21_seeds_table.md", "t21_run_meta_seeds.json"]
VOLATILE = ("run_date", "elapsed_seconds", "fit_seconds")
fails: list[str] = []
results: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def result(msg: str) -> None:
    print("RESULT " + msg)
    results.append(msg)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_without_volatile(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    for k in VOLATILE:
        d.pop(k, None)
    return d


def md_has_lines(md: Path, lines: list[str]) -> bool:
    text = md.read_text(encoding="utf-8").splitlines()
    return all(ln in text for ln in lines) and [ln for ln in text if ln in set(lines)] == lines


def main(scratch: Path | None) -> None:
    scratch = scratch or Path(tempfile.mkdtemp(prefix="t21_prev_"))
    print(f"scratch dir for first-run copies: {scratch}")
    for f in FILES:
        check((RES / f).exists(), f"first-run output exists: {f}")
        shutil.copy2(RES / f, scratch / f)
    if fails:
        sys.exit(1)

    print("\n=== determinism: second run under -W error ===")
    proc = subprocess.run([sys.executable, "-W", "error", str(SCRIPT), "--mode", "main"], cwd=ROOT, capture_output=True, text=True)
    check(proc.returncode == 0, f"second run exit code {proc.returncode}")
    check(proc.stderr.strip() == "", f"second run stderr empty ({len(proc.stderr)} chars)")
    if proc.stderr.strip():
        print(proc.stderr[-3000:])
    last = [ln for ln in proc.stdout.splitlines() if ln.startswith("elapsed")]
    print("  " + (last[-1] if last else "(no elapsed line)"))
    check("ALL CHECKS PASS" in proc.stdout, "second run reports ALL CHECKS PASS")
    differs: list[str] = []
    determinism: dict[str, str] = {}
    for n, f in PRED.items():
        same = sha256(RES / f) == sha256(scratch / f)
        if same:
            determinism[n] = "PASS (byte-identical)"
            print(f"PASS sha256 identical: {f} ({sha256(RES / f)[:16]})")
            continue
        differs.append(f)
        a, b = pd.read_csv(scratch / f), pd.read_csv(RES / f)
        n_days = int((a.y_pred.to_numpy() != b.y_pred.to_numpy()).sum())
        max_dp = float(np.abs(a[em.PROBA_COLS].to_numpy() - b[em.PROBA_COLS].to_numpy()).max())
        msg = f"{f} NOT byte-identical between two consecutive runs: y_pred differs on {n_days} / 192 days, max abs proba diff {max_dp:.2e}"
        if n in em.NETWORKS:
            determinism[n] = f"NON-DETERMINISTIC ({n_days} days, {max_dp:.2e})"
            result(msg + " (network: reported as a result, not a task failure; the seeds dispersion gives the scale)")
        else:
            determinism[n] = "FAIL"
            check(False, msg + " (tree model: FAIL)")
    for f in TABLES:
        same = sha256(RES / f) == sha256(scratch / f)
        if same or differs:
            print(("PASS " if same else "INFO ") + f"sha256 {'identical' if same else 'differs (expected when a prediction file differs)'}: {f}")
        else:
            check(same, f"sha256 identical: {f}")
        if not same:
            differs.append(f)
    same_json = json_without_volatile(RES / META) == json_without_volatile(scratch / META)
    if same_json or differs:
        print(("PASS " if same_json else "INFO ") + f"json {'identical' if same_json else 'differs'} without {VOLATILE}: {META}")
    else:
        check(same_json, f"json identical without {VOLATILE}: {META}")
    if not same_json:
        differs.append(META)
    print(f"  determinism per model: {determinism}")
    if differs:
        for f in FILES:                                   # keep the committed figure = first run
            shutil.copy2(scratch / f, RES / f)
        result(f"first-run outputs restored from {scratch} for {len(FILES)} files (differing: {differs})")

    print("\n=== LF, md == csv, predictions, run_meta ===")
    for f in FILES + [s for s in SEED_FILES if (RES / s).exists()]:
        check(b"\r\n" not in (RES / f).read_bytes(), f"LF only: {f}")
    check(b"\r\n" not in (ROOT / "revision" / "CIFRE_REVIZIE.md").read_bytes(), "CIFRE_REVIZIE.md LF only")
    table = pd.read_csv(RES / "t21_extra_models.csv")
    check(md_has_lines(RES / "t21_extra_models_table.md", em.metrics_md_lines(table)),
          "t21_extra_models_table.md == t21_extra_models.csv (table lines rebuilt from the csv)")
    comp = pd.read_csv(RES / "t21_comparisons.csv", keep_default_na=False)
    check(md_has_lines(RES / "t21_comparisons_table.md", em.comparisons_md_lines(comp)),
          "t21_comparisons_table.md == t21_comparisons.csv")
    check(len(table) == 9 * 11 and list(dict.fromkeys(table.system)) == em.SYSTEMS, "metrics csv: 9 systems x 11 metrics in the fixed order")
    check(len(comp) == 4 and list(comp.model) == em.NEW_MODELS, "comparisons csv: 4 rows (lgbm, catboost, tabnet, ftt)")
    g = {(r.system, r.metric): r for r in table.itertuples()}
    for r in comp.itertuples():
        check(abs(float(r.delta_rf_minus_model_f1_macro) - (float(g["rf", "f1_macro"].value) - float(g[r.model, "f1_macro"].value))) < 1e-9
              and abs(float(r.delta_model_minus_persistence_f1_macro) - (float(g[r.model, "f1_macro"].value) - float(g["persistence", "f1_macro"].value))) < 1e-9
              and int(r.n00) + int(r.n01) + int(r.n10) + int(r.n11) == 192
              and str(r.delta_rf_minus_model_f1_macro_include_zero) == str(float(r.delta_rf_minus_model_f1_macro_ci_low) <= 0 <= float(r.delta_rf_minus_model_f1_macro_ci_high))
              and str(r.recommended) == ("exact" if int(r.n01) + int(r.n10) < 25 else "cc")
              and float(r.p_cc_holm) >= float(r.p_cc) - 1e-12 and float(r.p_exact_holm) >= float(r.p_exact) - 1e-12,
              f"{r.model}: deltas == csv differences, McNemar cells sum to 192, include_zero / recommended flags, Holm >= raw")
    ref = pd.read_csv(RES / "test_predictions_rf.csv")
    meta = json.loads((RES / META).read_text(encoding="utf-8"))
    for n, f in PRED.items():
        df = pd.read_csv(RES / f)
        cm = meta["models"][n]["point"]
        sums = df[em.PROBA_COLS].sum(axis=1)
        check(len(df) == 192 and list(df.columns) == ["date", "y_true", "y_pred"] + em.PROBA_COLS
              and list(df.date) == list(ref.date) and np.array_equal(df.y_true.to_numpy(), ref.y_true.to_numpy())
              and float(np.abs(sums - 1).max()) < 1e-6
              and np.array_equal(df[em.PROBA_COLS].to_numpy().argmax(1), df.y_pred.to_numpy())
              and int((df.y_pred == 2).sum()) == cm["cm_ra"] + cm["cm_na"] + cm["cm_aa"]
              and abs(float(g[n, "f1_macro"].value) - cm["f1_macro"]) < 1e-9
              and meta["prediction_files_sha256"][f] == sha256(RES / f),
              f"{f}: 192 rows, schema, dates / y_true == rf csv, proba sum 1, argmax == y_pred, Active count == confusion matrix, sha256 == meta")
    for n in em.NETWORKS:
        mm = meta["models"][n]
        check(1 <= mm["best_epoch"] <= em.MAX_EPOCHS and mm["stage_a_epochs_run"] >= mm["best_epoch"]
              and len(mm["val_history"]) == mm["stage_a_epochs_run"] and mm["inner_active"] == [25, 13]
              and mm["inner_train_rows"] == 612 and mm["inner_val_rows"] == 154 and mm["scaled_input"] is True,
              f"{n}: best_epoch {mm['best_epoch']} <= stage A epochs {mm['stage_a_epochs_run']} <= 200, val history length, inner split 612 / 154 with 25 / 13 Active, scaled input")
    for n in ("lgbm", "catboost"):
        check(meta["models"][n]["scaled_input"] is False and "best_epoch" not in meta["models"][n], f"{n}: unscaled input, no early stopping")
    check(meta["sanity_checks_failed"] == [] and meta["seed"] == 42 and meta["smote_seed"] == 42 and meta["n_resamples"] == 1000
          and meta["torch_threads"] == 1 and meta["versions"]["torch_cuda_available"] is False
          and "vendor" in meta["versions"]["pytorch_tabnet_path"]
          and [round(v, 4) for v in meta["rf_f1_macro_marginal_ci"]] == meta["t11_rf_f1_macro_ci_from_cifre_revizie"]
          and round(meta["t210_identity"]["f1_macro"], 4) == 0.5063,
          f"run_meta: 0 failed checks, seed 42, SMOTE seed 42, 1000 resamples, 1 thread, CPU, vendored TabNet, RF CI == T1.1, "
          f"T2.1.0 0.5063 (elapsed {meta['elapsed_seconds']} s)")
    check(meta["elapsed_seconds"] < 30 * 60, "main-mode elapsed under the 30 min budget")
    if all((RES / s).exists() for s in SEED_FILES):
        seeds = pd.read_csv(RES / "t21_seeds.csv")
        check(len(seeds) == 20 and sorted(set(seeds.seed)) == [0, 1, 2, 3, 42] and list(dict.fromkeys(seeds.model)) == em.NEW_MODELS,
              "t21_seeds.csv: 20 rows, 4 models x seeds {42, 0, 1, 2, 3}")
        check(md_has_lines(RES / "t21_seeds_table.md", em.seeds_md_lines(em.seeds_summary(seeds))), "t21_seeds_table.md == t21_seeds.csv (summary rebuilt)")
        ms = json.loads((RES / "t21_run_meta_seeds.json").read_text(encoding="utf-8"))
        check(all(v["y_pred_identical"] and v["max_abs_proba_diff"] < 1e-9 for v in ms["identity_seed42_vs_main"].values())
              and ms["sanity_checks_failed"] == [] and ms["elapsed_seconds"] < 60 * 60,
              f"seeds run_meta: seed-42 identity with main for 4 / 4 models, 0 failed checks, under 60 min ({ms['elapsed_seconds']} s)")
        for n in em.NEW_MODELS:
            s42 = seeds[(seeds.model == n) & (seeds.seed == 42)].iloc[0]
            check(abs(float(s42.f1_macro) - float(g[n, "f1_macro"].value)) < 1e-9, f"{n}: seeds csv seed 42 macro F1 == main table")
    else:
        print("INFO seeds outputs not present (run --mode seeds); seeds checks skipped")

    print("\n=== git diff on protected files ===")
    tracked = subprocess.run(["git", "ls-files", "revision/results", "CIFRE_CANONICE.md"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    protected = [p for p in tracked if "/t21_" not in p]
    diff = subprocess.run(["git", "status", "--porcelain", "--"] + protected, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    check(diff == "", f"git diff empty on CIFRE_CANONICE.md and {len(protected) - 1} tracked T0.x-T3.3 result files" + (f"\n{diff}" if diff else ""))
    check(any("test_predictions_rf.csv" in p for p in protected), "test_predictions_*.csv are in the protected set")

    print(f"\ndeterminism: {determinism}")
    if results:
        print("RESULTS (reported, not failures): " + "; ".join(results))
    print(f"\n{'ALL PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=None)
    main(ap.parse_args().scratch)
