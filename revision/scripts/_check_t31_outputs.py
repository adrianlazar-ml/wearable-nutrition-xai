"""
_check_t31_outputs.py - T3.1.4: determinism and hygiene checks for threshold_sensitivity.py.

1. Copies the current T3.1 outputs (csv / md / png / pdf / json) to a scratch directory,
   then re-runs `python -W error threshold_sensitivity.py` (any warning becomes an error ->
   non-zero exit -> FAIL); stderr printed verbatim if non-empty.
2. sha256 identical between the two consecutive runs for every csv / md / png / pdf; json
   identical without run_date and elapsed_seconds.
3. LF only on all text outputs and on CIFRE_REVIZIE.md.
4. md == csv (every formatted cell of the csv appears in the md row of the same cell);
   9 prediction files x 192 rows, probabilities sum to 1, argmax consistency, dates ==
   test_predictions_rf.csv, Active count == confusion matrix; canonical row reproduces
   T1.5.0 / T0.4 / T1.1; shap csv consistent with the grid; run_meta sane.
5. git diff empty on CIFRE_CANONICE.md, results/test_predictions_*.csv and every tracked
   result file of T0.x-T1.5 (everything tracked in revision/results before T3.1).

Exits 1 on any FAIL. Evidence script, kept.
Run from repo root (after one full run):  python revision/scripts/_check_t31_outputs.py
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

import matplotlib.image
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
SCRIPT = ROOT / "revision" / "scripts" / "threshold_sensitivity.py"
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outputs() -> list[str]:
    preds = sorted(p.relative_to(RES).as_posix() for p in RES.glob("t31_predictions_rf_*.csv"))
    return ["t31_threshold_grid.csv", "t31_threshold_grid_table.md", "t31_shap_top3.csv", "t31_run_meta.json",
            "figures/t31_threshold_grid.png", "figures/t31_threshold_grid.pdf"] + preds


def json_without_volatile(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    d.pop("run_date", None)
    d.pop("elapsed_seconds", None)
    return d


def md_matches_csv(csv: Path, md: Path, key: str = "cell") -> bool:
    """Every csv cell, formatted as md_table formats it (%.4f for float columns), is in the md row of the same key."""
    df = pd.read_csv(csv, keep_default_na=False)      # empty table7_set cells stay "" as in the md
    text = md.read_text(encoding="utf-8")
    rows = {ln.strip().strip("|").split("|")[0].strip(): ln
            for ln in text.splitlines() if ln.startswith("| ") and " | " in ln}      # first column = cell
    ok = True
    for _, r in df.iterrows():
        line = rows.get(str(r[key]))
        if line is None:
            print(f"   md has no row for {key} = {r[key]!r}")
            return False
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for c in df.columns:
            v = r[c]
            cell = ("%.4f" % v) if df[c].dtype.kind == "f" else str(v)
            if cell not in cells:
                ok = False
                print(f"   md missing {c} = {cell!r} for {key} {r[key]}")
    return ok


def main(scratch: Path | None) -> None:
    scratch = scratch or Path(tempfile.mkdtemp(prefix="t31_prev_"))
    print(f"scratch dir for first-run copies: {scratch}")
    files = outputs()
    check(len(files) == 6 + 9, f"first run produced 6 + 9 files ({len(files)})")
    for f in files:
        check((RES / f).exists(), f"first-run output exists: {f}")
        (scratch / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RES / f, scratch / f)
    if fails:
        sys.exit(1)

    print("\n=== determinism: second run under -W error ===")
    proc = subprocess.run([sys.executable, "-W", "error", str(SCRIPT)], cwd=ROOT, capture_output=True, text=True)
    check(proc.returncode == 0, f"second run exit code {proc.returncode}")
    check(proc.stderr.strip() == "", f"second run stderr empty ({len(proc.stderr)} chars)")
    if proc.stderr.strip():
        print(proc.stderr)
    last = [ln for ln in proc.stdout.splitlines() if ln.startswith("elapsed")]
    print("  " + (last[-1] if last else "(no elapsed line)"))
    check("ALL CHECKS PASS" in proc.stdout, "second run reports ALL CHECKS PASS")
    for f in files:
        if f.endswith(".json"):
            check(json_without_volatile(RES / f) == json_without_volatile(scratch / f),
                  f"json identical without run_date / elapsed_seconds: {f}")
        else:
            check(sha256(RES / f) == sha256(scratch / f), f"sha256 identical: {f} ({sha256(RES / f)[:16]})")

    print("\n=== LF, md == csv, grid, predictions, shap, figure, meta ===")
    for f in files:
        if f.endswith((".csv", ".md", ".json")):
            check(b"\r\n" not in (RES / f).read_bytes(), f"LF only: {f}")
    check(b"\r\n" not in (ROOT / "revision" / "CIFRE_REVIZIE.md").read_bytes(), "CIFRE_REVIZIE.md LF only")
    check(md_matches_csv(RES / "t31_threshold_grid.csv", RES / "t31_threshold_grid_table.md"),
          "t31_threshold_grid_table.md == t31_threshold_grid.csv")
    grid = pd.read_csv(RES / "t31_threshold_grid.csv")
    check(len(grid) == 9 and sorted(grid.cell) == sorted(f"R{r}_A{a}" for r in (55, 60, 65) for a in (80, 85, 90))
          and (grid.A_s == 75).all(), "grid: 9 cells R{55,60,65} x A{80,85,90}, A_s = 75")
    can = grid[grid.canonical].iloc[0]
    check(can.cell == "R60_A85" and round(float(can.f1_macro), 4) == 0.5063 and round(float(can.accuracy), 4) == 0.6406
          and round(float(can.mcc), 4) == 0.1416 and round(float(can.pers_f1_macro), 4) == 0.3039
          and (round(float(can.pers_f1_macro_ci_low), 4), round(float(can.pers_f1_macro_ci_high), 4)) == (0.2612, 0.3501)
          and (round(float(can.f1_macro_ci_low), 4), round(float(can.f1_macro_ci_high), 4)) == (0.3784, 0.6130)
          and round(float(can.maj_f1_macro), 4) == 0.2811 and round(float(can.maj_accuracy), 4) == 0.7292
          and int(can.n_skipped) == 0,
          "canonical row: 0.5063 / 0.6406 / 0.1416; persistence 0.3039 [0.2612, 0.3501]; RF CI [0.3784, 0.6130]; "
          "majority 0.2811 / 0.7292; 0 skipped")
    t7 = {("R55_A85", 0.505, (97, 815, 46)), ("R60_A85", 0.506, (202, 710, 46)), ("R65_A85", 0.456, (360, 552, 46))}
    g = grid.set_index("cell")
    check(all(round(float(g.loc[c, "f1_macro"]), 3) == v and
              (int(g.loc[c, "n_rest"]), int(g.loc[c, "n_normal"]), int(g.loc[c, "n_active"])) == d for c, v, d in t7),
          "Table 7 cells T2 / T3 / T5 reproduced (0.505 / 0.506 / 0.456; 97/815/46, 202/710/46, 360/552/46)")
    check(bool((grid.n_rest + grid.n_normal + grid.n_active == 958).all())
          and bool((grid.train_rest + grid.train_normal + grid.train_active == 766).all())
          and bool((grid.test_rest + grid.test_normal + grid.test_active == 192).all()),
          "every cell sums to 958 / 766 / 192")
    check(bool((grid.comparable == ((grid.test_active >= 5) & (grid.train_active >= 4))).all())
          and int(grid.comparable.sum()) == 6, "comparable flag == (test Active >= 5 and train Active >= 4); 6 of 9")
    check(bool(((grid.n_skipped == 0) | (grid.test_active < 5)).all()), "skipped resamples only in cells with < 5 Active in test")
    ref = pd.read_csv(RES / "test_predictions_rf.csv")
    meta = json.loads((RES / "t31_run_meta.json").read_text(encoding="utf-8"))
    for f in files:
        if "t31_predictions_rf_" not in f:
            continue
        df = pd.read_csv(RES / f)
        cell = Path(f).stem.replace("t31_predictions_rf_", "")
        row = g.loc[cell]
        sums = df[["p_rest", "p_normal", "p_active"]].sum(axis=1)
        ok = (len(df) == 192 and list(df.columns) == ["date", "y_true", "y_pred", "p_rest", "p_normal", "p_active"]
              and list(df.date) == list(ref.date) and float(np.abs(sums - 1).max()) < 1e-6
              and np.array_equal(df[["p_rest", "p_normal", "p_active"]].to_numpy().argmax(1), df.y_pred.to_numpy())
              and int((df.y_pred == 2).sum()) == int(row.cm_ra + row.cm_na + row.cm_aa)
              and int((df.y_true == 2).sum()) == int(row.test_active)
              and meta["prediction_files_sha256"][Path(f).name] == sha256(RES / f))
        if cell == "R60_A85":
            ok = ok and np.array_equal(df.y_true.to_numpy(), ref.y_true.to_numpy()) \
                and np.array_equal(df.y_pred.to_numpy(), ref.y_pred.to_numpy())
        check(ok, f"{f}: 192 rows, dates == rf csv, proba sum 1, argmax == y_pred, Active counts == grid, sha256 == meta")
    shap_top = pd.read_csv(RES / "t31_shap_top3.csv")
    ok = len(shap_top) == 27 and sorted(shap_top["rank"].unique()) == [1, 2, 3]
    for cell, sub in shap_top.groupby("cell"):
        sub = sub.sort_values("rank")
        ok = ok and list(sub.feature) == [g.loc[cell, f"shap_top{j}"] for j in (1, 2, 3)] \
            and np.allclose(sub.mean_abs_shap.to_numpy(), [g.loc[cell, f"shap_top{j}_value"] for j in (1, 2, 3)])
    check(ok, "t31_shap_top3.csv: 27 rows, consistent with the grid's shap_top1..3 names and values")
    check(bool((grid[grid.comparable].aw_hr_mean_rank_1 == (grid[grid.comparable].shap_top1 == "aw_hr_mean")).all()),
          "aw_hr_mean_rank_1 flag consistent with shap_top1 on comparable cells")
    img = matplotlib.image.imread(RES / "figures/t31_threshold_grid.png")
    check(img.ndim == 3 and img.shape[1] > img.shape[0] > 700, f"png opens {img.shape}")
    check(meta["sanity_checks_failed"] == [] and meta["figure_axes"] == 4 and meta["seed"] == 42
          and meta["n_resamples"] == 1000 and meta["grid"] == {"R": [55, 60, 65], "A_r": [80, 85, 90], "A_s": 75, "n_cells": 9}
          and meta["n_comparable"] == 6
          and [round(v, 4) for v in meta["canonical_rf_f1_macro_marginal_ci"]] == meta["t11_rf_f1_macro_ci_from_cifre_revizie"]
          and all(meta["labels_identical_to_table7_90_80"].values()),
          f"run_meta: 0 failed checks, 4 axes, seed 42, 1000 resamples, grid 3 x 3, 6 comparable, CI == T1.1, "
          f"A_r = 90 labels == Table 7 90/80 (elapsed {meta['elapsed_seconds']} s)")
    check(meta["elapsed_seconds"] < 20 * 60, f"elapsed {meta['elapsed_seconds']} s under the 20 min budget")

    print("\n=== git diff on protected files ===")
    tracked = subprocess.run(["git", "ls-files", "revision/results", "CIFRE_CANONICE.md"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    # T3.1.6 (same commit) changes only the png / pdf of the t15 figure (titles, labels, marker) and the run_date /
    # elapsed_seconds of t15_run_meta.json; the t15 csv / md tables stay protected.
    protected = [t for t in tracked if "/t31_" not in t and "figures/t15_group_ablation." not in t
                 and not t.endswith("t15_run_meta.json")]
    diff = subprocess.run(["git", "status", "--porcelain", "--"] + protected, cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    check(diff == "", f"git diff empty on CIFRE_CANONICE.md and {len(protected) - 1} tracked T0.x-T1.5 result files"
          + (f"\n{diff}" if diff else ""))
    check(any("test_predictions_rf.csv" in t for t in protected), "test_predictions_*.csv are in the protected set")

    print(f"\n{'ALL PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=None)
    main(ap.parse_args().scratch)
