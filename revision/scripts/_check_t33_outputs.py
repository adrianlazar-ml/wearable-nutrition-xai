"""
_check_t33_outputs.py - T3.3.4: determinism and hygiene checks for expanding_window.py.

1. Copies the current T3.3 outputs to a scratch directory, then re-runs
   `python -W error expanding_window.py` (any warning -> error -> FAIL); stderr printed.
2. sha256 identical between the two consecutive runs for every csv / md / png / pdf; json
   identical without run_date and elapsed_seconds.
3. LF only on all text outputs and on CIFRE_REVIZIE.md.
4. md == csv (every formatted cell of the csv appears in the md row of the same fold);
   4 prediction files x 192 rows (fold 1 == test_predictions_rf.csv), probabilities sum to 1,
   argmax consistency, Active counts == table; fold 1 reproduces T1.5.0 / T0.4 / T1.1;
   pooled row == concatenation of the 4 prediction files; shap csv consistent; run_meta sane.
5. git diff empty on CIFRE_CANONICE.md, results/test_predictions_*.csv and every tracked result
   file of T0.x-T3.2 (everything tracked in revision/results before T3.3).

Exits 1 on any FAIL. Evidence script, kept.
Run from repo root (after one full run):  python revision/scripts/_check_t33_outputs.py
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
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expanding_window import fmt_cell  # noqa: E402  (same md formatting rule as the producer)

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
SCRIPT = ROOT / "revision" / "scripts" / "expanding_window.py"
FILES = ["t33_expanding_window.csv", "t33_expanding_window_table.md", "t33_shap_top3.csv", "t33_run_meta.json",
         "figures/t33_expanding_window.png", "figures/t33_expanding_window.pdf"] + \
        [f"t33_predictions_rf_fold{k}.csv" for k in (1, 2, 3, 4)]
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_without_volatile(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    d.pop("run_date", None)
    d.pop("elapsed_seconds", None)
    return d


def md_matches_csv(csv: Path, md: Path, key: str = "fold") -> bool:
    df = pd.read_csv(csv, keep_default_na=False)
    text = md.read_text(encoding="utf-8")
    rows = {ln.strip().strip("|").split("|")[0].strip(): ln for ln in text.splitlines() if ln.startswith("| ")}
    ok = True
    for _, r in df.iterrows():
        line = rows.get(str(r[key]))
        if line is None:
            print(f"   md has no row for {key} = {r[key]!r}")
            return False
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for c in df.columns:
            cell = fmt_cell(c, r[c])          # csv read as text: "" -> n/a, score columns -> %.4f (same rule as the script)
            if cell not in cells:
                ok = False
                print(f"   md missing {c} = {cell!r} for {key} {r[key]}")
    return ok


def main(scratch: Path | None) -> None:
    scratch = scratch or Path(tempfile.mkdtemp(prefix="t33_prev_"))
    print(f"scratch dir for first-run copies: {scratch}")
    for f in FILES:
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
    for f in FILES:
        if f.endswith(".json"):
            check(json_without_volatile(RES / f) == json_without_volatile(scratch / f),
                  f"json identical without run_date / elapsed_seconds: {f}")
        else:
            check(sha256(RES / f) == sha256(scratch / f), f"sha256 identical: {f} ({sha256(RES / f)[:16]})")

    print("\n=== LF, md == csv, folds, predictions, pooled, shap, figure, meta ===")
    for f in FILES:
        if f.endswith((".csv", ".md", ".json")):
            check(b"\r\n" not in (RES / f).read_bytes(), f"LF only: {f}")
    check(b"\r\n" not in (ROOT / "revision" / "CIFRE_REVIZIE.md").read_bytes(), "CIFRE_REVIZIE.md LF only")
    check(md_matches_csv(RES / "t33_expanding_window.csv", RES / "t33_expanding_window_table.md"),
          "t33_expanding_window_table.md == t33_expanding_window.csv")
    t = pd.read_csv(RES / "t33_expanding_window.csv", keep_default_na=False)
    check(len(t) == 5 and list(t.fold) == ["1", "2", "3", "4", "pooled"], "5 rows: folds 1-4 + pooled")
    g = t.set_index("fold")
    f1 = g.loc["1"]
    check(round(float(f1.f1_macro), 4) == 0.5063 and round(float(f1.accuracy), 4) == 0.6406 and round(float(f1.mcc), 4) == 0.1416
          and round(float(f1.pers_f1_macro), 4) == 0.3039
          and (round(float(f1.pers_f1_macro_ci_low), 4), round(float(f1.pers_f1_macro_ci_high), 4)) == (0.2612, 0.3501)
          and (round(float(f1.f1_macro_ci_low), 4), round(float(f1.f1_macro_ci_high), 4)) == (0.3784, 0.6130)
          and f1.test_rows == "766-957" and int(f1.n_train) == 766,
          "fold 1 row: 0.5063 / 0.6406 / 0.1416; persistence 0.3039 [0.2612, 0.3501]; RF CI [0.3784, 0.6130]; rows 766-957")
    check([g.loc[k, "test_rows"] for k in ("2", "3", "4")] == ["574-765", "382-573", "190-381"]
          and [int(g.loc[k, "n_train"]) for k in ("2", "3", "4")] == [574, 382, 190], "fold rows 2-4 as designed")
    ref = pd.read_csv(RES / "test_predictions_rf.csv")
    meta = json.loads((RES / "t33_run_meta.json").read_text(encoding="utf-8"))
    yt_all, yp_all = [], []
    for k in (1, 2, 3, 4):
        f = f"t33_predictions_rf_fold{k}.csv"
        df = pd.read_csv(RES / f)
        row = g.loc[str(k)]
        sums = df[["p_rest", "p_normal", "p_active"]].sum(axis=1)
        ok = (len(df) == 192 and list(df.columns) == ["fold", "date", "y_true", "y_pred", "p_rest", "p_normal", "p_active"]
              and (df.fold == k).all() and float(np.abs(sums - 1).max()) < 1e-6
              and np.array_equal(df[["p_rest", "p_normal", "p_active"]].to_numpy().argmax(1), df.y_pred.to_numpy())
              and int((df.y_pred == 2).sum()) == int(row.cm_ra) + int(row.cm_na) + int(row.cm_aa)
              and int((df.y_true == 2).sum()) == int(row.test_active)
              and df.date.iloc[0] == row.test_start and df.date.iloc[-1] == row.test_end
              and meta["prediction_files_sha256"][f] == sha256(RES / f))
        if k == 1:
            ok = ok and np.array_equal(df.y_true.to_numpy(), ref.y_true.to_numpy()) \
                and np.array_equal(df.y_pred.to_numpy(), ref.y_pred.to_numpy()) and list(df.date) == list(ref.date)
        check(ok, f"{f}: 192 rows, fold column, proba sum 1, argmax == y_pred, Active counts == table, dates == table, sha256 == meta")
        yt_all.append(df.y_true.to_numpy())
        yp_all.append(df.y_pred.to_numpy())
    yt, yp = np.concatenate(yt_all), np.concatenate(yp_all)
    pooled = g.loc["pooled"]
    check(int(pooled.n_test) == 768
          and abs(float(pooled.f1_macro) - f1_score(yt, yp, labels=[0, 1, 2], average="macro", zero_division=0)) < 1e-9
          and abs(float(pooled.accuracy) - float(np.mean(yt == yp))) < 1e-9
          and int(pooled.test_active) == int((yt == 2).sum())
          and float(pooled.f1_macro_ci_low) <= float(pooled.f1_macro) <= float(pooled.f1_macro_ci_high),
          "pooled row == concatenation of the 4 prediction files (768 days), CI contains the point")
    shap_top = pd.read_csv(RES / "t33_shap_top3.csv")
    ok = len(shap_top) == 12
    for k, sub in shap_top.groupby("fold"):
        sub = sub.sort_values("rank")
        ok = ok and list(sub.feature) == [g.loc[str(k), f"shap_top{j}"] for j in (1, 2, 3)]
    check(ok, "t33_shap_top3.csv: 12 rows, consistent with the table")
    for k in ("1", "2", "3", "4"):
        r = g.loc[k]
        check(abs(float(r.delta_f1_macro) - (float(r.f1_macro) - float(r.pers_f1_macro))) < 1e-9
              and (str(r.delta_f1_macro_include_zero) == str(float(r.delta_f1_macro_ci_low) <= 0 <= float(r.delta_f1_macro_ci_high)))
              and str(r.comparable) == str(int(r.test_active) >= 5 and int(r.train_active) >= 4),
              f"fold {k}: delta == RF minus persistence, include_zero and comparable flags consistent")
    img = matplotlib.image.imread(RES / "figures/t33_expanding_window.png")
    check(img.ndim == 3 and img.shape[1] > img.shape[0] > 700, f"png opens {img.shape}")
    check(meta["sanity_checks_failed"] == [] and meta["figure_axes"] == 1 and meta["seed"] == 42
          and meta["n_resamples"] == 1000 and meta["n_folds"] == 4 and meta["block"] == 192
          and [round(v, 4) for v in meta["fold1_rf_f1_macro_marginal_ci"]] == meta["t11_rf_f1_macro_ci_from_cifre_revizie"]
          and meta["pooled"]["n_test"] == 768,
          f"run_meta: 0 failed checks, 1 axis, seed 42, 1000 resamples, 4 folds x 192, CI == T1.1, pooled 768 "
          f"(elapsed {meta['elapsed_seconds']} s)")
    check(meta["elapsed_seconds"] < 20 * 60, "elapsed under the 20 min budget")

    print("\n=== git diff on protected files ===")
    tracked = subprocess.run(["git", "ls-files", "revision/results", "CIFRE_CANONICE.md"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    protected = [p for p in tracked if "/t33_" not in p]
    diff = subprocess.run(["git", "status", "--porcelain", "--"] + protected, cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    check(diff == "", f"git diff empty on CIFRE_CANONICE.md and {len(protected) - 1} tracked T0.x-T3.2 result files"
          + (f"\n{diff}" if diff else ""))
    check(any("test_predictions_rf.csv" in p for p in protected), "test_predictions_*.csv are in the protected set")

    print(f"\n{'ALL PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=None)
    main(ap.parse_args().scratch)
