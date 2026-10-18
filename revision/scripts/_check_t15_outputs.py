"""
_check_t15_outputs.py - T1.5.5: determinism and hygiene checks for group_ablation.py.

1. Copies the current T1.5 outputs (csv / md / png / pdf / json) to a scratch
   directory, then re-runs `python -W error group_ablation.py` (any warning
   becomes an error -> non-zero exit -> FAIL); stderr printed verbatim if non-empty.
2. sha256 identical between the two consecutive runs for every csv / md / png /
   pdf; json identical without run_date and elapsed_seconds.
3. LF only on all text outputs and on CIFRE_REVIZIE.md.
4. md == csv for both tables (every formatted cell of the csv appears in the md
   row of the same group); 12 prediction files x 192 rows, probabilities sum to 1,
   argmax consistency, y_true == test_predictions_rf.csv; run_meta sane.
5. git diff empty on CIFRE_CANONICE.md, results/test_predictions_*.csv and every
   tracked result file of T0.x-T1.4 (everything tracked in revision/results before T1.5).

Exits 1 on any FAIL. Evidence script, kept.
Run from repo root (after one full run):  python revision/scripts/_check_t15_outputs.py
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
SCRIPT = ROOT / "revision" / "scripts" / "group_ablation.py"
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outputs() -> list[str]:
    preds = sorted(p.relative_to(RES).as_posix() for p in RES.glob("t15_predictions_*.csv"))
    return ["t15_logo.csv", "t15_logo_table.md", "t15_kogi.csv", "t15_kogi_table.md", "t15_run_meta.json",
            "figures/t15_group_ablation.png", "figures/t15_group_ablation.pdf"] + preds


def json_without_volatile(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    d.pop("run_date", None)
    d.pop("elapsed_seconds", None)
    return d


def md_matches_csv(csv: Path, md: Path) -> bool:
    """Every csv cell, formatted as the md formats it, is in the md row of the same group."""
    df = pd.read_csv(csv, keep_default_na=False)
    text = md.read_text(encoding="utf-8")
    rows = {ln.split(" | ")[1]: ln for ln in text.splitlines() if ln.startswith("| ") and " | " in ln}
    ok = True
    for _, r in df.iterrows():
        line = rows.get(str(r["group"]))
        if line is None:
            return False
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for c in df.columns:
            v = str(r[c])                       # csv read as text: "" for the full row's n/a cells
            if v == "":
                cell = "n/a" if ("ci_" in c or "prop" in c) else ""
            elif c.startswith(("f1_", "accuracy", "mcc", "balanced", "delta_f1_macro", "delta_f1_active")) \
                    and not c.endswith("include_zero"):
                cell = "%.4f" % float(v)        # float columns, formatted as md_table does
            else:
                cell = v
            if cell not in cells:
                ok = False
                print(f"   md missing {c} = {cell!r} for group {r['group']}")
    return ok


def main(scratch: Path | None) -> None:
    scratch = scratch or Path(tempfile.mkdtemp(prefix="t15_prev_"))
    print(f"scratch dir for first-run copies: {scratch}")
    files = outputs()
    check(len(files) == 7 + 12, f"first run produced 7 + 12 files ({len(files)})")
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

    print("\n=== LF, md == csv, predictions, figure, meta ===")
    for f in files:
        if f.endswith((".csv", ".md", ".json")):
            check(b"\r\n" not in (RES / f).read_bytes(), f"LF only: {f}")
    check(b"\r\n" not in (ROOT / "revision" / "CIFRE_REVIZIE.md").read_bytes(), "CIFRE_REVIZIE.md LF only")
    check(md_matches_csv(RES / "t15_logo.csv", RES / "t15_logo_table.md"), "t15_logo_table.md == t15_logo.csv")
    check(md_matches_csv(RES / "t15_kogi.csv", RES / "t15_kogi_table.md"), "t15_kogi_table.md == t15_kogi.csv")
    logo = pd.read_csv(RES / "t15_logo.csv")
    kogi = pd.read_csv(RES / "t15_kogi.csv")
    check(len(logo) == 7 and logo.loc[0, "config"] == "full" and round(float(logo.loc[0, "f1_macro"]), 4) == 0.5063
          and round(float(logo.loc[0, "accuracy"]), 4) == 0.6406 and round(float(logo.loc[0, "mcc"]), 4) == 0.1416,
          "t15_logo.csv: 7 rows, full row = 0.5063 / 0.6406 / 0.1416")
    check(len(kogi) == 6, "t15_kogi.csv: 6 rows")
    ref = pd.read_csv(RES / "test_predictions_rf.csv")
    meta = json.loads((RES / "t15_run_meta.json").read_text(encoding="utf-8"))
    for f in files:
        if "t15_predictions_" not in f:
            continue
        df = pd.read_csv(RES / f)
        kind, grp = Path(f).stem.replace("t15_predictions_", "").split("_", 1)
        table = logo if kind == "logo" else kogi
        row = table[table.group == grp].iloc[0]
        sums = df[["p_rest", "p_normal", "p_active"]].sum(axis=1)
        ok = (len(df) == 192 and list(df.columns) == ["date", "y_true", "y_pred", "p_rest", "p_normal", "p_active"]
              and np.array_equal(df.y_true.to_numpy(), ref.y_true.to_numpy())
              and list(df.date) == list(ref.date) and float(np.abs(sums - 1).max()) < 1e-6
              and np.array_equal(df[["p_rest", "p_normal", "p_active"]].to_numpy().argmax(1), df.y_pred.to_numpy())
              and int((df.y_pred == 2).sum()) == int(row.cm_ra + row.cm_na + row.cm_aa)
              and meta["prediction_files_sha256"][Path(f).name] == sha256(RES / f))
        check(ok, f"{f}: 192 rows, dates / y_true == rf csv, proba sum 1, argmax == y_pred, Active count == cm, sha256 == meta")
    img = matplotlib.image.imread(RES / "figures/t15_group_ablation.png")
    check(img.ndim == 3 and img.shape[1] > img.shape[0] > 800, f"png opens {img.shape}")
    check(meta["sanity_checks_failed"] == [] and meta["figure_axes"] == 2 and meta["seed"] == 42
          and meta["n_resamples"] == 1000 and meta["SLEEP_SCORE_GROUP"] == "G1"
          and sum(len(v) for v in meta["groups"].values()) == 19
          and [round(v, 4) for v in meta["full_f1_macro_marginal_ci"]] == meta["t11_rf_f1_macro_ci_from_cifre_revizie"],
          f"run_meta: 0 failed checks, 2 axes, seed 42, 1000 resamples, G1, 19 features, marginal CI == T1.1 "
          f"(elapsed {meta['elapsed_seconds']} s)")

    print("\n=== git diff on protected files ===")
    tracked = subprocess.run(["git", "ls-files", "revision/results", "CIFRE_CANONICE.md"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    protected = [t for t in tracked if "/t15_" not in t]
    diff = subprocess.run(["git", "status", "--porcelain", "--"] + protected, cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    check(diff == "", f"git diff empty on CIFRE_CANONICE.md and {len(protected) - 1} tracked T0.x-T1.4 result files"
          + (f"\n{diff}" if diff else ""))
    check(any("test_predictions_rf.csv" in t for t in protected), "test_predictions_*.csv are in the protected set")

    print(f"\n{'ALL PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=None)
    main(ap.parse_args().scratch)
