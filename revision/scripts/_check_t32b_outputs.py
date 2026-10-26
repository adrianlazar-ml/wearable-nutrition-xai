"""
_check_t32b_outputs.py - T3.2b: determinism and hygiene checks for watch_only_sensitivity.py.

1. Copies the current T3.2b outputs (csv / md / json) to a scratch directory, then re-runs
   `python -W error watch_only_sensitivity.py` (any warning -> error -> FAIL); stderr printed.
2. sha256 identical between the two consecutive runs for csv and md; json identical without
   run_date and elapsed_seconds.
3. LF only on the three outputs and on CIFRE_REVIZIE.md.
4. md == csv (every formatted cell of the csv appears in the md row with the same series key);
   2 rows (parser_all, watch_only); n == n_rest + n_normal + n_active; Cliff's delta in [-1, 1]
   with the CI containing the point on the 3 pairs; Spearman in [-1, 1] and equal on both rows;
   run_meta consistent with the csv (n, distribution, Spearman, 0 failed checks, seed 42, 1,000
   resamples, per-year counts sum to n, identity mismatches within the documented exceptions).
5. git diff empty on CIFRE_CANONICE.md and every tracked file of revision/results that is not a
   T3.2b output (T0.x-T3.3 results, test_predictions_*.csv, the T3.2 outputs themselves).

Exits 1 on any FAIL. Evidence script, kept.
Run from repo root (after one full run):  python revision/scripts/_check_t32b_outputs.py
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

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
SCRIPT = ROOT / "revision" / "scripts" / "watch_only_sensitivity.py"
FILES = ["t32b_watch_only_hr.csv", "t32b_watch_only_hr_table.md", "t32b_run_meta.json"]
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


def md_matches_csv(csv: Path, md: Path, keys: list[str]) -> bool:
    """Every csv cell, formatted as md_table formats it (%.4f floats), is in the md row with the same key."""
    df = pd.read_csv(csv, keep_default_na=False, na_values=[""])
    cols = None
    rows: dict[tuple, list[str]] = {}
    for ln in md.read_text(encoding="utf-8").splitlines():
        if not ln.startswith("| "):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cols is None:
            cols = cells
            continue
        rows[tuple(cells[cols.index(k)] for k in keys)] = cells
    ok = cols == list(df.columns)
    if not ok:
        print(f"   md columns != csv columns")
    for _, r in df.iterrows():
        line = rows.get(tuple(str(r[k]) for k in keys))
        if line is None:
            print(f"   md has no row for {[r[k] for k in keys]}")
            return False
        for c in df.columns:
            v = r[c]
            cell = ("%.4f" % v) if df[c].dtype.kind == "f" else str(v)
            if cell not in line:
                ok = False
                print(f"   md missing {c} = {cell!r} for {[r[k] for k in keys]}")
    return ok


def main(scratch: Path | None) -> None:
    scratch = scratch or Path(tempfile.mkdtemp(prefix="t32b_prev_"))
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"scratch dir for first-run copies: {scratch}")
    for f in FILES:
        check((RES / f).exists(), f"first-run output exists: {f}")
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

    print("\n=== LF, md == csv, table, meta ===")
    for f in FILES:
        check(b"\r\n" not in (RES / f).read_bytes(), f"LF only: {f}")
    check(b"\r\n" not in (ROOT / "revision" / "CIFRE_REVIZIE.md").read_bytes(), "CIFRE_REVIZIE.md LF only")
    check(md_matches_csv(RES / FILES[0], RES / FILES[1], ["series"]), f"{FILES[1]} == {FILES[0]}")
    t = pd.read_csv(RES / FILES[0])
    check(list(t.series) == ["parser_all", "watch_only"], "2 rows: parser_all, watch_only")
    check(bool((t.n == t.n_rest + t.n_normal + t.n_active).all()) and t.n.nunique() == 1
          and bool((t[["n_rest", "n_normal", "n_active"]].nunique() == 1).all()),
          f"n == n_rest + n_normal + n_active, same days for both series (n {int(t.n.iloc[0])}, "
          f"{int(t.n_rest.iloc[0])} / {int(t.n_normal.iloc[0])} / {int(t.n_active.iloc[0])})")
    ok = True
    for nm in ("active_rest", "active_normal", "normal_rest"):
        c = t[f"cliff_{nm}"]
        ok &= bool(((c >= -1) & (c <= 1)).all() and ((t[f"cliff_{nm}_ci_low"] <= c) & (c <= t[f"cliff_{nm}_ci_high"])).all())
    check(ok, "Cliff's delta in [-1, 1], CI contains the point, 3 pairs x 2 series")
    rho = float(t.spearman_parser_vs_watch.iloc[0])
    check(-1 <= rho <= 1 and t.spearman_parser_vs_watch.nunique() == 1, f"Spearman in [-1, 1], equal on both rows ({rho:.4f})")
    for cls in ("rest", "normal", "active"):
        check(bool((t[f"p25_{cls}"] <= t[f"median_{cls}"]).all() and (t[f"median_{cls}"] <= t[f"p75_{cls}"]).all()),
              f"p25 <= median <= p75 ({cls})")
    meta = json.loads((RES / FILES[2]).read_text(encoding="utf-8"))
    n = int(t.n.iloc[0])
    check(meta["sanity_checks_failed"] == [] and meta["seed"] == 42 and meta["n_boot"] == 1000 and meta["n"] == n
          and meta["label_distribution"] == [int(t.n_rest.iloc[0]), int(t.n_normal.iloc[0]), int(t.n_active.iloc[0])]
          and sum(v["n_subset"] for v in meta["per_year"].values()) == n
          and sum(v["n_labelled"] for v in meta["per_year"].values()) == 958
          and abs(meta["spearman"]["rho"] - rho) < 1e-12
          and set(meta["identity"]["mismatch_days_all_labelled"]) <= set(meta["identity"]["documented_exceptions"])
          and all(abs(meta["results"][s]["cliff_active_rest"] - float(t.set_index("series").loc[s, "cliff_active_rest"])) < 1e-12
                  for s in ("parser_all", "watch_only")),
          f"run_meta: 0 failed checks, seed 42, 1000 resamples, n / distribution / Spearman / Cliff == csv, per-year "
          f"sums {n} / 958, identity mismatches within the documented exceptions (elapsed {meta['elapsed_seconds']} s)")
    check(meta["spearman"]["reproduced"] is True, f"Spearman reproduces the audit value {meta['spearman']['f4_reference']} "
                                                  f"on {meta['spearman']['f4_n']} days")

    print("\n=== git diff on protected files ===")
    tracked = subprocess.run(["git", "ls-files", "revision/results", "CIFRE_CANONICE.md"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    protected = [p for p in tracked if "/t32b_" not in p]
    diff = subprocess.run(["git", "status", "--porcelain", "--"] + protected, cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    check(diff == "", f"git diff empty on CIFRE_CANONICE.md and {len(protected) - 1} tracked T0.x-T3.3 result files"
          + (f"\n{diff}" if diff else ""))
    check(any("test_predictions_rf.csv" in p for p in protected) and any("/t32_tests.csv" in p for p in protected),
          "test_predictions_*.csv and the T3.2 outputs are in the protected set")

    print(f"\n{'ALL PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=None)
    main(ap.parse_args().scratch)
