"""
_check_t32_outputs.py - T3.2.5: determinism and hygiene checks for label_agreement.py.

1. Copies the current T3.2 outputs (csv / md / png / pdf / json) to a scratch directory, then
   re-runs `python -W error label_agreement.py` (any warning -> error -> FAIL); stderr printed.
2. sha256 identical between the two consecutive runs for every csv / md / png / pdf; json
   identical without run_date and elapsed_seconds.
3. LF only on all text outputs and on CIFRE_REVIZIE.md.
4. md == csv for both tables (every formatted cell of the csv appears in the md row of the same
   key: variable + cls for descriptives, variable for tests); inventory n per class == tests;
   p in [0, 1], epsilon2 in [0, 1], Cliff's delta CI contains the point; 13 rows; run_meta sane.
5. git diff empty on CIFRE_CANONICE.md, results/test_predictions_*.csv and every tracked result
   file of T0.x-T3.1 (everything tracked in revision/results before T3.2).

Exits 1 on any FAIL. Evidence script, kept.
Run from repo root (after one full run):  python revision/scripts/_check_t32_outputs.py
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
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
SCRIPT = ROOT / "revision" / "scripts" / "label_agreement.py"
FILES = ["t32_inventory.csv", "t32_descriptives.csv", "t32_descriptives_table.md", "t32_tests.csv",
         "t32_tests_table.md", "t32_run_meta.json", "figures/t32_indicator_boxplots.png",
         "figures/t32_indicator_boxplots.pdf", "figures/t32_indicator_boxplots_oura.png",
         "figures/t32_indicator_boxplots_oura.pdf"]
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


def md_matches_csv(csv: Path, md: Path, keys: list[str], fmt: dict[str, str]) -> bool:
    """Every csv cell, formatted as md_table formats it, is in the md row with the same key columns."""
    df = pd.read_csv(csv, keep_default_na=False)
    text = md.read_text(encoding="utf-8")
    cols = None
    rows: dict[tuple, list[str]] = {}
    for ln in text.splitlines():
        if not ln.startswith("| "):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if cols is None:
            cols = cells
            continue
        if set(c.strip("-") for c in cells) == {""}:
            continue
        rows[tuple(cells[cols.index(k)] for k in keys)] = cells
    ok = True
    for _, r in df.iterrows():
        line = rows.get(tuple(str(r[k]) for k in keys))
        if line is None:
            print(f"   md has no row for {[r[k] for k in keys]}")
            return False
        for c in df.columns:
            v = r[c]
            cell = (fmt[c] % v) if c in fmt else ("%.4f" % v if df[c].dtype.kind == "f" else str(v))
            if cell not in line:
                ok = False
                print(f"   md missing {c} = {cell!r} for {[r[k] for k in keys]}")
    return ok


def main(scratch: Path | None) -> None:
    scratch = scratch or Path(tempfile.mkdtemp(prefix="t32_prev_"))
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

    print("\n=== LF, md == csv, tables, figures, meta ===")
    for f in FILES:
        if f.endswith((".csv", ".md", ".json")):
            check(b"\r\n" not in (RES / f).read_bytes(), f"LF only: {f}")
    check(b"\r\n" not in (ROOT / "revision" / "CIFRE_REVIZIE.md").read_bytes(), "CIFRE_REVIZIE.md LF only")
    tests = pd.read_csv(RES / "t32_tests.csv")
    pcols = [c for c in tests.columns if c.startswith(("kw_p", "dunn_p"))]
    check(md_matches_csv(RES / "t32_descriptives.csv", RES / "t32_descriptives_table.md", ["variable", "cls"], {}),
          "t32_descriptives_table.md == t32_descriptives.csv")
    check(md_matches_csv(RES / "t32_tests.csv", RES / "t32_tests_table.md", ["variable"], {c: "%.3e" for c in pcols}),
          "t32_tests_table.md == t32_tests.csv")
    inv = pd.read_csv(RES / "t32_inventory.csv").set_index("variable")
    desc = pd.read_csv(RES / "t32_descriptives.csv")
    check(len(tests) == 13 and len(inv) == 13 and len(desc) == 39, "13 variables: 13 test rows, 13 inventory rows, 39 descriptive rows")
    t = tests.set_index("variable")
    check(bool((t.loc[inv.index, ["n_rest", "n_normal", "n_active"]].to_numpy()
                == inv[["n_rest", "n_normal", "n_active"]].to_numpy()).all()), "tests n per class == inventory")
    check(bool(((tests[pcols] >= 0) & (tests[pcols] <= 1)).all().all())
          and bool(((tests.epsilon2 >= 0) & (tests.epsilon2 <= 1)).all()), "p in [0, 1], epsilon2 in [0, 1]")
    ok = True
    for nm in ("active_rest", "active_normal", "normal_rest"):
        c = tests[f"cliff_{nm}"]
        ok &= bool(((c >= -1) & (c <= 1)).all()
                   and ((tests[f"cliff_{nm}_ci_low"] <= c) & (c <= tests[f"cliff_{nm}_ci_high"])).all())
    check(ok, "Cliff's delta in [-1, 1], CI contains the point, all 3 pairs")
    sign_ok = all((r.expected_active_vs_rest == "0" and r.agreement == "no expectation")
                  or (r.expected_active_vs_rest != "0" and r.agreement ==
                      ("concordant" if (r.cliff_active_rest > 0) == (r.expected_active_vs_rest == "+")
                       and r.cliff_active_rest != 0 else "discordant")) for r in tests.itertuples())
    check(sign_ok, "agreement flag == sign(Cliff A-R) vs expected direction on every row")
    check(sorted(tests.level.unique()) == ["A", "A_lag1", "B", "B_lag1"]
          and int((tests.level == "A").sum()) == 6 and int((tests.level == "B").sum()) == 4, "levels A 6 / B 4 / lag 3")
    for f in FILES:
        if f.endswith(".png"):
            img = matplotlib.image.imread(RES / f)
            check(img.ndim == 3 and img.shape[0] > 600 and img.shape[1] > 600, f"{f} opens {img.shape}")
    meta = json.loads((RES / "t32_run_meta.json").read_text(encoding="utf-8"))
    check(meta["sanity_checks_failed"] == [] and meta["seed"] == 42 and meta["n_boot"] == 1000
          and meta["label_distribution"] == [202, 710, 46] and meta["figure_axes"] == [6, 8]
          and len(meta["variables"]) == 13 and meta["versions"]["scikit_posthocs"] is None
          and all(meta["n_per_class"][v] == [int(inv.loc[v, "n_rest"]), int(inv.loc[v, "n_normal"]),
                                             int(inv.loc[v, "n_active"])] for v in inv.index),
          f"run_meta: 0 failed checks, seed 42, 1000 resamples, 202/710/46, 13 variables, n per class == inventory "
          f"(elapsed {meta['elapsed_seconds']} s)")

    print("\n=== git diff on protected files ===")
    tracked = subprocess.run(["git", "ls-files", "revision/results", "CIFRE_CANONICE.md"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    protected = [t for t in tracked if "/t32_" not in t]
    diff = subprocess.run(["git", "status", "--porcelain", "--"] + protected, cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    check(diff == "", f"git diff empty on CIFRE_CANONICE.md and {len(protected) - 1} tracked T0.x-T3.1 result files"
          + (f"\n{diff}" if diff else ""))
    check(any("test_predictions_rf.csv" in t for t in protected), "test_predictions_*.csv are in the protected set")

    print(f"\n{'ALL PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=None)
    main(ap.parse_args().scratch)
