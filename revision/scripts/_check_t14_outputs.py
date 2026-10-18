"""
_check_t14_outputs.py - T1.4 Task 4: determinism and output checks for
shap_stability.py.

For each mode (bootstrap, seeds):
1. Copies the current T1.4 outputs (csv / md / npy / png / pdf / json) to a
   scratch directory, then re-runs `python -W error shap_stability.py --mode M`
   (any warning becomes an error -> non-zero exit -> FAIL). The subprocess
   stderr is printed verbatim if non-empty.
2. sha256 identical between the two consecutive runs for csv / md / npy /
   png / pdf; json identical without run_date and elapsed_seconds.
3. Text outputs and CIFRE_REVIZIE.md contain no CRLF.
4. Markdown table carries the csv values (the key cells are re-derived from
   t14_shap_stability_{mode}.csv); the figure png opens; run_meta reports
   1 axis and no failed sanity checks.
5. bootstrap_train_indices_100x766_seed42.npy is byte-identical to
   np.random.default_rng(42).integers(0, 766, size=(100, 766)) and its sha256
   equals the one recorded in t14_run_meta_bootstrap.json.
6. t14_shap_reference.csv is identical after both modes.
Finally: git diff empty on CIFRE_CANONICE.md and on every tracked file of
revision/results/ produced by T0.x-T1.3 (everything tracked before T1.4).

Exits 1 on any FAIL. Evidence script, kept.
Run from repo root (after one full run of each mode):
    python revision/scripts/_check_t14_outputs.py [--modes bootstrap seeds] [--scratch DIR]
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
SCRIPT = ROOT / "revision" / "scripts" / "shap_stability.py"
NPY = "bootstrap_train_indices_100x766_seed42.npy"
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outputs(mode: str) -> list[str]:
    files = [f"t14_shap_values_{mode}.csv", f"t14_shap_stability_{mode}.csv", f"t14_shap_stability_{mode}_table.md",
             f"t14_run_meta_{mode}.json", f"figures/t14_shap_rank_stability_{mode}.png",
             f"figures/t14_shap_rank_stability_{mode}.pdf", "t14_shap_reference.csv"]
    if mode == "bootstrap":
        files.append(NPY)
    return files


def json_without_volatile(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    d.pop("run_date", None)
    d.pop("elapsed_seconds", None)
    return d


def check_mode(mode: str, scratch: Path) -> None:
    print(f"\n=== mode {mode}: determinism (second run under -W error) ===")
    prev = scratch / mode
    prev.mkdir(parents=True, exist_ok=True)
    for f in outputs(mode):
        check((RES / f).exists(), f"first-run output exists: {f}")
        if (RES / f).exists():
            (prev / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RES / f, prev / f)
    if fails:
        return
    proc = subprocess.run([sys.executable, "-W", "error", str(SCRIPT), "--mode", mode], cwd=ROOT,
                          capture_output=True, text=True)
    check(proc.returncode == 0, f"second run exit code {proc.returncode}")
    check(proc.stderr.strip() == "", f"second run stderr empty ({len(proc.stderr)} chars)")
    if proc.stderr.strip():
        print(proc.stderr)
    last = [ln for ln in proc.stdout.splitlines() if ln.startswith(f"[{mode}]")]
    print("  " + (last[-1] if last else "(no summary line)"))
    for f in outputs(mode):
        if f.endswith(".json"):
            check(json_without_volatile(RES / f) == json_without_volatile(prev / f),
                  f"json identical without run_date / elapsed_seconds: {f}")
        else:
            check(sha256(RES / f) == sha256(prev / f), f"sha256 identical: {f} ({sha256(RES / f)[:16]})")

    print(f"\n=== mode {mode}: LF, md == csv, figure, meta ===")
    for f in outputs(mode):
        if f.endswith((".csv", ".md", ".json")):
            check(b"\r\n" not in (RES / f).read_bytes(), f"LF only: {f}")
    table = pd.read_csv(RES / f"t14_shap_stability_{mode}.csv", keep_default_na=False)
    g = {(b, m, ft): v for b, m, ft, v in table.itertuples(index=False)}
    md = (RES / f"t14_shap_stability_{mode}_table.md").read_text(encoding="utf-8")
    n = int(json.loads((RES / f"t14_run_meta_{mode}.json").read_text(encoding="utf-8"))["n_runs"])
    expect = [
        f"| aw_hr_mean rank 1 | {int(g[('rank1_frequency', 'aw_hr_mean_rank1_count', '')])} / {n} "
        f"({g[('rank1_frequency', 'aw_hr_mean_rank1_fraction', '')]:.2f}) |",
        f"{g[('jaccard_top5_vs_reference', 'mean', '')]:.4f} / {g[('jaccard_top5_vs_reference', 'median', '')]:.4f} / "
        f"{g[('jaccard_top5_vs_reference', 'min', '')]:.4f} / {g[('jaccard_top5_vs_reference', 'max', '')]:.4f} |",
        f"{g[('kendall_tau_vs_reference', 'mean', '')]:.4f} [{g[('kendall_tau_vs_reference', 'ci_low_p2.5', '')]:.4f}, "
        f"{g[('kendall_tau_vs_reference', 'ci_high_p97.5', '')]:.4f}]",
        f"{g[('kendall_tau_pairwise', 'mean', '')]:.4f} [{g[('kendall_tau_pairwise', 'ci_low_p2.5', '')]:.4f}, "
        f"{g[('kendall_tau_pairwise', 'ci_high_p97.5', '')]:.4f}]",
        f"{g[('f1_test_informative_noncanonical', 'f1_macro_mean', '')]:.4f} "
        f"[{g[('f1_test_informative_noncanonical', 'f1_macro_p2.5', '')]:.4f}, "
        f"{g[('f1_test_informative_noncanonical', 'f1_macro_p97.5', '')]:.4f}]",
    ]
    for e in expect:
        check(e in md, f"md carries csv values: '{e[:60]}...'")
    pf = table[table.block == "per_feature"]
    check(len(pf) == 19 * 11 and sorted(pf[pf.metric == "reference_rank"].value.astype(int)) == list(range(1, 20)),
          "per-feature block: 19 features x 11 metrics, reference ranks 1..19")
    check(md.count("\n| ") >= 19 + 8, "md has the per-feature rows")
    vals = pd.read_csv(RES / f"t14_shap_values_{mode}.csv")
    check(len(vals) == n * 19 and vals["rank"].between(1, 19).all() and (vals.mean_abs_shap >= 0).all(),
          f"t14_shap_values_{mode}.csv: {n} x 19 rows, ranks in 1..19, mean |SHAP| >= 0")
    img = matplotlib.image.imread(RES / f"figures/t14_shap_rank_stability_{mode}.png")
    check(img.ndim == 3 and img.shape[0] > 500, f"png opens {img.shape}")
    meta = json.loads((RES / f"t14_run_meta_{mode}.json").read_text(encoding="utf-8"))
    check(meta["sanity_checks_failed"] == [] and meta["figure_axes"] == 1 and meta["n_runs"] == 100
          and meta["seed"] == 42 and meta["identity_test"]["ranking_equal"],
          f"run_meta: 0 failed sanity checks, 1 axis, n_runs 100, seed 42, identity test PASS "
          f"(elapsed {meta['elapsed_seconds']} s, shap {meta['versions']['shap']})")


def main(modes: list[str], scratch: Path | None) -> None:
    scratch = scratch or Path(tempfile.mkdtemp(prefix="t14_prev_"))
    print(f"scratch dir for first-run copies: {scratch}")
    ref_before = sha256(RES / "t14_shap_reference.csv")
    for mode in modes:
        check_mode(mode, scratch)
    check(sha256(RES / "t14_shap_reference.csv") == ref_before, "t14_shap_reference.csv identical across modes")

    print("\n=== bootstrap indices ===")
    if (RES / NPY).exists():
        idx = np.load(RES / NPY)
        regen = np.random.default_rng(42).integers(0, 766, size=(100, 766))
        check(idx.shape == (100, 766) and idx.dtype == regen.dtype and np.array_equal(idx, regen),
              f"{NPY} == default_rng(42).integers(0, 766, size=(100, 766)) ({idx.dtype})")
        meta_b = RES / "t14_run_meta_bootstrap.json"
        if meta_b.exists():
            rec = json.loads(meta_b.read_text(encoding="utf-8"))["bootstrap_indices"]["sha256"]
            check(rec == sha256(RES / NPY), f"npy sha256 {sha256(RES / NPY)[:16]} == run_meta record")

    print("\n=== CIFRE_REVIZIE.md LF and git diff on protected files ===")
    check(b"\r\n" not in (ROOT / "revision" / "CIFRE_REVIZIE.md").read_bytes(), "CIFRE_REVIZIE.md LF only")
    tracked = subprocess.run(["git", "ls-files", "revision/results", "CIFRE_CANONICE.md"], cwd=ROOT,
                             capture_output=True, text=True).stdout.split()
    protected = [t for t in tracked if "t14_" not in t and NPY not in t]
    diff = subprocess.run(["git", "status", "--porcelain", "--"] + protected, cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    check(diff == "", f"git diff empty on CIFRE_CANONICE.md and {len(protected) - 1} tracked T0.x-T1.3 result files"
          + (f"\n{diff}" if diff else ""))

    print(f"\n{'ALL PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["bootstrap", "seeds"])
    ap.add_argument("--scratch", type=Path, default=None)
    a = ap.parse_args()
    main(a.modes, a.scratch)
