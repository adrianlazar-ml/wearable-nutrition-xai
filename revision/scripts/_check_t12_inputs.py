"""
_check_t12_inputs.py - T1.2 Task 1: inventory of the inputs used by
statistical_tests.py and of the joblib artifacts produced at T0.2.

1. The four test_predictions_{rf,xgb,svm,persistence}.csv files: 192 rows,
   same `date` column sorted ascending, `y_true` identical element by element
   (same logic as _check_t11_inputs.py), int labels, support 44/140/8.
2. bootstrap_indices_1000x192_seed42.npy: shape (1000, 192), integer dtype,
   values in [0, 191], byte-identical to
   np.random.default_rng(42).integers(0, 192, size=(1000, 192)).
3. joblib artifacts (rf/xgb/svm/imputer/scaler_revision.joblib): real path,
   size in bytes, sha256, mtime, producing script (grep joblib.dump), git
   status (tracked / untracked / ignored).

Exits 1 on any FAIL in 1 or 2 (then T1.2 must stop). Evidence script, kept.
Run from repo root:  python revision/scripts/_check_t12_inputs.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
LABELS = [0, 1, 2]  # Rest, Normal, Active
SYSTEMS = ["rf", "xgb", "svm", "persistence"]
JOBLIBS = ["rf_revision.joblib", "xgb_revision.joblib", "svm_revision.joblib",
           "imputer_revision.joblib", "scaler_revision.joblib"]
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> tuple[int, str]:
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def git_status(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    if git("ls-files", "--error-unmatch", rel)[0] == 0:
        return "tracked"
    code, out = git("check-ignore", "-v", rel)
    return f"ignored ({out})" if code == 0 else "untracked"


def main() -> None:
    print("=== 1. test_predictions_*.csv ===")
    dfs = {k: pd.read_csv(RES / f"test_predictions_{k}.csv") for k in SYSTEMS}
    ref = dfs["rf"]
    for k, df in dfs.items():
        check(len(df) == 192, f"{k}: 192 rows (got {len(df)})")
        check(list(df.columns) == ["date", "y_true", "y_pred", "p_rest", "p_normal", "p_active"],
              f"{k}: expected columns")
        check(df.y_true.dtype.kind == "i" and df.y_pred.dtype.kind == "i", f"{k}: int labels")
        check(set(df.y_true.unique()) <= set(LABELS) and set(df.y_pred.unique()) <= set(LABELS),
              f"{k}: labels in {LABELS}")
        check(pd.to_datetime(df.date).is_monotonic_increasing, f"{k}: date sorted ascending")
        if k != "rf":
            check(ref.date.equals(df.date), f"date identical rf vs {k}")
            check(np.array_equal(ref.y_true.to_numpy(), df.y_true.to_numpy()), f"y_true identical rf vs {k}")
    support = tuple(int((ref.y_true == c).sum()) for c in LABELS)
    check(support == (44, 140, 8), f"support Rest/Normal/Active = {support}, expected (44, 140, 8)")
    for k, df in dfs.items():
        print(f"  {k}: sha256 {sha256(RES / f'test_predictions_{k}.csv')[:16]}..., "
              f"accuracy {float((df.y_true == df.y_pred).mean()):.4f}")

    print("\n=== 2. bootstrap_indices_1000x192_seed42.npy ===")
    npy = RES / "bootstrap_indices_1000x192_seed42.npy"
    idx = np.load(npy)
    check(idx.shape == (1000, 192), f"shape {idx.shape} == (1000, 192)")
    check(idx.dtype.kind in "iu", f"integer dtype ({idx.dtype})")
    check(int(idx.min()) >= 0 and int(idx.max()) <= 191, f"values in [0, 191] (min {idx.min()}, max {idx.max()})")
    expected = np.random.default_rng(42).integers(0, 192, size=(1000, 192))
    check(np.array_equal(idx, expected) and idx.dtype == expected.dtype,
          f"byte-identical to default_rng(42).integers(0, 192, size=(1000, 192)) (dtype {idx.dtype} vs {expected.dtype})")
    check(idx.tobytes() == expected.tobytes(), "tobytes() identical")
    print(f"  sha256 {sha256(npy)[:16]}...")

    if fails:
        print(f"\n{len(fails)} FAIL in 1/2: {fails}\nSTOP: do not continue with Task 2.")
        sys.exit(1)

    print("\n=== 3. joblib artifacts ===")
    producers: dict[str, str] = {}
    for script in sorted((ROOT / "revision" / "scripts").glob("*.py")):
        for line in script.read_text(encoding="utf-8").splitlines():
            if "joblib.dump" in line:
                for name in JOBLIBS:
                    stem = name.replace("_revision.joblib", "")
                    if name in line or ('{key}_revision.joblib' in line and stem in ("rf", "xgb", "svm")):
                        producers[name] = script.relative_to(ROOT).as_posix()
    rows = []
    total = 0
    for name in JOBLIBS:
        hits = [p for p in (RES / name, ROOT / "revision" / name, RES / "models" / name) if p.exists()]
        if not hits:
            print(f"MISSING {name}")
            continue
        p = hits[0]
        size = p.stat().st_size
        total += size
        rows.append(dict(path=p.relative_to(ROOT).as_posix(), bytes=size, sha256=sha256(p),
                         mtime=datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                         producer=producers.get(name, "?"), git=git_status(p)))
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    print(f"\ntotal bytes = {total} ({total / 1e6:.2f} MB)")
    print("| path | bytes | sha256[:16] | mtime | producer | git |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['path']} | {r['bytes']} | {r['sha256'][:16]} | {r['mtime']} | {r['producer']} | {r['git']} |")
    print("\nALL PASS (1, 2); joblib table above (3).")


if __name__ == "__main__":
    main()
