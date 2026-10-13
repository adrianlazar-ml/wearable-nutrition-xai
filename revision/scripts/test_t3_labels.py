"""
test_t3_labels.py - T0.3 pass/fail: parametrised t3_labels is bit-identical
to the frozen version at default thresholds, and the vectorised assign_labels
matches the row-wise function for arbitrary thresholds.

Run from repo root:  python revision/scripts/test_t3_labels.py
Pass: prints "ALL PASS". Any AssertionError = fail.
"""

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(old_path: Path, new_path: Path) -> None:
    old = _load("t3_old", old_path)
    new = _load("t3_new", new_path)

    # 1. Scalar identity on the full integer grid 0..100 x 0..100.
    grid = list(itertools.product(range(0, 101), range(0, 101)))
    for r, s in grid:
        assert old.classify_day(r, s) == new.classify_day(r, s), (r, s)
    print(f"[1] classify_day identical on {len(grid)} (readiness, sleep) pairs")

    # 2. Row-wise identity incl. NaN handling, half-integers, edges.
    vals = [np.nan, 0, 59, 59.5, 60, 74.9, 75, 84.9, 85, 100]
    df = pd.DataFrame(list(itertools.product(vals, vals)),
                      columns=["readiness_score", "sleep_score"])
    lab_old = df.apply(old.assign_nutrition_label, axis=1)
    lab_new = df.apply(new.assign_nutrition_label, axis=1)
    pd.testing.assert_series_equal(lab_old, lab_new)
    print(f"[2] assign_nutrition_label identical on {len(df)} rows (with NaN)")

    # 3. Vectorised == row-wise, for canonical and for grid thresholds.
    rng = np.random.default_rng(42)
    big = pd.DataFrame({
        "readiness_score": rng.uniform(30, 100, 5000),
        "sleep_score": rng.uniform(30, 100, 5000),
    })
    big.loc[rng.choice(5000, 200, replace=False), "readiness_score"] = np.nan
    big.loc[rng.choice(5000, 200, replace=False), "sleep_score"] = np.nan
    combos = [dict()] + [
        dict(rest_cutoff=rc, active_readiness=ar, active_sleep=asl)
        for rc in (55, 60, 65) for ar in (80, 85, 90) for asl in (70, 75, 80)
    ] + [dict(rest_cutoff=55, sleep_rest_cutoff=65)]
    for kw in combos:
        vec = new.assign_labels(big, **kw)
        row = big.apply(new.assign_nutrition_label, axis=1, **kw)
        pd.testing.assert_series_equal(vec, row, check_names=False)
    print(f"[3] assign_labels == row-wise on {len(combos)} threshold configs")

    # 4. Canonical distribution sanity on the same random data (old == new).
    assert lab_old.value_counts(dropna=False).equals(lab_new.value_counts(dropna=False))
    print("[4] label distributions identical")
    print("ALL PASS")


if __name__ == "__main__":
    old_p = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "t3_labels.py"
    new_p = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "revision" / "t3_labels.py"
    main(old_p, new_p)
