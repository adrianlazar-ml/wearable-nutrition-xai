"""
_check_t13_inputs.py - T1.3 Task 1: inventory of the inputs used by
calibration_rf.py (RF calibration, R1.9).

1. test_predictions_rf.csv: 192 rows, support 44/140/8, y_pred == argmax of
   (p_rest, p_normal, p_active) on every row, max |row sum - 1| < 1e-6.
2. rf_revision.joblib loads; predict_proba on the 192 test days rebuilt from
   models/dataset_snapshot_20260331.csv (imputer_revision.joblib, RF is not
   scaled) reproduces p_rest/p_normal/p_active from the csv at 1e-6, with the
   same dates. The scaler loads too (not used by RF, listed for completeness).
   FAIL here means the feature pipeline behind the csv is not the joblib one
   and T1.3 must stop.
3. Train window: 766 rows, label distribution (expected 158/570/38 from
   frozen_meta; reported either way), test dates == csv dates.
4. bootstrap_indices_1000x192_seed42.npy: (1000, 192) int64, byte-identical
   to np.random.default_rng(42).integers(0, 192, size=(1000, 192)).
5. Versions: python, sklearn, numpy, pandas, matplotlib, imblearn, joblib.

Exits 1 on any FAIL. Evidence script, kept.
Run from repo root:  python revision/scripts/_check_t13_inputs.py
"""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

import imblearn
import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
SNAPSHOT = ROOT / "models" / "dataset_snapshot_20260331.csv"
LABELS = [0, 1, 2]
PROBA_COLS = ["p_rest", "p_normal", "p_active"]
TRAIN_RATIO = 0.8
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    print("=== 1. test_predictions_rf.csv ===")
    csv = RES / "test_predictions_rf.csv"
    df = pd.read_csv(csv)
    check(len(df) == 192, f"192 rows (got {len(df)})")
    check(list(df.columns) == ["date", "y_true", "y_pred"] + PROBA_COLS, f"columns {list(df.columns)}")
    y_true = df.y_true.to_numpy(dtype=int)
    y_pred = df.y_pred.to_numpy(dtype=int)
    proba = df[PROBA_COLS].to_numpy(dtype=float)
    support = tuple(int((y_true == c).sum()) for c in LABELS)
    check(support == (44, 140, 8), f"support Rest/Normal/Active = {support}, expected (44, 140, 8)")
    n_agree = int((proba.argmax(axis=1) == y_pred).sum())
    check(n_agree == 192, f"y_pred == argmax(proba) on {n_agree}/192 rows")
    dev = float(np.abs(proba.sum(axis=1) - 1.0).max())
    check(dev < 1e-6, f"max |sum(proba) - 1| = {dev:.3e} < 1e-6")
    check(pd.to_datetime(df.date).is_monotonic_increasing, "date sorted ascending")
    print(f"  sha256 {sha256(csv)[:16]}...")

    print("\n=== 2. rf_revision.joblib reproduces the csv probabilities ===")
    rf = joblib.load(RES / "models" / "rf_revision.joblib")
    imputer = joblib.load(RES / "models" / "imputer_revision.joblib")
    scaler = joblib.load(RES / "models" / "scaler_revision.joblib")
    print(f"  loaded RF ({type(rf).__name__}, n_estimators={rf.n_estimators}, max_depth={rf.max_depth}, "
          f"min_samples_leaf={rf.min_samples_leaf}, random_state={rf.random_state}), "
          f"imputer ({imputer.strategy}), scaler ({type(scaler).__name__})")
    check(list(rf.classes_) == LABELS, f"rf.classes_ == {LABELS} (got {list(rf.classes_)})")
    features = list(imputer.feature_names_in_)
    check(len(features) == 19, f"imputer has 19 features (got {len(features)})")

    snap = pd.read_csv(SNAPSHOT)
    snap["date"] = pd.to_datetime(snap["date"])
    snap = snap.sort_values("date").reset_index(drop=True)
    cols_sorted = sorted(snap.columns.tolist())
    content = snap[cols_sorted].to_csv(index=False, lineterminator="\n").encode("utf-8")
    h = hashlib.sha256(content).hexdigest()[:16]
    check(h == "63bc3a2915bae864", f"snapshot hash {h} == 63bc3a2915bae864")
    check(len(snap) == 958, f"snapshot 958 labelled rows (got {len(snap)})")
    check(set(features) <= set(snap.columns), "all 19 imputer features present in snapshot")
    split = int(len(snap) * TRAIN_RATIO)
    check(split == 766, f"split index {split} == 766")
    X_te = snap.iloc[split:][features]
    X_te_imp = pd.DataFrame(imputer.transform(X_te), columns=features)
    p_joblib = rf.predict_proba(X_te_imp)
    max_abs = float(np.abs(p_joblib - proba).max())
    check(max_abs < 1e-6, f"predict_proba(joblib) vs csv: max abs diff {max_abs:.3e} < 1e-6")
    snap_dates = snap.iloc[split:]["date"].dt.strftime("%Y-%m-%d").to_numpy()
    check(np.array_equal(snap_dates, df.date.to_numpy()), "test dates snapshot == csv")
    y_snap = snap.iloc[split:]["nutrition_label"].astype(int).to_numpy()
    check(np.array_equal(y_snap, y_true), "y_true snapshot == csv")

    print("\n=== 3. train window ===")
    y_tr = snap.iloc[:split]["nutrition_label"].astype(int).to_numpy()
    check(len(y_tr) == 766, f"766 train rows (got {len(y_tr)})")
    dist = tuple(int((y_tr == c).sum()) for c in LABELS)
    print(f"  train label distribution Rest/Normal/Active = {dist} "
          f"(expected 158/570/38 from frozen_meta: {'match' if dist == (158, 570, 38) else 'DIFFERS, reported'})")
    check(sum(dist) == 766, "train distribution sums to 766")
    print(f"  train dates {snap.date.iloc[0].date()} .. {snap.date.iloc[split - 1].date()}; "
          f"test {snap.date.iloc[split].date()} .. {snap.date.iloc[-1].date()}")

    print("\n=== 4. bootstrap_indices_1000x192_seed42.npy ===")
    npy = RES / "bootstrap_indices_1000x192_seed42.npy"
    idx = np.load(npy)
    expected = np.random.default_rng(42).integers(0, 192, size=(1000, 192))
    check(idx.shape == (1000, 192), f"shape {idx.shape}")
    check(idx.dtype == np.int64, f"dtype {idx.dtype} == int64")
    check(idx.tobytes() == expected.tobytes() and idx.dtype == expected.dtype,
          "byte-identical to default_rng(42).integers(0, 192, size=(1000, 192))")
    print(f"  sha256 {sha256(npy)[:16]}...")

    print("\n=== 5. versions ===")
    print(f"  python {platform.python_version()}, scikit-learn {sklearn.__version__}, numpy {np.__version__}, "
          f"pandas {pd.__version__}, matplotlib {matplotlib.__version__}, imblearn {imblearn.__version__}, "
          f"joblib {joblib.__version__}")
    check(sklearn.__version__ == "1.8.0", "scikit-learn 1.8.0 (frozen environment)")

    print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}\nSTOP: do not continue with Task 2."))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
