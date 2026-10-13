"""
persistence_baseline.py - T0.4 (Reviewer 2: "no persistence baseline")
=====================================================================
Predicts label(t) = label(t-1) on the canonical temporal test set and
reports the same metrics as the RF model, with bootstrap 95% CI.

Data source (in order of preference):
  1. --snapshot CSV (default models/dataset_snapshot_20260331.csv) with a
     'label' (or 'nutrition_label') column  -> persistence = label.shift(1) over the 958-day series
     (first test day uses the last train day automatically).
  2. Same CSV with readiness_score/sleep_score -> label recomputed via
     revision/t3_labels.py, then shifted.
  3. Same CSV with readiness_score_lag1/sleep_score_lag1 -> persistence
     computed directly from the lag-1 scores (no shift needed).

Note for the response letter: on this dataset the persistence baseline is a
deterministic function of two of the 19 model features
(readiness_score_lag1, sleep_score_lag1), so RF has strictly more
information than the baseline.

Pass criteria (must match CIFRE_CANONICE.md):
  n_total = 958, n_train = 766, n_test = 192,
  test distribution Rest/Normal/Active = 44/140/8.
Any mismatch -> script exits with code 1 and prints what it found.

Outputs:
  revision/results/test_predictions_persistence.csv
  revision/results/test_predictions_majority.csv
  revision/results/persistence_baseline.csv   (metrics + CI, one row/model)

Run from repo root:  python revision/scripts/persistence_baseline.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score,
                             matthews_corrcoef)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision"))
import t3_labels  # noqa: E402  (parametrised version, revision/t3_labels.py)

SEED = 42
N_BOOT = 1000
TRAIN_RATIO = 0.8
EXPECTED = dict(n_total=958, n_train=766, n_test=192, test_dist=(44, 140, 8))


def load_labels(path: Path) -> pd.DataFrame:
    """Return DataFrame with columns date, y (int) and y_prev (persistence)."""
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    else:
        print("WARNING: no 'date' column; assuming rows are in temporal order")
        df["date"] = np.arange(len(df))

    if "nutrition_label" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"nutrition_label": "label"})   # frozen snapshot naming
    if "label" in df.columns:
        src = "label column"
        y = df["label"]
        y_prev = y.shift(1)
    elif {"readiness_score", "sleep_score"} <= set(df.columns):
        src = "recomputed from readiness_score/sleep_score"
        y = t3_labels.assign_labels(df)
        y_prev = y.shift(1)
    elif {"readiness_score_lag1", "sleep_score_lag1"} <= set(df.columns):
        src = "from lag-1 scores"
        y = None
        tmp = df.rename(columns={"readiness_score_lag1": "readiness_score",
                                 "sleep_score_lag1": "sleep_score"})
        y_prev = t3_labels.assign_labels(tmp)
        if "label" not in df.columns:
            sys.exit("Need a 'label' column when only lag-1 scores are present. "
                     f"Columns found: {list(df.columns)}")
    else:
        sys.exit(f"Cannot derive labels. Columns found: {list(df.columns)}")

    out = pd.DataFrame({"date": df["date"], "y": y, "y_prev": y_prev})
    out = out.dropna(subset=["y"]).reset_index(drop=True)
    out["y"] = out["y"].astype(int)
    print(f"Labels: {src}; n_total = {len(out)}")
    return out


def bootstrap_ci(y_true, y_pred, metric, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric(y_true[idx], y_pred[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    f1m = lambda a, b: f1_score(a, b, average="macro", zero_division=0)
    mcc = lambda a, b: matthews_corrcoef(a, b)
    per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None,
                         zero_division=0)
    lo_f1, hi_f1 = bootstrap_ci(y_true, y_pred, f1m)
    lo_mcc, hi_mcc = bootstrap_ci(y_true, y_pred, mcc)
    return dict(
        model=name,
        f1_macro=f1m(y_true, y_pred), f1_macro_ci_lo=lo_f1, f1_macro_ci_hi=hi_f1,
        f1_rest=per_class[0], f1_normal=per_class[1], f1_active=per_class[2],
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        mcc=mcc(y_true, y_pred), mcc_ci_lo=lo_mcc, mcc_ci_hi=hi_mcc,
        kappa=cohen_kappa_score(y_true, y_pred),
        n_test=len(y_true), seed=SEED, n_boot=N_BOOT,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=ROOT / "models" / "dataset_snapshot_20260331.csv")
    ap.add_argument("--out", default=ROOT / "revision" / "results")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the canonical-size pass criteria (synthetic data)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_labels(Path(args.snapshot))
    split = int(len(d) * TRAIN_RATIO)
    train, test = d.iloc[:split], d.iloc[split:].copy()

    # Persistence: first test row already holds last train label via shift(1).
    if test["y_prev"].isna().any():
        sys.exit("y_prev has NaN inside the test window; check data continuity")
    test["y_prev"] = test["y_prev"].astype(int)
    majority = int(train["y"].mode()[0])

    # Gap diagnostics: how often is "previous row" not the previous calendar day?
    if np.issubdtype(d["date"].dtype, np.datetime64):
        gaps = d["date"].diff().dt.days.iloc[split:]
        print(f"Test window: {test['date'].min().date()} -> {test['date'].max().date()}; "
              f"rows whose previous row is >1 day back: {(gaps > 1).sum()}")

    dist = tuple(int((test["y"] == c).sum()) for c in (0, 1, 2))
    print(f"n_train = {len(train)}, n_test = {len(test)}, test Rest/Normal/Active = {dist}")
    if not args.no_check:
        found = dict(n_total=len(d), n_train=len(train), n_test=len(test), test_dist=dist)
        if found != EXPECTED:
            sys.exit(f"FAIL pass criteria. expected {EXPECTED}, found {found}")
        print("PASS canonical sizes")

    y_true = test["y"].to_numpy()
    rows = [evaluate("persistence", y_true, test["y_prev"].to_numpy()),
            evaluate("majority", y_true, np.full(len(test), majority))]
    res = pd.DataFrame(rows)
    res["run_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    res.to_csv(out_dir / "persistence_baseline.csv", index=False)

    for name, pred in (("persistence", test["y_prev"]), ("majority", pd.Series(majority, index=test.index))):
        p = pd.DataFrame({"date": test["date"], "y_true": test["y"], "y_pred": pred.astype(int)})
        for c, col in enumerate(["p_rest", "p_normal", "p_active"]):
            p[col] = (p["y_pred"] == c).astype(float)   # hard predictions -> one-hot
        p.to_csv(out_dir / f"test_predictions_{name}.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["model", "f1_macro", "f1_macro_ci_lo", "f1_macro_ci_hi", "f1_rest",
            "f1_normal", "f1_active", "accuracy", "balanced_accuracy", "mcc",
            "mcc_ci_lo", "mcc_ci_hi", "kappa"]
    print("\n" + res[cols].round(4).to_string(index=False))
    print("\nReference RF (canonical): F1 0.5063, acc 0.6406, MCC 0.1416, "
          "F1 Active 0.44, MCC CI [-0.0045, 0.2843]")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
