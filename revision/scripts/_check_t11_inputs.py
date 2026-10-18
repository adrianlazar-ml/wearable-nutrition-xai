"""
_check_t11_inputs.py - T1.1 Task 1 + Task 2: inventory and sanity check of the
saved test predictions before computing any extended metric.

Task 1: shape, columns, dtypes, label values, p_* column order, date ordering
        and identity across files; persistence reconstruction from y_true with
        pred[0] = last train label read from the canonical snapshot; majority.
Task 2: macro F1 / accuracy / MCC / QWK recomputed from the csv files, compared
        with CIFRE_CANONICE.md at 4 decimals.

Exits 1 on any FAIL. Evidence script, kept on purpose.
Run from repo root:  python revision/scripts/_check_t11_inputs.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
sys.path.insert(0, str(ROOT / "revision"))
import t3_labels  # noqa: E402

LABELS = [0, 1, 2]  # Rest, Normal, Active
CANON = {
    "rf": (0.5063, 0.6406, 0.1416, 0.2551),
    "xgb": (0.4760, 0.6875, 0.1503, 0.1805),
    "svm": (0.4347, 0.6302, 0.0956, 0.1709),
}
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float]:
    return (f1_score(y, p, labels=LABELS, average="macro", zero_division=0),
            accuracy_score(y, p), matthews_corrcoef(y, p),
            cohen_kappa_score(y, p, labels=LABELS, weights="quadratic"))


def main() -> None:
    files = {k: pd.read_csv(RES / f"test_predictions_{k}.csv") for k in ("rf", "xgb", "svm")}
    base = pd.read_csv(RES / "persistence_baseline.csv")

    print("=== Task 1: inventory ===")
    for k, df in files.items():
        print(f"{k}: shape={df.shape}, columns={list(df.columns)}, "
              f"dtype y_true={df.y_true.dtype}, y_pred={df.y_pred.dtype}, "
              f"labels y_true={sorted(df.y_true.unique())}, y_pred={sorted(df.y_pred.unique())}, "
              f"p_* order={[c for c in df.columns if c.startswith('p_')]}, "
              f"date sorted={pd.to_datetime(df.date).is_monotonic_increasing}")
        check(df.shape == (192, 6), f"{k}: 192 rows x 6 cols")
        check(list(df.columns) == ["date", "y_true", "y_pred", "p_rest", "p_normal", "p_active"],
              f"{k}: expected column order")
        check(df.y_true.dtype.kind == "i" and df.y_pred.dtype.kind == "i", f"{k}: int labels")
    ref = files["rf"]
    for k in ("xgb", "svm"):
        check(ref.date.equals(files[k].date), f"date identical rf vs {k}")
        check(ref.y_true.equals(files[k].y_true), f"y_true identical rf vs {k}")
    y_true = ref.y_true.to_numpy()
    support = tuple(int((y_true == c).sum()) for c in LABELS)
    print(f"support Rest/Normal/Active = {support}")
    check(support == (44, 140, 8), "support 44/140/8")

    print(f"\npersistence_baseline.csv: shape={base.shape}, columns={list(base.columns)}")
    print("-> summary only (one row per baseline), no per-day predictions; reconstructing.")

    # Last train label from the canonical snapshot (not assumed).
    snap = pd.read_csv(ROOT / "models" / "dataset_snapshot_20260331.csv")
    snap["date"] = pd.to_datetime(snap["date"])
    snap = snap.sort_values("date").reset_index(drop=True)
    lab_col = "label" if "label" in snap.columns else "nutrition_label"
    if lab_col in snap.columns:
        lab = snap[lab_col]
    else:
        lab = t3_labels.assign_labels(snap)
    lab = lab.dropna().astype(int).reset_index(drop=True)
    snap_dates = snap.loc[snap[lab_col].notna(), "date"].reset_index(drop=True) if lab_col in snap.columns else None
    n = len(lab)
    split = int(n * 0.8)
    last_train = int(lab.iloc[split - 1])
    print(f"snapshot: n_labelled={n}, split={split}, last train label={last_train} "
          f"(label source: {lab_col if lab_col in snap.columns else 'recomputed'})")
    check(n == 958 and split == 766, "snapshot 958 labelled / 766 train")
    check(np.array_equal(lab.iloc[split:].to_numpy(), y_true), "snapshot test labels == y_true from csv")
    if snap_dates is not None:
        check(np.array_equal(pd.to_datetime(ref.date).to_numpy(), snap_dates.iloc[split:].to_numpy()),
              "snapshot test dates == csv dates")

    pers = np.concatenate([[last_train], y_true[:-1]])
    maj = np.full(192, 1)
    saved_pers = pd.read_csv(RES / "test_predictions_persistence.csv")
    saved_maj = pd.read_csv(RES / "test_predictions_majority.csv")
    check(np.array_equal(saved_pers.y_true.to_numpy(), y_true), "y_true identical in saved persistence file")
    check(np.array_equal(saved_pers.y_pred.to_numpy(), pers), "saved persistence y_pred == reconstruction")
    check(np.array_equal(saved_maj.y_pred.to_numpy(), maj), "saved majority y_pred == all Normal")

    f1p, accp, mccp, _ = metrics(y_true, pers)
    f1m, accm, _, _ = metrics(y_true, maj)
    print(f"persistence: F1 {f1p:.4f} acc {accp:.4f} MCC {mccp:.4f}; majority: F1 {f1m:.4f} acc {accm:.4f}")
    check(round(f1p, 4) == 0.3039 and round(accp, 4) == 0.5625, "persistence F1 0.3039 / acc 0.5625")
    check(round(f1m, 4) == 0.2811 and round(accm, 4) == 0.7292, "majority F1 0.2811 / acc 0.7292")

    print("\n=== Task 2: canonical metrics from csv ===")
    for k, df in files.items():
        got = tuple(round(v, 4) for v in metrics(y_true, df.y_pred.to_numpy()))
        print(f"{k}: F1 {got[0]:.4f} acc {got[1]:.4f} MCC {got[2]:.4f} QWK {got[3]:.4f}  (canon {CANON[k]})")
        check(got == CANON[k], f"{k} matches CIFRE_CANONICE at 4 decimals")

    print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}"))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
