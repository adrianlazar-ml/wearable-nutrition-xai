"""recompute_f1.py - AUDIT F1 (T3): independent recomputation from artefacts.
Imports only numpy, pandas, sklearn. Nothing from revision/scripts/.
Inputs : revision/results/test_predictions_rf.csv, models/dataset_snapshot_20260331.csv,
         revision/results/bootstrap_indices_1000x192_seed42.npy (comparison only)
Run from repo root: python revision/audit/recompute_f1.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]
LABELS = [0, 1, 2]
CLAIMS = dict(rf_f1=0.5063, rf_acc=0.6406, rf_mcc=0.1416,
              pers_f1=0.3039, pers_f1_active=0.0, pers_mcc=-0.0566, pers_acc=0.5625,
              maj_f1=0.2811, maj_acc=0.7292,
              pers_ci=(0.2612, 0.3501), rf_ci=(0.3784, 0.6130))


def f1m(a, b):
    return f1_score(a, b, labels=LABELS, average="macro", zero_division=0)


def f1m_nolabels(a, b):
    return f1_score(a, b, average="macro", zero_division=0)


rf = pd.read_csv(ROOT / "revision/results/test_predictions_rf.csv")
y_true = rf["y_true"].to_numpy(dtype=int)
y_rf = rf["y_pred"].to_numpy(dtype=int)
print("rows:", len(rf), "| test support Rest/Normal/Active:", [int((y_true == c).sum()) for c in LABELS])
print("test dates:", rf["date"].iloc[0], "->", rf["date"].iloc[-1])

# RF point metrics
rf_f1, rf_acc, rf_mcc = f1m(y_true, y_rf), accuracy_score(y_true, y_rf), matthews_corrcoef(y_true, y_rf)
print(f"RF  F1 macro {rf_f1:.6f} | acc {rf_acc:.6f} | MCC {rf_mcc:.6f}")

# Persistence from y_true + last train label from snapshot
snap = pd.read_csv(ROOT / "models/dataset_snapshot_20260331.csv")
snap["date"] = pd.to_datetime(snap["date"])
snap = snap.sort_values("date").reset_index(drop=True)
lab = snap["nutrition_label"].dropna().astype(int).reset_index(drop=True)
split = int(len(lab) * 0.8)
print("snapshot labelled rows:", len(lab), "| split:", split, "| train end:", snap["date"].iloc[split - 1].date(),
      "| test start:", snap["date"].iloc[split].date())
assert np.array_equal(lab.iloc[split:].to_numpy(), y_true), "snapshot test labels differ from csv y_true"
assert (snap["date"].iloc[split:].dt.strftime("%Y-%m-%d").to_numpy() == rf["date"].to_numpy()).all()
last_train = int(lab.iloc[split - 1])
y_pers = np.concatenate([[last_train], y_true[:-1]])
pers_f1 = f1m(y_true, y_pers)
pers_pc = f1_score(y_true, y_pers, labels=LABELS, average=None, zero_division=0)
pers_mcc, pers_acc = matthews_corrcoef(y_true, y_pers), accuracy_score(y_true, y_pers)
print(f"PERS F1 macro {pers_f1:.6f} | F1 Active {pers_pc[2]:.6f} | MCC {pers_mcc:.6f} | acc {pers_acc:.6f} | last train label {last_train}")

# Majority from train distribution
train_dist = lab.iloc[:split].value_counts().sort_index().to_dict()
maj_cls = int(lab.iloc[:split].mode()[0])
y_maj = np.full(len(y_true), maj_cls)
maj_f1, maj_acc = f1m(y_true, y_maj), accuracy_score(y_true, y_maj)
print(f"MAJ  class {maj_cls} | train dist {train_dist} | F1 macro {maj_f1:.6f} | acc {maj_acc:.6f}")

# Bootstrap CI (regenerated matrix, percentile 2.5/97.5)
idx = np.random.default_rng(42).integers(0, 192, size=(1000, 192))
saved = np.load(ROOT / "revision/results/bootstrap_indices_1000x192_seed42.npy")
print("bootstrap matrix identical to npy:", bool(np.array_equal(idx, saved)))


def ci(y, p, fn):
    vals = np.array([fn(y[row], p[row]) for row in idx])
    return tuple(float(v) for v in np.percentile(vals, [2.5, 97.5]))


rf_ci, pers_ci = ci(y_true, y_rf, f1m), ci(y_true, y_pers, f1m)
rf_ci_nl, pers_ci_nl = ci(y_true, y_rf, f1m_nolabels), ci(y_true, y_pers, f1m_nolabels)
print(f"RF   F1 CI (labels fixed) [{rf_ci[0]:.6f}, {rf_ci[1]:.6f}] | (labels not fixed) [{rf_ci_nl[0]:.6f}, {rf_ci_nl[1]:.6f}]")
print(f"PERS F1 CI (labels fixed) [{pers_ci[0]:.6f}, {pers_ci[1]:.6f}] | (labels not fixed) [{pers_ci_nl[0]:.6f}, {pers_ci_nl[1]:.6f}]")
absent = sum(1 for row in idx if len(np.unique(y_true[row])) < 3)
print("resamples with a class absent from y_true:", absent)

# Verdicts at 4 decimals
found = dict(rf_f1=rf_f1, rf_acc=rf_acc, rf_mcc=rf_mcc, pers_f1=pers_f1, pers_f1_active=pers_pc[2],
             pers_mcc=pers_mcc, pers_acc=pers_acc, maj_f1=maj_f1, maj_acc=maj_acc, pers_ci=pers_ci, rf_ci=rf_ci)
print("\nkey | claimed | recomputed | verdict")
all_ok = True
for k, v in CLAIMS.items():
    f = found[k]
    if isinstance(v, tuple):
        ok = all(abs(round(a, 4) - b) < 5e-5 for a, b in zip(f, v))
        fs = f"[{f[0]:.4f}, {f[1]:.4f}]"
        vs = f"[{v[0]:.4f}, {v[1]:.4f}]"
    else:
        ok = abs(round(f, 4) - v) < 5e-5
        fs = f"{f:.4f}"
        vs = f"{v:.4f}"
    all_ok &= ok
    print(f"{k} | {vs} | {fs} | {'PASS' if ok else 'FAIL'}")
print("T3 overall:", "PASS (all identical at 4 decimals)" if all_ok else "FAIL")
