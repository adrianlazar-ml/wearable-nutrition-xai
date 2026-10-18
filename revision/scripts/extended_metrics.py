"""
extended_metrics.py - T1.1 (Reviewer 1, R1.8): metrics suited to class imbalance,
computed from the saved test-set predictions (no retraining).

Systems: RF, XGB, SVM (revision/results/test_predictions_{rf,xgb,svm}.csv),
persistence (pred[t] = y_true[t-1], pred[0] = last train label read from the
canonical snapshot) and majority (all Normal).

Metrics (point estimate + bootstrap 95 percent CI, 1000 resamples, seed 42):
  balanced_accuracy  sklearn.metrics.balanced_accuracy_score
  cohen_kappa        sklearn.metrics.cohen_kappa_score, unweighted
  g_mean             geometric mean of the per-class recalls (manual, see gmean())
  roc_auc_ovr_macro  roc_auc_score(multi_class="ovr", average="macro"); RF/XGB/SVM only
  pr_auc_macro       average_precision_score on label_binarize(y_true), macro; RF/XGB/SVM only
  f1_macro, accuracy, mcc, qwk   canonical figures recomputed as a sanity check

Hard labels come from y_pred (for SVM: predict(), not argmax(proba), see the
T0.2 note in CIFRE_REVIZIE.md). Probabilities come from p_rest, p_normal,
p_active in that order, which is the class order [0, 1, 2] = [Rest, Normal, Active].

Bootstrap: one shared index matrix (1000 x 192) generated with
np.random.default_rng(42).integers(0, 192, size=(1000, 192)), saved to
revision/results/bootstrap_indices_1000x192_seed42.npy and reused by every
system and every metric (T1.2 reuses it for paired bootstrap). A resample in
which a metric is undefined (a class absent from the resampled y_true for
AUC and G-mean) is skipped for that metric and the skip count is reported.

Outputs (revision/results/):
  t11_extended_metrics.csv        one row per (system, metric)
  t11_extended_metrics_table.md   one markdown table, "0.xxxx [0.xxxx, 0.xxxx]"
  bootstrap_indices_1000x192_seed42.npy

Run from repo root:  python revision/scripts/extended_metrics.py
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
SNAPSHOT = ROOT / "models" / "dataset_snapshot_20260331.csv"

SEED = 42
N_BOOT = 1000
N_TEST = 192
TRAIN_RATIO = 0.8
LABELS = [0, 1, 2]                       # Rest, Normal, Active (as stored in the csv files)
CLASS_NAMES = ["Rest", "Normal", "Active"]
PROBA_COLS = ["p_rest", "p_normal", "p_active"]   # must follow LABELS order
TRAINED = ["rf", "xgb", "svm"]
SYSTEMS = TRAINED + ["persistence", "majority"]
SYSTEM_LABEL = {"rf": "RF", "xgb": "XGB", "svm": "SVM",
                "persistence": "Persistence", "majority": "Majority"}

LABEL_METRICS = ["balanced_accuracy", "cohen_kappa", "g_mean",
                 "f1_macro", "accuracy", "mcc", "qwk"]
PROBA_METRICS = ["roc_auc_ovr_macro", "pr_auc_macro"]
CANONICAL = {"f1_macro", "accuracy", "mcc", "qwk"}
METRIC_TITLE = {
    "balanced_accuracy": "Balanced accuracy", "cohen_kappa": "Cohen's kappa",
    "g_mean": "G-mean", "roc_auc_ovr_macro": "ROC-AUC (OvR, macro)",
    "pr_auc_macro": "PR-AUC (macro)", "f1_macro": "Macro F1 (canonical)",
    "accuracy": "Accuracy (canonical)", "mcc": "MCC (canonical)",
    "qwk": "QWK (canonical)",
}
LabelMetric = Callable[[np.ndarray, np.ndarray], float]
ProbaMetric = Callable[[np.ndarray, np.ndarray], float]


# --------------------------------------------------------------------------- metrics
def gmean(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] = LABELS) -> float:
    """Multiclass G-mean: geometric mean of the per-class recalls.

    G = (prod_{c in labels} recall_c) ** (1 / K), with
    recall_c = TP_c / (TP_c + FN_c) = correct predictions of class c / true size of class c,
    K = len(labels). Classes are fixed by `labels`, so the result does not depend on
    which classes happen to appear in y_true or y_pred. If a class has no true
    sample in y_true its recall is undefined and the function returns nan
    (used by the bootstrap to skip the resample). A class that is never predicted
    has recall 0, which makes G = 0.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)
    if (support == 0).any():
        return float("nan")
    recalls = np.diag(cm) / support
    return float(np.prod(recalls) ** (1.0 / len(labels)))


def label_metrics() -> dict[str, LabelMetric]:
    """Hard-label metrics, with `labels` fixed wherever the sklearn API accepts it."""
    return {
        "balanced_accuracy": lambda y, p: float(balanced_accuracy_score(y, p)),
        "cohen_kappa": lambda y, p: float(cohen_kappa_score(y, p, labels=LABELS)),
        "g_mean": lambda y, p: gmean(y, p, LABELS),
        "f1_macro": lambda y, p: float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)),
        "accuracy": lambda y, p: float(accuracy_score(y, p)),
        "mcc": lambda y, p: float(matthews_corrcoef(y, p)),
        "qwk": lambda y, p: float(cohen_kappa_score(y, p, labels=LABELS, weights="quadratic")),
    }


def proba_metrics() -> dict[str, ProbaMetric]:
    """Probability metrics; return nan when a class is absent from y_true (undefined)."""
    def roc(y: np.ndarray, proba: np.ndarray) -> float:
        if len(np.unique(y)) < len(LABELS):
            return float("nan")
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=LABELS))

    def pr(y: np.ndarray, proba: np.ndarray) -> float:
        if len(np.unique(y)) < len(LABELS):
            return float("nan")
        y_bin = label_binarize(y, classes=LABELS)
        return float(average_precision_score(y_bin, proba, average="macro"))

    return {"roc_auc_ovr_macro": roc, "pr_auc_macro": pr}


# --------------------------------------------------------------------------- inputs
def last_train_label(snapshot: Path = SNAPSHOT) -> int:
    """Last label of the canonical train window, read from the frozen snapshot."""
    snap = pd.read_csv(snapshot, usecols=["date", "nutrition_label"])
    snap["date"] = pd.to_datetime(snap["date"])
    lab = snap.sort_values("date")["nutrition_label"].dropna().astype(int).reset_index(drop=True)
    split = int(len(lab) * TRAIN_RATIO)
    if len(lab) != 958 or split != 766:
        sys.exit(f"FAIL snapshot sizes: n_labelled={len(lab)}, split={split}")
    return int(lab.iloc[split - 1])


def load_inputs(res: Path = RES) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Return y_true, y_pred per system, proba per trained system, max |row sum - 1|."""
    preds: dict[str, np.ndarray] = {}
    probas: dict[str, np.ndarray] = {}
    y_ref: np.ndarray | None = None
    max_dev = 0.0
    for k in TRAINED:
        df = pd.read_csv(res / f"test_predictions_{k}.csv")
        if list(df.columns) != ["date", "y_true", "y_pred"] + PROBA_COLS:
            sys.exit(f"FAIL {k}: unexpected columns {list(df.columns)}")
        if len(df) != N_TEST:
            sys.exit(f"FAIL {k}: {len(df)} rows, expected {N_TEST}")
        y = df["y_true"].to_numpy(dtype=int)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y, y_ref):
            sys.exit(f"FAIL {k}: y_true differs from rf")
        preds[k] = df["y_pred"].to_numpy(dtype=int)
        proba = df[PROBA_COLS].to_numpy(dtype=float)
        dev = float(np.abs(proba.sum(axis=1) - 1.0).max())
        max_dev = max(max_dev, dev)
        if dev > 1e-6:
            sys.exit(f"FAIL {k}: probability rows do not sum to 1 (max deviation {dev:.2e})")
        probas[k] = proba
    assert y_ref is not None
    support = tuple(int((y_ref == c).sum()) for c in LABELS)
    if support != (44, 140, 8):
        sys.exit(f"FAIL support {support}, expected (44, 140, 8)")

    preds["persistence"] = np.concatenate([[last_train_label()], y_ref[:-1]])
    preds["majority"] = np.full(N_TEST, 1, dtype=int)
    saved = pd.read_csv(res / "test_predictions_persistence.csv")
    if not np.array_equal(saved["y_pred"].to_numpy(dtype=int), preds["persistence"]):
        sys.exit("FAIL persistence reconstruction differs from test_predictions_persistence.csv")
    return y_ref, preds, probas, max_dev


# --------------------------------------------------------------------------- bootstrap
def bootstrap_indices(path: Path, n_boot: int = N_BOOT, n: int = N_TEST, seed: int = SEED) -> np.ndarray:
    """Generate (or load, if identical) the shared resampling index matrix."""
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    if path.exists():
        old = np.load(path)
        if not np.array_equal(old, idx):
            sys.exit(f"FAIL existing {path.name} differs from the seed-{seed} matrix")
    else:
        np.save(path, idx)
    return idx


def bootstrap(values_fn: Callable[[np.ndarray], float], idx: np.ndarray
              ) -> tuple[float, float, int, int, int]:
    """Percentile CI over resamples; nan results are skipped and counted.

    Returns (ci_low, ci_high, n_used, n_skipped, n_warnings). sklearn warnings for
    absent classes are suppressed only inside this loop and counted.
    """
    vals: list[float] = []
    n_warn = 0
    for row in idx:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = values_fn(row)
            n_warn += len(caught)
        if not np.isnan(v):
            vals.append(v)
    arr = np.asarray(vals)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi), len(vals), len(idx) - len(vals), n_warn


# --------------------------------------------------------------------------- main
def fmt(v: float, lo: float, hi: float) -> str:
    return f"{v:.4f} [{lo:.4f}, {hi:.4f}]"


def plausibility(table: pd.DataFrame, support: tuple[int, ...]) -> list[str]:
    """Task 5 checks: report, never correct."""
    g = {(r.system, r.metric): r.value for r in table.itertuples()}
    lines = []
    lines.append(f"persistence balanced_accuracy == f1_macro: {g['persistence','balanced_accuracy']:.6f} vs "
                 f"{g['persistence','f1_macro']:.6f} -> "
                 f"{'equal' if abs(g['persistence','balanced_accuracy'] - g['persistence','f1_macro']) < 1e-12 else 'NOT equal'}")
    lines.append(f"persistence cohen_kappa == mcc: {g['persistence','cohen_kappa']:.6f} vs {g['persistence','mcc']:.6f} -> "
                 f"{'equal' if abs(g['persistence','cohen_kappa'] - g['persistence','mcc']) < 1e-12 else 'NOT equal'}")
    lines.append(f"majority balanced_accuracy = {g['majority','balanced_accuracy']:.6f} (expected 1/3), "
                 f"kappa = {g['majority','cohen_kappa']:.6f} (expected 0), g_mean = {g['majority','g_mean']:.6f} (expected 0)")
    ba, f1 = g["rf", "balanced_accuracy"], g["rf", "f1_macro"]
    lines.append(f"rf balanced_accuracy {ba:.4f} in (f1_macro {f1:.4f}, 1.0]: {f1 < ba <= 1.0}")
    prevalence = np.mean([s / sum(support) for s in support])
    for k in TRAINED:
        roc, pr = g[k, "roc_auc_ovr_macro"], g[k, "pr_auc_macro"]
        flag = []
        if roc < 0.5:
            flag.append("ROC-AUC BELOW 0.5")
        if pr < prevalence:
            flag.append(f"PR-AUC BELOW mean prevalence {prevalence:.4f}")
        lines.append(f"{k}: roc_auc {roc:.4f}, pr_auc {pr:.4f} (mean class prevalence {prevalence:.4f})"
                     + (f"  <-- {'; '.join(flag)}" if flag else ""))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results", default=RES, type=Path)
    args = ap.parse_args()
    res: Path = args.results
    t0 = time.time()

    y_true, preds, probas, max_dev = load_inputs(res)
    support = tuple(int((y_true == c).sum()) for c in LABELS)
    print(f"y_true support Rest/Normal/Active = {support}; "
          f"max |sum(proba) - 1| over RF/XGB/SVM = {max_dev:.3e}")
    idx = bootstrap_indices(res / f"bootstrap_indices_{N_BOOT}x{N_TEST}_seed{SEED}.npy")
    missing = sum(1 for row in idx if len(np.unique(y_true[row])) < len(LABELS))
    print(f"bootstrap matrix {idx.shape}; resamples with a class absent from y_true: {missing}")

    lm, pm = label_metrics(), proba_metrics()
    rows: list[dict] = []
    skipped: dict[str, int] = {}
    for sysname in SYSTEMS:
        yp = preds[sysname]
        for name, fn in lm.items():
            value = fn(y_true, yp)
            lo, hi, used, skip, nwarn = bootstrap(lambda r, fn=fn, yp=yp: fn(y_true[r], yp[r]), idx)
            skipped[name] = skipped.get(name, 0) + skip
            note = "canonical, sanity check" if name in CANONICAL else ""
            if skip:
                note = (note + "; " if note else "") + f"{skip} resamples skipped (class absent)"
            if nwarn:
                note = (note + "; " if note else "") + f"{nwarn} sklearn warnings in bootstrap"
            rows.append(dict(system=sysname, metric=name, value=value, ci_low=lo, ci_high=hi,
                             n_resamples_used=used, note=note))
        for name, fn in pm.items():
            if sysname in TRAINED:
                pr = probas[sysname]
                value = fn(y_true, pr)
                lo, hi, used, skip, nwarn = bootstrap(lambda r, fn=fn, pr=pr: fn(y_true[r], pr[r]), idx)
                skipped[name] = skipped.get(name, 0) + skip
                note = f"{skip} resamples skipped (class absent)" if skip else ""
                if nwarn:
                    note = (note + "; " if note else "") + f"{nwarn} sklearn warnings in bootstrap"
                rows.append(dict(system=sysname, metric=name, value=value, ci_low=lo, ci_high=hi,
                                 n_resamples_used=used, note=note))
            else:
                rows.append(dict(system=sysname, metric=name, value=np.nan, ci_low=np.nan, ci_high=np.nan,
                                 n_resamples_used=0, note="n/a: hard labels only"))
        print(f"  {sysname} done ({time.time() - t0:.1f}s)")

    table = pd.DataFrame(rows)
    table["seed"] = SEED
    table["n_boot"] = N_BOOT
    table["run_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    table.to_csv(res / "t11_extended_metrics.csv", index=False, float_format="%.10g")

    # Markdown table: systems on rows, metrics on columns.
    order = ["balanced_accuracy", "cohen_kappa", "g_mean", "roc_auc_ovr_macro", "pr_auc_macro",
             "f1_macro", "accuracy", "mcc", "qwk"]
    md = ["| System | " + " | ".join(METRIC_TITLE[m] for m in order) + " |",
          "|---|" + "---|" * len(order)]
    for sysname in SYSTEMS:
        cells = []
        for m in order:
            r = table[(table.system == sysname) & (table.metric == m)].iloc[0]
            cells.append("n/a" if np.isnan(r.value) else fmt(r.value, r.ci_low, r.ci_high))
        md.append(f"| {SYSTEM_LABEL[sysname]} | " + " | ".join(cells) + " |")
    skip_txt = ", ".join(f"{m}: {skipped[m]}" for m in order)
    md += ["",
           f"Test set n = {N_TEST} (Rest/Normal/Active = {support[0]}/{support[1]}/{support[2]}). "
           f"Point estimate [bootstrap 95 percent CI], {N_BOOT} resamples, seed {SEED}, shared index matrix "
           f"bootstrap_indices_{N_BOOT}x{N_TEST}_seed{SEED}.npy. "
           "ROC-AUC and PR-AUC n/a for persistence and majority (hard labels only). "
           f"Resamples skipped per metric (summed over systems): {skip_txt}. "
           f"Script: revision/scripts/extended_metrics.py, run {table.run_date.iloc[0]}."]
    (res / "t11_extended_metrics_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n" + "\n".join(md))
    print("\nSkipped resamples per metric (summed over systems):", skipped)
    over = {m: n for m, n in skipped.items() if n > 10}
    print("WARNING: more than 10 skipped resamples for:", over) if over else print("No metric with more than 10 skipped resamples.")
    print("\nTask 5 plausibility checks (reported, not corrected):")
    print("\n".join("  " + s for s in plausibility(table, support)))
    print(f"\nSaved t11_extended_metrics.csv, t11_extended_metrics_table.md in {res}; "
          f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
