"""
calibration_rf.py - T1.3 (Reviewer 1, R1.9): calibration of the canonical RF.

Stage A (--stage A): out-of-fold (OOF) probabilities on the 766 train days.
  TimeSeriesSplit(n_splits=4), temporal order, no shuffle. Per fold: imputer +
  SMOTE(k=3) fit on the fold's train part only (common.preprocess_split, the
  function behind the canonical split), RF with the canonical hyperparameters
  (save_test_predictions.build_models()["rf"], seed 42), predict_proba on the
  fold's validation part. The canonical model rf_revision.joblib is never
  retrained or overwritten; the test set (192 days) is not touched here.
  Output: results/t13_oof_probabilities_rf.csv (date, fold, y_true, p_rest,
          p_normal, p_active), results/t13_oof_folds_rf.csv (fold summary).

Stage B (--stage B): calibrators fit on the OOF probabilities, applied to the
  canonical RF test probabilities (results/test_predictions_rf.csv, the only
  test probabilities used). Variants: raw, platt, isotonic. Two fit variants:
  fit_folds = "1-4" (all 612 OOF rows, 33 Active; primary) and "2-4" (459 rows,
  folds 2-4 only, 27 Active; sensitivity, because fold 1 trains on 154 days
  with 5 Active).
  Calibrators are manual one-vs-rest (no CalibratedClassifierCV):
    platt     LogisticRegression(C=np.inf, no regularisation) on logit(p_c), p clipped to
              [1e-6, 1 - 1e-6], fit on (y == c); then normalisation to sum 1
    isotonic  IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip") on p_c
              vs (y == c); then normalisation to sum 1 (rows whose calibrated
              probabilities are all 0 fall back to 1/3 each and are counted)
  Metrics (test, 192 rows, and OOF in-sample on the rows used for the fit):
    ECE / MCE top-label (confidence = max p, 10 equal-width bins on [0, 1],
    bin ids as sklearn.calibration.calibration_curve(strategy="uniform")),
    ECE per class (OvR, 10 bins on p_c), Brier multiclass (mean over samples of
    the sum over classes of (p_c - y_c)^2, range [0, 2], checked against
    sklearn brier_score_loss(scale_by_half=False)), Brier per class, log loss
    (sklearn log_loss, labels=[0, 1, 2]).
  Bootstrap 95 percent CI (percentile 2.5 / 97.5) on the shared matrix
  results/bootstrap_indices_1000x192_seed42.npy for ECE top-label, Brier
  multiclass, log loss and Brier Active on test. ECE Active on test is a point
  only (8 positives).
  Effect on predictions (informative, NON-canonical): days whose argmax changes
  vs raw; F1 macro / F1 Active / accuracy / MCC of the calibrated argmax.
  Outputs (results/): t13_calibration.csv (long format), t13_calibration_table.md,
    t13_reliability_bins.csv, t13_run_meta.json, figures/t13_reliability_rf.{pdf,png},
    models/calibrator_platt_rf.joblib, models/calibrator_isotonic_rf.joblib (fit 1-4).

Run from repo root:  python revision/scripts/calibration_rf.py --stage A
                     python revision/scripts/calibration_rf.py --stage B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import imblearn
import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss,
                             matthews_corrcoef)
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
FIG = RES / "figures"
SEED = 42
TRAIN_RATIO = 0.8
LABELS = [0, 1, 2]
CLASS_NAMES = ["Rest", "Normal", "Active"]
PROBA_COLS = ["p_rest", "p_normal", "p_active"]
N_SPLITS = 4
N_TRAIN = 766
N_TEST = 192
N_BINS = 10
EDGES = np.linspace(0.0, 1.0, N_BINS + 1)
PLATT_CLIP = 1e-6
VARIANTS = ["raw", "platt", "isotonic"]
FIT_FOLDS = {"1-4": [1, 2, 3, 4], "2-4": [2, 3, 4]}
CANON = dict(f1_macro=0.5063, accuracy=0.6406, mcc=0.1416)

OOF_CSV = RES / "t13_oof_probabilities_rf.csv"
FOLDS_CSV = RES / "t13_oof_folds_rf.csv"
TEST_CSV = RES / "test_predictions_rf.csv"
BOOT_NPY = RES / "bootstrap_indices_1000x192_seed42.npy"

METRIC_TITLE = {
    "ece_top": "ECE top-label", "mce_top": "MCE top-label",
    "brier_multiclass": "Brier multiclass [0, 2]", "log_loss": "Log loss",
    "ece_rest": "ECE Rest", "ece_normal": "ECE Normal", "ece_active": "ECE Active",
    "brier_rest": "Brier Rest", "brier_normal": "Brier Normal", "brier_active": "Brier Active",
    "n_argmax_changed_vs_raw": "Days with argmax changed vs raw",
    "f1_macro_argmax": "F1 macro (argmax, non-canonical)",
    "f1_active_argmax": "F1 Active (argmax, non-canonical)",
    "accuracy_argmax": "Accuracy (argmax, non-canonical)",
    "mcc_argmax": "MCC (argmax, non-canonical)",
    "p_active_min": "p_active min", "p_active_median": "p_active median",
    "p_active_max": "p_active max", "n_p_active_ge_050": "Days with p_active >= 0.50",
    "n_rows": "Rows", "n_active": "Active rows",
    "n_zero_sum_rows": "Rows with all calibrated p = 0 (fallback 1/3)",
    "n_true_class_p_zero": "Rows with p(true class) = 0",
}
CI_METRICS = ["ece_top", "brier_multiclass", "log_loss", "brier_active"]
INT_METRICS = {"n_argmax_changed_vs_raw", "n_p_active_ge_050", "n_rows", "n_active",
               "n_zero_sum_rows", "n_true_class_p_zero"}

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- stage A
def stage_a(n_splits: int = N_SPLITS, out: Path = OOF_CSV) -> pd.DataFrame:
    from common import load_canonical, preprocess_split  # noqa: E402  (loads raw data)
    from save_test_predictions import build_models  # noqa: E402

    t0 = time.time()
    X, y, dates, _split = load_canonical()
    i = int(len(X) * TRAIN_RATIO)
    X_tr, y_tr, d_tr = X.iloc[:i].reset_index(drop=True), y.iloc[:i].reset_index(drop=True), \
        dates.iloc[:i].reset_index(drop=True)
    check(len(X_tr) == N_TRAIN, f"train window {len(X_tr)} rows == {N_TRAIN}")
    dist = tuple(int((y_tr == c).sum()) for c in LABELS)
    print(f"train label distribution Rest/Normal/Active = {dist}; "
          f"dates {d_tr.iloc[0].date()} .. {d_tr.iloc[-1].date()}")

    tss = TimeSeriesSplit(n_splits=n_splits)
    rows: list[pd.DataFrame] = []
    fold_info: list[dict] = []
    for k, (tr, va) in enumerate(tss.split(X_tr), start=1):
        tf = time.time()
        sp = preprocess_split(X_tr.iloc[tr], X_tr.iloc[va], y_tr.iloc[tr], y_tr.iloc[va],
                              d_tr.iloc[va], k_neighbors=3, seed=SEED)
        rf = build_models()["rf"]
        rf.fit(sp.X_train_sm, sp.y_train_sm)
        if list(rf.classes_) != LABELS:
            check(False, f"fold {k}: rf.classes_ {list(rf.classes_)} != {LABELS}")
            continue
        proba = rf.predict_proba(sp.X_test)
        rows.append(pd.DataFrame({"date": sp.dates_test.dt.strftime("%Y-%m-%d"), "fold": k,
                                  "y_true": sp.y_test.to_numpy(dtype=int),
                                  "p_rest": proba[:, 0], "p_normal": proba[:, 1], "p_active": proba[:, 2]}))
        tr_dist = tuple(int((y_tr.iloc[tr] == c).sum()) for c in LABELS)
        va_dist = tuple(int((y_tr.iloc[va] == c).sum()) for c in LABELS)
        temporal_ok = d_tr.iloc[tr].max() < d_tr.iloc[va].min()
        info = dict(fold=k, n_train=len(tr), n_val=len(va),
                    train_start=str(d_tr.iloc[tr[0]].date()), train_end=str(d_tr.iloc[tr[-1]].date()),
                    val_start=str(d_tr.iloc[va[0]].date()), val_end=str(d_tr.iloc[va[-1]].date()),
                    train_rest=tr_dist[0], train_normal=tr_dist[1], train_active=tr_dist[2],
                    val_rest=va_dist[0], val_normal=va_dist[1], val_active=va_dist[2],
                    n_smote=len(sp.y_train_sm), seconds=round(time.time() - tf, 1))
        fold_info.append(info)
        check(temporal_ok, f"fold {k}: validation strictly after train "
                           f"({info['train_end']} < {info['val_start']})")
        print(f"  fold {k}: train {len(tr)} ({info['train_start']}..{info['train_end']}, "
              f"R/N/A {tr_dist}, SMOTE -> {len(sp.y_train_sm)}), "
              f"val {len(va)} ({info['val_start']}..{info['val_end']}, R/N/A {va_dist}), "
              f"{info['seconds']} s")
        if va_dist[2] == 0:
            print(f"  WARNING fold {k}: 0 Active in validation (reported, n_splits not changed)")

    oof = pd.concat(rows, ignore_index=True)
    first_train = N_TRAIN - sum(f["n_val"] for f in fold_info)
    check(len(oof) == N_TRAIN - first_train,
          f"OOF rows {len(oof)} == {N_TRAIN} - first train fold {first_train} = {N_TRAIN - first_train}")
    dev = float(np.abs(oof[PROBA_COLS].sum(axis=1) - 1.0).max())
    check(dev < 1e-6, f"max |sum(proba) - 1| over OOF = {dev:.3e} < 1e-6")
    check(pd.to_datetime(oof.date).is_monotonic_increasing, "OOF dates sorted ascending")
    check(oof.date.is_unique, "OOF dates unique (each validation day appears once)")
    oof_dist = tuple(int((oof.y_true == c).sum()) for c in LABELS)
    print(f"OOF label distribution Rest/Normal/Active = {oof_dist}; Active per fold (validation) = "
          f"{[f['val_active'] for f in fold_info]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out, index=False, lineterminator="\n")
    # timing stays on stdout only, so the fold table is byte-identical across runs
    pd.DataFrame(fold_info).drop(columns=["seconds"]).to_csv(
        out.with_name("t13_oof_folds_rf.csv"), index=False, lineterminator="\n")
    print(f"saved {out.name} and t13_oof_folds_rf.csv in {RES}; stage A elapsed {time.time() - t0:.1f} s")
    return oof


# --------------------------------------------------------------------------- calibration metrics
def bin_ids(p: np.ndarray) -> np.ndarray:
    """Bin index in 0..N_BINS-1 for probabilities in [0, 1], equal-width bins.
    Same rule as sklearn.calibration.calibration_curve(strategy="uniform"):
    bin 0 is [0, 0.1], bin b > 0 is (b/10, (b+1)/10]."""
    return np.searchsorted(EDGES[1:-1], np.asarray(p, dtype=float))


def reliability_table(conf: np.ndarray, hit: np.ndarray) -> pd.DataFrame:
    """Per bin: n_b, mean confidence, empirical accuracy (nan for empty bins)."""
    conf = np.asarray(conf, dtype=float)
    hit = np.asarray(hit, dtype=float)
    ids = bin_ids(conf)
    rows = []
    for b in range(N_BINS):
        m = ids == b
        n = int(m.sum())
        rows.append(dict(bin=b, bin_low=float(EDGES[b]), bin_high=float(EDGES[b + 1]), n_b=n,
                         conf_b=float(conf[m].mean()) if n else np.nan,
                         acc_b=float(hit[m].mean()) if n else np.nan))
    return pd.DataFrame(rows)


def ece_mce(conf: np.ndarray, hit: np.ndarray) -> tuple[float, float]:
    """ECE = sum_b (n_b / N) |acc_b - conf_b|; MCE = max_b |acc_b - conf_b| over non-empty bins."""
    tab = reliability_table(conf, hit)
    tab = tab[tab.n_b > 0]
    gaps = (tab.acc_b - tab.conf_b).abs()
    return float((tab.n_b / len(conf) * gaps).sum()), float(gaps.max())


def ece_top_label(y: np.ndarray, P: np.ndarray) -> float:
    P = np.asarray(P, dtype=float)
    return ece_mce(P.max(axis=1), P.argmax(axis=1) == np.asarray(y))[0]


def mce_top_label(y: np.ndarray, P: np.ndarray) -> float:
    P = np.asarray(P, dtype=float)
    return ece_mce(P.max(axis=1), P.argmax(axis=1) == np.asarray(y))[1]


def ece_binary(y_bin: np.ndarray, p: np.ndarray) -> float:
    """One-vs-rest ECE for one class: confidence = p_c, outcome = (y == c)."""
    return ece_mce(p, y_bin)[0]


def brier_multiclass(y: np.ndarray, P: np.ndarray, labels: list[int] = LABELS) -> float:
    """Mean over samples of the sum over classes of (p_c - y_c)^2; range [0, 2]."""
    P = np.asarray(P, dtype=float)
    Y = (np.asarray(y)[:, None] == np.asarray(labels)[None, :]).astype(float)
    return float(((P - Y) ** 2).sum(axis=1).mean())


def brier_binary(y_bin: np.ndarray, p: np.ndarray) -> float:
    return float(((np.asarray(p, dtype=float) - np.asarray(y_bin, dtype=float)) ** 2).mean())


def logloss(y: np.ndarray, P: np.ndarray) -> float:
    return float(log_loss(y, P, labels=LABELS))


def normalize_ovr(Q: np.ndarray) -> tuple[np.ndarray, int]:
    """Divide each row by its sum. Rows with sum 0 (possible after isotonic) get 1/K each.
    Returns (normalised matrix, number of fallback rows)."""
    Q = np.asarray(Q, dtype=float)
    s = Q.sum(axis=1)
    zero = s <= 0.0
    out = np.empty_like(Q)
    out[~zero] = Q[~zero] / s[~zero, None]
    out[zero] = 1.0 / Q.shape[1]
    return out, int(zero.sum())


def logit(p: np.ndarray, clip: float = PLATT_CLIP) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), clip, 1.0 - clip)
    return np.log(p / (1.0 - p))


class PlattOvR:
    """One-vs-rest Platt scaling: per class, LogisticRegression(penalty=None) with the
    single feature logit(p_c) (p clipped to [1e-6, 1 - 1e-6]); outputs normalised to sum 1."""

    def __init__(self, labels: list[int] = LABELS, clip: float = PLATT_CLIP) -> None:
        self.labels = list(labels)
        self.clip = clip
        self.models_: dict[int, LogisticRegression] = {}
        self.n_fallback_rows_ = 0

    def fit(self, P: np.ndarray, y: np.ndarray) -> "PlattOvR":
        P, y = np.asarray(P, dtype=float), np.asarray(y)
        for j, c in enumerate(self.labels):
            # C=np.inf, l1_ratio=0 == no regularisation in sklearn 1.8 (penalty=None is deprecated;
            # coefficients identical). sklearn 1.8 still emits "Setting penalty=None will ignore the C"
            # on this very path, so that one UserWarning is silenced here and nowhere else.
            lr = LogisticRegression(C=np.inf, l1_ratio=0, solver="lbfgs", max_iter=10000, tol=1e-8)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Setting penalty=None will ignore", category=UserWarning)
                lr.fit(logit(P[:, j], self.clip)[:, None], (y == c).astype(int))
            self.models_[c] = lr
        return self

    def predict_ovr(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        return np.column_stack([self.models_[c].predict_proba(logit(P[:, j], self.clip)[:, None])[:, 1]
                                for j, c in enumerate(self.labels)])

    def predict_proba(self, P: np.ndarray) -> np.ndarray:
        out, self.n_fallback_rows_ = normalize_ovr(self.predict_ovr(P))
        return out

    def coefficients(self) -> dict[str, dict[str, float]]:
        return {CLASS_NAMES[c]: dict(slope=float(m.coef_[0, 0]), intercept=float(m.intercept_[0]),
                                     n_iter=int(m.n_iter_[0]))
                for c, m in self.models_.items()}


class IsotonicOvR:
    """One-vs-rest isotonic regression: per class, IsotonicRegression(y_min=0, y_max=1,
    out_of_bounds="clip") of (y == c) on p_c; outputs normalised to sum 1."""

    def __init__(self, labels: list[int] = LABELS) -> None:
        self.labels = list(labels)
        self.models_: dict[int, IsotonicRegression] = {}
        self.n_fallback_rows_ = 0

    def fit(self, P: np.ndarray, y: np.ndarray) -> "IsotonicOvR":
        P, y = np.asarray(P, dtype=float), np.asarray(y)
        for j, c in enumerate(self.labels):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
            iso.fit(P[:, j], (y == c).astype(float))
            self.models_[c] = iso
        return self

    def predict_ovr(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        return np.column_stack([self.models_[c].predict(P[:, j]) for j, c in enumerate(self.labels)])

    def predict_proba(self, P: np.ndarray) -> np.ndarray:
        out, self.n_fallback_rows_ = normalize_ovr(self.predict_ovr(P))
        return out

    def is_monotone(self, grid: int = 1001) -> dict[str, bool]:
        g = np.linspace(0.0, 1.0, grid)
        return {CLASS_NAMES[c]: bool(np.all(np.diff(m.predict(g)) >= -1e-12)) for c, m in self.models_.items()}


# --------------------------------------------------------------------------- bootstrap
def bootstrap(fn: Callable[[np.ndarray], float], idx: np.ndarray) -> tuple[float, float, int, int]:
    """Percentile CI over resamples; nan results skipped and counted. Returns (lo, hi, n_skipped, n_warn)."""
    vals: list[float] = []
    n_warn = 0
    for row in idx:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            v = fn(row)
            n_warn += len(caught)
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(lo), float(hi), len(idx) - len(vals), n_warn


# --------------------------------------------------------------------------- stage B
def point_metrics(y: np.ndarray, P: np.ndarray) -> dict[str, float]:
    e, m = ece_mce(P.max(axis=1), P.argmax(axis=1) == y)
    out = dict(ece_top=e, mce_top=m, brier_multiclass=brier_multiclass(y, P), log_loss=logloss(y, P))
    for j, name in enumerate(CLASS_NAMES):
        out[f"ece_{name.lower()}"] = ece_binary((y == LABELS[j]).astype(float), P[:, j])
        out[f"brier_{name.lower()}"] = brier_binary((y == LABELS[j]).astype(float), P[:, j])
    out["n_true_class_p_zero"] = int((P[np.arange(len(y)), y] <= 0.0).sum())
    return out


def argmax_metrics(y: np.ndarray, yp: np.ndarray) -> dict[str, float]:
    return dict(f1_macro_argmax=float(f1_score(y, yp, labels=LABELS, average="macro", zero_division=0)),
                f1_active_argmax=float(f1_score(y, yp, labels=[2], average="macro", zero_division=0)),
                accuracy_argmax=float(accuracy_score(y, yp)),
                mcc_argmax=float(matthews_corrcoef(y, yp)))


def fmt(point: float, lo: float, hi: float, metric: str) -> str:
    if metric in INT_METRICS:
        return f"{int(round(point))}"
    if np.isnan(lo):
        return f"{point:.4f}"
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


def build_md(table: pd.DataFrame, meta: dict) -> str:
    """Readable markdown tables generated from the long csv (checked equal by _check_t13_outputs.py)."""
    def block(title: str, split: str, metrics: list[str], note: str = "") -> list[str]:
        lines = [f"### {title}", ""]
        lines.append("| fit_folds | variant | " + " | ".join(METRIC_TITLE[m] for m in metrics) + " |")
        lines.append("|---|---|" + "---|" * len(metrics))
        for ff in FIT_FOLDS:
            for v in VARIANTS:
                cells = []
                for m in metrics:
                    sel = table[(table.split == split) & (table.fit_folds == ff) & (table.variant == v)
                                & (table.metric == m)]
                    if len(sel) == 0:
                        cells.append("n/a")
                    else:
                        r = sel.iloc[0]
                        cells.append("n/a" if np.isnan(r.point) else fmt(r.point, r.ci_low, r.ci_high, m))
                lines.append(f"| {ff} | {v} | " + " | ".join(cells) + " |")
        if note:
            lines += ["", note]
        return lines + [""]

    md = ["# T1.3 RF calibration (R1.9)", "",
          f"Canonical RF test probabilities (192 days, Rest/Normal/Active 44/140/8) calibrated with one-vs-rest "
          f"Platt / isotonic calibrators fit on out-of-fold probabilities (TimeSeriesSplit {meta['n_splits']} on "
          f"the 766 train days; fit_folds 1-4 = {meta['oof_rows']['1-4']} rows / {meta['oof_active']['1-4']} Active, "
          f"primary; fit_folds 2-4 = {meta['oof_rows']['2-4']} rows / {meta['oof_active']['2-4']} Active, "
          f"sensitivity). raw does not depend on fit_folds (identical rows on test). "
          f"Point [bootstrap 95 percent CI], {meta['n_boot']} resamples, seed {meta['seed']}, shared matrix "
          f"bootstrap_indices_1000x192_seed42.npy; CI only for ECE top-label, Brier multiclass, log loss, "
          f"Brier Active on test. Brier multiclass = mean over samples of the sum over classes of (p_c - y_c)^2, "
          f"range [0, 2]. ECE = top-label, 10 equal-width bins. Script revision/scripts/calibration_rf.py, "
          f"run {meta['run_date'][:10]}.", ""]
    md += block("Test (192 days)", "test",
                ["ece_top", "mce_top", "brier_multiclass", "log_loss", "n_true_class_p_zero"])
    md += block("Test, class Active (8 positives; no CI for ECE Active)", "test",
                ["ece_active", "brier_active", "ece_rest", "brier_rest", "ece_normal", "brier_normal"])
    md += block("Test, effect on argmax (informative, NON-canonical; canonical RF figures unchanged)", "test",
                ["n_argmax_changed_vs_raw", "f1_macro_argmax", "f1_active_argmax", "accuracy_argmax", "mcc_argmax"],
                "raw argmax reproduces the canonical RF: F1 macro 0.5063, accuracy 0.6406, MCC 0.1416.")
    md += block("Test, distribution of p_active", "test",
                ["p_active_min", "p_active_median", "p_active_max", "n_p_active_ge_050"])
    md += block("OOF, in-sample reference for the calibrators (rows used for the fit)", "oof",
                ["n_rows", "n_active", "ece_top", "mce_top", "brier_multiclass", "log_loss", "ece_active",
                 "brier_active", "n_zero_sum_rows"],
                "In-sample: the calibrators were fit on these same rows, so these values are optimistic "
                "and only show what the calibrators do where they were fit.")
    return "\n".join(md)


def stage_b() -> None:
    import importlib
    import matplotlib.pyplot as plt  # noqa: E402

    # Instantiate the calibrators from the importable module (not __main__) so that joblib
    # pickles them as calibration_rf.PlattOvR / IsotonicOvR and other scripts can load them.
    mod = importlib.import_module("calibration_rf")
    PlattCls, IsoCls = mod.PlattOvR, mod.IsotonicOvR

    t0 = time.time()
    oof = pd.read_csv(OOF_CSV)
    folds = pd.read_csv(FOLDS_CSV)
    test = pd.read_csv(TEST_CSV)
    idx = np.load(BOOT_NPY)
    check(len(test) == N_TEST and len(oof) == 612, f"inputs: test {len(test)} rows, OOF {len(oof)} rows")
    check(idx.shape == (1000, N_TEST), f"bootstrap matrix {idx.shape}")
    y_te = test.y_true.to_numpy(dtype=int)
    P_raw_te = test[PROBA_COLS].to_numpy(dtype=float)
    yp_raw = P_raw_te.argmax(axis=1)
    check(np.array_equal(yp_raw, test.y_pred.to_numpy(dtype=int)), "test csv: argmax(proba) == y_pred on all rows")
    am = argmax_metrics(y_te, yp_raw)
    for k, v in CANON.items():
        check(round(am[f"{k}_argmax"], 4) == v, f"raw test argmax {k} {am[f'{k}_argmax']:.4f} == canonical {v}")
    # sklearn multiclass Brier (sum convention) must equal the manual implementation
    sk_b = float(brier_score_loss(y_te, P_raw_te, labels=LABELS, scale_by_half=False))
    man_b = brier_multiclass(y_te, P_raw_te)
    check(abs(sk_b - man_b) < 1e-12, f"Brier multiclass manual {man_b:.6f} == sklearn scale_by_half=False {sk_b:.6f}")

    rows: list[dict] = []
    bins: list[pd.DataFrame] = []
    probs: dict[tuple[str, str, str], np.ndarray] = {}   # (split, fit_folds, variant) -> P
    meta: dict = dict(run_date=pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S"), seed=SEED, n_splits=N_SPLITS,
                      n_boot=int(idx.shape[0]), n_test=N_TEST, n_bins=N_BINS,
                      python=platform.python_version(), scikit_learn=sklearn.__version__, numpy=np.__version__,
                      pandas=pd.__version__, matplotlib=matplotlib.__version__, imblearn=imblearn.__version__,
                      joblib=joblib.__version__,
                      platt=dict(C="inf", l1_ratio=0,
                                 note="LogisticRegression(C=np.inf, l1_ratio=0) = no regularisation (sklearn 1.8 "
                                 "deprecates penalty=None; coefficients identical), solver lbfgs, max_iter 10000, "
                                 "tol 1e-8, feature logit(p) with p clipped to [1e-6, 1 - 1e-6]"),
                      isotonic=dict(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True),
                      normalisation="per-class calibrated probabilities divided by their row sum; rows with sum 0 "
                                    "fall back to 1/3 each and are counted (n_zero_sum_rows)",
                      brier_convention="mean over samples of the sum over classes of (p_c - y_c)^2, range [0, 2]; "
                                       "equal to sklearn brier_score_loss(scale_by_half=False)",
                      ece_bins="top-label, 10 equal-width bins on [0, 1], bin ids as "
                               "sklearn.calibration.calibration_curve(strategy='uniform')",
                      oof_rows_total=int(len(oof)), oof_active_total=int((oof.y_true == 2).sum()),
                      train_active_total=38,
                      oof_active_note="OOF covers 33 of the 38 Active train days: the 5 Active days in the fold-1 "
                                      "train part (first 154 days) have no out-of-fold probability",
                      active_per_fold_validation={int(r.fold): int(r.val_active) for r in folds.itertuples()},
                      active_per_fold_train={int(r.fold): int(r.train_active) for r in folds.itertuples()},
                      oof_rows={}, oof_active={}, platt_coefficients={}, isotonic_monotone={},
                      n_fallback_rows={}, bootstrap_warnings=0, n_resamples_skipped_total=0,
                      inputs_sha256={p.name: sha256(p) for p in (TEST_CSV, OOF_CSV, FOLDS_CSV, BOOT_NPY,
                                                                   RES / "models" / "rf_revision.joblib")})
    skipped_total, warn_total = 0, 0

    for ff, fold_list in FIT_FOLDS.items():
        sub = oof[oof.fold.isin(fold_list)].reset_index(drop=True)
        y_oof = sub.y_true.to_numpy(dtype=int)
        P_raw_oof = sub[PROBA_COLS].to_numpy(dtype=float)
        meta["oof_rows"][ff] = int(len(sub))
        meta["oof_active"][ff] = int((y_oof == 2).sum())
        platt = PlattCls().fit(P_raw_oof, y_oof)
        iso = IsoCls().fit(P_raw_oof, y_oof)
        meta["platt_coefficients"][ff] = platt.coefficients()
        meta["isotonic_monotone"][ff] = iso.is_monotone()
        check(all(iso.is_monotone().values()), f"fit {ff}: isotonic non-decreasing on a 1001-point grid, all classes")
        if ff == "1-4":
            (RES / "models").mkdir(exist_ok=True)
            joblib.dump(platt, RES / "models" / "calibrator_platt_rf.joblib")
            joblib.dump(iso, RES / "models" / "calibrator_isotonic_rf.joblib")

        for split, y, P_raw in (("test", y_te, P_raw_te), ("oof", y_oof, P_raw_oof)):
            P_by = {"raw": P_raw, "platt": platt.predict_proba(P_raw), "isotonic": iso.predict_proba(P_raw)}
            fallback = {"platt": platt.n_fallback_rows_, "isotonic": iso.n_fallback_rows_}
            for v, P in P_by.items():
                probs[(split, ff, v)] = P
                dev = float(np.abs(P.sum(axis=1) - 1.0).max())
                check(dev < 1e-6, f"{split} fit {ff} {v}: max |sum(p) - 1| = {dev:.2e} < 1e-6")
                pm = point_metrics(y, P)
                check(0.0 <= pm["ece_top"] <= 1.0 and 0.0 <= pm["brier_multiclass"] <= 2.0
                      and np.isfinite(pm["log_loss"]),
                      f"{split} fit {ff} {v}: ECE {pm['ece_top']:.4f} in [0,1], Brier {pm['brier_multiclass']:.4f} "
                      f"in [0,2], log loss {pm['log_loss']:.4f} finite")
                pm["n_zero_sum_rows"] = fallback.get(v, 0)
                meta["n_fallback_rows"][f"{split}/{ff}/{v}"] = fallback.get(v, 0)
                pm["n_rows"] = len(y)
                pm["n_active"] = int((y == 2).sum())
                if split == "test":
                    pm.update(argmax_metrics(y, P.argmax(axis=1)))
                    pm["n_argmax_changed_vs_raw"] = int((P.argmax(axis=1) != yp_raw).sum())
                    pa = P[:, 2]
                    pm.update(p_active_min=float(pa.min()), p_active_median=float(np.median(pa)),
                              p_active_max=float(pa.max()), n_p_active_ge_050=int((pa >= 0.5).sum()))
                for m, val in pm.items():
                    lo = hi = np.nan
                    nskip = np.nan
                    note = ""
                    if split == "test" and m in CI_METRICS:
                        fn = {"ece_top": lambda r, P=P: ece_top_label(y_te[r], P[r]),
                              "brier_multiclass": lambda r, P=P: brier_multiclass(y_te[r], P[r]),
                              "log_loss": lambda r, P=P: logloss(y_te[r], P[r]),
                              "brier_active": lambda r, P=P: brier_binary((y_te[r] == 2).astype(float), P[r, 2])}[m]
                        lo, hi, nskip, nwarn = bootstrap(fn, idx)
                        skipped_total += nskip
                        warn_total += nwarn
                        check(lo - 1e-12 <= val <= hi + 1e-12, f"test fit {ff} {v} {m}: CI [{lo:.4f}, {hi:.4f}] "
                                                                f"contains point {val:.4f}")
                    if split == "oof":
                        note = "in-sample for platt/isotonic"
                    if m == "ece_active" and split == "test":
                        note = "point only; 8 positives, bins practically empty, no CI"
                    if m.endswith("_argmax") or m == "n_argmax_changed_vs_raw":
                        note = "informative, NON-canonical"
                    rows.append(dict(split=split, fit_folds=ff, variant=v, metric=m, point=val, ci_low=lo,
                                     ci_high=hi, n_skipped=nskip, note=note))
                # reliability bins: top-label and Active
                for kind, conf, hit in (("top_label", P.max(axis=1), (P.argmax(axis=1) == y).astype(float)),
                                        ("active", P[:, 2], (y == 2).astype(float))):
                    tab = reliability_table(conf, hit)
                    tab.insert(0, "kind", kind)
                    tab.insert(0, "variant", v)
                    tab.insert(0, "fit_folds", ff)
                    tab.insert(0, "split", split)
                    bins.append(tab)
            # isotonic in-sample must not increase top-label ECE vs raw on the fit rows (otherwise bug)
            if split == "oof":
                e_raw = ece_top_label(y, P_by["raw"])
                e_iso = ece_top_label(y, P_by["isotonic"])
                e_pl = ece_top_label(y, P_by["platt"])
                check(e_iso <= e_raw + 1e-12, f"OOF fit {ff}: isotonic in-sample ECE top {e_iso:.4f} <= raw {e_raw:.4f}")
                print(f"  OOF fit {ff}: platt in-sample ECE top {e_pl:.4f} vs raw {e_raw:.4f} "
                      f"({'lower' if e_pl < e_raw else 'NOT lower'}; not guaranteed, reported)")
        print(f"  fit {ff} done ({time.time() - t0:.1f} s)")

    meta["n_resamples_skipped_total"] = int(skipped_total)
    meta["bootstrap_warnings"] = int(warn_total)
    check(skipped_total == 0, f"resamples skipped in bootstrap: {skipped_total} (expected 0)")
    # sanity: raw on test identical under both fit_folds
    check(np.array_equal(probs[("test", "1-4", "raw")], probs[("test", "2-4", "raw")]), "raw test identical for both fit_folds")

    table = pd.DataFrame(rows)
    table["seed"] = SEED
    table.to_csv(RES / "t13_calibration.csv", index=False, float_format="%.10g", lineterminator="\n")
    bins_df = pd.concat(bins, ignore_index=True)
    bins_df.to_csv(RES / "t13_reliability_bins.csv", index=False, float_format="%.10g", lineterminator="\n")
    md = build_md(table, meta)
    (RES / "t13_calibration_table.md").write_text(md + "\n", encoding="utf-8", newline="\n")

    # ------------------------------------------------------------------ figure
    FIG.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    colors = {"raw": "#555555", "platt": "#2196F3", "isotonic": "#E65100"}
    markers = {"raw": "o", "platt": "s", "isotonic": "^"}
    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.1], hspace=0.08, wspace=0.25)
    panels = [("(a) Top-label reliability, test set\n(n = 192; calibrators fit on OOF folds 1-4)", "test", "top_label",
               "Confidence (max p)", "Accuracy in bin"),
              ("(b) Class Active reliability, OOF in-sample\n(n = 612, 33 Active; fit folds 1-4)", "oof", "active",
               "Predicted p(Active)", "Observed Active fraction in bin")]
    for col, (title, split, kind, xlabel, ylabel) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        axh = fig.add_subplot(gs[1, col], sharex=ax)
        ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1, label="perfect calibration")
        width = 0.1 / 3.6
        for k, v in enumerate(VARIANTS):
            t = bins_df[(bins_df.split == split) & (bins_df.fit_folds == "1-4") & (bins_df.variant == v)
                        & (bins_df.kind == kind) & (bins_df.n_b > 0)]
            ax.plot(t.conf_b, t.acc_b, marker=markers[v], color=colors[v], lw=1.4, ms=5, label=v)
            full = bins_df[(bins_df.split == split) & (bins_df.fit_folds == "1-4") & (bins_df.variant == v)
                           & (bins_df.kind == kind)]
            centers = (full.bin_low + full.bin_high) / 2 + (k - 1) * width
            axh.bar(centers, full.n_b, width=width, color=colors[v], alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=9, loc="upper left")
        axh.set_xlabel(xlabel, fontsize=10)
        axh.set_ylabel("n days", fontsize=9)
        axh.set_xticks(EDGES)
        axh.grid(alpha=0.3, axis="y")
    fig.savefig(FIG / "t13_reliability_rf.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    fig.savefig(FIG / "t13_reliability_rf.png", dpi=300, bbox_inches="tight", metadata={"Software": None})
    meta["figure_panels"] = 2
    meta["figure_axes"] = len(fig.axes)
    plt.close(fig)

    meta["sanity_checks_failed"] = list(fails)
    meta["elapsed_seconds"] = round(time.time() - t0, 1)
    (RES / "t13_run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")

    print("\n" + md)
    print(f"\nsaved t13_calibration.csv, t13_calibration_table.md, t13_reliability_bins.csv, t13_run_meta.json, "
          f"figures/t13_reliability_rf.pdf/.png, models/calibrator_{{platt,isotonic}}_rf.joblib in {RES}; "
          f"stage B elapsed {time.time() - t0:.1f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description="T1.3 RF calibration")
    ap.add_argument("--stage", choices=["A", "B"], default="B")
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    args = ap.parse_args()
    if args.stage == "A":
        stage_a(n_splits=args.n_splits)
    else:
        stage_b()
    print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}"))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
