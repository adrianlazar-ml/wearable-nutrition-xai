"""
expanding_window.py - T3.3 (Reviewer 1, R1.10): temporal robustness of the canonical RF
with an expanding training window on the 958 labelled days (canonical labels R 60,
A_r 85, A_s 75). Nothing canonical is modified; every figure outside fold 1 is NON-canonical.

Design (fixed before any result was seen):
  the 958 days in temporal order (the order of common.canonical_split); test blocks of 192
  rows taken from the tail: fold 1 = rows 766-957 (== the canonical test set), fold 2 =
  574-765, fold 3 = 382-573, fold 4 = 190-381; train = every row before the block
  (766 / 574 / 382 / 190 rows). Per fold: common.preprocess_split (median imputer and
  SMOTE fit on the train part, seed 42, k from threshold_sensitivity.smote_k, the
  canonical Table 7 convention) and the canonical RF (save_test_predictions.build_models()
  ["rf"], 100 trees / depth 10 / min leaf 5 / seed 42) on all 19 lag-1 features. No choice
  depends on the test block. Comparability as in T3.1: >= 5 Active in the test block and
  >= 4 in train; non-comparable folds are still run when SMOTE allows and flagged.

Per fold: calendar range of train and test, class distribution, RF metrics (macro F1,
accuracy, MCC, balanced accuracy, F1 per class, confusion matrix), bootstrap 95 percent
CI on macro F1 and F1 Active with the shared matrix results/bootstrap_indices_1000x192_seed42.npy
(row indices 0..191 relative to the block; resamples with a class absent are skipped and
counted), persistence on the block (T0.4 rule: label of the previous row of the 958-day
series, the first block day uses the last train day; point metrics through the unmodified
persistence_baseline.evaluate()), majority (train mode) on the block, paired bootstrap RF
minus persistence on macro F1 (CI, include_zero), SHAP top-3 (T1.4 procedure).
Pooled: the 4 blocks concatenated (768 days): macro F1, F1 Active, accuracy, MCC with a
percentile CI from 1000 resamples of the 768 days (own np.random.default_rng(42), NOT the
192 matrix). Pooled is reported once, with the caveat that the folds have train sets of
different sizes.

Identity (abort on FAIL): fold 1 reproduces T1.5.0 (0.5063 / 0.6406 / 0.1416, y_pred
element-wise == results/test_predictions_rf.csv, probabilities at 1e-9), persistence
0.3039 [0.2612, 0.3501], RF marginal CI on macro F1 == T1.1 row of CIFRE_REVIZIE.md.

Outputs (revision/results/, LF):
  t33_expanding_window.csv (4 fold rows + 1 pooled row) + _table.md, t33_shap_top3.csv,
  t33_predictions_rf_fold{k}.csv (schema of test_predictions_*.csv plus a fold column),
  t33_run_meta.json, figures/t33_expanding_window.{png,pdf}

Run from repo root:  python revision/scripts/expanding_window.py
Exits 1 on any failed check.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import imblearn
import matplotlib
import numpy as np
import pandas as pd
import shap
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES, ROOT, SEED, TRAIN_RATIO, build_dataset, load_raw, preprocess_split  # noqa: E402
from group_ablation import f1_active, f1_macro, metrics, read_t11_rf_f1_ci, sha256  # noqa: E402
from persistence_baseline import evaluate as t04_evaluate  # noqa: E402  (T0.4, unmodified)
from save_test_predictions import build_models  # noqa: E402
from shap_stability import mean_abs_shap, rank_desc, short_name  # noqa: E402
from statistical_tests import INDEX_FILE, LABELS, paired_bootstrap, write_text  # noqa: E402
from threshold_sensitivity import (HR_FEATURE, is_comparable, label_counts, majority_predictions,  # noqa: E402
                                   persistence_predictions, smote_k, top_shap)

RES = ROOT / "revision" / "results"
FIG = RES / "figures"

N_BOOT, BLOCK, N_TOTAL = 1000, 192, 958
N_FOLDS = 4
CANON_RF = dict(f1_macro=0.5063, accuracy=0.6406, mcc=0.1416)
CANON_PERSISTENCE = dict(f1_macro=0.3039, ci=(0.2612, 0.3501))


# ------------------------------------------------------------------ pure helpers (unit-tested)
def fold_bounds(k: int, n: int = N_TOTAL, block: int = BLOCK) -> tuple[int, int, int]:
    """(train_end, test_start, test_end) for fold k (1 = the tail block); train = rows [0, train_end)."""
    if k < 1 or n - block * k < 1:
        raise ValueError(f"fold {k} is not cuttable from {n} rows with blocks of {block}")
    te0 = n - block * k
    return te0, te0, te0 + block


def all_fold_bounds(n_folds: int = N_FOLDS, n: int = N_TOTAL, block: int = BLOCK) -> list[tuple[int, int, int]]:
    out = [fold_bounds(k, n, block) for k in range(1, n_folds + 1)]
    tests = [range(a, b) for _, a, b in out]
    for i in range(len(tests)):
        for j in range(i + 1, len(tests)):
            if set(tests[i]) & set(tests[j]):
                raise AssertionError("test blocks overlap")
    return out


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    m = metrics(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int))
    return {k: m[k] for k in ("f1_macro", "f1_active", "accuracy", "mcc", "balanced_accuracy", "f1_rest", "f1_normal")}


def pooled_bootstrap(y_true: np.ndarray, y_pred: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT,
                     labels: list[int] = LABELS) -> dict[str, float | int]:
    """Percentile CIs of macro F1, F1 Active, accuracy and MCC over resamples of all pooled days."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    n = len(y_true)
    vals: dict[str, list[float]] = {k: [] for k in ("f1_macro", "f1_active", "accuracy", "mcc")}
    skipped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < len(labels):
            skipped += 1
            continue
        m = metrics(yt, yp)
        for k in vals:
            vals[k].append(m[k])
    out: dict[str, float | int] = {"n_skipped": skipped, "n_used": n_boot - skipped}
    for k, v in vals.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = float(lo), float(hi)
    return out


def is_float_col(c: str) -> bool:
    """Columns of the fold table that hold scores (formatted %.4f in the md); shared with _check_t33_outputs."""
    if c in ("shap_top1", "shap_top2", "shap_top3") or c.endswith("include_zero"):
        return False
    return c.startswith(("f1_", "accuracy", "mcc", "balanced_accuracy", "pers_f1", "pers_acc", "pers_mcc",
                         "delta_f1_macro", "maj_f1", "maj_acc", "shap_top"))


def fmt_cell(c: str, v) -> str:
    """md formatting of one csv cell: n/a for empty / NaN, %.4f for score columns, ints without decimals."""
    if v is None or (isinstance(v, str) and v == "") or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "n/a"
    if is_float_col(c):
        return "%.4f" % float(v)
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def table_to_md(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt_cell(c, r[c]) for c in cols) + " |")
    return lines


# ------------------------------------------------------------------ one fold
def run_fold(k: int, X: pd.DataFrame, y: pd.Series, dates: pd.Series, idx: np.ndarray) -> tuple[dict, dict, list[dict]]:
    """Fit the canonical RF on rows [0, train_end) and evaluate on the block; returns (row, arrays, shap_rows)."""
    tr_end, te0, te1 = fold_bounds(k, len(X))
    y_tr, y_te = y.iloc[:tr_end], y.iloc[te0:te1]
    c_tr, c_te = label_counts(y_tr), label_counts(y_te)
    kk = smote_k(min(c_tr))
    row: dict = dict(fold=k, train_rows=f"0-{tr_end - 1}", test_rows=f"{te0}-{te1 - 1}", n_train=tr_end, n_test=te1 - te0,
                     train_start=dates.iloc[0].strftime("%Y-%m-%d"), train_end=dates.iloc[tr_end - 1].strftime("%Y-%m-%d"),
                     test_start=dates.iloc[te0].strftime("%Y-%m-%d"), test_end=dates.iloc[te1 - 1].strftime("%Y-%m-%d"),
                     train_rest=c_tr[0], train_normal=c_tr[1], train_active=c_tr[2],
                     test_rest=c_te[0], test_normal=c_te[1], test_active=c_te[2],
                     comparable=is_comparable(c_te[2], c_tr[2]), smote_k=kk, trainable=kk is not None and min(c_tr) > 0)
    y_true = y_te.to_numpy(dtype=int)
    p_pers = persistence_predictions(y, te0)[:te1 - te0]
    t04 = t04_evaluate("persistence", y_true, p_pers)
    row.update(pers_f1_macro=float(t04["f1_macro"]), pers_f1_active=float(t04["f1_active"]),
               pers_accuracy=float(t04["accuracy"]), pers_mcc=float(t04["mcc"]),
               pers_f1_macro_ci_low_t04=float(t04["f1_macro_ci_lo"]), pers_f1_macro_ci_high_t04=float(t04["f1_macro_ci_hi"]))
    p_maj, maj = majority_predictions(y_tr, len(y_true))
    row.update(majority_class=maj, maj_f1_macro=f1_macro(y_true, p_maj), maj_accuracy=float(np.mean(p_maj == y_true)))
    arrays = dict(y_true=y_true, p_pers=p_pers, dates_test=dates.iloc[te0:te1].reset_index(drop=True))
    if not row["trainable"]:
        return row, arrays, []
    split = preprocess_split(X.iloc[:tr_end], X.iloc[te0:te1], y_tr, y_te, dates.iloc[te0:te1], k_neighbors=kk, seed=SEED)
    rf = build_models()["rf"]
    rf.fit(split.X_train_sm, split.y_train_sm)
    assert list(rf.classes_) == LABELS
    y_pred = rf.predict(split.X_test).astype(int)
    proba = rf.predict_proba(split.X_test)
    arrays.update(y_pred=y_pred, proba=proba)
    row.update(metrics(y_true, y_pred))
    b = paired_bootstrap(y_true, y_pred, p_pers, f1_macro, idx)
    row.update(f1_macro_ci_low=b["ci_A_low"], f1_macro_ci_high=b["ci_A_high"],
               pers_f1_macro_ci_low=b["ci_B_low"], pers_f1_macro_ci_high=b["ci_B_high"],
               delta_f1_macro=b["diff_point"], delta_f1_macro_ci_low=b["ci_low"], delta_f1_macro_ci_high=b["ci_high"],
               delta_f1_macro_include_zero=bool(b["ci_low"] <= 0.0 <= b["ci_high"]),
               delta_f1_macro_prop_gt_0=b["prop_diff_gt_0"], n_used=b["n_used"], n_skipped=b["n_skipped"])
    ba = paired_bootstrap(y_true, y_pred, p_pers, f1_active, idx)
    row.update(f1_active_ci_low=ba["ci_A_low"], f1_active_ci_high=ba["ci_A_high"])
    row.update(rf_gt_persistence=bool(row["f1_macro"] > row["pers_f1_macro"]),
               rf_gt_majority=bool(row["f1_macro"] > row["maj_f1_macro"]))
    mean_abs = mean_abs_shap(rf, split.X_test)
    ranks = rank_desc(mean_abs)
    hr_rank = int(ranks[FEATURES.index(HR_FEATURE)])
    for j, (name, val, _) in enumerate(top_shap(mean_abs), start=1):
        row[f"shap_top{j}"] = name
        row[f"shap_top{j}_value"] = val
    row.update(aw_hr_mean_rank=hr_rank, aw_hr_mean_rank_1=bool(hr_rank == 1))
    shap_rows = [dict(fold=k, feature=short_name(f), mean_abs_shap=float(mean_abs[j]), rank=int(ranks[j]))
                 for j, f in enumerate(FEATURES)]
    return row, arrays, shap_rows


def save_fold_predictions(path: Path, k: int, arrays: dict) -> None:
    df = pd.DataFrame({"fold": k, "date": pd.to_datetime(arrays["dates_test"]).dt.strftime("%Y-%m-%d").to_numpy(),
                       "y_true": arrays["y_true"], "y_pred": arrays["y_pred"], "p_rest": arrays["proba"][:, 0],
                       "p_normal": arrays["proba"][:, 1], "p_active": arrays["proba"][:, 2]})
    df.to_csv(path, index=False, lineterminator="\n")


# ------------------------------------------------------------------ figure
def make_figure(folds: pd.DataFrame, path_stem: Path) -> int:
    """One panel, folds in chronological order of the test block (fold 4 first): RF macro F1 (circle) and
    F1 Active (square) with 95 percent CI, persistence macro F1 (diamond, CI) and majority (cross)."""
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    f = folds.sort_values("test_start").reset_index(drop=True)
    x = np.arange(len(f))
    c_macro, c_active, c_grey = "#1565C0", "#E65100", "#616161"
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.errorbar(x - 0.12, f.f1_macro, yerr=[f.f1_macro - f.f1_macro_ci_low, f.f1_macro_ci_high - f.f1_macro],
                fmt="o", color=c_macro, ms=7, lw=1.6, capsize=4, capthick=1.6, label="RF macro F1")
    ax.errorbar(x + 0.12, f.f1_active, yerr=[f.f1_active - f.f1_active_ci_low, f.f1_active_ci_high - f.f1_active],
                fmt="s", color=c_active, ms=7, lw=1.6, capsize=4, capthick=1.6, label="RF F1 Active")
    ax.errorbar(x - 0.12, f.pers_f1_macro, yerr=[f.pers_f1_macro - f.pers_f1_macro_ci_low,
                                                  f.pers_f1_macro_ci_high - f.pers_f1_macro],
                fmt="D", color=c_grey, ms=6, lw=1.2, capsize=3, capthick=1.2, mfc="white", label="persistence macro F1")
    ax.plot(x - 0.12, f.maj_f1_macro, "x", color=c_grey, ms=8, mew=1.8, label="majority macro F1")
    for i, r in f.iterrows():
        if not bool(r.comparable):
            ax.add_patch(Rectangle((i - 0.5, 0), 1, 1, fill=False, hatch="///", edgecolor="#999999", lw=0))
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {int(r.fold)}\n{r.test_start} to {r.test_end}\ntrain {int(r.n_train)} d, "
                        f"Active in test {int(r.test_active)}" + ("" if r.comparable else " *") for _, r in f.iterrows()],
                       fontsize=8)
    ax.set_ylim(-0.03, 1)
    ax.set_ylabel("Score on the 192-day test block", fontsize=9)
    ax.set_title("Expanding-window evaluation", fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8.5, loc="upper left", ncol=2)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", metadata={"Software": None})
    n = len(fig.axes)
    plt.close(fig)
    return n


# ------------------------------------------------------------------ main
def main(res: Path = RES) -> None:
    t0 = time.time()
    fails: list[str] = []

    def check(cond: bool, msg: str, abort: bool = False) -> None:
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)
            if abort:
                sys.exit(f"ABORT: {msg}")

    X, y, dates = build_dataset(load_raw())
    check(len(X) == N_TOTAL and list(X.columns) == FEATURES and label_counts(y) == (202, 710, 46),
          "958 labelled days, canonical order and distribution", abort=True)
    check(bool(dates.is_monotonic_increasing), "dates strictly increasing", abort=True)
    bounds = all_fold_bounds()
    check(bounds[0] == (766, 766, 958) and bounds[0][1] == int(N_TOTAL * TRAIN_RATIO),
          "fold 1 == canonical split 766 / 192", abort=True)
    idx = np.load(res / INDEX_FILE)
    regen = np.random.default_rng(SEED).integers(0, BLOCK, size=(N_BOOT, BLOCK))
    check(np.array_equal(idx, regen) and idx.dtype == regen.dtype, f"{INDEX_FILE} loaded, byte-identical to default_rng({SEED})",
          abort=True)
    t11_ci = read_t11_rf_f1_ci()

    # ---- T3.3.0 identity (fold 1)
    print("\n=== T3.3.0 identity, fold 1 ===")
    rows, shap_all, pred_files = [], [], {}
    pooled_true, pooled_pred = [], []
    row1, arr1, shap1 = run_fold(1, X, y, dates, idx)
    ref = pd.read_csv(res / "test_predictions_rf.csv")
    check(np.array_equal(ref["y_true"].to_numpy(), arr1["y_true"]) and np.array_equal(ref["y_pred"].to_numpy(), arr1["y_pred"]),
          "fold 1 y_true and y_pred element-wise == test_predictions_rf.csv", abort=True)
    max_dp = float(np.abs(ref[["p_rest", "p_normal", "p_active"]].to_numpy() - arr1["proba"]).max())
    check(max_dp < 1e-9, f"fold 1 probabilities == csv (max abs diff {max_dp:.2e})", abort=True)
    for k, v in CANON_RF.items():
        check(round(row1[k], 4) == v, f"fold 1 {k} {row1[k]:.6f} == canonical {v}", abort=True)
    ref_p = pd.read_csv(res / "test_predictions_persistence.csv")
    check(np.array_equal(ref_p["y_pred"].to_numpy(), arr1["p_pers"]), "fold 1 persistence == test_predictions_persistence.csv",
          abort=True)
    check(round(row1["pers_f1_macro"], 4) == CANON_PERSISTENCE["f1_macro"]
          and (round(row1["pers_f1_macro_ci_low"], 4), round(row1["pers_f1_macro_ci_high"], 4)) == CANON_PERSISTENCE["ci"]
          and (round(row1["pers_f1_macro_ci_low_t04"], 4), round(row1["pers_f1_macro_ci_high_t04"], 4)) == CANON_PERSISTENCE["ci"],
          f"fold 1 persistence {row1['pers_f1_macro']:.4f} [{row1['pers_f1_macro_ci_low']:.4f}, {row1['pers_f1_macro_ci_high']:.4f}] == T0.4",
          abort=True)
    got = (round(row1["f1_macro_ci_low"], 4), round(row1["f1_macro_ci_high"], 4))
    check(got == t11_ci, f"fold 1 RF marginal CI {got} == T1.1 {t11_ci}", abort=True)
    check(round(row1["delta_f1_macro"], 4) == 0.2024, f"fold 1 delta RF minus persistence {row1['delta_f1_macro']:.4f} == T1.2b")
    check(row1["shap_top1"] == "aw_hr_mean" and row1["shap_top2"] == "oura_hrv_balance", "fold 1 SHAP top-2 == T1.4 reference")

    # ---- T3.3.1 folds
    print("\n=== T3.3.1 folds ===")
    for k in range(1, N_FOLDS + 1):
        row, arr, srows = (row1, arr1, shap1) if k == 1 else run_fold(k, X, y, dates, idx)
        check(all(c > 0 for c in (row["test_rest"], row["test_normal"], row["test_active"])),
              f"fold {k}: all three classes in the test block")
        if row["trainable"]:
            f = res / f"t33_predictions_rf_fold{k}.csv"
            save_fold_predictions(f, k, arr)
            pred_files[f.name] = sha256(f)
            pooled_true.append(arr["y_true"])
            pooled_pred.append(arr["y_pred"])
            check(row["n_skipped"] <= 0.02 * N_BOOT,          # reported, allowed up to 2 percent (class absent)
                  f"fold {k}: {row['n_skipped']} resamples skipped (class absent from the resample; reported)")
        rows.append(row)
        shap_all.extend(srows)
        print(f"  fold {k}: test {row['test_start']}..{row['test_end']} train {row['n_train']} "
              f"A train/test {row['train_active']}/{row['test_active']} comparable={row['comparable']} "
              f"RF F1 {row.get('f1_macro', np.nan):.4f} [{row.get('f1_macro_ci_low', np.nan):.4f}, {row.get('f1_macro_ci_high', np.nan):.4f}] "
              f"Active {row.get('f1_active', np.nan):.4f} [{row.get('f1_active_ci_low', np.nan):.4f}, {row.get('f1_active_ci_high', np.nan):.4f}] "
              f"pers {row['pers_f1_macro']:.4f} [{row.get('pers_f1_macro_ci_low', np.nan):.4f}, {row.get('pers_f1_macro_ci_high', np.nan):.4f}] "
              f"delta {row.get('delta_f1_macro', np.nan):+.4f} [{row.get('delta_f1_macro_ci_low', np.nan):+.4f}, {row.get('delta_f1_macro_ci_high', np.nan):+.4f}] "
              f"maj {row['maj_f1_macro']:.4f} top1 {row.get('shap_top1', 'n/a')} (hr rank {row.get('aw_hr_mean_rank', 'n/a')}) skipped {row.get('n_skipped', 'n/a')}")
    folds = pd.DataFrame(rows)
    check(len(folds) == N_FOLDS, f"{N_FOLDS} folds", abort=True)
    check(bool(folds.trainable.all()), "all folds trainable")
    tr = folds[folds.trainable]
    check(bool(((tr.f1_macro_ci_low <= tr.f1_macro) & (tr.f1_macro <= tr.f1_macro_ci_high)).all()
               and ((tr.f1_active_ci_low <= tr.f1_active) & (tr.f1_active <= tr.f1_active_ci_high)).all()
               and ((tr.pers_f1_macro_ci_low <= tr.pers_f1_macro) & (tr.pers_f1_macro <= tr.pers_f1_macro_ci_high)).all()
               and ((tr.delta_f1_macro_ci_low <= tr.delta_f1_macro) & (tr.delta_f1_macro <= tr.delta_f1_macro_ci_high)).all()),
          "every CI contains its point (RF macro / Active, persistence, delta)")
    check(bool((tr.delta_f1_macro - (tr.f1_macro - tr.pers_f1_macro)).abs().max() < 1e-12), "delta == RF minus persistence")

    # ---- pooled
    yt, yp = np.concatenate(pooled_true), np.concatenate(pooled_pred)
    check(len(yt) == BLOCK * int(folds.trainable.sum()), f"pooled over {len(yt)} days")
    pooled = dict(fold="pooled", n_test=int(len(yt)), n_train="", train_rows="", test_rows="",
                  test_start=str(folds.test_start.min()), test_end=str(folds.test_end.max()),
                  test_rest=int((yt == 0).sum()), test_normal=int((yt == 1).sum()), test_active=int((yt == 2).sum()),
                  comparable="", trainable="", smote_k="")
    pooled.update(pooled_metrics(yt, yp))
    pooled.update(pooled_bootstrap(yt, yp, np.random.default_rng(SEED)))
    print(f"  pooled ({len(yt)} d): F1 {pooled['f1_macro']:.4f} [{pooled['f1_macro_ci_low']:.4f}, {pooled['f1_macro_ci_high']:.4f}] "
          f"Active {pooled['f1_active']:.4f} [{pooled['f1_active_ci_low']:.4f}, {pooled['f1_active_ci_high']:.4f}] "
          f"acc {pooled['accuracy']:.4f} MCC {pooled['mcc']:.4f} skipped {pooled['n_skipped']}")
    comp = folds[folds.comparable & folds.trainable]
    print(f"  comparable folds {len(comp)} / {N_FOLDS}: RF > persistence {int(comp.rf_gt_persistence.sum())}, "
          f"RF > majority {int(comp.rf_gt_majority.sum())}, aw_hr_mean rank 1 {int(comp.aw_hr_mean_rank_1.sum())}; "
          f"macro F1 [{comp.f1_macro.min():.4f}, {comp.f1_macro.max():.4f}], F1 Active [{comp.f1_active.min():.4f}, {comp.f1_active.max():.4f}]")

    # ---- outputs
    cols = ["fold", "train_rows", "test_rows", "n_train", "n_test", "train_start", "train_end", "test_start", "test_end",
            "train_rest", "train_normal", "train_active", "test_rest", "test_normal", "test_active", "comparable",
            "trainable", "smote_k", "f1_macro", "f1_macro_ci_low", "f1_macro_ci_high", "f1_active", "f1_active_ci_low",
            "f1_active_ci_high", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "mcc", "mcc_ci_low", "mcc_ci_high",
            "balanced_accuracy", "f1_rest", "f1_normal"]
    cols += [c for c in folds.columns if c.startswith("cm_")]
    cols += ["pers_f1_macro", "pers_f1_macro_ci_low", "pers_f1_macro_ci_high", "pers_f1_macro_ci_low_t04",
             "pers_f1_macro_ci_high_t04", "pers_f1_active", "pers_accuracy", "pers_mcc", "delta_f1_macro",
             "delta_f1_macro_ci_low", "delta_f1_macro_ci_high", "delta_f1_macro_include_zero", "delta_f1_macro_prop_gt_0",
             "n_used", "n_skipped", "majority_class", "maj_f1_macro", "maj_accuracy", "rf_gt_persistence", "rf_gt_majority",
             "shap_top1", "shap_top1_value", "shap_top2", "shap_top2_value", "shap_top3", "shap_top3_value",
             "aw_hr_mean_rank", "aw_hr_mean_rank_1"]
    table = pd.concat([folds, pd.DataFrame([pooled])], ignore_index=True)
    for c in cols:
        if c not in table.columns:
            table[c] = np.nan
    table = table[cols]
    run_date = datetime.now().isoformat(timespec="seconds")
    day = run_date[:10]
    table.to_csv(res / "t33_expanding_window.csv", index=False, float_format="%.10g", lineterminator="\n")
    md = table_to_md(table)
    md += ["", "Expanding window on the 958 canonically labelled days in temporal order: test blocks of 192 rows from "
               "the tail (fold 1 = rows 766-957 = the canonical test set; fold 2 = 574-765; fold 3 = 382-573; fold 4 = "
               "190-381), train = every row before the block. Per fold: median imputer + SMOTE (k = smote_k of the "
               "smallest train class, canonical Table 7 convention) fit on train (common.preprocess_split), canonical RF "
               "(100 trees, depth 10, min leaf 5, seed 42), 19 lag-1 features. comparable = Active in test >= 5 and in "
               "train >= 4. [ci_low, ci_high] = percentiles 2.5 / 97.5 over 1000 resamples of the 192 block days on the "
               f"shared matrix {INDEX_FILE} (indices relative to the block); n_skipped = resamples with a class absent. "
               "Persistence = label of the previous row of the 958-day series (T0.4 rule; the first block day uses the "
               "last train day), point metrics from the unmodified persistence_baseline.evaluate(); *_t04 CI = its own RNG "
               "stream. delta = RF minus persistence on macro F1, paired on the same indices; prop_gt_0 descriptive. "
               "Majority = train mode. Pooled row = the 4 blocks concatenated (768 days), CI from 1000 resamples of the "
               "768 days with np.random.default_rng(42) (not the 192 matrix); the folds have train sets of different "
               "sizes (766 / 574 / 382 / 190), so pooled mixes models of unequal training history and does not replace "
               "the canonical figures. SHAP = TreeExplainer on the imputed block, mean |SHAP| over days x 3 classes "
               "(T1.4). All figures outside fold 1 are NON-canonical. Script: revision/scripts/expanding_window.py, "
               f"run {day}."]
    write_text(res / "t33_expanding_window_table.md", "\n".join(md) + "\n")
    shap_df = pd.DataFrame(shap_all)
    shap_top = shap_df[shap_df["rank"] <= 3].sort_values(["fold", "rank"]).reset_index(drop=True)
    shap_top.to_csv(res / "t33_shap_top3.csv", index=False, float_format="%.10g", lineterminator="\n")
    check(len(shap_top) == 3 * int(folds.trainable.sum()), f"t33_shap_top3.csv has 3 rows per fold ({len(shap_top)})")
    n_axes = make_figure(folds[folds.trainable], FIG / "t33_expanding_window")
    check(n_axes == 1, f"figure has 1 axis ({n_axes})")

    meta = {
        "run_date": run_date, "elapsed_seconds": round(time.time() - t0, 1), "seed": SEED, "n_resamples": N_BOOT,
        "block": BLOCK, "n_total": N_TOTAL, "n_folds": N_FOLDS,
        "fold_bounds": {f"fold{k}": {"train_rows": [0, b[0] - 1], "test_rows": [b[1], b[2] - 1]} for k, b in enumerate(bounds, 1)},
        "comparability": {"min_active_test": 5, "min_active_train": 4},
        "smote_k_per_fold": {f"fold{int(r.fold)}": int(r.smote_k) for _, r in folds.iterrows()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "imbalanced_learn": imblearn.__version__,
                     "shap": shap.__version__, "matplotlib": matplotlib.__version__},
        "t330_fold1": {k: row1[k] for k in ("f1_macro", "accuracy", "mcc", "f1_active", "pers_f1_macro", "delta_f1_macro")},
        "t11_rf_f1_macro_ci_from_cifre_revizie": list(t11_ci),
        "fold1_rf_f1_macro_marginal_ci": [row1["f1_macro_ci_low"], row1["f1_macro_ci_high"]],
        "pooled": {k: pooled[k] for k in ("n_test", "f1_macro", "f1_macro_ci_low", "f1_macro_ci_high", "f1_active",
                                          "f1_active_ci_low", "f1_active_ci_high", "accuracy", "mcc", "n_skipped")},
        "bootstrap_indices_sha256": sha256(res / INDEX_FILE),
        "prediction_files_sha256": pred_files,
        "figure_axes": n_axes,
        "sanity_checks_failed": fails,
    }
    with open(res / "t33_run_meta.json", "w", newline="\n", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print(f"\nelapsed {time.time() - t0:.1f}s; outputs in {res}")
    if fails:
        print(f"{len(fails)} check(s) FAILED: {fails}")
        sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="T3.3 expanding-window temporal robustness")
    ap.add_argument("--results", default=RES, type=Path)
    main(ap.parse_args().results)
