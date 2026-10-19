"""
threshold_sensitivity.py - T3.1 (Reviewer 1, R1.5; Reviewer 2 partial): sensitivity
of the canonical RF to the labelling thresholds, 3 x 3 grid, RF vs persistence and
majority baselines per cell, SHAP top-3 per cell.

Labelling rule (revision/t3_labels.py, thresholds as parameters since T0.3):
  Rest   = readiness < R  OR  sleep < R
  Active = readiness >= A_r  AND  sleep >= A_s
  Normal = everything else
Canonical: R = 60, A_r = 85, A_s = 75. Grid (KBS_Revision_Plan section 8.11: the
sleep grid is NOT run): R in {55, 60, 65} x A_r in {80, 85, 90}, A_s = 75 fixed.
The 958 labelled days and the temporal cut 766 / 192 do not depend on the
thresholds (they depend only on the presence of readiness and sleep).

Per cell, one function fit_eval_labels(R, A_r, A_s) regenerates the labels on the
958 days, keeps the 766 / 192 cut, applies common.preprocess_split (median imputer
fit on train, SMOTE on train, seed 42) and the canonical RF
(save_test_predictions.build_models()["rf"], 100 trees / depth 10 / min leaf 5 /
seed 42) on all 19 lag-1 features. No copy-paste of the preprocessing.

Convention replicated from the canonical Table 7 script (threshold_sensitivity.py
at the repo root, commit f7cc4ca, source of CIFRE_CANONICE.md section 5): SMOTE
k_neighbors = 3 when the smallest train class has >= 4 rows, k = n_min - 1 when it
has 2 or 3 rows (the canonical script trains without SMOTE below 2; such a cell is
marked "not trainable" here, none occurs in this grid). Macro F1 uses the fixed
labels [0, 1, 2] with zero_division = 0 (revision convention); it coincides with the
canonical call f1_score(average="macro") whenever all three classes appear in the
test labels, which holds for all 9 cells (checked at run time).

Comparability (Table 7 convention, T4 / T6 with asterisk): a cell is "comparable"
when Active in test >= 5 and Active in train >= 4 (SMOTE k = 3 needs 4). Non-
comparable cells are still run when SMOTE allows and flagged.

Identity checks (abort on FAIL):
  T3.1.0a cell (60, 85, 75) reproduces results/test_predictions_rf.csv (y_pred
          element-wise, probabilities at 1e-9) and 0.5063 / 0.6406 / 0.1416;
          its persistence predictions reproduce results/test_predictions_persistence.csv
          and 0.3039 [0.2612, 0.3501]; majority 0.2811 / 0.7292; RF marginal CI on
          the shared matrix == T1.1 row of CIFRE_REVIZIE.md.
  T3.1.0b cells (55, 85), (60, 85), (65, 85) reproduce Table 7 (T2, T3, T5): macro F1
          0.505 / 0.506 / 0.456 at 3 decimals, distributions 97/815/46, 202/710/46,
          360/552/46.
  Informative (non-abort): with A_s = 75 the cells (60, 90) and (65, 90) have the same
          labels as Table 7 T4 (60/60-90/80) and T6 (65/65-90/80) (every day with
          readiness >= 90 has sleep >= 80), so they are checked against 0.687 / 0.682.

Per cell: class distribution (958 / train / test), RF metrics (macro F1, accuracy,
MCC, balanced accuracy, F1 per class, confusion matrix), persistence baseline on the
cell's labels (label(t-1) over the 958-day series, evaluated with the unmodified
persistence_baseline.evaluate()), majority baseline (train mode), bootstrap 95
percent CI on macro F1 for RF and persistence on the shared matrix
results/bootstrap_indices_1000x192_seed42.npy (loaded, not regenerated), paired
bootstrap RF minus persistence on macro F1 (CI, include_zero), SHAP top-3 (canonical
procedure of T1.4: TreeExplainer, mean |SHAP| over the 192 test days x 3 classes).

Outputs (revision/results/, LF):
  t31_threshold_grid.csv (9 rows), t31_threshold_grid_table.md, t31_shap_top3.csv,
  t31_predictions_rf_R{R}_A{A_r}.csv (9 files, schema of test_predictions_*.csv),
  t31_run_meta.json, figures/t31_threshold_grid.{png,pdf}

Run from repo root:  python revision/scripts/threshold_sensitivity.py
Exits 1 on any failed check. Nothing canonical is modified; all figures outside the
cell (60, 85) are NON-canonical.
"""

from __future__ import annotations

import argparse
import hashlib
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
from group_ablation import (f1_active, f1_macro, metrics, read_t11_rf_f1_ci, save_predictions,  # noqa: E402
                            sha256)
from persistence_baseline import evaluate as t04_evaluate  # noqa: E402  (T0.4, unmodified)
from save_test_predictions import build_models  # noqa: E402
from shap_stability import mean_abs_shap, rank_desc, short_name  # noqa: E402
from statistical_tests import INDEX_FILE, LABELS, md_table, paired_bootstrap, write_text  # noqa: E402

RES = ROOT / "revision" / "results"
FIG = RES / "figures"

N_BOOT, N_TEST, N_TRAIN, N_TOTAL = 1000, 192, 766, 958
REST_GRID = (55, 60, 65)
ACTIVE_R_GRID = (80, 85, 90)
ACTIVE_SLEEP = 75
CANONICAL = (60, 85, ACTIVE_SLEEP)
MIN_ACTIVE_TEST = 5            # Table 7 convention (T4 / T6 with asterisk)
MIN_ACTIVE_TRAIN = 4           # SMOTE k = 3 needs 4 rows in the minority class
K_CANON = 3
TOP_N_SHAP = 3
HR_FEATURE = "aw_hr_mean_lag1"

# Reference figures (CIFRE_CANONICE.md and T0.4 / T1.1 rows of CIFRE_REVIZIE.md).
CANON_RF = dict(f1_macro=0.5063, accuracy=0.6406, mcc=0.1416)
CANON_PERSISTENCE = dict(f1_macro=0.3039, ci=(0.2612, 0.3501))
CANON_MAJORITY = dict(f1_macro=0.2811, accuracy=0.7292)
# Table 7 (CIFRE_CANONICE.md section 5), 3 decimals, distributions on the 958 days.
TABLE7 = {
    (55, 85): dict(set="T2", f1_macro=0.505, dist=(97, 815, 46), abort=True),
    (60, 85): dict(set="T3", f1_macro=0.506, dist=(202, 710, 46), abort=True),
    (65, 85): dict(set="T5", f1_macro=0.456, dist=(360, 552, 46), abort=True),
    # same labels as T4 / T6 (90/80) because every readiness >= 90 day has sleep >= 80; informative
    (60, 90): dict(set="T4*", f1_macro=0.687, dist=(202, 753, 3), abort=False),
    (65, 90): dict(set="T6*", f1_macro=0.682, dist=(360, 595, 3), abort=False),
}


# ------------------------------------------------------------------ pure helpers (unit-tested)
def label_counts(y) -> tuple[int, int, int]:
    """(n_rest, n_normal, n_active) with the classes fixed to 0 / 1 / 2."""
    y = np.asarray(y, dtype=int)
    return tuple(int((y == c).sum()) for c in LABELS)


def smote_k(min_class_count: int, k_canon: int = K_CANON) -> int | None:
    """SMOTE k_neighbors of the canonical Table 7 script: 3 when the smallest train
    class has >= 4 rows, n_min - 1 when it has 2 or 3 rows, None (no SMOTE) below 2."""
    if min_class_count >= 4:
        return min(k_canon, min_class_count - 1)
    if min_class_count >= 2:
        return min_class_count - 1
    return None


def is_comparable(n_active_test: int, n_active_train: int,
                  min_test: int = MIN_ACTIVE_TEST, min_train: int = MIN_ACTIVE_TRAIN) -> bool:
    """Table 7 convention: Active in test >= 5; plus Active in train >= 4 (SMOTE k = 3)."""
    return n_active_test >= min_test and n_active_train >= min_train


def persistence_predictions(y: pd.Series, i_split: int) -> np.ndarray:
    """T0.4 rule: prediction for row t = label of row t-1 of the labelled series;
    the first test row uses the last train row. Returns the test part (int)."""
    y_prev = pd.Series(np.asarray(y, dtype=int)).shift(1)
    out = y_prev.iloc[i_split:]
    if out.isna().any():
        raise ValueError("persistence undefined: split at row 0")
    return out.to_numpy(dtype=int)


def majority_predictions(y_train, n_test: int) -> tuple[np.ndarray, int]:
    """Majority baseline: the train mode repeated on the test rows."""
    maj = int(pd.Series(np.asarray(y_train, dtype=int)).mode()[0])
    return np.full(n_test, maj, dtype=int), maj


def top_shap(mean_abs: np.ndarray, features: list[str] = FEATURES, n: int = TOP_N_SHAP) -> list[tuple[str, float, int]]:
    """[(short feature name, mean |SHAP|, rank)] for the n largest values (stable tie-break)."""
    order = np.argsort(-np.asarray(mean_abs, dtype=float), kind="stable")[:n]
    ranks = rank_desc(mean_abs)
    return [(short_name(features[j]), float(mean_abs[j]), int(ranks[j])) for j in order]


def cell_name(R: int, A_r: int) -> str:
    return f"R{R}_A{A_r}"


# ------------------------------------------------------------------ core
def fit_eval_labels(R: int, A_r: int, A_s: int = ACTIVE_SLEEP, df: pd.DataFrame | None = None,
                    seed: int = SEED) -> dict:
    """Regenerate the labels with thresholds (R, A_r, A_s) on the 958 labelled days, keep the
    temporal cut 766 / 192, preprocess with common.preprocess_split (imputer + SMOTE fit on
    train; k from the canonical Table 7 convention) and fit the canonical RF on all 19
    features. Returns a dict with the test predictions, probabilities, the fitted RF, the
    Split and the per-part label counts. 'trainable' is False (no RF) when SMOTE cannot run."""
    if df is None:
        df = load_raw()
    X, y, dates = build_dataset(df, rest_cutoff=R, active_readiness=A_r, active_sleep=A_s)
    if len(X) != N_TOTAL or list(X.columns) != FEATURES:
        raise ValueError(f"dataset {X.shape} differs from the canonical 958 x 19")
    i = int(len(X) * TRAIN_RATIO)
    if i != N_TRAIN:
        raise ValueError(f"split {i} != {N_TRAIN}")
    y_tr, y_te = y.iloc[:i], y.iloc[i:]
    counts = dict(total=label_counts(y), train=label_counts(y_tr), test=label_counts(y_te))
    out = dict(R=R, A_r=A_r, A_s=A_s, X=X, y=y, dates=dates, i_split=i, counts=counts,
               y_true=y_te.to_numpy(dtype=int), dates_test=dates.iloc[i:].reset_index(drop=True))
    k = smote_k(min(counts["train"]))
    out["smote_k"] = k
    out["trainable"] = k is not None and min(counts["train"]) > 0
    if not out["trainable"]:
        out.update(split=None, rf=None, y_pred=None, proba=None)
        return out
    split = preprocess_split(X.iloc[:i], X.iloc[i:], y_tr, y_te, dates.iloc[i:], k_neighbors=k, seed=seed)
    rf = build_models()["rf"]
    rf.fit(split.X_train_sm, split.y_train_sm)
    if list(rf.classes_) != LABELS:
        raise ValueError(f"RF classes {list(rf.classes_)} != {LABELS}")
    out.update(split=split, rf=rf, y_pred=rf.predict(split.X_test).astype(int), proba=rf.predict_proba(split.X_test))
    return out


def evaluate_cell(cell: dict, idx: np.ndarray, t11_ci: tuple[float, float] | None = None) -> tuple[dict, list[dict]]:
    """All per-cell figures: RF metrics, persistence (T0.4 evaluate, unmodified), majority,
    marginal and paired bootstrap on the shared matrix, SHAP top-3. Returns (row, shap_rows)."""
    R, A_r, A_s = cell["R"], cell["A_r"], cell["A_s"]
    y_true = cell["y_true"]
    n_tot, n_tr, n_te = cell["counts"]["total"], cell["counts"]["train"], cell["counts"]["test"]
    row: dict = dict(cell=cell_name(R, A_r), R=R, A_r=A_r, A_s=A_s, canonical=(R, A_r, A_s) == CANONICAL,
                     table7_set=TABLE7.get((R, A_r), {}).get("set", ""),
                     n_rest=n_tot[0], n_normal=n_tot[1], n_active=n_tot[2],
                     train_rest=n_tr[0], train_normal=n_tr[1], train_active=n_tr[2],
                     test_rest=n_te[0], test_normal=n_te[1], test_active=n_te[2],
                     comparable=is_comparable(n_te[2], n_tr[2]), smote_k=cell["smote_k"], trainable=cell["trainable"])
    # persistence (T0.4 rule + unmodified evaluate(): point metrics and its own-RNG CI, cross-check)
    p_pers = persistence_predictions(cell["y"], cell["i_split"])
    t04 = t04_evaluate("persistence", y_true, p_pers)
    row.update(pers_f1_macro=float(t04["f1_macro"]), pers_f1_active=float(t04["f1_active"]),
               pers_accuracy=float(t04["accuracy"]), pers_mcc=float(t04["mcc"]),
               pers_f1_macro_ci_low_t04=float(t04["f1_macro_ci_lo"]), pers_f1_macro_ci_high_t04=float(t04["f1_macro_ci_hi"]))
    # majority
    p_maj, maj = majority_predictions(cell["y"].iloc[:cell["i_split"]], len(y_true))
    row.update(majority_class=maj, maj_f1_macro=f1_macro(y_true, p_maj),
               maj_accuracy=float(np.mean(p_maj == y_true)))
    shap_rows: list[dict] = []
    if not cell["trainable"]:
        row.update({k: np.nan for k in ("f1_macro", "accuracy", "mcc", "balanced_accuracy", "f1_rest", "f1_normal",
                                        "f1_active")})
        return row, shap_rows
    m = metrics(y_true, cell["y_pred"])
    row.update(m)
    # marginal CIs (RF, persistence) and the paired delta on the shared matrix
    b = paired_bootstrap(y_true, cell["y_pred"], p_pers, f1_macro, idx)
    row.update(f1_macro_ci_low=b["ci_A_low"], f1_macro_ci_high=b["ci_A_high"],
               pers_f1_macro_ci_low=b["ci_B_low"], pers_f1_macro_ci_high=b["ci_B_high"],
               delta_f1_macro=b["diff_point"], delta_f1_macro_ci_low=b["ci_low"], delta_f1_macro_ci_high=b["ci_high"],
               delta_f1_macro_include_zero=bool(b["ci_low"] <= 0.0 <= b["ci_high"]),
               delta_f1_macro_prop_gt_0=b["prop_diff_gt_0"], n_skipped=b["n_skipped"], n_used=b["n_used"])
    ba = paired_bootstrap(y_true, cell["y_pred"], p_pers, f1_active, idx)
    row.update(f1_active_ci_low=ba["ci_A_low"], f1_active_ci_high=ba["ci_A_high"])
    row.update(rf_gt_persistence=bool(m["f1_macro"] > row["pers_f1_macro"]),
               rf_gt_majority=bool(m["f1_macro"] > row["maj_f1_macro"]))
    # SHAP (canonical T1.4 procedure) on the imputed, unscaled test set
    mean_abs = mean_abs_shap(cell["rf"], cell["split"].X_test)
    ranks = rank_desc(mean_abs)
    hr_rank = int(ranks[FEATURES.index(HR_FEATURE)])
    for j, (name, val, rk) in enumerate(top_shap(mean_abs), start=1):
        row[f"shap_top{j}"] = name
        row[f"shap_top{j}_value"] = val
    row.update(aw_hr_mean_rank=hr_rank, aw_hr_mean_rank_1=bool(hr_rank == 1))
    for j, f in enumerate(FEATURES):
        shap_rows.append(dict(cell=row["cell"], R=R, A_r=A_r, A_s=A_s, feature=short_name(f),
                              mean_abs_shap=float(mean_abs[j]), rank=int(ranks[j])))
    return row, shap_rows


# ------------------------------------------------------------------ figure
def make_figure(grid: pd.DataFrame, path_stem: Path) -> int:
    """Two 3 x 3 heatmaps: (a) macro F1, (b) F1 Active of the RF; value and Active-in-test
    count in each cell; non-comparable cells hatched and starred; canonical cell outlined."""
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    g = grid.set_index(["R", "A_r"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    panels = [("f1_macro", "(a) Macro F1", "Blues"), ("f1_active", "(b) F1 Active", "Oranges")]
    for ax, (col, title, cmap) in zip(axes, panels):
        mat = np.array([[g.loc[(R, A), col] for A in ACTIVE_R_GRID] for R in REST_GRID], dtype=float)
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
        for r, R in enumerate(REST_GRID):
            for c, A in enumerate(ACTIVE_R_GRID):
                row = g.loc[(R, A)]
                v = float(row[col])
                comp = bool(row["comparable"])
                if not comp:
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="///",
                                           edgecolor="#777777", lw=0))
                txt_color = "white" if v > 0.6 else "#222222"
                ax.text(c, r - 0.08, ("%.3f" % v) + ("" if comp else "*"), ha="center", va="center",
                        fontsize=10.5, fontweight="bold", color=txt_color)
                ax.text(c, r + 0.26, f"Active in test: {int(row['test_active'])}", ha="center", va="center",
                        fontsize=7.5, color=txt_color)
        rc, cc = REST_GRID.index(CANONICAL[0]), ACTIVE_R_GRID.index(CANONICAL[1])
        ax.add_patch(Rectangle((cc - 0.5, rc - 0.5), 1, 1, fill=False, edgecolor="#1565C0" if col == "f1_macro"
                               else "#E65100", lw=2.6))
        ax.set_xticks(range(len(ACTIVE_R_GRID)))
        ax.set_xticklabels([str(a) for a in ACTIVE_R_GRID], fontsize=9)
        ax.set_yticks(range(len(REST_GRID)))
        ax.set_yticklabels([str(r) for r in REST_GRID], fontsize=9)
        ax.set_xlabel("Active readiness threshold A_r (sleep >= 75)", fontsize=9)
        ax.set_ylabel("Rest threshold R (readiness or sleep < R)", fontsize=9)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", metadata={"Software": None})
    n_axes = len(fig.axes)
    plt.close(fig)
    return n_axes


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

    df = load_raw()
    idx = np.load(res / INDEX_FILE)
    regen = np.random.default_rng(SEED).integers(0, N_TEST, size=(N_BOOT, N_TEST))
    check(idx.shape == (N_BOOT, N_TEST) and idx.dtype == regen.dtype and np.array_equal(idx, regen),
          f"{INDEX_FILE} loaded, byte-identical to default_rng({SEED}).integers(0, {N_TEST}, size=({N_BOOT}, {N_TEST}))",
          abort=True)
    t11_ci = read_t11_rf_f1_ci()

    # ---- T3.1.0a identity on the canonical cell
    print("\n=== T3.1.0a identity, cell (60, 85, 75) ===")
    canon = fit_eval_labels(*CANONICAL, df=df)
    check(canon["counts"]["total"] == (202, 710, 46) and canon["counts"]["train"] == (158, 570, 38)
          and canon["counts"]["test"] == (44, 140, 8), f"canonical distributions {canon['counts']}", abort=True)
    check(canon["smote_k"] == K_CANON, f"canonical SMOTE k = {canon['smote_k']}", abort=True)
    ref = pd.read_csv(res / "test_predictions_rf.csv")
    check(np.array_equal(ref["y_true"].to_numpy(), canon["y_true"]), "y_true == test_predictions_rf.csv", abort=True)
    check(np.array_equal(ref["y_pred"].to_numpy(), canon["y_pred"]), "y_pred element-wise == test_predictions_rf.csv",
          abort=True)
    max_dp = float(np.abs(ref[["p_rest", "p_normal", "p_active"]].to_numpy() - canon["proba"]).max())
    check(max_dp < 1e-9, f"probabilities == csv (max abs diff {max_dp:.2e} < 1e-9)", abort=True)
    check(list(pd.to_datetime(ref["date"]).dt.strftime("%Y-%m-%d")) == list(canon["dates_test"].dt.strftime("%Y-%m-%d")),
          "test dates == csv", abort=True)
    m_canon = metrics(canon["y_true"], canon["y_pred"])
    for k, v in CANON_RF.items():
        check(round(m_canon[k], 4) == v, f"RF {k} {m_canon[k]:.6f} == canonical {v}", abort=True)
    ref_p = pd.read_csv(res / "test_predictions_persistence.csv")
    p_pers = persistence_predictions(canon["y"], canon["i_split"])
    check(np.array_equal(ref_p["y_pred"].to_numpy(), p_pers), "persistence y_pred == test_predictions_persistence.csv",
          abort=True)
    canon_row, canon_shap = evaluate_cell(canon, idx)
    check(round(canon_row["pers_f1_macro"], 4) == CANON_PERSISTENCE["f1_macro"]
          and (round(canon_row["pers_f1_macro_ci_low_t04"], 4), round(canon_row["pers_f1_macro_ci_high_t04"], 4))
          == CANON_PERSISTENCE["ci"],
          f"persistence (T0.4 evaluate) {canon_row['pers_f1_macro']:.4f} [{canon_row['pers_f1_macro_ci_low_t04']:.4f}, "
          f"{canon_row['pers_f1_macro_ci_high_t04']:.4f}] == T0.4 0.3039 [0.2612, 0.3501]", abort=True)
    check((round(canon_row["pers_f1_macro_ci_low"], 4), round(canon_row["pers_f1_macro_ci_high"], 4))
          == CANON_PERSISTENCE["ci"],
          f"persistence CI on the shared matrix [{canon_row['pers_f1_macro_ci_low']:.4f}, "
          f"{canon_row['pers_f1_macro_ci_high']:.4f}] == T0.4", abort=True)
    got = (round(canon_row["f1_macro_ci_low"], 4), round(canon_row["f1_macro_ci_high"], 4))
    check(got == t11_ci, f"RF macro F1 marginal CI {got} == T1.1 CIFRE_REVIZIE.md {t11_ci}", abort=True)
    check(round(canon_row["maj_f1_macro"], 4) == CANON_MAJORITY["f1_macro"]
          and round(canon_row["maj_accuracy"], 4) == CANON_MAJORITY["accuracy"],
          f"majority {canon_row['maj_f1_macro']:.4f} / {canon_row['maj_accuracy']:.4f} == 0.2811 / 0.7292", abort=True)
    check(round(canon_row["delta_f1_macro"], 4) == 0.2024 and round(canon_row["delta_f1_macro_ci_low"], 4) == 0.0642
          and round(canon_row["delta_f1_macro_ci_high"], 4) == 0.3173,
          f"paired RF minus persistence {canon_row['delta_f1_macro']:.4f} [{canon_row['delta_f1_macro_ci_low']:.4f}, "
          f"{canon_row['delta_f1_macro_ci_high']:.4f}] == T1.2b 0.2024 [0.0642, 0.3173]")
    ref_shap = pd.read_csv(res / "t14_shap_reference.csv")
    top3_ref = [short_name(f) for f in ref_shap.sort_values("rank").head(3)["feature"]]
    top3_here = [canon_row[f"shap_top{j}"] for j in (1, 2, 3)]
    check(top3_here == top3_ref, f"SHAP top-3 {top3_here} == T1.4 reference {top3_ref} (t14_shap_reference.csv)")
    ref_by_short = {short_name(f): float(v) for f, v in zip(ref_shap["feature"], ref_shap["mean_abs_shap"])}
    max_ds = max(abs(canon_row[f"shap_top{j}_value"] - ref_by_short[canon_row[f"shap_top{j}"]]) for j in (1, 2, 3))
    check(max_ds < 1e-6, f"SHAP top-3 values == T1.4 reference (max abs diff {max_ds:.2e} < 1e-6)")
    canon_shap_sorted = sorted(canon_shap, key=lambda r: r["rank"])
    check([r["feature"] for r in canon_shap_sorted[:3]] == top3_ref, "t31_shap_top3 rows of the canonical cell == reference")

    # ---- T3.1.1 grid
    print("\n=== T3.1.1 grid 3 x 3 (A_s = 75) ===")
    rows: list[dict] = []
    shap_all: list[dict] = []
    pred_files: dict[str, str] = {}
    label_identity_90_80: dict[str, bool] = {}
    for R in REST_GRID:
        for A_r in ACTIVE_R_GRID:
            if (R, A_r, ACTIVE_SLEEP) == CANONICAL:
                cell, row, srows = canon, canon_row, canon_shap
            else:
                cell = fit_eval_labels(R, A_r, ACTIVE_SLEEP, df=df)
                row, srows = evaluate_cell(cell, idx)
            check(cell["counts"]["test"][2] >= 1 and all(c > 0 for c in cell["counts"]["test"]),
                  f"{row['cell']}: all three classes present in the test labels (F1 convention identical to Table 7)")
            if A_r == 90:   # same labels as Table 7 T4 / T6 (90/80)?
                _, y80, _ = build_dataset(df, rest_cutoff=R, active_readiness=90, active_sleep=80)
                label_identity_90_80[row["cell"]] = bool(np.array_equal(y80.to_numpy(), cell["y"].to_numpy()))
            t7 = TABLE7.get((R, A_r))
            if t7 is not None:
                check(cell["counts"]["total"] == t7["dist"],
                      f"{row['cell']}: distribution {cell['counts']['total']} == Table 7 {t7['set']} {t7['dist']}",
                      abort=t7["abort"])
                if cell["trainable"]:
                    check(round(row["f1_macro"], 3) == t7["f1_macro"],
                          f"{row['cell']}: macro F1 {row['f1_macro']:.6f} == Table 7 {t7['set']} {t7['f1_macro']:.3f}",
                          abort=t7["abort"])
            if cell["trainable"]:
                f = res / f"t31_predictions_rf_{row['cell']}.csv"
                save_predictions(f, cell["dates_test"], cell["y_true"], cell["y_pred"], cell["proba"])
                pred_files[f.name] = sha256(f)
                check(row["n_skipped"] == 0 or row["test_active"] < MIN_ACTIVE_TEST,
                      f"{row['cell']}: {row['n_skipped']} resamples skipped (class absent from the resample)")
            rows.append(row)
            shap_all.extend(srows)
            print(f"  {row['cell']:<9} {cell['counts']['total']} train A {row['train_active']:>3} test A {row['test_active']:>2} "
                  f"k={row['smote_k']} comparable={row['comparable']!s:<5} RF F1 {row.get('f1_macro', np.nan):.4f} "
                  f"[{row.get('f1_macro_ci_low', np.nan):.4f}, {row.get('f1_macro_ci_high', np.nan):.4f}] "
                  f"Active {row.get('f1_active', np.nan):.4f}  pers {row['pers_f1_macro']:.4f} "
                  f"[{row.get('pers_f1_macro_ci_low', np.nan):.4f}, {row.get('pers_f1_macro_ci_high', np.nan):.4f}]  "
                  f"delta {row.get('delta_f1_macro', np.nan):+.4f} [{row.get('delta_f1_macro_ci_low', np.nan):+.4f}, "
                  f"{row.get('delta_f1_macro_ci_high', np.nan):+.4f}]  maj {row['maj_f1_macro']:.4f}  "
                  f"top1 {row.get('shap_top1', 'n/a')} (hr rank {row.get('aw_hr_mean_rank', 'n/a')})  skipped {row.get('n_skipped', 'n/a')}")
    grid = pd.DataFrame(rows)
    check(len(grid) == 9, f"grid has 9 rows ({len(grid)})", abort=True)
    check(int(grid["canonical"].sum()) == 1, "exactly one canonical cell")
    tr = grid[grid.trainable]
    check(bool(((tr.f1_macro_ci_low <= tr.f1_macro) & (tr.f1_macro <= tr.f1_macro_ci_high)).all())
          and bool(((tr.pers_f1_macro_ci_low <= tr.pers_f1_macro) & (tr.pers_f1_macro <= tr.pers_f1_macro_ci_high)).all())
          and bool(((tr.delta_f1_macro_ci_low <= tr.delta_f1_macro) & (tr.delta_f1_macro <= tr.delta_f1_macro_ci_high)).all()),
          "sanity: every CI contains its point estimate (RF, persistence, delta)")
    check(bool((tr.delta_f1_macro - (tr.f1_macro - tr.pers_f1_macro)).abs().max() < 1e-12),
          "sanity: delta == RF minus persistence at the point")
    comp = grid[grid.comparable]
    print(f"\n  comparable cells: {len(comp)} / 9; RF > persistence in {int(comp.rf_gt_persistence.sum())}, "
          f"RF > majority in {int(comp.rf_gt_majority.sum())}, aw_hr_mean rank 1 in {int(comp.aw_hr_mean_rank_1.sum())}; "
          f"macro F1 range [{comp.f1_macro.min():.4f}, {comp.f1_macro.max():.4f}], "
          f"F1 Active range [{comp.f1_active.min():.4f}, {comp.f1_active.max():.4f}]")

    # ---- outputs
    cols = ["cell", "R", "A_r", "A_s", "canonical", "table7_set", "comparable", "trainable", "smote_k",
            "n_rest", "n_normal", "n_active", "train_rest", "train_normal", "train_active",
            "test_rest", "test_normal", "test_active",
            "f1_macro", "f1_macro_ci_low", "f1_macro_ci_high", "accuracy", "mcc", "balanced_accuracy",
            "f1_rest", "f1_normal", "f1_active", "f1_active_ci_low", "f1_active_ci_high"]
    cols += [c for c in grid.columns if c.startswith("cm_")]
    cols += ["pers_f1_macro", "pers_f1_macro_ci_low", "pers_f1_macro_ci_high", "pers_f1_macro_ci_low_t04",
             "pers_f1_macro_ci_high_t04", "pers_f1_active", "pers_accuracy", "pers_mcc",
             "delta_f1_macro", "delta_f1_macro_ci_low", "delta_f1_macro_ci_high", "delta_f1_macro_include_zero",
             "delta_f1_macro_prop_gt_0", "n_used", "n_skipped",
             "majority_class", "maj_f1_macro", "maj_accuracy", "rf_gt_persistence", "rf_gt_majority",
             "shap_top1", "shap_top1_value", "shap_top2", "shap_top2_value", "shap_top3", "shap_top3_value",
             "aw_hr_mean_rank", "aw_hr_mean_rank_1"]
    grid = grid[cols]
    run_date = datetime.now().isoformat(timespec="seconds")
    day = run_date[:10]
    grid.to_csv(res / "t31_threshold_grid.csv", index=False, float_format="%.10g", lineterminator="\n")
    fcols = [c for c in grid.columns if grid[c].dtype.kind == "f"]
    md = md_table(grid, {c: "%.4f" for c in fcols})
    md += ["",
           "Labelling rule: Rest = readiness < R or sleep < R; Active = readiness >= A_r and sleep >= A_s; Normal = rest. "
           f"Grid R in {list(REST_GRID)} x A_r in {list(ACTIVE_R_GRID)}, A_s = {ACTIVE_SLEEP} fixed (sleep grid not run, "
           "KBS_Revision_Plan section 8.11). Canonical cell R60_A85 (202 / 710 / 46; CIFRE_CANONICE.md). "
           "Same 958 labelled days and temporal cut 766 / 192 in every cell. RF = canonical model (100 trees, depth 10, "
           "min leaf 5, seed 42) on all 19 lag-1 features, median imputer + SMOTE fit on train "
           "(common.preprocess_split); smote_k follows the canonical Table 7 script (3 when the smallest train class "
           "has >= 4 rows, n_min - 1 for 2-3 rows). comparable = Active in test >= 5 (Table 7 convention) and Active "
           "in train >= 4. table7_set = row of CIFRE_CANONICE.md section 5 with identical labels (T4* / T6*: the 90/80 "
           "sets coincide with A_r = 90, A_s = 75 because every readiness >= 90 day has sleep >= 80). "
           "Persistence = label of the previous row of the 958-day series (T0.4 rule, evaluated with the unmodified "
           "persistence_baseline.evaluate(); *_t04 CI = its own RNG stream). Majority = train mode. "
           f"[ci_low, ci_high] = percentiles 2.5 / 97.5 over {N_BOOT} resamples of the 192 test days on the shared matrix "
           f"{INDEX_FILE} (loaded, not regenerated); delta = RF minus persistence on macro F1, paired (same indices for "
           "both), prop_gt_0 descriptive (not a p-value); n_skipped = resamples with a class absent from the resampled "
           "y_true (only in cells with 1 Active in test). CIs are computed on DIFFERENT labels in every cell and are not "
           "comparable across cells as if on one test set. cm_xy = confusion count, true x, predicted y (r / n / a). "
           "SHAP = TreeExplainer on the imputed test set, mean |SHAP| over 192 days x 3 classes (T1.4 procedure); "
           "names use the Figure 2 device prefix. All figures outside R60_A85 are NON-canonical. "
           f"Script: revision/scripts/threshold_sensitivity.py, run {day}."]
    write_text(res / "t31_threshold_grid_table.md", "\n".join(md) + "\n")
    shap_df = pd.DataFrame(shap_all)
    shap_top = shap_df[shap_df["rank"] <= TOP_N_SHAP].sort_values(["R", "A_r", "rank"]).reset_index(drop=True)
    shap_top.to_csv(res / "t31_shap_top3.csv", index=False, float_format="%.10g", lineterminator="\n")
    check(len(shap_top) == 3 * int(grid.trainable.sum()), f"t31_shap_top3.csv has 3 rows per trainable cell ({len(shap_top)})")

    n_axes = make_figure(grid, FIG / "t31_threshold_grid")
    check(n_axes == 4, f"figure has 2 heatmaps + 2 colorbars ({n_axes} axes)")

    meta = {
        "run_date": run_date, "elapsed_seconds": round(time.time() - t0, 1),
        "seed": SEED, "n_resamples": N_BOOT, "n_test": N_TEST, "n_train": N_TRAIN, "n_total": N_TOTAL,
        "grid": {"R": list(REST_GRID), "A_r": list(ACTIVE_R_GRID), "A_s": ACTIVE_SLEEP, "n_cells": 9},
        "canonical_cell": list(CANONICAL),
        "comparability": {"min_active_test": MIN_ACTIVE_TEST, "min_active_train": MIN_ACTIVE_TRAIN},
        "smote_k_convention": "canonical Table 7 script: 3 if n_min >= 4 else n_min - 1 if n_min >= 2 else no SMOTE",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "imbalanced_learn": imblearn.__version__,
                     "shap": shap.__version__, "matplotlib": matplotlib.__version__},
        "t310a_canonical": {k: canon_row[k] for k in ("f1_macro", "accuracy", "mcc", "f1_active", "pers_f1_macro",
                                                       "maj_f1_macro", "maj_accuracy")},
        "t11_rf_f1_macro_ci_from_cifre_revizie": list(t11_ci),
        "canonical_rf_f1_macro_marginal_ci": [canon_row["f1_macro_ci_low"], canon_row["f1_macro_ci_high"]],
        "table7_checks": {f"R{R}_A{A}": v["set"] for (R, A), v in TABLE7.items()},
        "labels_identical_to_table7_90_80": label_identity_90_80,
        "cells_per_smote_k": {str(k): int((grid.smote_k == k).sum()) for k in sorted(grid.smote_k.unique())},
        "n_comparable": int(grid.comparable.sum()),
        "bootstrap_indices_sha256": sha256(res / INDEX_FILE),
        "prediction_files_sha256": pred_files,
        "figure_axes": n_axes,
        "sanity_checks_failed": fails,
    }
    with open(res / "t31_run_meta.json", "w", newline="\n", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    print(f"\nelapsed {time.time() - t0:.1f}s; outputs in {res}")
    if fails:
        print(f"{len(fails)} check(s) FAILED: {fails}")
        sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="T3.1 labelling-threshold sensitivity (3 x 3 grid)")
    ap.add_argument("--results", default=RES, type=Path)
    main(ap.parse_args().results)
