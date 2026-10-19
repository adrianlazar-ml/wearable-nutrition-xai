"""
group_ablation.py - T1.5 (Reviewer 1, R1.7): feature-group ablation of the
canonical RF with leave-one-group-out (LOGO) and keep-one-group-in (KOGI).

Groups (complete partition of the 19 lag-1 features, 2 + 4 + 3 + 2 + 4 + 4):
  G1 composite_scores   readiness_score, sleep_score
  G2 heart_rate         aw_hr_mean, rhr_lowest, rhr_avg, recovery_index
  G3 hrv                hrv_mean, hrv_balance, aw_hrv_mean
  G4 sleep_architecture aw_sleep_deep_hours, aw_sleep_rem_hours
  G5 activity_energy    activity_score, steps_oura, aw_steps, aw_active_calories
  G6 resp_thermal_spo2  respiratory_rate, aw_resp_rate, temp_deviation, aw_spo2
SLEEP_SCORE_GROUP = "G1" puts sleep_score with readiness_score (the two scores
from which the T3 labels and the persistence baseline are derived); the
documented alternative "G4" would move it to sleep architecture.

Procedure in reduced feature space = the canonical device ablation
(ablation_study.py, CIFRE_CANONICE.md section 4): same 958 labelled days, same
temporal 80/20 split (766/192), SimpleImputer(median) and SMOTE(k=3, seed 42)
refit on the remaining features only, RF 100 trees / depth 10 / min leaf 5 /
seed 42 (save_test_predictions.build_models()["rf"]), features kept in the
canonical order. One function, fit_eval(features), is used for everything.

Stages (all in one run, each with abort on FAIL):
  T1.5.0  full 19 features: macro F1 0.5063 / acc 0.6406 / MCC 0.1416 at 4
          decimals and predictions identical to results/test_predictions_rf.csv
          (y_pred element-wise, probabilities at 1e-9).
  T1.5.0b Oura-only (11, no aw_ prefix) / Watch-only (8, aw_ prefix) through the
          same fit_eval: macro F1 0.428 / 0.324 and F1 Active 0.286 / 0.100 at
          3 decimals (canonical ablation).
  T1.5.1  LOGO: RF on 19 minus g for each group; metrics, confusion matrix,
          delta = full minus LOGO (positive = the group matters) on macro F1
          and F1 Active, paired bootstrap CI of the delta on the shared index
          matrix results/bootstrap_indices_1000x192_seed42.npy (loaded, not
          regenerated; verified byte-identical to default_rng(42)). Sanity:
          marginal CI of the full model on macro F1 == T1.1 row of
          CIFRE_REVIZIE.md at 4 decimals (read from the file).
  T1.5.2  KOGI: RF on g only; same metrics, no CI (descriptive).
  T1.5.3  figure results/figures/t15_group_ablation.{png,pdf}.

Outputs (revision/results/, LF):
  t15_logo.csv, t15_logo_table.md, t15_kogi.csv, t15_kogi_table.md
  t15_predictions_logo_{group}.csv, t15_predictions_kogi_{group}.csv
  t15_run_meta.json, figures/t15_group_ablation.{png,pdf}

Run from repo root:  python revision/scripts/group_ablation.py
Exits 1 on any failed check. Nothing canonical is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import imblearn
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES, ROOT, SEED, TRAIN_RATIO, build_dataset, load_raw  # noqa: E402
from save_test_predictions import build_models  # noqa: E402
from shap_stability import short_name  # noqa: E402
from statistical_tests import INDEX_FILE, LABELS, md_table, paired_bootstrap, write_text  # noqa: E402

RES = ROOT / "revision" / "results"
FIG = RES / "figures"
CIFRE_REVIZIE = ROOT / "revision" / "CIFRE_REVIZIE.md"

N_BOOT, N_TEST = 1000, 192
K_NEIGHBORS = 3
SLEEP_SCORE_GROUP = "G1"          # documented alternative: "G4"
RUN_KOGI = True

GROUP_TITLES = {
    "G1": "composite_scores", "G2": "heart_rate", "G3": "hrv",
    "G4": "sleep_architecture", "G5": "activity_energy", "G6": "resp_thermal_spo2",
}
GROUP_LABELS = {                  # figure labels, no underscores
    "G1": "Composite\nscores", "G2": "Heart rate", "G3": "HRV",
    "G4": "Sleep\narchitecture", "G5": "Activity /\nenergy", "G6": "Resp / thermal /\nSpO2",
}
_BASE_GROUPS = {
    "G1": ["readiness_score"],
    "G2": ["aw_hr_mean", "rhr_lowest", "rhr_avg", "recovery_index"],
    "G3": ["hrv_mean", "hrv_balance", "aw_hrv_mean"],
    "G4": ["aw_sleep_deep_hours", "aw_sleep_rem_hours"],
    "G5": ["activity_score", "steps_oura", "aw_steps", "aw_active_calories"],
    "G6": ["respiratory_rate", "aw_resp_rate", "temp_deviation", "aw_spo2"],
}

# Reference figures the script must reproduce (CIFRE_CANONICE.md).
CANON_FULL = dict(f1_macro=0.5063, accuracy=0.6406, mcc=0.1416)
CANON_DEVICE = {"oura_only": dict(f1_macro=0.428, f1_active=0.286),
                "watch_only": dict(f1_macro=0.324, f1_active=0.100)}
MAJORITY_F1 = 0.2811


# ------------------------------------------------------------------ groups
def build_groups(sleep_score_group: str = SLEEP_SCORE_GROUP) -> dict[str, list[str]]:
    """Group id -> list of lag-1 feature names (canonical order within each group)."""
    if sleep_score_group not in ("G1", "G4"):
        raise ValueError("SLEEP_SCORE_GROUP must be 'G1' or 'G4'")
    base = {g: list(f) for g, f in _BASE_GROUPS.items()}
    base[sleep_score_group].append("sleep_score")
    return {g: [f"{f}_lag1" for f in FEATURES_BASE_ORDER(base[g])] for g in base}


def FEATURES_BASE_ORDER(names: list[str]) -> list[str]:
    """Sort base names by the canonical feature order."""
    order = {f.replace("_lag1", ""): i for i, f in enumerate(FEATURES)}
    return sorted(names, key=lambda n: order[n])


def check_partition(groups: dict[str, list[str]], features: list[str] = FEATURES) -> None:
    """Assert: union == the 19 features, pairwise disjoint."""
    all_feats = [f for g in groups.values() for f in g]
    if len(all_feats) != len(set(all_feats)):
        raise AssertionError("groups overlap")
    if set(all_feats) != set(features) or len(all_feats) != len(features):
        raise AssertionError(f"union of groups != features: {sorted(set(all_feats) ^ set(features))}")


def group_name(g: str) -> str:
    return f"{g}_{GROUP_TITLES[g]}"


# ------------------------------------------------------------------ core
def subset_in_canonical_order(features: list[str]) -> list[str]:
    return [f for f in FEATURES if f in set(features)]


def fit_eval(X: pd.DataFrame, y: pd.Series, features: list[str], seed: int = SEED,
             k_neighbors: int = K_NEIGHBORS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical reduced-space procedure (ablation_study.run_ablation):
    temporal split -> median imputer fit on train -> SMOTE(k=3) on train -> RF.
    Returns (y_true, y_pred, proba) on the 192 test days."""
    feats = subset_in_canonical_order(features)
    if not feats:
        raise ValueError("empty feature list")
    i = int(len(X) * TRAIN_RATIO)
    X_tr, X_te = X.iloc[:i][feats], X.iloc[i:][feats]
    y_tr, y_te = y.iloc[:i], y.iloc[i:]
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=feats)
    X_te_imp = pd.DataFrame(imputer.transform(X_te), columns=feats)
    X_sm, y_sm = SMOTE(random_state=seed, k_neighbors=k_neighbors).fit_resample(X_tr_imp, y_tr)
    rf = build_models()["rf"]
    rf.fit(X_sm, y_sm)
    assert list(rf.classes_) == LABELS
    return y_te.to_numpy(dtype=int), rf.predict(X_te_imp).astype(int), rf.predict_proba(X_te_imp)


def f1_class(y: np.ndarray, p: np.ndarray, c: int) -> float:
    return float(f1_score(y, p, labels=[c], average="macro", zero_division=0))


def f1_macro(y: np.ndarray, p: np.ndarray) -> float:
    return float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0))


def f1_active(y: np.ndarray, p: np.ndarray) -> float:
    return f1_class(y, p, 2)


CI_METRICS = {"f1_macro": f1_macro, "f1_active": f1_active}


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    cm = confusion_matrix(y, p, labels=LABELS)
    names = "rna"
    out: dict[str, float | int] = dict(
        f1_macro=f1_macro(y, p), accuracy=float(accuracy_score(y, p)),
        mcc=float(matthews_corrcoef(y, p)), balanced_accuracy=float(balanced_accuracy_score(y, p)),
        f1_rest=f1_class(y, p, 0), f1_normal=f1_class(y, p, 1), f1_active=f1_class(y, p, 2))
    for a, ra in enumerate(names):
        for b, cb in enumerate(names):
            out[f"cm_{ra}{cb}"] = int(cm[a, b])        # rows = true, columns = predicted
    return out


def delta(full_value: float, ablated_value: float) -> float:
    """full minus ablated: positive = removing the group hurts = the group matters."""
    return float(full_value - ablated_value)


def read_t11_rf_f1_ci(path: Path = CIFRE_REVIZIE) -> tuple[float, float]:
    """Marginal 95 percent CI of RF macro F1 on the shared matrix, from the T1.1 sanity row."""
    pat = re.compile(r"^\| T1\.1 \| RF F1 / acc / MCC / QWK \(sanity[^|]*\| [0-9.]+ \[([0-9.]+), ([0-9.]+)\]")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            return float(m.group(1)), float(m.group(2))
    raise ValueError("T1.1 RF sanity row not found in CIFRE_REVIZIE.md")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_predictions(path: Path, dates: pd.Series, y: np.ndarray, p: np.ndarray, proba: np.ndarray) -> None:
    df = pd.DataFrame({"date": pd.to_datetime(dates).dt.strftime("%Y-%m-%d").to_numpy(), "y_true": y,
                       "y_pred": p, "p_rest": proba[:, 0], "p_normal": proba[:, 1], "p_active": proba[:, 2]})
    df.to_csv(path, index=False, lineterminator="\n")


# ------------------------------------------------------------------ figure
def make_figure(logo: pd.DataFrame, kogi: pd.DataFrame, full_f1: float, groups: dict[str, list[str]]) -> int:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    gids = list(groups)
    x = np.arange(len(gids))
    labels = [GROUP_LABELS[g] for g in gids]
    c_macro, c_active = "#1565C0", "#E65100"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1]))

    lg = logo.set_index("group")
    off = 0.17
    # T3.1.6 figure hygiene (2026-10-19): short panel titles, plain y label, square marker for the
    # F1 Active series (same colours); method details go to the caption / CIFRE_REVIZIE. Numbers unchanged.
    for k, (col, color, marker, lab) in enumerate([("f1_macro", c_macro, "o", "delta macro F1"),
                                                   ("f1_active", c_active, "s", "delta F1 Active")]):
        pos = x + (off if k else -off)
        d = lg.loc[[group_name(g) for g in gids], f"delta_{col}"].to_numpy()
        lo = lg.loc[[group_name(g) for g in gids], f"delta_{col}_ci_low"].to_numpy()
        hi = lg.loc[[group_name(g) for g in gids], f"delta_{col}_ci_high"].to_numpy()
        ax1.errorbar(pos, d, yerr=[d - lo, hi - d], fmt=marker, color=color, ms=7, lw=1.6, capsize=4,
                     capthick=1.6, label=lab)
    ax1.axhline(0, color="#555555", lw=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Delta (full RF minus ablated RF)", fontsize=8.5)
    ax1.set_title("(a) Leave-one-group-out", fontsize=9.5, fontweight="bold")
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(fontsize=8.5, loc="upper left")

    kg = kogi.set_index("group")
    vals = kg.loc[[group_name(g) for g in gids], "f1_macro"].to_numpy()
    act = kg.loc[[group_name(g) for g in gids], "f1_active"].to_numpy()
    bars = ax2.bar(x, vals, width=0.58, color=c_macro, alpha=0.9, label="macro F1, RF on the group only")
    for b, v, a in zip(bars, vals, act):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}\n(Active {a:.2f})", ha="center",
                 va="bottom", fontsize=7.5, color="#333333")
    ax2.axhline(full_f1, color="#333333", lw=1.2, ls="--", label=f"full RF, 19 features: macro F1 {full_f1:.4f}")
    ax2.axhline(MAJORITY_F1, color="#888888", lw=1.2, ls=":", label=f"majority baseline: macro F1 {MAJORITY_F1:.4f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylim(0, max(0.72, float(vals.max()) + 0.12))
    ax2.set_ylabel("macro F1 on the test set (absolute, non-canonical)", fontsize=8.5)
    ax2.set_title("(b) Keep-one-group-in", fontsize=9.5, fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(fontsize=8.5, loc="upper left")

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "t15_group_ablation.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    fig.savefig(FIG / "t15_group_ablation.png", dpi=300, bbox_inches="tight", metadata={"Software": None})
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

    groups = build_groups(SLEEP_SCORE_GROUP)
    check_partition(groups)
    sizes = [len(v) for v in groups.values()]
    check(sizes == [2, 4, 3, 2, 4, 4] if SLEEP_SCORE_GROUP == "G1" else sizes == [1, 4, 3, 3, 4, 4],
          f"partition of the 19 features into 6 groups, sizes {sizes} (SLEEP_SCORE_GROUP = {SLEEP_SCORE_GROUP})",
          abort=True)

    X, y, dates = build_dataset(load_raw())
    check(len(X) == 958 and list(X.columns) == FEATURES, f"dataset {X.shape}, canonical feature order", abort=True)
    i = int(len(X) * TRAIN_RATIO)
    dates_te = dates.iloc[i:].reset_index(drop=True)
    check(i == 766 and len(dates_te) == N_TEST, "split 766 / 192", abort=True)

    # ---- T1.5.0 reference reproduction
    print("\n=== T1.5.0 full RF, 19 features ===")
    y_true, p_full, pr_full = fit_eval(X, y, FEATURES)
    m_full = metrics(y_true, p_full)
    for k, v in CANON_FULL.items():
        check(round(m_full[k], 4) == v, f"T1.5.0 {k} {m_full[k]:.6f} == canonical {v}", abort=True)
    ref = pd.read_csv(res / "test_predictions_rf.csv")
    check(np.array_equal(ref["y_true"].to_numpy(), y_true), "T1.5.0 y_true == test_predictions_rf.csv", abort=True)
    check(np.array_equal(ref["y_pred"].to_numpy(), p_full), "T1.5.0 y_pred element-wise == test_predictions_rf.csv",
          abort=True)
    max_dp = float(np.abs(ref[["p_rest", "p_normal", "p_active"]].to_numpy() - pr_full).max())
    check(max_dp < 1e-9, f"T1.5.0 probabilities == csv (max abs diff {max_dp:.2e} < 1e-9)", abort=True)
    check(list(pd.to_datetime(ref["date"]).dt.strftime("%Y-%m-%d")) == list(dates_te.dt.strftime("%Y-%m-%d")),
          "T1.5.0 test dates == csv", abort=True)

    # ---- T1.5.0b identity of the reduced-space procedure (canonical device ablation)
    print("\n=== T1.5.0b device ablation through fit_eval ===")
    device = {"oura_only": [f for f in FEATURES if not f.startswith("aw_")],
              "watch_only": [f for f in FEATURES if f.startswith("aw_")]}
    device_rows = []
    for name, feats in device.items():
        _, p, _ = fit_eval(X, y, feats)
        m = metrics(y_true, p)
        device_rows.append(dict(config=name, n_features=len(feats), **m))
        for k, v in CANON_DEVICE[name].items():
            check(round(m[k], 3) == v, f"T1.5.0b {name} ({len(feats)} features) {k} {m[k]:.6f} == canonical {v:.3f}",
                  abort=True)

    # ---- shared bootstrap matrix
    idx = np.load(res / INDEX_FILE)
    regen = np.random.default_rng(SEED).integers(0, N_TEST, size=(N_BOOT, N_TEST))
    check(idx.shape == (N_BOOT, N_TEST) and idx.dtype == regen.dtype and np.array_equal(idx, regen),
          f"{INDEX_FILE} loaded, byte-identical to default_rng({SEED}).integers(0, {N_TEST}, size=({N_BOOT}, {N_TEST}))",
          abort=True)
    t11_ci = read_t11_rf_f1_ci()

    # ---- T1.5.1 LOGO
    print("\n=== T1.5.1 leave-one-group-out ===")
    pred_files: dict[str, str] = {}
    full_row = dict(config="full", group="none", n_features=len(FEATURES), features_removed="", **m_full)
    rows = [full_row]
    for g, feats in groups.items():
        kept = [f for f in FEATURES if f not in set(feats)]
        _, p, pr = fit_eval(X, y, kept)
        m = metrics(y_true, p)
        row = dict(config="logo", group=group_name(g), n_features=len(kept),
                   features_removed=";".join(short_name(f) for f in feats), **m)
        for mname, mfn in CI_METRICS.items():
            b = paired_bootstrap(y_true, p_full, p, mfn, idx)
            check(abs(b["diff_point"] - delta(m_full[mname], m[mname])) < 1e-12,
                  f"{group_name(g)} {mname}: diff_point == full minus LOGO ({b['diff_point']:+.4f})")
            row[f"delta_{mname}"] = b["diff_point"]
            row[f"delta_{mname}_ci_low"] = b["ci_low"]
            row[f"delta_{mname}_ci_high"] = b["ci_high"]
            row[f"delta_{mname}_include_zero"] = bool(b["ci_low"] <= 0.0 <= b["ci_high"])
            row[f"delta_{mname}_prop_gt_0"] = b["prop_diff_gt_0"]
            row[f"{mname}_ci_low"] = b["ci_B_low"]
            row[f"{mname}_ci_high"] = b["ci_B_high"]
            row[f"n_skipped_{mname}"] = b["n_skipped"]
            full_row[f"{mname}_ci_low"] = b["ci_A_low"]
            full_row[f"{mname}_ci_high"] = b["ci_A_high"]
        rows.append(row)
        f = res / f"t15_predictions_logo_{group_name(g)}.csv"
        save_predictions(f, dates_te, y_true, p, pr)
        pred_files[f.name] = sha256(f)
        print(f"  {group_name(g):<24} F1 {m['f1_macro']:.4f} (delta {row['delta_f1_macro']:+.4f} "
              f"[{row['delta_f1_macro_ci_low']:+.4f}, {row['delta_f1_macro_ci_high']:+.4f}])  "
              f"Active {m['f1_active']:.4f} (delta {row['delta_f1_active']:+.4f} "
              f"[{row['delta_f1_active_ci_low']:+.4f}, {row['delta_f1_active_ci_high']:+.4f}])")
    # sanity (abort): marginal CI of the full model on macro F1 == T1.1 (read from CIFRE_REVIZIE.md)
    got = (round(full_row["f1_macro_ci_low"], 4), round(full_row["f1_macro_ci_high"], 4))
    check(got == t11_ci, f"sanity: full RF macro F1 marginal CI {got} == T1.1 CIFRE_REVIZIE.md {t11_ci}", abort=True)
    for mname in CI_METRICS:          # full vs itself: delta 0 by definition, no CI columns
        full_row[f"delta_{mname}"] = 0.0
        full_row[f"delta_{mname}_include_zero"] = ""
        full_row[f"n_skipped_{mname}"] = 0
    logo = pd.DataFrame(rows)
    check(int(logo.filter(like="n_skipped").to_numpy().sum()) == 0, "LOGO: 0 resamples skipped")
    cols = ["config", "group", "n_features", "features_removed", "f1_macro", "accuracy", "mcc", "balanced_accuracy",
            "f1_rest", "f1_normal", "f1_active"] + [c for c in logo.columns if c.startswith("cm_")]
    cols += ["f1_macro_ci_low", "f1_macro_ci_high", "f1_active_ci_low", "f1_active_ci_high",
             "delta_f1_macro", "delta_f1_macro_ci_low", "delta_f1_macro_ci_high", "delta_f1_macro_include_zero",
             "delta_f1_macro_prop_gt_0", "delta_f1_active", "delta_f1_active_ci_low", "delta_f1_active_ci_high",
             "delta_f1_active_include_zero", "delta_f1_active_prop_gt_0", "n_skipped_f1_macro", "n_skipped_f1_active"]
    logo = logo[cols]
    check(len(logo) == 7 and round(float(logo.loc[0, "f1_macro"]), 4) == CANON_FULL["f1_macro"],
          "t15_logo: 7 rows, full row reproduces T1.5.0")

    # ---- T1.5.2 KOGI
    kogi = pd.DataFrame()
    if RUN_KOGI:
        print("\n=== T1.5.2 keep-one-group-in ===")
        krows = []
        for g, feats in groups.items():
            _, p, pr = fit_eval(X, y, feats)
            m = metrics(y_true, p)
            krows.append(dict(config="kogi", group=group_name(g), n_features=len(feats),
                              features_kept=";".join(short_name(f) for f in feats), **m))
            f = res / f"t15_predictions_kogi_{group_name(g)}.csv"
            save_predictions(f, dates_te, y_true, p, pr)
            pred_files[f.name] = sha256(f)
            print(f"  {group_name(g):<24} F1 {m['f1_macro']:.4f}  Active {m['f1_active']:.4f}  acc {m['accuracy']:.4f}")
        kogi = pd.DataFrame(krows)
        check(len(kogi) == 6, "t15_kogi: 6 rows")

    # ---- outputs
    run_date = datetime.now().isoformat(timespec="seconds")
    day = run_date[:10]
    ffmt = "%.4f"
    logo.to_csv(res / "t15_logo.csv", index=False, float_format="%.10g", lineterminator="\n")
    fcols = [c for c in logo.columns if logo[c].dtype.kind == "f"]
    logo_md = md_table(logo, {c: ffmt for c in fcols})
    logo_md += ["",
                "LOGO = RF (100 trees, depth 10, min leaf 5, seed 42) trained on the 19 lag-1 features minus one group; "
                "imputer (median) and SMOTE (k = 3, seed 42) refit on the remaining features, as in the canonical device "
                "ablation (CIFRE_CANONICE.md section 4, reproduced by this script: Oura-only 0.428 / Watch-only 0.324). "
                "delta = full minus LOGO (positive = removing the group lowers the test score). [ci_low, ci_high] = "
                f"percentile 2.5 / 97.5 of the paired bootstrap of the difference over {N_BOOT} resamples of the 192 test "
                f"days ({INDEX_FILE}, seed {SEED}, same indices for both models); prop_gt_0 = proportion of resamples "
                "with delta > 0 (descriptive, not a p-value); CIs are NOT adjusted for the 6 comparisons. "
                "cm_xy = confusion matrix count, true class x, predicted class y (r = Rest, n = Normal, a = Active). "
                "All scores on the test set (44 / 140 / 8) are informative and NON-canonical; the canonical RF figures "
                "(0.5063 / 0.6406 / 0.1416) remain. Groups: "
                + "; ".join(f"{group_name(g)} = {', '.join(short_name(f) for f in feats)}" for g, feats in groups.items())
                + f". SLEEP_SCORE_GROUP = {SLEEP_SCORE_GROUP}. Script: revision/scripts/group_ablation.py, run {day}."]
    write_text(res / "t15_logo_table.md", "\n".join(logo_md) + "\n")
    if RUN_KOGI:
        kogi.to_csv(res / "t15_kogi.csv", index=False, float_format="%.10g", lineterminator="\n")
        kogi_md = md_table(kogi, {c: ffmt for c in kogi.columns if kogi[c].dtype.kind == "f"})
        kogi_md += ["",
                    "KOGI = RF trained on one group only (same procedure as LOGO, descriptive, no CI). "
                    f"Reference: full RF macro F1 {CANON_FULL['f1_macro']:.4f}, majority baseline {MAJORITY_F1:.4f}. "
                    "Informative, NON-canonical. "
                    f"Script: revision/scripts/group_ablation.py, run {day}."]
        write_text(res / "t15_kogi_table.md", "\n".join(kogi_md) + "\n")

    n_axes = make_figure(logo[logo.config == "logo"], kogi, m_full["f1_macro"], groups) if RUN_KOGI else 0
    check(n_axes == 2, f"figure has 2 axes ({n_axes})")

    meta = {
        "run_date": run_date, "elapsed_seconds": round(time.time() - t0, 1),
        "seed": SEED, "n_resamples": N_BOOT, "n_test": N_TEST, "k_neighbors": K_NEIGHBORS,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "imbalanced_learn": imblearn.__version__,
                     "matplotlib": matplotlib.__version__},
        "SLEEP_SCORE_GROUP": SLEEP_SCORE_GROUP, "RUN_KOGI": RUN_KOGI,
        "groups": groups,
        "t150_full": {k: m_full[k] for k in ("f1_macro", "accuracy", "mcc", "f1_active")},
        "t150b_device": device_rows,
        "t11_rf_f1_macro_ci_from_cifre_revizie": list(t11_ci),
        "full_f1_macro_marginal_ci": [full_row["f1_macro_ci_low"], full_row["f1_macro_ci_high"]],
        "bootstrap_indices_sha256": sha256(res / INDEX_FILE),
        "prediction_files_sha256": pred_files,
        "figure_axes": n_axes,
        "sanity_checks_failed": fails,
    }
    with open(res / "t15_run_meta.json", "w", newline="\n", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    print(f"\nelapsed {time.time() - t0:.1f}s; outputs in {res}")
    if fails:
        print(f"{len(fails)} check(s) FAILED: {fails}")
        sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="T1.5 feature-group ablation (LOGO + KOGI)")
    ap.add_argument("--results", default=RES, type=Path)
    main(ap.parse_args().results)
