"""
shap_stability.py - T1.4: stability of the SHAP feature ranking of the canonical
RF (R1.6, Reviewer 1 point 6).

Reference ranking: mean |SHAP| of rf_revision.joblib on the 192 test days with
the canonical procedure (cap5_compare_algorithms_json_ext.run_shap(), see
_check_t14_inputs.py): shap.TreeExplainer(rf).shap_values(X_test), array
(192, 19, 3), mean of |SHAP| over samples and classes. Rank 1 = largest
mean |SHAP|; ties get the smallest rank ("min"); top-k = the k largest values
with stable tie-breaking by feature order.

Modes (--mode, default bootstrap; --n-runs, default 100):
  bootstrap : 100 bootstrap samples of the 766 raw train rows (with
              replacement, before imputation / SMOTE), indices
              np.random.default_rng(42).integers(0, 766, size=(100, 766)),
              saved to results/bootstrap_train_indices_100x766_seed42.npy.
              Per run: common.preprocess_split (imputer, SMOTE k=3 seed 42,
              scaler fit on the bootstrap rows only), RF from
              save_test_predictions.build_models()["rf"] with random_state 42
              fixed; only the data vary.
  seeds     : full train, canonical preprocessing (SMOTE seed 42), RF with
              random_state 0..99; only the model varies. Run 42 is the
              canonical model and must reproduce the reference exactly.
In both modes the test set is used only for the SHAP values and for the
informative, NON-canonical F1 of each retrained model: nothing is selected on
it, and rf_revision.joblib is never retrained.

Internal sanity checks (all must PASS, otherwise exit 1): n_runs results;
every run has 19 ranks forming a permutation of 1..19; Jaccard in [0, 1];
tau in [-1, 1]; mean |SHAP| >= 0; identity test (bootstrap: indices 0..765
without duplicates = original train; seeds: run 42) reproduces the reference
ranking exactly, mean |SHAP| at 1e-12 and F1 macro 0.5063.

Outputs (LF), results/:
  t14_shap_reference.csv                 feature, mean_abs_shap, rank (identical from both modes)
  t14_shap_values_{mode}.csv             run, feature, mean_abs_shap, rank
  t14_shap_stability_{mode}.csv / _table.md
  t14_run_meta_{mode}.json
  figures/t14_shap_rank_stability_{mode}.pdf / .png
  bootstrap_train_indices_100x766_seed42.npy   (bootstrap mode only)

Run from repo root:
    python revision/scripts/shap_stability.py --mode bootstrap
    python revision/scripts/shap_stability.py --mode seeds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from itertools import combinations
from pathlib import Path

import imblearn
import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
from scipy.stats import kendalltau, rankdata
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (FEATURES, ROOT, SEED, Split, build_dataset, canonical_split,  # noqa: E402
                    check_canonical_sizes, load_raw, preprocess_split)
from save_test_predictions import build_models  # noqa: E402

RES = ROOT / "revision" / "results"
FIG = RES / "figures"
MODES = ("bootstrap", "seeds")
N_TRAIN = 766
N_FEAT = len(FEATURES)
TOP_K = 5
FOCUS = "aw_hr_mean_lag1"
F1_CANON = 0.5063
SHAP_PROCEDURE = {
    "explainer": "shap.TreeExplainer(rf), defaults (feature_perturbation tree_path_dependent, "
                 "model_output raw = class probabilities)",
    "data": "imputed test set, 192 days, unscaled (imputer fit on the run's train part)",
    "aggregation": "np.abs(shap_values).mean(axis=(0, 2)) on the (192, 19, 3) array: mean |SHAP| "
                   "over samples and the 3 classes (identical to run_shap() in "
                   "cap5_compare_algorithms_json_ext.py, the script behind Figure 2)",
    "rank": "1 = largest mean |SHAP|; ties -> smallest rank (scipy rankdata method min); "
            "top-5 = 5 largest values, stable tie-break by feature order",
}
# Jaccard of two 5-sets with k common elements = k / (10 - k)
JACCARD_LEVELS = {k: k / (2 * TOP_K - k) for k in range(TOP_K + 1)}


# ------------------------------------------------------------------ pure helpers (unit-tested)
def short_name(feature: str) -> str:
    """aw_hr_mean_lag1 -> aw_hr_mean; hrv_balance_lag1 -> oura_hrv_balance (device prefix as in Figure 2)."""
    base = feature.replace("_lag1", "")
    if base == "steps_oura":          # avoid oura_steps_oura
        return "oura_steps"
    return base if base.startswith("aw_") else f"oura_{base}"


def aggregate_multiclass(shap_values, n_samples: int) -> np.ndarray:
    """Canonical aggregation: mean of |SHAP| over samples and classes -> (n_features,).

    Accepts the (n_samples, n_features, n_classes) array of shap >= 0.45 or a
    list of n_classes arrays (n_samples, n_features) from older versions, exactly
    as the canonical run_shap() does via np.array() + ndim/shape dispatch.
    """
    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-d SHAP array or list of 2-d arrays, got ndim {arr.ndim}")
    if arr.shape[0] == n_samples:          # (samples, features, classes)
        return np.abs(arr).mean(axis=(0, 2))
    return np.abs(arr).mean(axis=(0, 1))   # (classes, samples, features)


def mean_abs_shap(model, X: pd.DataFrame) -> np.ndarray:
    return aggregate_multiclass(shap.TreeExplainer(model).shap_values(X), n_samples=len(X))


def rank_desc(values) -> np.ndarray:
    """Rank 1 = largest value; ties share the smallest rank (deterministic)."""
    return rankdata(-np.asarray(values, dtype=float), method="min").astype(int)


def top_k(values, k: int = TOP_K) -> set[int]:
    """Indices of the k largest values; ties broken by feature order (stable)."""
    return set(np.argsort(-np.asarray(values, dtype=float), kind="stable")[:k].tolist())


def jaccard(a: set, b: set) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def kendall(ranks_a, ranks_b) -> float:
    return float(kendalltau(np.asarray(ranks_a), np.asarray(ranks_b)).statistic)


def percentile_ci(values, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    p = np.percentile(np.asarray(values, dtype=float), [lo, hi])
    return float(p[0]), float(p[1])


# ------------------------------------------------------------------ one retrain
def fit_rf(split: Split, rf_seed: int):
    rf = build_models()["rf"]
    rf.set_params(random_state=rf_seed)
    rf.fit(split.X_train_sm, split.y_train_sm)
    return rf


def run_one(split: Split, rf_seed: int) -> dict:
    rf = fit_rf(split, rf_seed)
    y_pred = rf.predict(split.X_test)
    y_true = split.y_test.to_numpy()
    f1_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return dict(mean_abs=mean_abs_shap(rf, split.X_test),
                f1_macro=float(f1_per.mean()), f1_active=float(f1_per[2]),
                n_train_active=int((split.y_train == 2).sum()),
                n_train_rest=int((split.y_train == 0).sum()))


def bootstrap_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series, idx: np.ndarray) -> Split:
    """Bootstrap the raw 766 train rows (duplicates allowed), test unchanged; preprocessing refit."""
    X_tr, y_tr = X.iloc[:N_TRAIN].iloc[idx], y.iloc[:N_TRAIN].iloc[idx]
    return preprocess_split(X_tr, X.iloc[N_TRAIN:], y_tr, y.iloc[N_TRAIN:], dates.iloc[N_TRAIN:],
                            k_neighbors=3, seed=SEED)


# ------------------------------------------------------------------ summary
def summarize(mean_abs_runs: np.ndarray, ref_mean: np.ndarray, f1_macro: np.ndarray,
              f1_active: np.ndarray) -> tuple[pd.DataFrame, dict]:
    n_runs = mean_abs_runs.shape[0]
    ref_rank = rank_desc(ref_mean)
    ref_top = top_k(ref_mean)
    ranks = np.vstack([rank_desc(v) for v in mean_abs_runs])
    tops = [top_k(v) for v in mean_abs_runs]
    jac = np.array([jaccard(t, ref_top) for t in tops])
    overlap = np.array([len(t & ref_top) for t in tops])
    tau_ref = np.array([kendall(r, ref_rank) for r in ranks])
    tau_pair = np.array([kendall(ranks[i], ranks[j]) for i, j in combinations(range(n_runs), 2)])
    focus = FEATURES.index(FOCUS)

    rows: list[tuple] = []

    def add(block, metric, value, feature=""):
        rows.append((block, metric, feature, float(value)))

    add("rank1_frequency", "aw_hr_mean_rank1_count", (ranks[:, focus] == 1).sum())
    add("rank1_frequency", "aw_hr_mean_rank1_fraction", (ranks[:, focus] == 1).mean())
    add("rank1_frequency", "aw_hr_mean_top3_count", (ranks[:, focus] <= 3).sum())
    add("rank1_frequency", "aw_hr_mean_top3_fraction", (ranks[:, focus] <= 3).mean())
    for m, v in (("mean", jac.mean()), ("median", np.median(jac)), ("min", jac.min()), ("max", jac.max())):
        add("jaccard_top5_vs_reference", m, v)
    for k, level in JACCARD_LEVELS.items():
        add("jaccard_top5_vs_reference", f"count_overlap{k}_jaccard{level:.4f}", (overlap == k).sum())
    lo, hi = percentile_ci(tau_ref)
    for m, v in (("mean", tau_ref.mean()), ("ci_low_p2.5", lo), ("ci_high_p97.5", hi),
                 ("median", np.median(tau_ref)), ("min", tau_ref.min()), ("max", tau_ref.max())):
        add("kendall_tau_vs_reference", m, v)
    lo, hi = percentile_ci(tau_pair)
    for m, v in (("mean", tau_pair.mean()), ("ci_low_p2.5", lo), ("ci_high_p97.5", hi),
                 ("median", np.median(tau_pair)), ("min", tau_pair.min()), ("max", tau_pair.max()),
                 ("n_pairs", len(tau_pair))):
        add("kendall_tau_pairwise", m, v)
    for j, f in enumerate(FEATURES):
        r = ranks[:, j]
        q25, q75 = np.percentile(r, [25, 75])
        for m, v in (("reference_rank", ref_rank[j]), ("median_rank", np.median(r)), ("q25_rank", q25),
                     ("q75_rank", q75), ("iqr_rank", q75 - q25), ("min_rank", r.min()), ("max_rank", r.max()),
                     ("top5_frequency", np.mean([j in t for t in tops])), ("rank1_frequency", (r == 1).mean()),
                     ("mean_abs_shap_reference", ref_mean[j]), ("mean_abs_shap_mean_runs", mean_abs_runs[:, j].mean())):
            add("per_feature", m, v, f)
    for name, arr in (("f1_macro", f1_macro), ("f1_active", f1_active)):
        lo, hi = percentile_ci(arr)
        for m, v in (("mean", arr.mean()), ("p2.5", lo), ("p97.5", hi), ("min", arr.min()), ("max", arr.max())):
            add("f1_test_informative_noncanonical", f"{name}_{m}", v)
    add("f1_test_informative_noncanonical", "f1_macro_canonical_reference", F1_CANON)

    table = pd.DataFrame(rows, columns=["block", "metric", "feature", "value"])
    extra = dict(ranks=ranks, jaccard=jac, tau_ref=tau_ref, tau_pair=tau_pair, ref_rank=ref_rank)
    return table, extra


def markdown_table(table: pd.DataFrame, mode: str, n_runs: int) -> str:
    g = {(b, m, f): v for b, m, f, v in table.itertuples(index=False)}
    lines = [f"# T1.4 SHAP ranking stability, mode `{mode}` ({n_runs} runs, seed 42)", ""]
    lines += ["Reference = canonical RF (rf_revision.joblib), mean |SHAP| on the 192 test days.", ""]
    lines += ["| Metric | Value |", "|---|---|"]
    lines.append(f"| aw_hr_mean rank 1 | {int(g[('rank1_frequency', 'aw_hr_mean_rank1_count', '')])} / {n_runs} "
                 f"({g[('rank1_frequency', 'aw_hr_mean_rank1_fraction', '')]:.2f}) |")
    lines.append(f"| aw_hr_mean top-3 | {int(g[('rank1_frequency', 'aw_hr_mean_top3_count', '')])} / {n_runs} "
                 f"({g[('rank1_frequency', 'aw_hr_mean_top3_fraction', '')]:.2f}) |")
    j = lambda m: g[("jaccard_top5_vs_reference", m, "")]  # noqa: E731
    lines.append(f"| Jaccard top-5 vs reference: mean / median / min / max | "
                 f"{j('mean'):.4f} / {j('median'):.4f} / {j('min'):.4f} / {j('max'):.4f} |")
    dist = ", ".join(f"{level:.2f}: {int(g[('jaccard_top5_vs_reference', f'count_overlap{k}_jaccard{level:.4f}', '')])}"
                     for k, level in sorted(JACCARD_LEVELS.items(), reverse=True))
    lines.append(f"| Jaccard distribution (value: n runs) | {dist} |")
    t = lambda m: g[("kendall_tau_vs_reference", m, "")]  # noqa: E731
    lines.append(f"| Kendall tau vs reference: mean [p2.5, p97.5], min / max | "
                 f"{t('mean'):.4f} [{t('ci_low_p2.5'):.4f}, {t('ci_high_p97.5'):.4f}], {t('min'):.4f} / {t('max'):.4f} |")
    p = lambda m: g[("kendall_tau_pairwise", m, "")]  # noqa: E731
    lines.append(f"| Kendall tau between runs ({int(p('n_pairs'))} pairs): mean [p2.5, p97.5], min / max | "
                 f"{p('mean'):.4f} [{p('ci_low_p2.5'):.4f}, {p('ci_high_p97.5'):.4f}], {p('min'):.4f} / {p('max'):.4f} |")
    f = lambda m: g[("f1_test_informative_noncanonical", m, "")]  # noqa: E731
    lines.append(f"| F1 macro on test, retrained models (informative, NON-canonical): mean [p2.5, p97.5], min / max | "
                 f"{f('f1_macro_mean'):.4f} [{f('f1_macro_p2.5'):.4f}, {f('f1_macro_p97.5'):.4f}], "
                 f"{f('f1_macro_min'):.4f} / {f('f1_macro_max'):.4f} (canonical {F1_CANON}) |")
    lines.append(f"| F1 Active on test, retrained models (informative, NON-canonical): mean [p2.5, p97.5], min / max | "
                 f"{f('f1_active_mean'):.4f} [{f('f1_active_p2.5'):.4f}, {f('f1_active_p97.5'):.4f}], "
                 f"{f('f1_active_min'):.4f} / {f('f1_active_max'):.4f} |")
    lines += ["", "## Per feature (ordered by reference rank)", "",
              "| Ref. rank | Feature | Ref. mean abs SHAP | Median rank | IQR (q25-q75) | Min-max rank | Top-5 freq. | Rank-1 freq. |",
              "|---|---|---|---|---|---|---|---|"]
    pf = table[table.block == "per_feature"].pivot(index="feature", columns="metric", values="value")
    pf = pf.loc[sorted(FEATURES, key=lambda x: pf.loc[x, "reference_rank"])]
    for feat, r in pf.iterrows():
        lines.append(f"| {int(r.reference_rank)} | {short_name(feat)} | {r.mean_abs_shap_reference:.4f} | "
                     f"{r.median_rank:g} | {r.iqr_rank:g} ({r.q25_rank:g}-{r.q75_rank:g}) | "
                     f"{int(r.min_rank)}-{int(r.max_rank)} | {r.top5_frequency:.2f} | {r.rank1_frequency:.2f} |")
    return "\n".join(lines) + "\n"


def make_figure(ranks: np.ndarray, ref_rank: np.ndarray, mode: str, n_runs: int) -> int:
    order = np.argsort(ref_rank, kind="stable")          # reference rank 1 first
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8.5, 7))
    pos = np.arange(len(order))[::-1] + 1                 # rank-1 feature at the top
    ax.boxplot([ranks[:, j] for j in order], positions=pos, orientation="horizontal", widths=0.6,
               patch_artist=True, showfliers=True,
               boxprops=dict(facecolor="#BBDEFB", edgecolor="#1565C0", linewidth=1.0),
               medianprops=dict(color="#0D47A1", linewidth=1.8),
               whiskerprops=dict(color="#1565C0", linewidth=1.0),
               capprops=dict(color="#1565C0", linewidth=1.0),
               flierprops=dict(marker="o", markersize=3, markerfacecolor="#1565C0",
                               markeredgecolor="none", alpha=0.6))
    ax.scatter(ref_rank[order], pos, marker="D", s=42, color="#E65100", zorder=5,
               label="reference rank (canonical RF, rf_revision.joblib)")
    ax.set_yticks(pos)
    ax.set_yticklabels([short_name(FEATURES[j]) for j in order], fontsize=9)
    ax.set_xticks(np.arange(1, N_FEAT + 1))
    ax.set_xlim(0.5, N_FEAT + 0.5)
    ax.set_ylim(0.3, len(order) + 0.7)
    ax.set_xlabel("SHAP rank across runs (1 = largest mean |SHAP| on the test set)", fontsize=10)
    what = ("100 RF retrains on bootstrap samples of the 766 train days (RF seed 42)" if mode == "bootstrap"
            else "100 RF retrains on the full train with random_state 0-99")
    ax.set_title(f"SHAP rank stability, {what}\nboxes: median and IQR over {n_runs} runs; "
                 f"features ordered by reference rank", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=9, loc="upper right")   # top-right is empty: rank-1/2 features sit at the left
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"t14_shap_rank_stability_{mode}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    fig.savefig(FIG / f"t14_shap_rank_stability_{mode}.png", dpi=300, bbox_inches="tight",
                metadata={"Software": None})
    n_axes = len(fig.axes)
    plt.close(fig)
    return n_axes


# ------------------------------------------------------------------ main
def main(mode: str = "bootstrap", n_runs: int = 100) -> None:
    if mode not in MODES:
        sys.exit(f"unknown mode {mode}")
    t0 = time.time()
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    X, y, dates = build_dataset(load_raw())
    canon = canonical_split(X, y, dates)
    check_canonical_sizes(X, y, canon)
    rf_ref = joblib.load(RES / "models" / "rf_revision.joblib")
    ref_mean = mean_abs_shap(rf_ref, canon.X_test)
    ref_rank = rank_desc(ref_mean)
    check(FEATURES[int(np.argmin(ref_rank))] == FOCUS, f"reference rank 1 is {FOCUS}")
    print("reference top-5:", [short_name(FEATURES[i]) for i in np.argsort(ref_rank, kind="stable")[:5]])

    # identity test: original train (indices 0..765) / canonical seed 42 reproduces the reference
    ident = run_one(bootstrap_split(X, y, dates, np.arange(N_TRAIN)), SEED) if mode == "bootstrap" \
        else run_one(canon, SEED)
    ident_diff = float(np.abs(ident["mean_abs"] - ref_mean).max())
    check(np.array_equal(rank_desc(ident["mean_abs"]), ref_rank), "identity test: ranking == reference")
    check(ident_diff < 1e-12, f"identity test: max |mean abs SHAP - reference| = {ident_diff:.1e} < 1e-12")
    check(round(ident["f1_macro"], 4) == F1_CANON, f"identity test: F1 macro {ident['f1_macro']:.4f} == {F1_CANON}")

    meta = dict(task="T1.4", mode=mode, n_runs=n_runs, seed=SEED, run_date=time.strftime("%Y-%m-%d"),
                versions=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                              numpy=np.__version__, pandas=pd.__version__, shap=shap.__version__,
                              imblearn=imblearn.__version__, scipy=scipy.__version__,
                              matplotlib=matplotlib.__version__),
                shap_procedure=SHAP_PROCEDURE,
                rf_hyperparameters=dict(n_estimators=100, max_depth=10, min_samples_leaf=5),
                reference_model="revision/results/models/rf_revision.joblib",
                reference_ranking=[short_name(FEATURES[i]) for i in np.argsort(ref_rank, kind="stable")],
                identity_test=dict(ranking_equal=bool(np.array_equal(rank_desc(ident["mean_abs"]), ref_rank)),
                                   max_abs_diff_mean_abs_shap=ident_diff, f1_macro=round(ident["f1_macro"], 6),
                                   f1_active=round(ident["f1_active"], 6)))

    # the runs
    mean_abs_runs, f1m, f1a, n_act, n_rest = [], [], [], [], []
    if mode == "bootstrap":
        idx = np.random.default_rng(SEED).integers(0, N_TRAIN, size=(n_runs, N_TRAIN))
        npy = RES / f"bootstrap_train_indices_{n_runs}x{N_TRAIN}_seed{SEED}.npy"
        np.save(npy, idx)
        meta["bootstrap_indices"] = dict(path=str(npy.relative_to(ROOT)).replace("\\", "/"), shape=list(idx.shape),
                                         dtype=str(idx.dtype), rng="np.random.default_rng(42).integers(0, 766, size=(100, 766))",
                                         sha256=hashlib.sha256(npy.read_bytes()).hexdigest(),
                                         mean_unique_rows=float(np.mean([len(np.unique(r)) for r in idx])))
        meta["what_varies"] = "train rows (bootstrap, before imputation / SMOTE); SMOTE seed 42 and RF seed 42 fixed"
        for r in range(n_runs):
            out = run_one(bootstrap_split(X, y, dates, idx[r]), SEED)
            mean_abs_runs.append(out["mean_abs"]); f1m.append(out["f1_macro"]); f1a.append(out["f1_active"])
            n_act.append(out["n_train_active"]); n_rest.append(out["n_train_rest"])
            if (r + 1) % 10 == 0:
                print(f"  run {r + 1}/{n_runs}  {time.time() - t0:.0f}s  f1 {out['f1_macro']:.4f}  "
                      f"rank1 {short_name(FEATURES[int(np.argmax(out['mean_abs']))])}")
        meta["train_active_per_run"] = dict(min=int(min(n_act)), mean=float(np.mean(n_act)), max=int(max(n_act)))
        meta["train_rest_per_run"] = dict(min=int(min(n_rest)), mean=float(np.mean(n_rest)), max=int(max(n_rest)))
    else:
        meta["what_varies"] = "RF random_state 0..n_runs-1; data and preprocessing (SMOTE seed 42) fixed"
        for r in range(n_runs):
            out = run_one(canon, r)
            mean_abs_runs.append(out["mean_abs"]); f1m.append(out["f1_macro"]); f1a.append(out["f1_active"])
            if (r + 1) % 10 == 0:
                print(f"  run {r + 1}/{n_runs}  {time.time() - t0:.0f}s  f1 {out['f1_macro']:.4f}  "
                      f"rank1 {short_name(FEATURES[int(np.argmax(out['mean_abs']))])}")
        if n_runs > SEED:
            check(np.array_equal(rank_desc(mean_abs_runs[SEED]), ref_rank)
                  and float(np.abs(mean_abs_runs[SEED] - ref_mean).max()) < 1e-12,
                  "seeds mode: run 42 (canonical seed) == reference")
    mean_abs_runs = np.vstack(mean_abs_runs)
    f1m, f1a = np.array(f1m), np.array(f1a)

    table, extra = summarize(mean_abs_runs, ref_mean, f1m, f1a)
    ranks = extra["ranks"]

    # sanity checks
    check(mean_abs_runs.shape == (n_runs, N_FEAT), f"{n_runs} runs x {N_FEAT} features")
    check(all(sorted(r.tolist()) == list(range(1, N_FEAT + 1)) for r in ranks),
          "every run: ranks are a permutation of 1..19 (no ties)")
    check(bool((mean_abs_runs >= 0).all()), "mean |SHAP| >= 0 in every run")
    check(bool(((extra["jaccard"] >= 0) & (extra["jaccard"] <= 1)).all()), "Jaccard in [0, 1]")
    check(bool(((extra["tau_ref"] >= -1) & (extra["tau_ref"] <= 1)).all())
          and bool(((extra["tau_pair"] >= -1) & (extra["tau_pair"] <= 1)).all()), "Kendall tau in [-1, 1]")
    check(len(extra["tau_pair"]) == n_runs * (n_runs - 1) // 2, f"{len(extra['tau_pair'])} run pairs")

    # outputs
    RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": FEATURES, "mean_abs_shap": ref_mean, "rank": ref_rank}).to_csv(
        RES / "t14_shap_reference.csv", index=False, float_format="%.10g", lineterminator="\n")
    long = pd.DataFrame({"run": np.repeat(np.arange(n_runs), N_FEAT), "feature": FEATURES * n_runs,
                         "mean_abs_shap": mean_abs_runs.ravel(), "rank": ranks.ravel()})
    long.to_csv(RES / f"t14_shap_values_{mode}.csv", index=False, float_format="%.10g", lineterminator="\n")
    table.to_csv(RES / f"t14_shap_stability_{mode}.csv", index=False, float_format="%.10g", lineterminator="\n")
    (RES / f"t14_shap_stability_{mode}_table.md").write_text(markdown_table(table, mode, n_runs),
                                                             encoding="utf-8", newline="\n")
    meta["figure_axes"] = make_figure(ranks, ref_rank, mode, n_runs)
    meta["sanity_checks_failed"] = list(fails)
    meta["elapsed_seconds"] = round(time.time() - t0, 1)
    (RES / f"t14_run_meta_{mode}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")

    g = table.set_index(["block", "metric", "feature"]).value
    print(f"\n[{mode}] aw_hr_mean rank 1: {int(g[('rank1_frequency', 'aw_hr_mean_rank1_count', '')])}/{n_runs}; "
          f"Jaccard top-5 mean {g[('jaccard_top5_vs_reference', 'mean', '')]:.4f} "
          f"[{g[('jaccard_top5_vs_reference', 'min', '')]:.2f}, {g[('jaccard_top5_vs_reference', 'max', '')]:.2f}]; "
          f"tau vs ref {g[('kendall_tau_vs_reference', 'mean', '')]:.4f} "
          f"[{g[('kendall_tau_vs_reference', 'ci_low_p2.5', '')]:.4f}, {g[('kendall_tau_vs_reference', 'ci_high_p97.5', '')]:.4f}]; "
          f"tau pairwise {g[('kendall_tau_pairwise', 'mean', '')]:.4f} "
          f"[{g[('kendall_tau_pairwise', 'ci_low_p2.5', '')]:.4f}, {g[('kendall_tau_pairwise', 'ci_high_p97.5', '')]:.4f}]; "
          f"F1 macro {f1m.mean():.4f} [{np.percentile(f1m, 2.5):.4f}, {np.percentile(f1m, 97.5):.4f}]; "
          f"F1 Active {f1a.mean():.4f} [{np.percentile(f1a, 2.5):.4f}, {np.percentile(f1a, 97.5):.4f}]; "
          f"{meta['elapsed_seconds']} s")
    print("ALL PASS" if not fails else "FAIL: " + "; ".join(fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, default="bootstrap")
    ap.add_argument("--n-runs", type=int, default=100)
    a = ap.parse_args()
    main(a.mode, a.n_runs)
