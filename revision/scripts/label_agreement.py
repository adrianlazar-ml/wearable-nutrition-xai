"""
label_agreement.py - T3.2 (Reviewer 1, R1.5): agreement between the canonical T3 labels
(Rest / Normal / Active) and indicators that do NOT enter the labelling rule, on all 958
labelled days. T3.2 validates the DEFINITION of the labels, not the model; nothing
canonical is modified.

Labels: canonical (R 60, A_r 85, A_s 75) on the 958 labelled days (202 / 710 / 46), from
common.build_dataset(). Indicators are the DAILY values of day t (not lag-1), read from the
same daily frame (common.load_raw()); identity checked at run time: value(t).shift(1) ==
canonical lag-1 feature on every row where both exist (abort otherwise).

Two levels (fixed before any result was seen):
  A  device-independent (Apple Watch): aw_hr_mean, aw_hrv_mean, aw_active_calories,
     aw_steps, aw_resp_rate, aw_spo2
  B  raw Oura signals that do not enter the rule but feed the readiness score (internal
     consistency only, not independent validation): hrv_mean, rhr_lowest, temp_deviation,
     steps_oura
  Excluded: readiness_score, sleep_score (define the label), hrv_balance, recovery_index,
  activity_score (derived Oura scores), sleep architecture (Watch; no directional expectation).
  Lag-1 extension, activity only: aw_active_calories, aw_steps, steps_oura at day t-1 vs
  the class of day t (test of the T1.5 hypothesis "activity precedes Active").
Expected direction Active vs Rest, fixed BEFORE running (in the table and in run_meta):
  aw_hrv_mean +, hrv_mean +, aw_hr_mean -, rhr_lowest -, temp_deviation -,
  aw_active_calories +, aw_steps +, steps_oura +, aw_resp_rate 0, aw_spo2 0; lag-1 activity +.
  Result per variable: concordant / discordant (sign of Cliff's delta Active minus Rest) or
  "no expectation".

Statistics per variable (days with a missing value are dropped per variable, no imputation;
n per class is reported):
  - descriptives per class: n, mean, median, p25, p75 (IQR = p75 - p25)
  - Kruskal-Wallis H and p over the 3 classes (scipy.stats.kruskal, tie-corrected)
  - epsilon-squared effect size: eps2 = H / (n - 1)  (Tomczak and Tomczak 2014, equal to
    H / ((n^2 - 1) / (n + 1))), in [0, 1]
  - post-hoc Dunn on the 3 pairs with the tie-corrected z statistic
    z = (Rbar_i - Rbar_j) / sqrt((N (N + 1) / 12 - sum(t^3 - t) / (12 (N - 1))) (1/n_i + 1/n_j)),
    two-sided normal p, Holm over the 3 pairs (direct implementation; scikit-posthocs is not
    installed and no new dependency is added)
  - Cliff's delta per pair, delta(X, Y) = P(x > y) - P(x < y) in [-1, 1], with a percentile
    95 percent CI from 1000 bootstrap resamples of the days within each class (own
    np.random.default_rng(42), NOT the 1000 x 192 test matrix: n differs here)
Caveat (also in the md): consecutive days are autocorrelated, so the Kruskal-Wallis and Dunn
p-values treat dependent days as independent and are approximate (optimistic); inference
rests on the effect sizes (epsilon-squared, Cliff's delta with CI).

Outputs (revision/results/, LF):
  t32_inventory.csv, t32_descriptives.csv + _table.md, t32_tests.csv + _table.md,
  t32_run_meta.json, figures/t32_indicator_boxplots.{png,pdf} (level A, 2 x 3),
  figures/t32_indicator_boxplots_oura.{png,pdf} (level B + lag-1 activity, 2 x 4)

Run from repo root:  python revision/scripts/label_agreement.py
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

import matplotlib
import numpy as np
import pandas as pd
import scipy
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SEED, build_dataset, load_raw  # noqa: E402
from statistical_tests import LABELS, md_table, write_text  # noqa: E402

RES = ROOT / "revision" / "results"
FIG = RES / "figures"

N_BOOT = 1000
CLASS_NAMES = {0: "Rest", 1: "Normal", 2: "Active"}
PAIRS = [(2, 0), (2, 1), (1, 0)]            # Active-Rest, Active-Normal, Normal-Rest (first minus second)
PAIR_NAMES = {(2, 0): "active_rest", (2, 1): "active_normal", (1, 0): "normal_rest"}
CANON_DIST = (202, 710, 46)

# variable -> (level, source column / file, unit, expected sign Active vs Rest: "+", "-", "0")
VARIABLES: dict[str, dict] = {
    "aw_hr_mean":         dict(level="A", source="apple_health_daily.csv: aw_hr_mean", unit="bpm", expected="-"),
    "aw_hrv_mean":        dict(level="A", source="apple_health_daily.csv: aw_hrv_mean", unit="ms", expected="+"),
    "aw_active_calories": dict(level="A", source="apple_health_daily.csv: aw_active_energy_kj / 4.184", unit="kcal",
                               expected="+"),
    "aw_steps":           dict(level="A", source="apple_health_daily.csv: aw_steps", unit="steps", expected="+"),
    "aw_resp_rate":       dict(level="A", source="apple_health_daily.csv: aw_resp_rate", unit="breaths/min",
                               expected="0"),
    "aw_spo2":            dict(level="A", source="apple_health_daily.csv: aw_spo2", unit="%", expected="0"),
    "hrv_mean":           dict(level="B", source="oura_combined.csv: hrv_mean", unit="ms", expected="+"),
    "rhr_lowest":         dict(level="B", source="oura_combined.csv: rhr_lowest", unit="bpm", expected="-"),
    "temp_deviation":     dict(level="B", source="oura_combined.csv: temp_deviation", unit="deg C", expected="-"),
    "steps_oura":         dict(level="B", source="oura_combined.csv: steps_oura", unit="steps", expected="+"),
}
LAG_VARIABLES = {                              # day t-1 value vs class of day t (T1.5 hypothesis)
    "aw_active_calories_lag1": dict(level="A_lag1", source="shift(1) of aw_active_calories (== canonical lag-1 feature)",
                                    unit="kcal", expected="+"),
    "aw_steps_lag1":           dict(level="A_lag1", source="shift(1) of aw_steps (== canonical lag-1 feature)",
                                    unit="steps", expected="+"),
    "steps_oura_lag1":         dict(level="B_lag1", source="shift(1) of steps_oura (== canonical lag-1 feature)",
                                    unit="steps", expected="+"),
}
ALL_VARIABLES = {**VARIABLES, **LAG_VARIABLES}
SHORT = {"aw_hr_mean": "aw_hr_mean", "aw_hrv_mean": "aw_hrv_mean", "aw_active_calories": "aw_active_calories",
         "aw_steps": "aw_steps", "aw_resp_rate": "aw_resp_rate", "aw_spo2": "aw_spo2", "hrv_mean": "oura_hrv_mean",
         "rhr_lowest": "oura_rhr_lowest", "temp_deviation": "oura_temp_deviation", "steps_oura": "oura_steps",
         "aw_active_calories_lag1": "aw_active_calories (t-1)", "aw_steps_lag1": "aw_steps (t-1)",
         "steps_oura_lag1": "oura_steps (t-1)"}


# ------------------------------------------------------------------ pure helpers (unit-tested)
def cliffs_delta(x, y) -> float:
    """delta(X, Y) = P(x > y) - P(x < y), in [-1, 1]; +1 = every x above every y."""
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    if len(x) == 0 or len(y) == 0:
        raise ValueError("empty sample")
    gt = np.searchsorted(y, x, side="left").sum()      # number of y strictly below each x
    lt = (len(y) - np.searchsorted(y, x, side="right")).sum()   # number of y strictly above each x
    return float((gt - lt) / (len(x) * len(y)))


def cliffs_delta_ci(x, y, rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile 95 percent CI: resample the days within each group independently."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = cliffs_delta(x[rng.integers(0, len(x), len(x))], y[rng.integers(0, len(y), len(y))])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def epsilon_squared(h: float, n: int) -> float:
    """Kruskal-Wallis effect size eps2 = H / (n - 1) = H / ((n^2 - 1) / (n + 1))."""
    if n < 2:
        raise ValueError("n must be >= 2")
    return float(h / (n - 1))


def dunn_test(groups: list[np.ndarray]) -> dict[tuple[int, int], dict[str, float]]:
    """Dunn's post-hoc z tests on all pairs, tie-corrected, two-sided normal p; keys (i, j) = group indices."""
    sizes = [len(g) for g in groups]
    pooled = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    n = len(pooled)
    ranks = stats.rankdata(pooled)
    _, counts = np.unique(pooled, return_counts=True)
    tie = float(np.sum(counts.astype(float) ** 3 - counts)) / (12.0 * (n - 1))
    var_base = n * (n + 1) / 12.0 - tie
    means = []
    start = 0
    for s in sizes:
        means.append(float(ranks[start:start + s].mean()))
        start += s
    out: dict[tuple[int, int], dict[str, float]] = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            se = np.sqrt(var_base * (1.0 / sizes[i] + 1.0 / sizes[j]))
            z = (means[i] - means[j]) / se
            p = float(2.0 * stats.norm.sf(abs(z)))
            out[(i, j)] = dict(z=float(z), p=p, mean_rank_i=means[i], mean_rank_j=means[j])
    return out


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for k, idx in enumerate(order):
        val = min(1.0, (m - k) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def concordance(delta_active_rest: float, expected: str) -> str:
    """concordant / discordant from the sign of Cliff's delta Active minus Rest; 'no expectation' for '0'."""
    if expected == "0":
        return "no expectation"
    if delta_active_rest == 0.0:
        return "discordant"      # no difference where one was expected
    return "concordant" if (delta_active_rest > 0) == (expected == "+") else "discordant"


def lag_identity(daily: pd.Series, lag_feature: pd.Series) -> bool:
    """value(t).shift(1) == lag-1 feature on every row where both exist (and same missingness)."""
    sh = daily.shift(1).to_numpy(dtype=float)
    lg = lag_feature.to_numpy(dtype=float)
    both = ~np.isnan(sh) & ~np.isnan(lg)
    return bool(np.array_equal(sh[both], lg[both]) and np.array_equal(np.isnan(sh), np.isnan(lg)))


# ------------------------------------------------------------------ data
def build_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """958 labelled days with the day-t values of every variable (and the lag-1 activity extension).
    Returns (frame indexed 0..957 with columns date, label, variables; identity flags)."""
    X, y, dates = build_dataset(df)
    daily = df.set_index("date")
    frame = pd.DataFrame({"date": dates.to_numpy(), "label": y.to_numpy(dtype=int)})
    ident = {}
    for v in VARIABLES:
        frame[v] = daily.loc[dates, v].to_numpy(dtype=float)
    for lv in LAG_VARIABLES:
        base = lv.replace("_lag1", "")
        frame[lv] = X[f"{base}_lag1"].to_numpy(dtype=float)           # canonical lag-1 feature of day t
    # identity on the full daily frame: shift(1) of the day-t value == canonical lag-1 feature
    lag_all = pd.DataFrame({f"{v}_lag1": df[v].shift(1) for v in VARIABLES})
    sel = df["date"].isin(dates).to_numpy()
    for v in VARIABLES:
        # Audit F4 (2026-10-25) observation, kept as is: the first clause compares df[v].shift(1) with lag_all,
        # which IS df[v].shift(1), so it is always True (tautological); the second clause, array_equal against the
        # canonical lag-1 feature X[f"{v}_lag1"] on the 958 labeled dates, is the identity actually tested.
        ident[v] = lag_identity(df[v], lag_all[f"{v}_lag1"]) and bool(
            np.array_equal(np.nan_to_num(lag_all[f"{v}_lag1"].to_numpy(dtype=float)[sel], nan=-1e30),
                           np.nan_to_num(X[f"{v}_lag1"].to_numpy(dtype=float), nan=-1e30)))
    return frame, pd.Series(ident)


# ------------------------------------------------------------------ figure
def boxplot_figure(frame: pd.DataFrame, variables: list[str], shape: tuple[int, int], path_stem: Path) -> int:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    colors = {0: "#E65100", 1: "#9E9E9E", 2: "#1565C0"}       # Rest orange, Normal grey, Active blue (t14 / t15 palette)
    nr, nc = shape
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 3.0 * nr))
    axes = np.atleast_1d(axes).ravel()
    for ax, v in zip(axes, variables):
        data = [frame.loc[frame.label == c, v].dropna().to_numpy() for c in LABELS]
        bp = ax.boxplot(data, orientation="vertical", widths=0.55, patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, markeredgecolor="#777777", alpha=0.6),
                        medianprops=dict(color="#222222", lw=1.4))
        for patch, c in zip(bp["boxes"], LABELS):
            patch.set_facecolor(colors[c])
            patch.set_alpha(0.55)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f"{CLASS_NAMES[c]}\n(n = {len(d)})" for c, d in zip(LABELS, data)], fontsize=8)
        ax.set_ylabel(ALL_VARIABLES[v]["unit"], fontsize=8.5)
        ax.set_title(SHORT[v], fontsize=9.5, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes[len(variables):]:
        ax.set_visible(False)
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

    # ---- T3.2.0 inventory
    print("=== T3.2.0 inventory ===")
    df = load_raw()
    frame, ident = build_frame(df)
    dist = tuple(int((frame.label == c).sum()) for c in LABELS)
    check(len(frame) == 958 and dist == CANON_DIST, f"958 labelled days, canonical distribution {dist}", abort=True)
    check(bool(ident.all()), f"identity value(t).shift(1) == canonical lag-1 feature for all {len(ident)} variables",
          abort=True)
    inv_rows = []
    for v, meta in ALL_VARIABLES.items():
        n_per = [int(frame.loc[frame.label == c, v].notna().sum()) for c in LABELS]
        inv_rows.append(dict(variable=v, display=SHORT[v], level=meta["level"], source=meta["source"],
                             unit=meta["unit"], expected_active_vs_rest=meta["expected"],
                             n_valid=sum(n_per), n_rest=n_per[0], n_normal=n_per[1], n_active=n_per[2],
                             n_missing=958 - sum(n_per)))
        print(f"  {v:26s} level {meta['level']:6s} n {n_per} missing {958 - sum(n_per)}")
    inventory = pd.DataFrame(inv_rows)
    inventory.to_csv(res / "t32_inventory.csv", index=False, lineterminator="\n")

    # ---- T3.2.1 descriptives
    print("\n=== T3.2.1 descriptives ===")
    desc_rows = []
    for v in ALL_VARIABLES:
        for c in LABELS:
            s = frame.loc[frame.label == c, v].dropna()
            desc_rows.append(dict(variable=v, display=SHORT[v], level=ALL_VARIABLES[v]["level"],
                                  unit=ALL_VARIABLES[v]["unit"], cls=CLASS_NAMES[c], n=int(len(s)),
                                  mean=float(s.mean()), median=float(s.median()), p25=float(s.quantile(0.25)),
                                  p75=float(s.quantile(0.75)), iqr=float(s.quantile(0.75) - s.quantile(0.25))))
    desc = pd.DataFrame(desc_rows)
    check(bool((desc.groupby("variable")["n"].sum().reindex(ALL_VARIABLES).to_numpy()
                == inventory.set_index("variable").loc[list(ALL_VARIABLES), "n_valid"].to_numpy()).all()),
          "descriptives n per class == inventory")

    # ---- T3.2.2 tests
    print("\n=== T3.2.2 Kruskal-Wallis, Dunn-Holm, Cliff's delta ===")
    rng = np.random.default_rng(SEED)
    test_rows = []
    for v, meta in ALL_VARIABLES.items():
        groups = [frame.loc[frame.label == c, v].dropna().to_numpy(dtype=float) for c in LABELS]
        n = sum(len(g) for g in groups)
        h, p = stats.kruskal(*groups)
        eps2 = epsilon_squared(float(h), n)
        dunn = dunn_test(groups)
        pair_p = [dunn[(min(a, b), max(a, b))]["p"] for a, b in PAIRS]
        pair_p_holm = holm(pair_p)
        row = dict(variable=v, display=SHORT[v], level=meta["level"], unit=meta["unit"], n=n,
                   n_rest=len(groups[0]), n_normal=len(groups[1]), n_active=len(groups[2]),
                   median_rest=float(np.median(groups[0])), median_normal=float(np.median(groups[1])),
                   median_active=float(np.median(groups[2])),
                   kw_h=float(h), kw_p=float(p), epsilon2=eps2)
        for (a, b), pr, ph in zip(PAIRS, pair_p, pair_p_holm):
            name = PAIR_NAMES[(a, b)]
            d = dunn[(min(a, b), max(a, b))]
            z = d["z"] if a < b else -d["z"]          # orient as first minus second
            row[f"dunn_z_{name}"] = float(z)
            row[f"dunn_p_{name}"] = float(pr)
            row[f"dunn_p_holm_{name}"] = float(ph)
            cd = cliffs_delta(groups[a], groups[b])
            lo, hi = cliffs_delta_ci(groups[a], groups[b], rng)
            row[f"cliff_{name}"] = cd
            row[f"cliff_{name}_ci_low"] = lo
            row[f"cliff_{name}_ci_high"] = hi
        row["expected_active_vs_rest"] = meta["expected"]
        row["agreement"] = concordance(row["cliff_active_rest"], meta["expected"])
        test_rows.append(row)
        print(f"  {SHORT[v]:26s} H {h:8.2f} p {p:.2e} eps2 {eps2:.4f}  Cliff A-R {row['cliff_active_rest']:+.3f} "
              f"[{row['cliff_active_rest_ci_low']:+.3f}, {row['cliff_active_rest_ci_high']:+.3f}]  "
              f"expected {meta['expected']}  {row['agreement']}")
    tests = pd.DataFrame(test_rows)
    pcols = [c for c in tests.columns if c.startswith(("kw_p", "dunn_p"))]
    check(bool(((tests[pcols] >= 0) & (tests[pcols] <= 1)).all().all()), "all p-values in [0, 1]")
    check(bool(((tests.epsilon2 >= 0) & (tests.epsilon2 <= 1)).all()), "epsilon2 in [0, 1]")
    ok_cd = True
    for name in PAIR_NAMES.values():
        c = tests[f"cliff_{name}"]
        ok_cd &= bool(((c >= -1) & (c <= 1)).all()
                      and ((tests[f"cliff_{name}_ci_low"] <= c) & (c <= tests[f"cliff_{name}_ci_high"])).all())
    check(ok_cd, "Cliff's delta in [-1, 1] and every CI contains its point")
    check(bool((tests.set_index("variable").loc[list(ALL_VARIABLES), ["n_rest", "n_normal", "n_active"]].to_numpy()
                == inventory.set_index("variable").loc[list(ALL_VARIABLES), ["n_rest", "n_normal", "n_active"]]
                .to_numpy()).all()), "tests n per class == inventory")
    # Holm sanity: adjusted >= raw, monotone
    ok_h = all(bool((tests[f"dunn_p_holm_{nm}"] >= tests[f"dunn_p_{nm}"] - 1e-15).all()) for nm in PAIR_NAMES.values())
    check(ok_h, "Holm-adjusted p >= raw p on every pair")

    # ---- outputs
    run_date = datetime.now().isoformat(timespec="seconds")
    day = run_date[:10]
    caveat = ("Caveat: consecutive days are autocorrelated; the Kruskal-Wallis and Dunn p-values treat the days as "
              "independent and are approximate (optimistic). Inference rests on the effect sizes (epsilon-squared, "
              "Cliff's delta with bootstrap CI).")
    desc.to_csv(res / "t32_descriptives.csv", index=False, float_format="%.10g", lineterminator="\n")
    dmd = md_table(desc, {c: "%.4f" for c in ("mean", "median", "p25", "p75", "iqr")})
    dmd += ["", "Day-t values of each indicator on the 958 canonically labelled days (R 60, A_r 85, A_s 75; 202 / "
                "710 / 46), per class: n (days with a value; no imputation), mean, median, p25, p75, iqr = p75 - p25. "
                "Level A = Apple Watch (device independent of the Oura scores that define the label); level B = raw "
                "Oura signals outside the rule but feeding readiness (internal consistency only); *_lag1 = value of "
                "day t-1 vs the class of day t (T1.5 hypothesis). Labels on all 958 days, not the test set. "
                f"Script: revision/scripts/label_agreement.py, run {day}."]
    write_text(res / "t32_descriptives_table.md", "\n".join(dmd) + "\n")
    tests.to_csv(res / "t32_tests.csv", index=False, float_format="%.10g", lineterminator="\n")
    fmt = {c: "%.4f" for c in tests.columns if tests[c].dtype.kind == "f" and not c.startswith(("kw_p", "dunn_p"))}
    fmt.update({c: "%.3e" for c in pcols})
    tmd = md_table(tests, fmt)
    tmd += ["", "One row per variable: Kruskal-Wallis H and p over Rest / Normal / Active (scipy, tie-corrected); "
                "epsilon2 = H / (n - 1); Dunn post-hoc z (first class minus second, tie-corrected) with raw and "
                "Holm-adjusted p over the 3 pairs (direct implementation, no scikit-posthocs); Cliff's delta = "
                "P(first > second) - P(first < second) with percentile 95 percent CI from 1000 bootstrap resamples of "
                f"the days within each class (np.random.default_rng({SEED}), sequential over variables and pairs). "
                "expected_active_vs_rest fixed before running (+ / - / 0 = no expectation); agreement = sign of "
                "Cliff's delta Active minus Rest vs the expectation. " + caveat + " Labels on all 958 days (canonical "
                f"thresholds), not the test set; nothing canonical changes. Script: revision/scripts/label_agreement.py, "
                f"run {day}."]
    write_text(res / "t32_tests_table.md", "\n".join(tmd) + "\n")

    n_a = boxplot_figure(frame, [v for v in VARIABLES if VARIABLES[v]["level"] == "A"], (2, 3),
                         FIG / "t32_indicator_boxplots")
    n_b = boxplot_figure(frame, [v for v in VARIABLES if VARIABLES[v]["level"] == "B"] + list(LAG_VARIABLES), (2, 4),
                         FIG / "t32_indicator_boxplots_oura")
    check(n_a == 6 and n_b == 8, f"figures: level A 6 panels ({n_a}), level B + lag 8 panels ({n_b}, 7 used)")

    summary = {lvl: {"concordant": int(((tests.level == lvl) & (tests.agreement == "concordant")).sum()),
                     "discordant": int(((tests.level == lvl) & (tests.agreement == "discordant")).sum()),
                     "no_expectation": int(((tests.level == lvl) & (tests.agreement == "no expectation")).sum())}
               for lvl in sorted(tests.level.unique())}
    meta = {
        "run_date": run_date, "elapsed_seconds": round(time.time() - t0, 1), "seed": SEED, "n_boot": N_BOOT,
        "n_days": 958, "label_distribution": list(dist), "thresholds": {"R": 60, "A_r": 85, "A_s": 75},
        "variables": {v: {k: m[k] for k in ("level", "source", "unit", "expected")} for v, m in ALL_VARIABLES.items()},
        "n_per_class": {v: [int(r.n_rest), int(r.n_normal), int(r.n_active)] for v, r in
                        inventory.set_index("variable").iterrows()},
        "excluded": ["readiness_score", "sleep_score", "hrv_balance", "recovery_index", "activity_score",
                     "aw_sleep_deep_hours", "aw_sleep_rem_hours"],
        "effect_size": "epsilon2 = H / (n - 1); Cliff's delta with within-class bootstrap CI",
        "posthoc": "Dunn z, tie-corrected, two-sided normal p, Holm over 3 pairs (direct implementation)",
        "caveat": caveat,
        "agreement_summary": summary,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__, "scikit_posthocs": None},
        "figure_axes": [n_a, n_b],
        "sanity_checks_failed": fails,
    }
    with open(res / "t32_run_meta.json", "w", newline="\n", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print(f"\nagreement summary: {summary}")
    print(f"elapsed {time.time() - t0:.1f}s; outputs in {res}")
    if fails:
        print(f"{len(fails)} check(s) FAILED: {fails}")
        sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="T3.2 label agreement with label-independent indicators")
    ap.add_argument("--results", default=RES, type=Path)
    main(ap.parse_args().results)
