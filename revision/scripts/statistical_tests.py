"""
statistical_tests.py - T1.2 (Reviewer 1, R1.3): statistical comparisons between
RF, XGB and SVM computed from the saved test-set predictions (no retraining, no
model or dataset access). Inputs are only the csv files written at T0.2/T0.4
and the shared bootstrap index matrix written at T1.1.

2a. McNemar on pairs (RF-XGB, RF-SVM, XGB-SVM), on correct/incorrect per day
    (correct = y_pred == y_true). Convention for the 2x2 table of a pair (A, B):
        n11 = A correct and B correct      n01 = A correct and B wrong
        n10 = A wrong   and B correct      n00 = A wrong   and B wrong
    i.e. the first index is A (1 = correct, 0 = wrong), the second is B.
    The table passed to statsmodels is [[n11, n01], [n10, n00]]; only the
    discordant cells n01, n10 enter the statistic. Reported: chi-square with
    continuity correction (statsmodels mcnemar(exact=False, correction=True)),
    the exact binomial p (exact=True), and Holm-adjusted p over the family of 3
    pairs (multipletests(method="holm")). When n01 + n10 < 25 the exact
    variant is flagged as the recommended one.
2b. Paired bootstrap of the metric difference for the same 3 pairs, plus
    RF-persistence (T1.2b, baseline comparison outside the Holm family), and two
    metrics (macro F1 with labels=[0,1,2], zero_division=0; balanced accuracy).
    For every row r of the shared index matrix, the same indices are applied
    to y_true, y_pred_A and y_pred_B and diff_r = metric_A - metric_B.
    Reported: point difference on the full test set, percentile 95 percent CI
    [2.5, 97.5] of the difference, and the "proportion of resamples with
    diff > 0" (a descriptive quantity, NOT a p-value). Resamples in which a
    class is absent from the resampled y_true are skipped and counted
    (expected 0, as at T1.1). The marginal CIs of each model (percentiles of
    metric_A alone) are also reported and checked against T1.1 at 4 decimals.
2c. Cochran's Q over the three models simultaneously on the correct/incorrect
    matrix (192 x 3), statsmodels cochrans_q. No further post hoc (2a is it).
2d. McNemar RF vs persistence, same procedure as 2a, reported separately as a
    baseline comparison, outside the Holm family of 3.

Outputs (revision/results/, LF line endings):
  t12_mcnemar.csv, t12_mcnemar_table.md
  t12_paired_bootstrap.csv, t12_paired_bootstrap_table.md
  t12_cochran_q.csv
  t12_run_meta.json

Run from repo root:  python revision/scripts/statistical_tests.py
Exits 1 if any built-in sanity check fails (inputs, marginal CIs vs T1.1,
canonical point differences, p in [0, 1], warnings inside the bootstrap loop).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
import statsmodels
from sklearn.metrics import balanced_accuracy_score, f1_score
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"

SEED = 42
N_BOOT = 1000
N_TEST = 192
LABELS = [0, 1, 2]                      # Rest, Normal, Active
SUPPORT = (44, 140, 8)
MODELS = ["rf", "xgb", "svm"]
BASELINE = "persistence"
LABEL = {"rf": "RF", "xgb": "XGB", "svm": "SVM", "persistence": "persistence"}
PAIRS = [("rf", "xgb"), ("rf", "svm"), ("xgb", "svm")]   # Holm family
BASELINE_PAIR = ("rf", BASELINE)                        # outside the family
DISCORDANT_MIN_FOR_CC = 25                              # below: exact recommended
INDEX_FILE = f"bootstrap_indices_{N_BOOT}x{N_TEST}_seed{SEED}.npy"

Metric = Callable[[np.ndarray, np.ndarray], float]
METRICS: dict[str, Metric] = {
    "f1_macro": lambda y, p: float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)),
    "balanced_accuracy": lambda y, p: float(balanced_accuracy_score(y, p)),
}
METRIC_TITLE = {"f1_macro": "Macro F1", "balanced_accuracy": "Balanced accuracy"}

# Reference figures (CIFRE_CANONICE.md and T1.1): point values and marginal CIs on
# the shared index matrix. The script must reproduce them exactly (sanity check).
CANON_POINT = {
    "f1_macro": {"rf": 0.5063, "xgb": 0.4760, "svm": 0.4347, "persistence": 0.3039},
    "balanced_accuracy": {"rf": 0.5227, "xgb": 0.4500, "svm": 0.4342, "persistence": 0.3039},
}
T11_CI = {
    "f1_macro": {"rf": (0.3784, 0.6130), "xgb": (0.3406, 0.5933), "svm": (0.3279, 0.5395),
                 "persistence": (0.2612, 0.3501)},
    "balanced_accuracy": {"rf": (0.3851, 0.6634), "xgb": (0.3439, 0.5763), "svm": (0.3288, 0.5683),
                          "persistence": (0.2602, 0.3520)},
}


# --------------------------------------------------------------------------- core functions
def mcnemar_counts(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
                   ) -> tuple[int, int, int, int]:
    """Return (n00, n01, n10, n11) for models A and B.

    Convention: first index = A, second index = B, 1 = correct, 0 = wrong.
    n01 = A correct and B wrong; n10 = A wrong and B correct.
    """
    a = np.asarray(pred_a) == np.asarray(y_true)
    b = np.asarray(pred_b) == np.asarray(y_true)
    n11 = int(np.sum(a & b))
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n00 = int(np.sum(~a & ~b))
    return n00, n01, n10, n11


def mcnemar_tests(n00: int, n01: int, n10: int, n11: int) -> dict[str, float]:
    """Chi-square McNemar with continuity correction and the exact binomial variant.

    Table layout passed to statsmodels: [[n11, n01], [n10, n00]]. The statistic
    depends only on the discordant cells n01 and n10.
    """
    table = [[n11, n01], [n10, n00]]
    cc = mcnemar(table, exact=False, correction=True)
    ex = mcnemar(table, exact=True)
    return {"chi2_cc": float(cc.statistic), "p_cc": float(cc.pvalue),
            "stat_exact": float(ex.statistic), "p_exact": float(ex.pvalue)}


def paired_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                     metric: Metric, idx: np.ndarray, labels: list[int] = LABELS
                     ) -> dict[str, float | int]:
    """Paired bootstrap of metric(A) - metric(B) on a shared index matrix.

    Every row of `idx` is applied to y_true, pred_a and pred_b alike. A row in
    which some class of `labels` is absent from the resampled y_true is skipped
    (the metric would be undefined for that class) and counted in n_skipped.
    Returns the point difference on the full data, the percentile CI of the
    difference, the proportion of used resamples with diff > 0 (descriptive,
    not a p-value), the skip count and the marginal CIs of A and B.
    """
    y_true, pred_a, pred_b = (np.asarray(v) for v in (y_true, pred_a, pred_b))
    diffs: list[float] = []
    va: list[float] = []
    vb: list[float] = []
    n_skipped = 0
    for row in idx:
        yt = y_true[row]
        if len(np.unique(yt)) < len(labels):
            n_skipped += 1
            continue
        ma = metric(yt, pred_a[row])
        mb = metric(yt, pred_b[row])
        va.append(ma)
        vb.append(mb)
        diffs.append(ma - mb)
    d = np.asarray(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    a_lo, a_hi = np.percentile(np.asarray(va), [2.5, 97.5])
    b_lo, b_hi = np.percentile(np.asarray(vb), [2.5, 97.5])
    return {"diff_point": metric(y_true, pred_a) - metric(y_true, pred_b),
            "ci_low": float(lo), "ci_high": float(hi),
            "prop_diff_gt_0": float(np.mean(d > 0)),
            "n_used": int(len(d)), "n_skipped": int(n_skipped),
            "ci_A_low": float(a_lo), "ci_A_high": float(a_hi),
            "ci_B_low": float(b_lo), "ci_B_high": float(b_hi)}


def cochran_q(y_true: np.ndarray, preds: list[np.ndarray]) -> dict[str, float]:
    """Cochran's Q on the (n_days x n_models) correct/incorrect matrix."""
    x = np.column_stack([np.asarray(p) == np.asarray(y_true) for p in preds]).astype(int)
    r = cochrans_q(x)
    return {"statistic": float(r.statistic), "df": int(r.df), "p": float(r.pvalue),
            "n_models": int(x.shape[1]), "n_days": int(x.shape[0])}


# --------------------------------------------------------------------------- inputs / io
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(res: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, str]]:
    """Read y_true, y_pred per system and the shared index matrix; verify them."""
    preds: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    y_ref: np.ndarray | None = None
    for k in MODELS + [BASELINE]:
        f = res / f"test_predictions_{k}.csv"
        df = pd.read_csv(f)
        hashes[f.name] = sha256(f)
        if len(df) != N_TEST:
            sys.exit(f"FAIL {k}: {len(df)} rows, expected {N_TEST}")
        y = df["y_true"].to_numpy(dtype=int)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y, y_ref):
            sys.exit(f"FAIL {k}: y_true differs from rf")
        preds[k] = df["y_pred"].to_numpy(dtype=int)
    assert y_ref is not None
    if tuple(int((y_ref == c).sum()) for c in LABELS) != SUPPORT:
        sys.exit("FAIL support differs from 44/140/8")
    f = res / INDEX_FILE
    idx = np.load(f)
    hashes[f.name] = sha256(f)
    if idx.shape != (N_BOOT, N_TEST) or idx.dtype.kind not in "iu" or idx.min() < 0 or idx.max() >= N_TEST:
        sys.exit(f"FAIL index matrix: shape {idx.shape}, dtype {idx.dtype}, range [{idx.min()}, {idx.max()}]")
    if not np.array_equal(idx, np.random.default_rng(SEED).integers(0, N_TEST, size=(N_BOOT, N_TEST))):
        sys.exit(f"FAIL index matrix differs from default_rng({SEED}) stream")
    return y_ref, preds, idx, hashes


def write_text(path: Path, text: str) -> None:
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(text)


def md_table(df: pd.DataFrame, fmt: dict[str, str]) -> list[str]:
    """Markdown table with the csv columns; floats formatted per `fmt`."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append("n/a" if pd.isna(v) else fmt[c] % v)   # nan (outside Holm family) -> n/a
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="T1.2 statistical tests on saved predictions")
    ap.add_argument("--results", default=RES, type=Path)
    args = ap.parse_args()
    res: Path = args.results
    t0 = time.time()
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    y_true, preds, idx, hashes = load_inputs(res)
    print(f"inputs OK: n = {N_TEST}, support {SUPPORT}, index matrix {idx.shape}")

    # ---- 2a + 2d: McNemar
    mc_rows: list[dict] = []
    for a, b in PAIRS + [BASELINE_PAIR]:
        n00, n01, n10, n11 = mcnemar_counts(y_true, preds[a], preds[b])
        t = mcnemar_tests(n00, n01, n10, n11)
        disc = n01 + n10
        in_family = (a, b) in PAIRS
        note = ("Holm family of 3" if in_family else "baseline comparison, outside the Holm family")
        if disc < DISCORDANT_MIN_FOR_CC:
            note += f"; n01+n10 = {disc} < {DISCORDANT_MIN_FOR_CC}: exact variant recommended"
        mc_rows.append(dict(pair=f"{LABEL[a]}-{LABEL[b]}", n00=n00, n01=n01, n10=n10, n11=n11,
                            chi2_cc=t["chi2_cc"], p_cc=t["p_cc"], p_exact=t["p_exact"],
                            p_cc_holm=np.nan, p_exact_holm=np.nan,
                            recommended="exact" if disc < DISCORDANT_MIN_FOR_CC else "cc",
                            note=note))
        check(n00 + n01 + n10 + n11 == N_TEST, f"{LABEL[a]}-{LABEL[b]}: n00+n01+n10+n11 = {N_TEST}")
    mc = pd.DataFrame(mc_rows)
    fam = mc.index[: len(PAIRS)]
    mc.loc[fam, "p_cc_holm"] = multipletests(mc.loc[fam, "p_cc"].to_numpy(), method="holm")[1]
    mc.loc[fam, "p_exact_holm"] = multipletests(mc.loc[fam, "p_exact"].to_numpy(), method="holm")[1]
    for c in ("p_cc", "p_exact", "p_cc_holm", "p_exact_holm"):
        vals = mc[c].dropna()
        check(bool(((vals >= 0) & (vals <= 1)).all()), f"McNemar {c} in [0, 1]")

    # ---- 2b: paired bootstrap (warnings inside the loop are counted and must be 0)
    pb_rows: list[dict] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for a, b in PAIRS + [BASELINE_PAIR]:        # T1.2b: RF-persistence added (outside the Holm family)
            for mname, mfn in METRICS.items():
                r = paired_bootstrap(y_true, preds[a], preds[b], mfn, idx)
                pb_rows.append(dict(pair=f"{LABEL[a]}-{LABEL[b]}", metric=mname, **r))
    n_warn = len(caught)
    check(n_warn == 0, f"sklearn/statsmodels warnings inside the bootstrap loop: {n_warn}")
    pb = pd.DataFrame(pb_rows)
    pb = pb[["pair", "metric", "diff_point", "ci_low", "ci_high", "prop_diff_gt_0", "n_skipped",
             "ci_A_low", "ci_A_high", "ci_B_low", "ci_B_high", "n_used"]]
    check(int(pb.n_skipped.sum()) == 0, f"resamples skipped (sum over pairs x metrics): {int(pb.n_skipped.sum())}")
    # sanity: marginal CIs == T1.1 and point differences == canonical, at 4 decimals
    # Audit F2 (2026-10-25) observation, kept as is: equality of two values each rounded to 4 decimals is fragile
    # near a rounding boundary (a difference of 1e-9 can flip the fourth decimal); it passes on the recorded
    # values and is not changed, a tolerance comparison (abs diff < 5e-5) would be the robust form.
    for r in pb.itertuples():
        a, b = r.pair.split("-")
        ka, kb = a.lower(), b.lower()
        exp_d = round(CANON_POINT[r.metric][ka] - CANON_POINT[r.metric][kb], 4)
        check(round(r.diff_point, 4) == exp_d,
              f"{r.pair} {r.metric}: diff_point {r.diff_point:.4f} == canonical {exp_d:.4f}")
        check((round(r.ci_A_low, 4), round(r.ci_A_high, 4)) == T11_CI[r.metric][ka],
              f"{r.pair} {r.metric}: marginal CI {a} [{r.ci_A_low:.4f}, {r.ci_A_high:.4f}] == T1.1 {T11_CI[r.metric][ka]}")
        check((round(r.ci_B_low, 4), round(r.ci_B_high, 4)) == T11_CI[r.metric][kb],
              f"{r.pair} {r.metric}: marginal CI {b} [{r.ci_B_low:.4f}, {r.ci_B_high:.4f}] == T1.1 {T11_CI[r.metric][kb]}")

    # ---- 2c: Cochran's Q
    q = cochran_q(y_true, [preds[k] for k in MODELS])
    cq = pd.DataFrame([dict(statistic=q["statistic"], df=q["df"], p=q["p"],
                            n_models=q["n_models"], n_days=q["n_days"])])
    check(0.0 <= q["p"] <= 1.0, "Cochran's Q p in [0, 1]")

    # ---- outputs
    run_date = datetime.now().isoformat(timespec="seconds")
    day = run_date[:10]
    mc.to_csv(res / "t12_mcnemar.csv", index=False, float_format="%.10g", lineterminator="\n")
    pb.to_csv(res / "t12_paired_bootstrap.csv", index=False, float_format="%.10g", lineterminator="\n")
    cq.to_csv(res / "t12_cochran_q.csv", index=False, float_format="%.10g", lineterminator="\n")

    pfmt = "%.4f"
    mc_md = md_table(mc, {c: pfmt for c in ("chi2_cc", "p_cc", "p_exact", "p_cc_holm", "p_exact_holm")})
    mc_md += ["",
              "Convention: n01 = A correct and B wrong, n10 = A wrong and B correct, n11 = both correct, "
              "n00 = both wrong (pair = A-B). chi2_cc / p_cc: McNemar chi-square with continuity correction "
              "(statsmodels mcnemar(exact=False, correction=True)); p_exact: exact binomial (exact=True); "
              f"Holm adjustment over the family of {len(PAIRS)} model pairs only (RF-persistence is reported "
              "outside the family, unadjusted). recommended = exact when n01 + n10 < "
              f"{DISCORDANT_MIN_FOR_CC}. Test set n = {N_TEST}. Script: revision/scripts/statistical_tests.py, "
              f"statsmodels {statsmodels.__version__}, run {day}."]
    write_text(res / "t12_mcnemar_table.md", "\n".join(mc_md) + "\n")

    pb_md = md_table(pb, {c: pfmt for c in ("diff_point", "ci_low", "ci_high", "prop_diff_gt_0",
                                             "ci_A_low", "ci_A_high", "ci_B_low", "ci_B_high")})
    pb_md += ["",
              "diff = metric(A) - metric(B) for pair A-B; diff_point on the full test set; [ci_low, ci_high] = "
              f"percentile 2.5 / 97.5 of the paired bootstrap differences over {N_BOOT} resamples (shared index "
              f"matrix {INDEX_FILE}, seed {SEED}, same indices applied to y_true, y_pred_A, y_pred_B). "
              "prop_diff_gt_0 = proportion of resamples with diff > 0 (descriptive, not a p-value). "
              "ci_A / ci_B = marginal CIs of each model on the same resamples (identical to T1.1). "
              "n_skipped = resamples with a class absent from y_true. RF-persistence (T1.2b) is a baseline "
              "comparison, outside the Holm family of 3 model pairs; the intervals are not adjusted for multiple comparisons. "
              f"Script: revision/scripts/statistical_tests.py, run {day}."]
    write_text(res / "t12_paired_bootstrap_table.md", "\n".join(pb_md) + "\n")

    meta = {
        "run_date": run_date,
        "seed": SEED,
        "n_boot": N_BOOT,
        "n_test": N_TEST,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "inputs_sha256": hashes,
        "pairs_holm_family": [f"{LABEL[a]}-{LABEL[b]}" for a, b in PAIRS],
        "baseline_pair": f"{LABEL[BASELINE_PAIR[0]]}-{LABEL[BASELINE_PAIR[1]]}",
        "n_warnings_in_bootstrap_loop": n_warn,
        "n_resamples_skipped_total": int(pb.n_skipped.sum()),
        "sanity_checks_failed": fails,
    }
    with open(res / "t12_run_meta.json", "w", newline="\n", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    print("\n" + "\n".join(mc_md[: len(mc) + 2]))
    print("\n" + "\n".join(pb_md[: len(pb) + 2]))
    print(f"\nCochran's Q: statistic {q['statistic']:.4f}, df {q['df']}, p {q['p']:.4f} "
          f"({q['n_models']} models, {q['n_days']} days)")
    print(f"\nelapsed {time.time() - t0:.1f}s; outputs in {res}")
    if fails:
        print(f"\n{len(fails)} sanity check(s) FAILED: {fails}")
        sys.exit(1)
    print("\nALL SANITY CHECKS PASS")


if __name__ == "__main__":
    main()
