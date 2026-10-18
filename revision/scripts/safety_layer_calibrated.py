"""
safety_layer_calibrated.py - T1.3 stage C: the canonical safety layer applied to
raw and calibrated RF probabilities on the test set (192 days).

Safety layer = confidence_threshold.apply_threshold (repo root), imported and
used unchanged: confidence = max(p); if max(p) < threshold the prediction is
overridden with Normal. Caloric error = confidence_threshold.caloric_error_analysis,
unchanged: mean / max of |offset[pred] - offset[true]| with offsets
Rest -200, Normal 0, Active +200, over all 192 days (overridden days count as
Normal). Coverage = (192 - overridden) / 192.

Reference raw = the canonical RF (19 features, results/test_predictions_rf.csv).
NOTE: CIFRE_CANONICE.md section 7 (coverage 80.2 percent, 68.8 kcal at 0.50)
was produced by confidence_threshold.main() on the 16-feature XML pipeline, not
by the canonical RF (see _check_t13_safety_layer.py). The canonical RF gives
coverage 65.6 percent (66 overridden), mean error 58.3 kcal, max 200 kcal at
0.50; these are the sanity values checked here.

Variants: raw, platt, isotonic (calibrators fit on OOF folds 1-4, loaded from
results/models/calibrator_{platt,isotonic}_rf.joblib). The prediction fed to the
safety layer is argmax of the variant's probabilities (for raw == y_pred).
Threshold grid {0.00, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80}; 0.50 is
the canonical threshold and is not re-chosen here.

Outputs (results/): t13_safety_layer_variants.csv, t13_safety_layer_variants_table.md,
  figures/t13_safety_layer_coverage.{pdf,png}

Run from repo root:  python revision/scripts/safety_layer_calibrated.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import confidence_threshold as ct  # noqa: E402  (functions only; main() is never called)
import calibration_rf as cal  # noqa: E402  (PlattOvR / IsotonicOvR classes for joblib.load)

RES = ROOT / "revision" / "results"
FIG = RES / "figures"
PROBA_COLS = ["p_rest", "p_normal", "p_active"]
VARIANTS = ["raw", "platt", "isotonic"]
THRESHOLDS = [0.00, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80]
CANONICAL_THRESHOLD = 0.50
# canonical RF through the unchanged safety layer at 0.50 (from _check_t13_safety_layer.py)
EXPECTED_RAW_050 = dict(coverage_pct=65.625, n_overridden=66, mean_cal_error=58.3333, max_cal_error=200)
COLS = ["variant", "threshold", "coverage_pct", "n_overridden", "mean_cal_error", "max_cal_error",
        "f1_macro", "accuracy", "f1_active", "n_pred_rest", "n_pred_normal", "n_pred_active"]
FLOAT_COLS = {"threshold", "coverage_pct", "mean_cal_error", "f1_macro", "accuracy", "f1_active"}
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def main() -> None:
    test = pd.read_csv(RES / "test_predictions_rf.csv")
    y = test.y_true.to_numpy(dtype=int)
    P_raw = test[PROBA_COLS].to_numpy(dtype=float)
    check(len(test) == 192, f"test csv 192 rows ({len(test)})")
    check(np.array_equal(P_raw.argmax(axis=1), test.y_pred.to_numpy(dtype=int)), "raw argmax == y_pred (192/192)")
    platt = joblib.load(RES / "models" / "calibrator_platt_rf.joblib")
    iso = joblib.load(RES / "models" / "calibrator_isotonic_rf.joblib")
    probs = {"raw": P_raw, "platt": platt.predict_proba(P_raw), "isotonic": iso.predict_proba(P_raw)}
    for v, P in probs.items():
        check(float(np.abs(P.sum(axis=1) - 1).max()) < 1e-6, f"{v}: rows sum to 1")

    rows = []
    for v in VARIANTS:
        P = probs[v]
        yp = P.argmax(axis=1)
        for t in THRESHOLDS:
            ya, n_over, _ = ct.apply_threshold(yp.copy(), P, t)
            mean_err, max_err, _ = ct.caloric_error_analysis(y, ya)
            rows.append(dict(variant=v, threshold=t, coverage_pct=(len(y) - n_over) / len(y) * 100,
                             n_overridden=int(n_over), mean_cal_error=float(mean_err), max_cal_error=int(max_err),
                             f1_macro=float(f1_score(y, ya, average="macro", zero_division=0)),
                             accuracy=float(accuracy_score(y, ya)),
                             f1_active=float(f1_score(y, ya, labels=[2], average="macro", zero_division=0)),
                             n_pred_rest=int((ya == 0).sum()), n_pred_normal=int((ya == 1).sum()),
                             n_pred_active=int((ya == 2).sum())))
    df = pd.DataFrame(rows)[COLS]

    # sanity: raw at 0.50 reproduces the canonical-RF safety-layer figures; raw at 0 == canonical RF
    r = df[(df.variant == "raw") & (df.threshold == CANONICAL_THRESHOLD)].iloc[0]
    check(abs(r.coverage_pct - EXPECTED_RAW_050["coverage_pct"]) < 1e-6 and int(r.n_overridden) == 66,
          f"raw @0.50 coverage {r.coverage_pct:.4f}% / overridden {int(r.n_overridden)} == 65.625% / 66")
    check(abs(r.mean_cal_error - EXPECTED_RAW_050["mean_cal_error"]) < 1e-3 and int(r.max_cal_error) == 200,
          f"raw @0.50 mean caloric error {r.mean_cal_error:.4f} / max {int(r.max_cal_error)} == 58.3333 / 200")
    r0 = df[(df.variant == "raw") & (df.threshold == 0.0)].iloc[0]
    check(round(r0.f1_macro, 4) == 0.5063 and round(r0.accuracy, 4) == 0.6406,
          f"raw @0.00 F1 {r0.f1_macro:.4f} / acc {r0.accuracy:.4f} == canonical 0.5063 / 0.6406")
    for v in VARIANTS:
        sub = df[df.variant == v]
        check(sub.coverage_pct.is_monotonic_decreasing, f"{v}: coverage non-increasing in threshold")
        check((sub.n_pred_rest + sub.n_pred_normal + sub.n_pred_active == 192).all(), f"{v}: emitted states sum to 192")

    df.to_csv(RES / "t13_safety_layer_variants.csv", index=False, float_format="%.10g", lineterminator="\n")

    def cell(row: pd.Series, c: str) -> str:
        if c == "variant":
            return str(row[c])
        return f"{row[c]:.4f}" if c in FLOAT_COLS else f"{int(row[c])}"

    md = ["# T1.3 stage C: safety layer with raw vs calibrated RF probabilities (test, 192 days)", "",
          "Safety layer = confidence_threshold.apply_threshold / caloric_error_analysis, unchanged: "
          "if max(p) < threshold the prediction becomes Normal; caloric error = |offset[pred] - offset[true]| with "
          "Rest -200 / Normal 0 / Active +200 kcal over all 192 days; coverage = (192 - overridden) / 192. "
          "raw = canonical RF (19 features, test_predictions_rf.csv). platt / isotonic = one-vs-rest calibrators "
          "fit on OOF folds 1-4 (calibration_rf.py); prediction = argmax of the calibrated probabilities. "
          "Canonical threshold 0.50 (not re-chosen here). NOTE: CIFRE_CANONICE.md section 7 (80.2 percent, "
          "68.8 kcal) comes from the 16-feature XML pipeline, not from the canonical RF; the canonical RF gives "
          "65.6 percent / 58.3 kcal at 0.50 (see _check_t13_safety_layer.py). "
          "Script revision/scripts/safety_layer_calibrated.py.", "",
          "## Canonical threshold 0.50", "",
          "| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for _, row in df[df.threshold == CANONICAL_THRESHOLD].iterrows():
        md.append("| " + " | ".join(cell(row, c) for c in COLS) + " |")
    md += ["", "## Full threshold grid", "", "| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for _, row in df.iterrows():
        md.append("| " + " | ".join(cell(row, c) for c in COLS) + " |")
    (RES / "t13_safety_layer_variants_table.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")

    # figure: coverage vs threshold, 3 variants
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    colors = {"raw": "#555555", "platt": "#2196F3", "isotonic": "#E65100"}
    markers = {"raw": "o", "platt": "s", "isotonic": "^"}
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for v in VARIANTS:
        sub = df[df.variant == v]
        ax.plot(sub.threshold, sub.coverage_pct, marker=markers[v], color=colors[v], lw=1.4, ms=5, label=v)
    ax.axvline(CANONICAL_THRESHOLD, color="#999999", ls=":", lw=1)
    ax.text(CANONICAL_THRESHOLD + 0.01, 3, "canonical 0.50", fontsize=8, color="#666666")
    ax.set_xlabel("Confidence threshold (max p)", fontsize=10)
    ax.set_ylabel("Coverage (% of test days)", fontsize=10)
    ax.set_title("Safety-layer coverage vs threshold, test (n = 192)", fontsize=10, fontweight="bold")
    ax.set_xticks(THRESHOLDS)
    ax.set_xticklabels([f"{t:.2f}" for t in THRESHOLDS], rotation=45, ha="right")
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.savefig(FIG / "t13_safety_layer_coverage.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None})
    fig.savefig(FIG / "t13_safety_layer_coverage.png", dpi=300, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)

    print("\n" + "\n".join(md[5:]))
    print(f"\nsaved t13_safety_layer_variants.csv / _table.md, figures/t13_safety_layer_coverage.pdf/.png in {RES}")
    print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}"))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
