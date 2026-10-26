"""
_check_t13_safety_layer.py - T1.3 Task 4 pre-check: does the canonical RF
(19 features, JSON source, revision/results/test_predictions_rf.csv) reproduce
the canonical safety-layer figures of CIFRE_CANONICE.md section 7
(threshold 0.50: coverage 80.2 percent, 38 overridden, mean caloric error
68.8 kcal, max 200 kcal)?

Safety layer located in confidence_threshold.py (repo root):
  apply_threshold(y_pred_raw, y_proba, threshold)  confidence = max(y_proba[i]);
      if max < threshold the prediction is overridden with Normal (1)
  caloric_error_analysis(y_true, y_pred)  mean and max of
      |CALORIC_MAP[pred] - CALORIC_MAP[true]|, CALORIC_MAP = {Rest: -200,
      Normal: 0, Active: +200}, over ALL test days (overridden days count as
      Normal); the reference is the true label's caloric offset, not a measured
      energy expenditure
  coverage = (N - n_overridden) / N on the test set
Both functions take hard predictions and probabilities as parameters, so they
are imported and applied unchanged here (no retraining, no modification).

The stored canonical table outputs/tables/confidence_threshold_analysis.csv
(commit f7cc4ca, 2026-05-26) was produced by confidence_threshold.main(),
whose pipeline differs from the canonical one: 16 lag features (no
aw_sleep_deep_hours, aw_sleep_rem_hours, aw_resp_rate), Apple Watch data from
the XML exports data/apple_health/*.csv instead of
apple_watch_api/apple_health_daily.csv, n_jobs=-1. Its threshold-0 row
(F1 0.520, accuracy 62.0 percent, 3 Active predicted) differs from the
canonical RF (F1 0.5063, accuracy 64.1 percent, 10 Active predicted).

Prints both tables side by side and PASS/FAIL on the 0.50 row. EXIT 1 IS THE
EXPECTED, INTENTIONAL OUTCOME: the canonical RF does NOT reproduce the stored
table (coverage 65.6 percent vs 80.2, 66 vs 38 overridden, 58.3 vs 68.8 kcal).
Decision at STOP 2b (2026-10-18): stage C runs with the canonical RF as the raw
reference (safety_layer_calibrated.py); the mismatch is recorded as a T1.3
finding in CIFRE_REVIZIE.md; CIFRE_CANONICE.md stays untouched and the erratum
is written at T4.3. Evidence script, kept; do not "fix" it to exit 0.
Run from repo root:  python revision/scripts/_check_t13_safety_layer.py

What this script does and does not do (clarified 2026-10-26, audit F3 observation): it does NOT rerun
confidence_threshold.main() or the old 16-feature pipeline; the "canonical" column is read from the stored
table outputs/tables/confidence_threshold_analysis.csv, whose figures are the ones printed in
CIFRE_CANONICE.md section 7, and the "canonical RF" column is recomputed here from
revision/results/test_predictions_rf.csv with the two unchanged functions above. The audit (F3, 2026-10-25)
executed the old pipeline's logic itself and confirmed the erratum of Section 4.9 by execution. The exit
code 1 is the intended outcome of this evidence script, not a failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import confidence_threshold as ct  # noqa: E402  (module import only; main() is not called)

RES = ROOT / "revision" / "results"
CANON_TABLE = ROOT / "outputs" / "tables" / "confidence_threshold_analysis.csv"
THRESHOLDS = [0.0, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def main() -> None:
    df = pd.read_csv(RES / "test_predictions_rf.csv")
    y = df.y_true.to_numpy(dtype=int)
    P = df[["p_rest", "p_normal", "p_active"]].to_numpy(dtype=float)
    yp = df.y_pred.to_numpy(dtype=int)
    print(f"confidence_threshold.py: CALORIC_MAP = {ct.CALORIC_MAP}, FEATURES = {len(ct.FEATURES)} "
          f"(canonical RF uses 19); apply_threshold uses max(y_proba[i]) < threshold -> Normal")

    rows = []
    for t in THRESHOLDS:
        ya, n_over, _ = ct.apply_threshold(yp.copy(), P, t)
        mean_err, max_err, _ = ct.caloric_error_analysis(y, ya)
        rows.append(dict(threshold=t, f1_macro=f1_score(y, ya, average="macro", zero_division=0),
                         accuracy=accuracy_score(y, ya), coverage_pct=(len(y) - n_over) / len(y) * 100,
                         n_overridden=n_over, mean_cal_error=mean_err, max_cal_error=max_err,
                         n_active_pred=int((ya == 2).sum())))
    mine = pd.DataFrame(rows)
    canon = pd.read_csv(CANON_TABLE)
    cols = ["threshold", "f1_macro", "accuracy", "coverage_pct", "n_overridden", "mean_cal_error",
            "max_cal_error", "n_active_pred"]
    print("\n=== canonical RF (19 feat, JSON) + confidence_threshold.apply_threshold ===")
    print(mine[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== stored outputs/tables/confidence_threshold_analysis.csv (confidence_threshold.main, 16 feat, XML) ===")
    print(canon[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    m = mine[mine.threshold == 0.5].iloc[0]
    c = canon[canon.threshold == 0.5].iloc[0]
    check(abs(m.coverage_pct - c.coverage_pct) < 0.05,
          f"threshold 0.50 coverage: canonical RF {m.coverage_pct:.1f}% vs stored {c.coverage_pct:.1f}%")
    check(abs(m.mean_cal_error - c.mean_cal_error) < 0.05,
          f"threshold 0.50 mean caloric error: canonical RF {m.mean_cal_error:.1f} vs stored {c.mean_cal_error:.1f} kcal")
    check(int(m.n_overridden) == int(c.n_overridden),
          f"threshold 0.50 overridden: canonical RF {int(m.n_overridden)} vs stored {int(c.n_overridden)}")
    b, bc = mine.iloc[0], canon.iloc[0]
    print(f"\nthreshold 0 baseline: canonical RF F1 {b.f1_macro:.4f} / acc {b.accuracy:.4f} / Active predicted "
          f"{int(b.n_active_pred)} vs stored {bc.f1_macro:.4f} / {bc.accuracy:.4f} / {int(bc.n_active_pred)}")
    n_below = int((P.max(axis=1) < 0.5).sum())
    print(f"days with max p < 0.50 in test_predictions_rf.csv: {n_below} (= 1 - coverage at 0.50)")

    print("\n" + ("ALL PASS" if not fails else
                  f"{len(fails)} FAIL: the canonical RF does not reproduce the stored safety-layer table; "
                  "stage C stops here (STOP, report)."))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
