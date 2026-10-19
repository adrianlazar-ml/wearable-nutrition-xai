"""
test_threshold_sensitivity.py - unit tests for threshold_sensitivity.py (T3.1.3).

- labelling rule on a synthetic frame with explicit thresholds, including the equalities
  (readiness == A_r with sleep >= A_s is Active; readiness == R is NOT Rest; sleep == R is
  NOT Rest; Rest has priority over Active)
- canonical cell (60, 85, 75) == canonical labels 202 / 710 / 46 on the real data, with the
  train / test counts 158 / 570 / 38 and 44 / 140 / 8
- comparability (>= 5 Active in test, >= 4 in train) on synthetic cases
- SMOTE k convention of the canonical Table 7 script
- persistence on a synthetic 5-day series with a known result
- majority baseline, label counts, top-k SHAP ordering
- the imported module is revision/scripts/threshold_sensitivity.py, not the root script

Run from repo root:  python -m pytest revision/scripts/test_threshold_sensitivity.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # revision/ before the repo root (frozen t3_labels.py)
import t3_labels  # noqa: E402
import threshold_sensitivity as ts  # noqa: E402
from common import build_dataset, load_raw  # noqa: E402


def test_modules_are_the_revision_ones() -> None:
    assert Path(ts.__file__).resolve().parent == Path(__file__).resolve().parent
    assert Path(t3_labels.__file__).resolve().parent == Path(__file__).resolve().parents[1]
    assert hasattr(t3_labels, "assign_labels")


def test_labelling_rule_with_explicit_thresholds_and_equalities() -> None:
    R, A_r, A_s = 60, 85, 75
    df = pd.DataFrame({
        "readiness_score": [59, 60, 90, 85, 84, 85, 100, 59, np.nan, 61],
        "sleep_score":     [90, 60, 59, 75, 90, 74, 100, 100, 80, 61],
    })
    # expected: Rest (readiness 59 < 60); Normal (60 == R is not Rest, below A_r); Rest (sleep 59 < R, priority
    # over readiness 90); Active (85 == A_r and 75 == A_s); Normal (84 < A_r); Normal (sleep 74 < A_s);
    # Active; Rest; NaN; Normal
    expected = [0, 1, 0, 2, 1, 1, 2, 0, np.nan, 1]
    got = t3_labels.assign_labels(df, rest_cutoff=R, active_readiness=A_r, active_sleep=A_s)
    assert got.tolist()[:8] == expected[:8]
    assert np.isnan(got.iloc[8])
    assert got.iloc[9] == 1
    # same rule through the scalar function
    assert t3_labels.classify_day(60, 60, rest_cutoff=60, active_readiness=85, active_sleep=75) == 1
    assert t3_labels.classify_day(85, 75, rest_cutoff=60, active_readiness=85, active_sleep=75) == 2
    assert t3_labels.classify_day(59, 100, rest_cutoff=60, active_readiness=85, active_sleep=75) == 0
    # thresholds really are parameters: the same frame under (65, 80, 75)
    got2 = t3_labels.assign_labels(df, rest_cutoff=65, active_readiness=80, active_sleep=75)
    assert got2.tolist()[:8] == [0, 0, 0, 2, 2, 1, 2, 0]


def test_canonical_cell_reproduces_canonical_labels() -> None:
    X, y, dates = build_dataset(load_raw(), rest_cutoff=60, active_readiness=85, active_sleep=75)
    assert len(X) == 958
    assert ts.label_counts(y) == (202, 710, 46)
    i = int(len(X) * ts.TRAIN_RATIO)
    assert i == 766
    assert ts.label_counts(y.iloc[:i]) == (158, 570, 38)
    assert ts.label_counts(y.iloc[i:]) == (44, 140, 8)


def test_rows_and_cut_do_not_depend_on_thresholds() -> None:
    df = load_raw()
    X1, y1, d1 = build_dataset(df, rest_cutoff=55, active_readiness=90, active_sleep=75)
    X2, y2, d2 = build_dataset(df, rest_cutoff=65, active_readiness=80, active_sleep=75)
    assert len(X1) == len(X2) == 958
    assert list(d1) == list(d2)
    assert ts.label_counts(y1) != ts.label_counts(y2)


@pytest.mark.parametrize("n_test, n_train, expected", [
    (5, 4, True), (8, 38, True), (4, 38, False), (5, 3, False), (1, 2, False), (0, 0, False), (19, 116, True),
])
def test_is_comparable(n_test: int, n_train: int, expected: bool) -> None:
    assert ts.is_comparable(n_test, n_train) is expected


@pytest.mark.parametrize("n_min, k", [(38, 3), (4, 3), (3, 2), (2, 1), (1, None), (0, None), (116, 3)])
def test_smote_k_convention(n_min: int, k: int | None) -> None:
    assert ts.smote_k(n_min) == k


def test_persistence_on_synthetic_series() -> None:
    y = pd.Series([2, 0, 1, 1, 0])          # 5 days; split after day 2 -> test = days 3, 4, 5
    pred = ts.persistence_predictions(y, 2)
    assert pred.tolist() == [0, 1, 1]        # label of the previous day; first test day uses the last train day
    y_true = y.iloc[2:].to_numpy()
    assert y_true.tolist() == [1, 1, 0]
    # Rest: 1 predicted (wrong), 1 true (missed) -> F1 0; Normal: TP 1, FP 1, FN 1 -> P = R = 0.5 -> F1 0.5; Active 0
    assert ts.f1_macro(y_true, pred) == pytest.approx(np.mean([0.0, 0.5, 0.0]))
    assert ts.f1_active(y_true, pred) == 0.0
    assert float(np.mean(pred == y_true)) == pytest.approx(1 / 3)                   # persistence accuracy 1 / 3
    with pytest.raises(ValueError):
        ts.persistence_predictions(y, 0)


def test_majority_and_counts() -> None:
    y_tr = [1, 1, 0, 2, 1]
    pred, maj = ts.majority_predictions(y_tr, 4)
    assert maj == 1 and pred.tolist() == [1, 1, 1, 1]
    assert ts.label_counts([0, 0, 1, 2, 2, 2]) == (2, 1, 3)
    assert ts.label_counts([]) == (0, 0, 0)


def test_top_shap_ordering() -> None:
    vals = np.zeros(len(ts.FEATURES))
    vals[ts.FEATURES.index("aw_hr_mean_lag1")] = 0.06
    vals[ts.FEATURES.index("hrv_balance_lag1")] = 0.03
    vals[ts.FEATURES.index("steps_oura_lag1")] = 0.02
    top = ts.top_shap(vals)
    assert [t[0] for t in top] == ["aw_hr_mean", "oura_hrv_balance", "oura_steps"]
    assert [t[2] for t in top] == [1, 2, 3]
    assert top[0][1] == pytest.approx(0.06)


def test_cell_name() -> None:
    assert ts.cell_name(60, 85) == "R60_A85"
    assert ts.CANONICAL == (60, 85, 75)
