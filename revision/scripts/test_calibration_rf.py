"""
test_calibration_rf.py - unit tests for the calibration metrics and calibrators in
calibration_rf.py (T1.3). Synthetic data only; no project files are read.

Run from repo root:  python -m pytest revision/scripts/test_calibration_rf.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration_rf as cal  # noqa: E402


def test_ece_zero_for_perfectly_calibrated_bins():
    # bin (0.7, 0.8]: 10 samples at confidence 0.75, exactly 7.5 correct is impossible, so use
    # 4 samples at 0.75 with 3 correct -> acc 0.75; bin (0.2, 0.3]: 4 samples at 0.25, 1 correct.
    conf = np.array([0.75] * 4 + [0.25] * 4)
    hit = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=float)
    ece, mce = cal.ece_mce(conf, hit)
    assert ece == pytest.approx(0.0, abs=1e-12)
    assert mce == pytest.approx(0.0, abs=1e-12)


def test_ece_known_two_bins():
    # bin 1 (0.1, 0.2]: conf 0.2, acc 0.5 -> gap 0.3, weight 0.5
    # bin 8 (0.8, 0.9]: conf 0.9, acc 1.0 -> gap 0.1, weight 0.5   => ECE 0.2, MCE 0.3
    conf = np.array([0.2, 0.2, 0.9, 0.9])
    hit = np.array([1, 0, 1, 1], dtype=float)
    ece, mce = cal.ece_mce(conf, hit)
    assert ece == pytest.approx(0.2, abs=1e-12)
    assert mce == pytest.approx(0.3, abs=1e-12)


def test_ece_top_label_uses_max_probability_and_argmax():
    y = np.array([1, 1, 0, 2])
    P = np.array([[0.1, 0.8, 0.1],    # conf 0.8, correct
                  [0.1, 0.8, 0.1],    # conf 0.8, correct
                  [0.8, 0.1, 0.1],    # conf 0.8, correct
                  [0.8, 0.1, 0.1]])   # conf 0.8, wrong -> bin (0.7, 0.8]: acc 0.75, conf 0.8
    assert cal.ece_top_label(y, P) == pytest.approx(0.05, abs=1e-12)
    assert cal.mce_top_label(y, P) == pytest.approx(0.05, abs=1e-12)


def test_bin_ids_match_sklearn_uniform_rule():
    p = np.array([0.0, 0.05, 0.1, 0.1000001, 0.55, 0.9, 0.95, 1.0])
    assert cal.bin_ids(p).tolist() == [0, 0, 0, 1, 5, 8, 9, 9]


def test_brier_multiclass_bounds():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    wrong = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    assert cal.brier_multiclass(y, perfect) == pytest.approx(0.0)
    assert cal.brier_multiclass(y, wrong) == pytest.approx(2.0)
    # mixed: row 0 perfect, row 1 wrong -> (0 + 2) / 2 = 1
    assert cal.brier_multiclass(np.array([0, 1]), np.array([[1, 0, 0], [1, 0, 0]], float)) == pytest.approx(1.0)


def test_brier_binary_known():
    assert cal.brier_binary(np.array([1, 0, 1]), np.array([0.5, 0.5, 1.0])) == pytest.approx((0.25 + 0.25 + 0) / 3)


def test_normalize_ovr_sums_to_one_and_fallback():
    Q = np.array([[0.2, 0.3, 0.5], [0.9, 0.9, 0.9], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    out, n_fallback = cal.normalize_ovr(Q)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-12)
    assert n_fallback == 1
    assert np.allclose(out[2], 1 / 3)
    assert np.allclose(out[3], [0, 1, 0])
    assert np.allclose(out[1], 1 / 3)


def _logistic_distorted(n: int = 4000, seed: int = 0):
    """True probability q ~ U(0.02, 0.98); labels ~ Bernoulli(q); reported probability
    overconfident: logit(p) = 3 * logit(q) + 0.5."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.02, 0.98, size=n)
    y = (rng.random(n) < q).astype(int)
    p = 1 / (1 + np.exp(-(3 * np.log(q / (1 - q)) + 0.5)))
    return p, y


def test_platt_reduces_ece_on_logistic_distortion():
    p, y = _logistic_distorted()
    P = np.column_stack([1 - p, p])             # two classes (labels 0 / 1), OvR on both columns
    platt = cal.PlattOvR(labels=[0, 1]).fit(P, y)
    Pc = platt.predict_proba(P)
    assert np.allclose(Pc.sum(axis=1), 1.0, atol=1e-12)
    e_raw = cal.ece_binary(y, P[:, 1])
    e_cal = cal.ece_binary(y, Pc[:, 1])
    assert e_raw > 0.1
    assert e_cal < 0.5 * e_raw
    coef = platt.coefficients()["Normal"]       # label 1 -> CLASS_NAMES[1]
    assert coef["slope"] == pytest.approx(1 / 3, abs=0.05)   # inverts the x3 distortion


def test_isotonic_reduces_ece_and_is_monotone():
    p, y = _logistic_distorted(seed=1)
    P = np.column_stack([1 - p, p])
    iso = cal.IsotonicOvR(labels=[0, 1]).fit(P, y)
    Pc = iso.predict_proba(P)
    assert np.allclose(Pc.sum(axis=1), 1.0, atol=1e-12)
    assert cal.ece_binary(y, Pc[:, 1]) < 0.5 * cal.ece_binary(y, P[:, 1])
    assert all(iso.is_monotone().values())


def test_reliability_table_counts_sum_to_n():
    rng = np.random.default_rng(3)
    conf = rng.random(500)
    hit = (rng.random(500) < conf).astype(float)
    tab = cal.reliability_table(conf, hit)
    assert len(tab) == 10
    assert int(tab.n_b.sum()) == 500
    assert tab.loc[tab.n_b > 0, "conf_b"].between(0, 1).all()
