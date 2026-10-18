"""
test_statistical_tests.py - unit tests for the core functions of
statistical_tests.py (McNemar counts, paired bootstrap, Cochran's Q helper).

Run from repo root:  python -m pytest revision/scripts/test_statistical_tests.py -q
(or: python revision/scripts/test_statistical_tests.py)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from statistical_tests import (METRICS, cochran_q, mcnemar_counts,  # noqa: E402
                               mcnemar_tests, paired_bootstrap)

L = [0, 1, 2]
F1 = METRICS["f1_macro"]
BA = METRICS["balanced_accuracy"]


def _balanced_idx(n: int, n_boot: int = 50, seed: int = 0) -> np.ndarray:
    """Index matrix whose rows are permutations of arange(n): every class present in every row."""
    rng = np.random.default_rng(seed)
    return np.vstack([rng.permutation(n) for _ in range(n_boot)])


def test_mcnemar_counts_manual_case() -> None:
    # 10 days. A correct on days 0-6 (7 days); B correct on days 0-3 and 7-8 (6 days).
    y = np.zeros(10, dtype=int)
    a = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])   # wrong on 7, 8, 9
    b = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])   # wrong on 4, 5, 6, 9
    n00, n01, n10, n11 = mcnemar_counts(y, a, b)
    assert (n00, n01, n10, n11) == (1, 3, 2, 4)      # both wrong: day 9; A only: 4,5,6; B only: 7,8
    assert n00 + n01 + n10 + n11 == 10
    t = mcnemar_tests(n00, n01, n10, n11)
    assert 0.0 <= t["p_cc"] <= 1.0 and 0.0 <= t["p_exact"] <= 1.0
    # chi-square with continuity correction: (|n01 - n10| - 1)^2 / (n01 + n10) = 0 for 3 vs 2
    assert math.isclose(t["chi2_cc"], 0.0, abs_tol=1e-12)


def test_identical_models_give_zero_everything() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 60)
    p = rng.integers(0, 3, 60)
    n00, n01, n10, n11 = mcnemar_counts(y, p, p)
    assert n01 == 0 and n10 == 0 and n00 + n11 == 60
    idx = _balanced_idx(60)
    for metric in (F1, BA):
        r = paired_bootstrap(y, p, p, metric, idx, L)
        assert r["diff_point"] == 0.0
        assert (r["ci_low"], r["ci_high"]) == (0.0, 0.0)
        assert r["prop_diff_gt_0"] == 0.0
        assert r["n_skipped"] == 0
        assert (r["ci_A_low"], r["ci_A_high"]) == (r["ci_B_low"], r["ci_B_high"])


def test_perfect_vs_all_wrong_gives_diff_one() -> None:
    # Every row of the index matrix is a permutation, so all 3 classes are present in each resample.
    y = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2])
    a = y.copy()                 # perfect: F1 macro = 1
    b = (y + 1) % 3              # every day wrong: F1 macro = 0 (labels fixed, zero_division=0)
    idx = _balanced_idx(len(y), n_boot=40, seed=3)
    r = paired_bootstrap(y, a, b, F1, idx, L)
    assert r["diff_point"] == 1.0
    assert (r["ci_low"], r["ci_high"]) == (1.0, 1.0)
    assert r["prop_diff_gt_0"] == 1.0
    assert r["n_skipped"] == 0 and r["n_used"] == 40
    assert (r["ci_A_low"], r["ci_A_high"]) == (1.0, 1.0)
    assert (r["ci_B_low"], r["ci_B_high"]) == (0.0, 0.0)
    assert mcnemar_counts(y, a, b) == (0, len(y), 0, 0)


def test_swap_models_flips_sign_and_discordant_cells() -> None:
    rng = np.random.default_rng(7)
    y = rng.integers(0, 3, 80)
    a = rng.integers(0, 3, 80)
    b = rng.integers(0, 3, 80)
    n00, n01, n10, n11 = mcnemar_counts(y, a, b)
    m00, m01, m10, m11 = mcnemar_counts(y, b, a)
    assert (m00, m01, m10, m11) == (n00, n10, n01, n11)
    ta, tb = mcnemar_tests(n00, n01, n10, n11), mcnemar_tests(m00, m01, m10, m11)
    assert ta == tb                                   # McNemar is symmetric in A, B
    idx = _balanced_idx(80, n_boot=60, seed=8)
    for metric in (F1, BA):
        ab = paired_bootstrap(y, a, b, metric, idx, L)
        ba = paired_bootstrap(y, b, a, metric, idx, L)
        assert math.isclose(ab["diff_point"], -ba["diff_point"], abs_tol=1e-12)
        assert math.isclose(ab["ci_low"], -ba["ci_high"], abs_tol=1e-12)
        assert math.isclose(ab["ci_high"], -ba["ci_low"], abs_tol=1e-12)
        assert (ab["ci_A_low"], ab["ci_A_high"]) == (ba["ci_B_low"], ba["ci_B_high"])
        assert (ab["ci_B_low"], ab["ci_B_high"]) == (ba["ci_A_low"], ba["ci_A_high"])


def test_determinism_same_index_matrix() -> None:
    rng = np.random.default_rng(11)
    y = rng.integers(0, 3, 100)
    a = rng.integers(0, 3, 100)
    b = rng.integers(0, 3, 100)
    idx = np.random.default_rng(42).integers(0, 100, size=(200, 100))
    r1 = paired_bootstrap(y, a, b, F1, idx, L)
    r2 = paired_bootstrap(y, a, b, F1, idx.copy(), L)
    assert r1 == r2
    assert mcnemar_counts(y, a, b) == mcnemar_counts(y, a, b)
    assert cochran_q(y, [a, b, a]) == cochran_q(y, [a, b, a])


def test_skipped_resamples_are_counted() -> None:
    # Rows that drop class 2 entirely must be skipped, not silently scored.
    y = np.array([0, 0, 1, 1, 2, 2])
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([0, 1, 1, 0, 2, 2])
    idx = np.array([[0, 1, 2, 3, 0, 1],        # no class 2 -> skipped
                    [0, 1, 2, 3, 4, 5],        # all classes
                    [4, 5, 0, 2, 1, 3]])       # all classes
    r = paired_bootstrap(y, a, b, F1, idx, L)
    assert r["n_skipped"] == 1 and r["n_used"] == 2


def test_cochran_q_two_identical_plus_one() -> None:
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    a = np.array([0, 1, 2, 0, 1, 2, 1, 2])     # 6/8 correct
    b = np.array([0, 1, 2, 0, 1, 2, 1, 2])     # identical to a
    c = np.array([1, 2, 0, 1, 2, 0, 0, 1])     # 2/8 correct
    q = cochran_q(y, [a, b, c])
    assert q["df"] == 2 and q["n_models"] == 3 and q["n_days"] == 8
    assert 0.0 <= q["p"] <= 1.0 and q["statistic"] > 0
    # Q is invariant to the order of the models
    assert q == cochran_q(y, [c, a, b])


def test_arbitrary_pair_prop_in_unit_interval() -> None:
    # Any pair of models, whatever the names: prop_diff_gt_0 is a proportion in [0, 1],
    # equal to the fraction of used resamples with a positive difference.
    rng = np.random.default_rng(21)
    y = rng.integers(0, 3, 90)
    baseline = np.concatenate([[1], y[:-1]])          # persistence-like: previous day's label
    model = np.where(rng.random(90) < 0.7, y, rng.integers(0, 3, 90))
    idx = _balanced_idx(90, n_boot=70, seed=22)
    for metric in (F1, BA):
        r = paired_bootstrap(y, model, baseline, metric, idx, L)
        assert 0.0 <= r["prop_diff_gt_0"] <= 1.0
        assert r["n_used"] + r["n_skipped"] == 70
        assert r["ci_low"] <= r["ci_high"]
        # the proportion is a multiple of 1 / n_used
        assert math.isclose(r["prop_diff_gt_0"] * r["n_used"], round(r["prop_diff_gt_0"] * r["n_used"]), abs_tol=1e-9)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"ALL PASS ({len(tests)} tests)")
