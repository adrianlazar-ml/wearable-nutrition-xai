"""
test_shap_stability.py - unit tests for the pure helpers of shap_stability.py (T1.4).

Run from repo root:  python -m pytest revision/scripts/test_shap_stability.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shap_stability import (JACCARD_LEVELS, aggregate_multiclass, jaccard, kendall,  # noqa: E402
                            percentile_ci, rank_desc, short_name, top_k)


# ---------------------------------------------------------------- Jaccard
def test_jaccard_identical_sets():
    assert jaccard({1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}) == 1.0


def test_jaccard_disjoint_sets():
    assert jaccard({1, 2, 3}, {4, 5, 6}) == 0.0


def test_jaccard_three_of_seven():
    # |A ∩ B| = 3, |A ∪ B| = 7  ->  3/7
    assert jaccard({1, 2, 3, 4, 5}, {3, 4, 5, 6, 7}) == pytest.approx(3 / 7)


def test_jaccard_levels_for_two_5_sets():
    assert JACCARD_LEVELS == {0: 0.0, 1: 1 / 9, 2: 0.25, 3: 3 / 7, 4: 2 / 3, 5: 1.0}


# ---------------------------------------------------------------- Kendall tau
def test_kendall_identical_ranking_is_one():
    r = np.arange(1, 20)
    assert kendall(r, r) == pytest.approx(1.0)


def test_kendall_reversed_ranking_is_minus_one():
    r = np.arange(1, 20)
    assert kendall(r, r[::-1]) == pytest.approx(-1.0)


def test_kendall_matches_scipy_on_random_case():
    rng = np.random.default_rng(0)
    a, b = rng.permutation(19) + 1, rng.permutation(19) + 1
    assert kendall(a, b) == pytest.approx(kendalltau(a, b).statistic)
    assert -1.0 <= kendall(a, b) <= 1.0


# ---------------------------------------------------------------- ranking
def test_rank_desc_largest_is_rank_one():
    assert rank_desc([0.1, 0.5, 0.3]).tolist() == [3, 1, 2]


def test_rank_desc_ties_deterministic_min():
    v = [3.0, 1.0, 3.0, 2.0]
    r1, r2 = rank_desc(v), rank_desc(list(v))
    assert r1.tolist() == [1, 4, 1, 3]          # tied maxima share rank 1, next is 3
    assert r1.tolist() == r2.tolist()
    assert r1.dtype.kind == "i"


def test_rank_desc_permutation_without_ties():
    rng = np.random.default_rng(1)
    v = rng.random(19)
    assert sorted(rank_desc(v).tolist()) == list(range(1, 20))
    assert rank_desc(v)[int(np.argmax(v))] == 1


def test_top_k_stable_tie_break_by_feature_order():
    v = np.array([0.2, 0.9, 0.2, 0.9, 0.1])
    assert top_k(v, 2) == {1, 3}
    assert top_k(v, 3) == {1, 3, 0}             # first of the tied 0.2 values wins


# ---------------------------------------------------------------- multiclass SHAP aggregation
def test_aggregate_array_samples_features_classes():
    rng = np.random.default_rng(2)
    sv = rng.normal(size=(50, 19, 3))
    manual = np.array([np.abs(sv[:, j, :]).mean() for j in range(19)])
    out = aggregate_multiclass(sv, n_samples=50)
    assert out.shape == (19,)
    np.testing.assert_allclose(out, manual, rtol=0, atol=1e-15)


def test_aggregate_list_of_class_arrays_equals_array_form():
    rng = np.random.default_rng(3)
    sv = rng.normal(size=(50, 19, 3))
    as_list = [sv[:, :, c] for c in range(3)]      # older shap: list of 3 x (n, 19)
    np.testing.assert_allclose(aggregate_multiclass(as_list, n_samples=50),
                               aggregate_multiclass(sv, n_samples=50), rtol=0, atol=1e-15)


def test_aggregate_rejects_2d():
    with pytest.raises(ValueError):
        aggregate_multiclass(np.zeros((50, 19)), n_samples=50)


# ---------------------------------------------------------------- misc
def test_percentile_ci_bounds():
    lo, hi = percentile_ci(np.arange(101))
    assert lo == pytest.approx(2.5) and hi == pytest.approx(97.5)


def test_short_name_prefixes():
    assert short_name("aw_hr_mean_lag1") == "aw_hr_mean"
    assert short_name("hrv_balance_lag1") == "oura_hrv_balance"
    assert short_name("steps_oura_lag1") == "oura_steps"
