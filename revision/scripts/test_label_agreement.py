"""
test_label_agreement.py - unit tests for label_agreement.py (T3.2.4).

- Cliff's delta on known cases: identical samples 0; fully separated +1 / -1; a hand-computed
  mixed case; bootstrap CI contains the point and stays inside [-1, 1]
- epsilon-squared on a synthetic case with a computable value and against scipy's H
- Dunn z against the known formula on a small case; Holm adjustment on known p-values
- concordance flags from the sign of Cliff's delta vs the expected direction
- identity value(t).shift(1) == lag-1 on a synthetic daily frame (and a broken one)
- fixed design: 10 day-t variables + 3 lag-1 activity variables, expected directions as declared

Run from repo root:  python -m pytest revision/scripts/test_label_agreement.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_agreement as la  # noqa: E402


def test_cliffs_delta_known_cases() -> None:
    assert la.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
    assert la.cliffs_delta([4, 5, 6], [1, 2, 3]) == 1.0
    assert la.cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0
    # x = [1, 2, 3], y = [2, 2, 4]: pairs x > y: (3,2),(3,2) = 2; x < y: (1,2),(1,2),(1,4),(2,4),(3,4) = 5; ties 2
    assert la.cliffs_delta([1, 2, 3], [2, 2, 4]) == pytest.approx((2 - 5) / 9)
    assert la.cliffs_delta([5.0], [1.0, 9.0]) == 0.0
    with pytest.raises(ValueError):
        la.cliffs_delta([], [1])


def test_cliffs_delta_ci_contains_point_and_is_bounded() -> None:
    rng0 = np.random.default_rng(0)
    x, y = rng0.normal(1, 1, 60), rng0.normal(0, 1, 40)
    d = la.cliffs_delta(x, y)
    lo, hi = la.cliffs_delta_ci(x, y, np.random.default_rng(42), n_boot=200)
    assert -1 <= lo <= d <= hi <= 1
    lo2, hi2 = la.cliffs_delta_ci(x, y, np.random.default_rng(42), n_boot=200)
    assert (lo, hi) == (lo2, hi2)                       # deterministic given the rng seed
    lo3, hi3 = la.cliffs_delta_ci([4, 5, 6, 7], [1, 2, 3], np.random.default_rng(1), n_boot=50)
    assert lo3 == hi3 == 1.0                            # fully separated stays +1 under resampling


def test_epsilon_squared() -> None:
    assert la.epsilon_squared(h=9.0, n=10) == pytest.approx(1.0)
    assert la.epsilon_squared(h=0.0, n=10) == 0.0
    g1, g2, g3 = [1, 2, 3], [4, 5, 6], [7, 8, 9]       # perfectly separated: H maximal
    h, _ = stats.kruskal(g1, g2, g3)
    eps = la.epsilon_squared(float(h), 9)
    assert 0 <= eps <= 1 and eps == pytest.approx(h / 8)
    assert eps == pytest.approx(h / ((9 ** 2 - 1) / (9 + 1)))   # the (n^2 - 1) / (n + 1) form
    with pytest.raises(ValueError):
        la.epsilon_squared(1.0, 1)


def test_dunn_matches_formula_and_is_symmetric() -> None:
    g = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), np.array([2.5, 7.0, 8.0])]
    out = la.dunn_test(g)
    pooled = np.concatenate(g)
    n = len(pooled)
    ranks = stats.rankdata(pooled)
    r0, r1 = ranks[:3].mean(), ranks[3:6].mean()
    z = (r0 - r1) / np.sqrt(n * (n + 1) / 12 * (1 / 3 + 1 / 3))   # no ties here
    assert out[(0, 1)]["z"] == pytest.approx(z)
    assert out[(0, 1)]["p"] == pytest.approx(2 * stats.norm.sf(abs(z)))
    assert out[(0, 1)]["z"] < 0 and out[(1, 2)]["z"] < 0
    assert set(out) == {(0, 1), (0, 2), (1, 2)}
    # identical groups -> z = 0, p = 1
    same = la.dunn_test([np.array([1.0, 2.0]), np.array([1.0, 2.0])])
    assert same[(0, 1)]["z"] == 0.0 and same[(0, 1)]["p"] == 1.0


def test_holm_known_values() -> None:
    from statsmodels.stats.multitest import multipletests
    # sorted p 0.01, 0.03, 0.04 -> 3 x 0.01, 2 x 0.03, max(0.06, 1 x 0.04) = 0.06 (step-down monotone)
    assert la.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert la.holm([0.5, 0.6, 0.7]) == pytest.approx([1.0, 1.0, 1.0])
    adj = la.holm([0.001, 0.5, 0.02])
    assert adj == pytest.approx([0.003, 0.5, 0.04])
    assert all(a >= p for a, p in zip(adj, [0.001, 0.5, 0.02]))
    for p in ([0.01, 0.04, 0.03], [0.001, 0.5, 0.02], [0.2, 0.01, 0.015], [1e-6, 1e-3, 0.9]):
        assert la.holm(p) == pytest.approx(multipletests(p, method="holm")[1].tolist())


@pytest.mark.parametrize("delta, expected, result", [
    (0.5, "+", "concordant"), (-0.5, "+", "discordant"), (-0.5, "-", "concordant"), (0.5, "-", "discordant"),
    (0.0, "+", "discordant"), (0.3, "0", "no expectation"), (-0.3, "0", "no expectation"),
])
def test_concordance(delta: float, expected: str, result: str) -> None:
    assert la.concordance(delta, expected) == result


def test_lag_identity_on_synthetic_frame() -> None:
    daily = pd.Series([10.0, np.nan, 12.0, 13.0, 14.0])
    lag = daily.shift(1)
    assert la.lag_identity(daily, lag)
    broken = lag.copy()
    broken.iloc[3] = 99.0
    assert not la.lag_identity(daily, broken)
    broken2 = lag.copy()
    broken2.iloc[2] = 11.0                 # value where the lag is missing -> missingness differs
    assert not la.lag_identity(daily, broken2)


def test_fixed_design() -> None:
    assert list(la.VARIABLES) == ["aw_hr_mean", "aw_hrv_mean", "aw_active_calories", "aw_steps", "aw_resp_rate",
                                  "aw_spo2", "hrv_mean", "rhr_lowest", "temp_deviation", "steps_oura"]
    assert list(la.LAG_VARIABLES) == ["aw_active_calories_lag1", "aw_steps_lag1", "steps_oura_lag1"]
    exp = {v: m["expected"] for v, m in la.ALL_VARIABLES.items()}
    assert exp == {"aw_hr_mean": "-", "aw_hrv_mean": "+", "aw_active_calories": "+", "aw_steps": "+",
                   "aw_resp_rate": "0", "aw_spo2": "0", "hrv_mean": "+", "rhr_lowest": "-", "temp_deviation": "-",
                   "steps_oura": "+", "aw_active_calories_lag1": "+", "aw_steps_lag1": "+", "steps_oura_lag1": "+"}
    assert sum(m["level"] == "A" for m in la.VARIABLES.values()) == 6
    assert sum(m["level"] == "B" for m in la.VARIABLES.values()) == 4
    for excluded in ("readiness_score", "sleep_score", "hrv_balance", "recovery_index", "activity_score"):
        assert excluded not in la.ALL_VARIABLES
    assert la.PAIRS == [(2, 0), (2, 1), (1, 0)] and la.N_BOOT == 1000 and la.SEED == 42
