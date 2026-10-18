"""
test_group_ablation.py - unit tests for group_ablation.py (T1.5.4).

- partition: union of the 6 groups == the 19 features, pairwise disjoint (G1 and G4 variants)
- delta: synthetic case, delta = full minus LOGO with the correct sign
- paired bootstrap: identical predictions -> diff 0 and CI [0, 0]; A strictly better on
  every day -> CI excludes 0
- marginal: CI of the full RF from the saved npy matrix == the T1.1 value in CIFRE_REVIZIE.md
  at 4 decimals

Run from repo root:  python -m pytest revision/scripts/test_group_ablation.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES  # noqa: E402
from group_ablation import (CIFRE_REVIZIE, INDEX_FILE, LABELS, RES, build_groups,  # noqa: E402
                            check_partition, delta, f1_active, f1_macro, metrics,
                            paired_bootstrap, read_t11_rf_f1_ci, subset_in_canonical_order)


def _balanced_idx(n: int, n_boot: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([rng.permutation(n) for _ in range(n_boot)])


def test_partition_g1_default() -> None:
    groups = build_groups("G1")
    check_partition(groups)
    assert [len(v) for v in groups.values()] == [2, 4, 3, 2, 4, 4]
    assert sum(len(v) for v in groups.values()) == 19 == len(FEATURES)
    assert "sleep_score_lag1" in groups["G1"] and "readiness_score_lag1" in groups["G1"]
    flat = [f for v in groups.values() for f in v]
    assert len(set(flat)) == 19 and set(flat) == set(FEATURES)
    keys = list(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert not set(groups[a]) & set(groups[b])


def test_partition_g4_alternative() -> None:
    groups = build_groups("G4")
    check_partition(groups)
    assert [len(v) for v in groups.values()] == [1, 4, 3, 3, 4, 4]
    assert "sleep_score_lag1" in groups["G4"]
    with pytest.raises(ValueError):
        build_groups("G2")


def test_check_partition_detects_overlap_and_gap() -> None:
    groups = build_groups("G1")
    bad = {k: list(v) for k, v in groups.items()}
    bad["G2"].append(bad["G1"][0])               # overlap
    with pytest.raises(AssertionError):
        check_partition(bad)
    bad = {k: list(v) for k, v in groups.items()}
    bad["G3"].pop()                              # gap
    with pytest.raises(AssertionError):
        check_partition(bad)


def test_subset_keeps_canonical_order() -> None:
    sub = subset_in_canonical_order(["aw_spo2_lag1", "readiness_score_lag1", "hrv_mean_lag1"])
    assert sub == ["readiness_score_lag1", "hrv_mean_lag1", "aw_spo2_lag1"]


def test_delta_sign_synthetic() -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    full = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])      # perfect: macro F1 = 1
    logo = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 1])      # two mistakes
    assert f1_macro(y, full) == 1.0 and f1_macro(y, logo) < 1.0
    d = delta(f1_macro(y, full), f1_macro(y, logo))
    assert d > 0                                          # removing the group hurts -> positive
    assert delta(f1_macro(y, logo), f1_macro(y, full)) == pytest.approx(-d)
    assert delta(0.5063, 0.5148) == pytest.approx(-0.0085)   # group whose removal helps -> negative
    assert delta(0.5063, 0.5063) == 0.0


def test_metrics_keys_and_confusion_matrix() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.array([0, 1, 2, 1, 1, 0])
    m = metrics(y, p)
    cm = np.array([[m[f"cm_{a}{b}"] for b in "rna"] for a in "rna"])
    assert cm.sum() == 6 and cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["f1_active"] == f1_active(y, p) == pytest.approx(2 / 3)


def test_paired_bootstrap_identical_predictions() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 60)
    p = rng.integers(0, 3, 60)
    idx = _balanced_idx(60)
    for metric in (f1_macro, f1_active):
        r = paired_bootstrap(y, p, p, metric, idx, LABELS)
        assert r["diff_point"] == 0.0
        assert (r["ci_low"], r["ci_high"]) == (0.0, 0.0)
        assert r["prop_diff_gt_0"] == 0.0 and r["n_skipped"] == 0


def test_paired_bootstrap_strictly_better_excludes_zero() -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2])
    a = y.copy()                          # perfect on every day
    b = (y + 1) % 3                       # wrong on every day
    idx = _balanced_idx(len(y), n_boot=40, seed=3)
    for metric in (f1_macro, f1_active):
        r = paired_bootstrap(y, a, b, metric, idx, LABELS)
        assert r["diff_point"] == 1.0
        assert r["ci_low"] > 0.0 and (r["ci_low"], r["ci_high"]) == (1.0, 1.0)
        assert r["prop_diff_gt_0"] == 1.0


def test_marginal_ci_full_rf_matches_t11() -> None:
    idx = np.load(RES / INDEX_FILE)
    assert idx.shape == (1000, 192)
    ref = pd.read_csv(RES / "test_predictions_rf.csv")
    y, p = ref["y_true"].to_numpy(dtype=int), ref["y_pred"].to_numpy(dtype=int)
    r = paired_bootstrap(y, p, p, f1_macro, idx, LABELS)
    got = (round(r["ci_A_low"], 4), round(r["ci_A_high"], 4))
    assert got == read_t11_rf_f1_ci(CIFRE_REVIZIE)
    assert round(f1_macro(y, p), 4) == 0.5063


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
