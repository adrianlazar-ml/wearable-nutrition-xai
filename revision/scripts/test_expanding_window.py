"""
test_expanding_window.py - unit tests for expanding_window.py (T3.3.3).

- fold bounds: correct indices, disjoint test blocks, train strictly before test, fold 1 ==
  canonical_split (766 / 192), fold 5 not cuttable from 958 rows
- persistence on a synthetic series: the first block day uses the last train day
- pooled metrics on synthetic predictions (perfect, degenerate) and the pooled bootstrap CI
  contains the point and is deterministic for a given seed
- fixed design constants

Run from repo root:  python -m pytest revision/scripts/test_expanding_window.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import expanding_window as ew  # noqa: E402
from common import TRAIN_RATIO  # noqa: E402


def test_fold_bounds_and_disjointness() -> None:
    b = ew.all_fold_bounds()
    assert b == [(766, 766, 958), (574, 574, 766), (382, 382, 574), (190, 190, 382)]
    assert b[0][1] == int(958 * TRAIN_RATIO)                  # fold 1 == canonical split
    for tr_end, te0, te1 in b:
        assert tr_end == te0 and te1 - te0 == 192 and tr_end >= 1
    tests = [set(range(a, c)) for _, a, c in b]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not tests[i] & tests[j]
    assert set().union(*tests) == set(range(190, 958))
    with pytest.raises(ValueError):
        ew.fold_bounds(5)                                      # 958 - 960 < 1
    with pytest.raises(ValueError):
        ew.fold_bounds(0)


def test_fold_bounds_generic() -> None:
    assert ew.fold_bounds(1, n=10, block=3) == (7, 7, 10)
    assert ew.fold_bounds(3, n=10, block=3) == (1, 1, 4)
    assert ew.all_fold_bounds(n_folds=3, n=10, block=3)[-1] == (1, 1, 4)


def test_persistence_first_block_day_uses_last_train_day() -> None:
    y = pd.Series([2, 0, 1, 1, 0, 2])
    tr_end, te0, te1 = ew.fold_bounds(1, n=6, block=3)         # train rows 0-2, test rows 3-5
    pred = ew.persistence_predictions(y, te0)[:te1 - te0]
    assert pred.tolist() == [1, 1, 0]                          # row 2 (last train), row 3, row 4
    tr_end, te0, te1 = ew.fold_bounds(2, n=6, block=2)         # train rows 0-1, test rows 2-3
    assert ew.persistence_predictions(y, te0)[:te1 - te0].tolist() == [0, 1]


def test_pooled_metrics_and_bootstrap() -> None:
    y_true = np.array([0, 1, 2] * 40)
    assert ew.pooled_metrics(y_true, y_true)["f1_macro"] == 1.0
    assert ew.pooled_metrics(y_true, y_true)["f1_active"] == 1.0
    y_pred = np.ones_like(y_true)                             # always Normal
    m = ew.pooled_metrics(y_true, y_pred)
    assert m["f1_active"] == 0.0 and m["accuracy"] == pytest.approx(1 / 3)
    rng = np.random.default_rng(42)
    b = ew.pooled_bootstrap(y_true, y_true, rng, n_boot=100)
    assert b["f1_macro_ci_low"] == b["f1_macro_ci_high"] == 1.0 and b["n_skipped"] == 0
    rng0 = np.random.default_rng(0)
    y_pred2 = np.where(rng0.random(len(y_true)) < 0.7, y_true, (y_true + 1) % 3)
    point = ew.pooled_metrics(y_true, y_pred2)["f1_macro"]
    b1 = ew.pooled_bootstrap(y_true, y_pred2, np.random.default_rng(42), n_boot=200)
    b2 = ew.pooled_bootstrap(y_true, y_pred2, np.random.default_rng(42), n_boot=200)
    assert b1 == b2                                            # deterministic
    assert b1["f1_macro_ci_low"] <= point <= b1["f1_macro_ci_high"]
    assert b1["n_used"] + b1["n_skipped"] == 200


def test_fixed_design() -> None:
    assert (ew.N_FOLDS, ew.BLOCK, ew.N_TOTAL, ew.N_BOOT, ew.SEED) == (4, 192, 958, 1000, 42)
    assert ew.CANON_RF == dict(f1_macro=0.5063, accuracy=0.6406, mcc=0.1416)
    assert ew.is_comparable(5, 4) and not ew.is_comparable(4, 4) and not ew.is_comparable(5, 3)
