"""
test_extended_metrics.py - unit tests for the manual G-mean in extended_metrics.py.

Run from repo root:  python revision/scripts/test_extended_metrics.py
(or: python -m pytest revision/scripts/test_extended_metrics.py)
"""

import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extended_metrics import gmean  # noqa: E402

L = [0, 1, 2]


def test_perfect_prediction_is_one() -> None:
    y = np.array([0, 0, 1, 1, 1, 2, 2])
    assert gmean(y, y, L) == 1.0


def test_one_class_never_predicted_is_zero() -> None:
    y = np.array([0, 0, 1, 1, 1, 2, 2])
    p = np.array([0, 0, 1, 1, 1, 1, 1])      # Active never predicted
    assert gmean(y, p, L) == 0.0


def test_known_recalls() -> None:
    # recalls: Rest 2/4 = 0.5, Normal 2/2 = 1.0, Active 1/4 = 0.25 -> (0.125) ** (1/3) = 0.5
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2])
    p = np.array([0, 0, 1, 1, 1, 1, 2, 0, 0, 1])
    assert math.isclose(gmean(y, p, L), 0.5, rel_tol=1e-12)


def test_matches_geometric_mean_of_sklearn_recalls() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.integers(0, 3, 60)
        p = rng.integers(0, 3, 60)
        if len(np.unique(y)) < 3:
            continue
        rec = recall_score(y, p, labels=L, average=None, zero_division=0)
        assert math.isclose(gmean(y, p, L), float(np.prod(rec) ** (1 / 3)), rel_tol=1e-12)
        # arithmetic mean (balanced accuracy) >= geometric mean (AM-GM)
        assert gmean(y, p, L) <= balanced_accuracy_score(y, p) + 1e-12


def test_missing_class_in_truth_is_nan() -> None:
    y = np.array([0, 0, 1, 1])        # no Active
    p = np.array([0, 1, 1, 2])
    assert math.isnan(gmean(y, p, L))


def test_label_order_independent_of_data_order() -> None:
    y = np.array([2, 2, 0, 1, 1, 0])
    p = np.array([2, 0, 0, 1, 0, 0])
    assert math.isclose(gmean(y, p, L), gmean(y, p, [2, 0, 1]), rel_tol=1e-12)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"ALL PASS ({len(tests)} tests)")
