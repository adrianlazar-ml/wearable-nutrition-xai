"""
_check_t14_inputs.py - T1.4 Task 1: inventory of the inputs of shap_stability.py
and identification / reproduction of the canonical SHAP procedure (R1.6).

Canonical SHAP procedure (cap5_compare_algorithms_json_ext.py, run_shap(), the
function behind Figure 2 = outputs/figures/json_extended_pipeline/
ext_shap_bar_random_forest.png, file dated 2026-06-11, commit b15862a):
  explainer   : shap.TreeExplainer(model), defaults (feature_perturbation
                "tree_path_dependent", model_output "raw" = class probabilities
                for a sklearn RandomForestClassifier)
  data        : the imputed TEST set (192 days), unscaled; train is not used
  aggregation : np.abs(shap_values).mean(axis=(0, 2)) on the (192, 19, 3)
                array returned by shap >= 0.45 (mean of |SHAP| over the 192
                samples AND the 3 classes); for an older shap returning a list
                of 3 (192, 19) arrays np.array() gives (3, 192, 19) and the
                canonical code averages axes (0, 1): the same quantity.
CIFRE_CANONICE.md has NO SHAP section and models/frozen_meta_20260331.json
records no shap version. The reference top-5 is therefore read off Figure 2
(bar lengths 0.061 / 0.027 / 0.0205 / 0.0192 / 0.0191): aw_hr_mean,
hrv_balance, aw_active_calories, readiness_score, activity_score.

Checks (exit 1 on any FAIL):
1. rf_revision.joblib, imputer_revision.joblib, scaler_revision.joblib load;
   RF hyperparameters are the canonical ones (100 / 10 / 5 / seed 42).
2. common.canonical_split(): 958 / 766 / 192 days, test 44/140/8,
   train 158/570/38, test dates == test_predictions_rf.csv dates.
3. predict_proba of the joblib on the canonical imputed test reproduces
   test_predictions_rf.csv (max abs diff < 1e-9); imputer_revision.joblib
   gives the same test matrix as the split's imputer.
4. mean |SHAP| with EXACTLY the canonical procedure, under -W error inside
   the script (0 warnings): additivity sum(SHAP) + expected_value == proba,
   top-5 == Figure 2, aw_hr_mean rank 1; full ranking printed.
5. Versions: python, sklearn, numpy, pandas, imblearn, shap, scipy, matplotlib.

Evidence script, kept. Run from repo root:
    python revision/scripts/_check_t14_inputs.py
"""

from __future__ import annotations

import hashlib
import platform
import sys
import warnings
from pathlib import Path

import imblearn
import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CANON, FEATURES, load_canonical  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
PROBA_COLS = ["p_rest", "p_normal", "p_active"]
FIGURE2_TOP5 = ["aw_hr_mean_lag1", "hrv_balance_lag1", "aw_active_calories_lag1",
                "readiness_score_lag1", "activity_score_lag1"]
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_mean_abs_shap(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Verbatim logic of run_shap() in cap5_compare_algorithms_json_ext.py (bar plot branch)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    sv_array = np.array(shap_values)
    if sv_array.ndim == 3:
        if sv_array.shape[0] == X.shape[0]:
            mean_shap = np.abs(sv_array).mean(axis=(0, 2))
        else:
            mean_shap = np.abs(sv_array).mean(axis=(0, 1))
    elif isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)
    return np.array(mean_shap).flatten(), sv_array, np.asarray(explainer.expected_value)


def main() -> None:
    print("=== 1. joblib artefacts ===")
    rf = joblib.load(RES / "models" / "rf_revision.joblib")
    imputer = joblib.load(RES / "models" / "imputer_revision.joblib")
    scaler = joblib.load(RES / "models" / "scaler_revision.joblib")
    for name in ("rf_revision", "imputer_revision", "scaler_revision"):
        print(f"  {name}.joblib sha256 {sha256(RES / 'models' / (name + '.joblib'))[:16]}")
    check(type(rf).__name__ == "RandomForestClassifier", f"model type {type(rf).__name__}")
    hp = (rf.n_estimators, rf.max_depth, rf.min_samples_leaf, rf.random_state)
    check(hp == (100, 10, 5, 42), f"RF hyperparameters (n_estimators, max_depth, min_samples_leaf, random_state) = {hp}")
    check(list(rf.classes_) == [0, 1, 2], f"classes_ {list(rf.classes_)}")
    check(imputer.strategy == "median" and type(scaler).__name__ == "StandardScaler",
          f"imputer {imputer.strategy}, scaler {type(scaler).__name__} (scaler not used by RF)")

    print("\n=== 2. canonical split ===")
    X, y, dates, s = load_canonical(check=False)
    sizes = dict(n_total=len(X), n_train=len(s.y_train), n_test=len(s.y_test),
                 test_dist=s.y_test.value_counts().sort_index().to_dict())
    check(sizes == CANON, f"sizes {sizes}")
    train_dist = tuple(int((s.y_train == c).sum()) for c in (0, 1, 2))
    check(train_dist == (158, 570, 38), f"train Rest/Normal/Active = {train_dist}")
    check(tuple(s.y_train_sm.value_counts().sort_index()) == (570, 570, 570),
          f"train post-SMOTE {s.y_train_sm.value_counts().sort_index().to_dict()}")
    csv = pd.read_csv(RES / "test_predictions_rf.csv")
    check(list(pd.to_datetime(csv.date)) == list(pd.to_datetime(s.dates_test)), "test dates == csv dates")
    check(list(s.X_test.columns) == FEATURES and len(FEATURES) == 19, "19 lag-1 features in canonical order")

    print("\n=== 3. joblib reproduces test_predictions_rf.csv ===")
    proba = rf.predict_proba(s.X_test)
    diff = float(np.abs(proba - csv[PROBA_COLS].to_numpy()).max())
    check(diff < 1e-9, f"max |predict_proba - csv| = {diff:.3e} < 1e-9")
    check(int((rf.predict(s.X_test) == csv.y_pred.to_numpy()).sum()) == 192, "predict() == csv y_pred on 192/192 days")
    imp_diff = float(np.abs(imputer.transform(X.iloc[766:]) - s.X_test.to_numpy()).max())
    check(imp_diff == 0.0, f"imputer_revision.joblib test matrix == split imputer test matrix (max diff {imp_diff:.1e})")

    print("\n=== 4. canonical SHAP procedure on the test set ===")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean_shap, sv, expected = canonical_mean_abs_shap(rf, s.X_test)
    print(f"  shap_values shape {sv.shape}, expected_value {np.round(expected, 6).tolist()}")
    check(sv.shape == (192, 19, 3), f"shap array shape {sv.shape} == (192, 19, 3) (samples, features, classes)")
    add = float(np.abs(sv.sum(axis=1) + expected[None, :] - proba).max())
    check(add < 1e-9, f"additivity: max |sum_f SHAP + expected - predict_proba| = {add:.3e}")
    check(bool((mean_shap >= 0).all()) and mean_shap.shape == (19,), "mean |SHAP| >= 0, 19 values")
    order = np.argsort(-mean_shap, kind="stable")
    ranking = [FEATURES[i] for i in order]
    print("  reproduced ranking (mean |SHAP| over 192 test days x 3 classes):")
    for r, i in enumerate(order, 1):
        print(f"    {r:>2}. {FEATURES[i]:<28} {mean_shap[i]:.6f}")
    check(ranking[:5] == FIGURE2_TOP5, f"top-5 == Figure 2 top-5 {[f.replace('_lag1', '') for f in FIGURE2_TOP5]}")
    check(ranking[0] == "aw_hr_mean_lag1", "aw_hr_mean rank 1")
    gap45 = mean_shap[order[3]] - mean_shap[order[4]]
    print(f"  note: gap between rank 4 and rank 5 = {gap45:.6f} (readiness_score vs activity_score), "
          f"rank 5 vs rank 6 = {mean_shap[order[4]] - mean_shap[order[5]]:.6f}")

    print("\n=== 5. versions ===")
    print(f"  python {platform.python_version()} | sklearn {sklearn.__version__} | numpy {np.__version__} | "
          f"pandas {pd.__version__} | imblearn {imblearn.__version__} | shap {shap.__version__} | "
          f"scipy {scipy.__version__} | matplotlib {matplotlib.__version__}")
    check(sklearn.__version__ == "1.8.0" and np.__version__ == "2.4.3", "frozen sklearn 1.8.0 / numpy 2.4.3")

    print(f"\n{'ALL PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
