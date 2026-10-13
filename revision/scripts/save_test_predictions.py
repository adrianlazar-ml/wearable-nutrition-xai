"""
save_test_predictions.py - T0.2: train the three canonical models and save
test-set predictions + class probabilities for reuse in Phase 1
(metrics, McNemar, calibration) without retraining.

Models and preprocessing identical to cap5_compare_algorithms_json_ext.py:
  RF  (100 trees, depth 10, min leaf 5, seed 42)          on imputed+SMOTE
  XGB (100 est, depth 6, lr 0.1, mlogloss, seed 42)       on imputed+SMOTE
  SVM (RBF, C=1, gamma=scale, probability=True, seed 42)  on imputed+scaled+SMOTE

Pass criteria (CIFRE_CANONICE.md, 4 decimals):
  RF  F1 0.5063  acc 0.6406  MCC 0.1416
  XGB F1 0.4760  acc 0.6875  MCC 0.1503
  SVM F1 0.4347  acc 0.6302  MCC 0.0956
Script prints PASS/FAIL per model and exits 1 on any FAIL.

Outputs (revision/results/):
  test_predictions_{rf,xgb,svm}.csv : date, y_true, y_pred, p_rest, p_normal, p_active
  models/{rf,xgb,svm}_revision.joblib + imputer/scaler (for T1.3/T1.4)
  test_predictions_metrics.csv

Run from repo root:  python revision/scripts/save_test_predictions.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SEED, load_canonical  # noqa: E402

EXPECTED = {
    "rf":  dict(f1=0.5063, acc=0.6406, mcc=0.1416),
    "xgb": dict(f1=0.4760, acc=0.6875, mcc=0.1503),
    "svm": dict(f1=0.4347, acc=0.6302, mcc=0.0956),
}


def build_models() -> dict:
    return {
        "rf": RandomForestClassifier(n_estimators=100, random_state=SEED,
                                     max_depth=10, min_samples_leaf=5),
        "xgb": XGBClassifier(n_estimators=100, random_state=SEED, max_depth=6,
                             learning_rate=0.1, eval_metric="mlogloss", verbosity=0),
        "svm": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED, probability=True),
    }


def main(out_dir: Path = ROOT / "revision" / "results") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    X, y, dates, s = load_canonical()
    y_true = s.y_test.to_numpy()

    rows, all_pass = [], True
    for key, model in build_models().items():
        if key == "svm":
            model.fit(s.X_train_scaled_sm, s.y_train_scaled_sm)
            X_te = s.X_test_scaled
        else:
            model.fit(s.X_train_sm, s.y_train_sm)
            X_te = s.X_test
        y_pred = model.predict(X_te)
        proba = model.predict_proba(X_te)      # columns follow model.classes_ = [0,1,2]
        assert list(model.classes_) == [0, 1, 2]

        pred = pd.DataFrame({"date": s.dates_test, "y_true": y_true, "y_pred": y_pred,
                             "p_rest": proba[:, 0], "p_normal": proba[:, 1], "p_active": proba[:, 2]})
        pred.to_csv(out_dir / f"test_predictions_{key}.csv", index=False)
        joblib.dump(model, out_dir / "models" / f"{key}_revision.joblib")

        m = dict(model=key,
                 f1=round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
                 acc=round(accuracy_score(y_true, y_pred), 4),
                 mcc=round(matthews_corrcoef(y_true, y_pred), 4))
        ok = all(abs(m[k] - EXPECTED[key][k]) < 5e-5 for k in ("f1", "acc", "mcc"))
        all_pass &= ok
        m["status"] = "PASS" if ok else f"FAIL (expected {EXPECTED[key]})"
        # NOTE: SVM predict() and predict_proba() can disagree (Platt scaling);
        # y_pred here is predict(), as in the canonical script.
        m["argmax_proba_agrees_with_predict"] = float((proba.argmax(1) == y_pred).mean())
        rows.append(m)
        print(m)

    joblib.dump(s.imputer, out_dir / "models" / "imputer_revision.joblib")
    joblib.dump(s.scaler, out_dir / "models" / "scaler_revision.joblib")
    res = pd.DataFrame(rows)
    res["seed"] = SEED
    res["run_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    res.to_csv(out_dir / "test_predictions_metrics.csv", index=False)
    print("ALL PASS" if all_pass else "SOME FAIL: do not proceed to Phase 1 until fixed")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
