"""
recompute_f5.py - independent recomputation for audit slice F5 (T2.1 extra models, T2.2 search).

Imports: numpy, pandas, scipy, scikit-learn, statsmodels, lightgbm, imbalanced-learn, joblib.
Nothing from revision/scripts/ or the repo root is imported; no torch.

Inputs (read only):
  revision/results/test_predictions_{rf,xgb,svm}.csv, t21_predictions_{lgbm,catboost,tabnet,ftt}.csv,
  t22_predictions_{rf,xgb,svm,lgbm,catboost,tabnet,ftt}.csv, t21_extra_models.csv, t21_comparisons.csv,
  t22_cv_results.csv, t22_comparisons.csv, t22_mcnemar.csv, t22_test_results.csv, t22_search_space.json,
  t22_run_meta_test.json, bootstrap_indices_1000x192_seed42.npy (compared with a regenerated matrix),
  models/dataset_snapshot_20260331.csv, revision/results/models/imputer_revision.joblib.

Sections: R1 T2.1 points and CIs; R2 McNemar + Holm (T2.1 and T2.2); R3 T2.2 CV selection and rank;
R4 T2.2 tuned points and the two LightGBM deltas, comparison labels (a)-(d); R5 search space
regeneration and fold bounds; R6 independent LightGBM default refit from the snapshot.
Run:  python -I revision/audit/recompute_f5.py > revision/audit/recompute_f5_output.txt
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from scipy.stats import binom, chi2, loguniform
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from statsmodels.stats.multitest import multipletests

import sys

sys.stdout.reconfigure(newline="\n")                      # LF output on Windows
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
LABELS = [0, 1, 2]
PROBA = ["p_rest", "p_normal", "p_active"]
N_TEST, N_BOOT, SEED = 192, 1000, 42
FEATURES = ["readiness_score", "sleep_score", "hrv_mean", "hrv_balance", "recovery_index", "rhr_lowest", "rhr_avg",
            "activity_score", "steps_oura", "respiratory_rate", "temp_deviation", "aw_hr_mean", "aw_hrv_mean", "aw_steps",
            "aw_active_calories", "aw_spo2", "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate"]
FEATURES = [f + "_lag1" for f in FEATURES]

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def f1m(y, p) -> float:
    return float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0))


def bacc(y, p) -> float:
    return float(balanced_accuracy_score(y, p))


def paired_boot(y, a, b, fn, idx):
    """diff = fn(A) - fn(B); resamples with a class absent from y[r] are skipped (T1.2 rule)."""
    d, skipped = [], 0
    for r in idx:
        yr = y[r]
        if any((yr == c).sum() == 0 for c in LABELS):
            skipped += 1
            continue
        d.append(fn(yr, a[r]) - fn(yr, b[r]))
    d = np.asarray(d)
    return dict(point=fn(y, a) - fn(y, b), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                prop=float((d > 0).mean()), skipped=skipped)


def mcnemar(y, a, b):
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n11, n00 = int((ca & cb).sum()), int((~ca & ~cb).sum())
    n = n01 + n10
    chi2_cc = (abs(n01 - n10) - 1) ** 2 / n if n > 0 else 0.0
    p_cc = float(chi2.sf(chi2_cc, 1)) if n > 0 else 1.0
    k = min(n01, n10)
    p_exact = float(min(1.0, 2 * binom.cdf(k, n, 0.5))) if n > 0 else 1.0
    return dict(n00=n00, n01=n01, n10=n10, n11=n11, chi2_cc=chi2_cc, p_cc=p_cc, p_exact=p_exact)


def holm(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def load_pred(path: Path):
    df = pd.read_csv(path)
    assert list(df.columns) == ["date", "y_true", "y_pred"] + PROBA and len(df) == N_TEST, path
    return df


print("=" * 100)
print("recompute_f5.py  (independent; no imports from revision/scripts)")
print("=" * 100)

# ------------------------------------------------------------------ inputs and index matrix
rf = load_pred(RES / "test_predictions_rf.csv")
y = rf.y_true.to_numpy(int)
pred = {"rf": rf.y_pred.to_numpy(int)}
proba = {"rf": rf[PROBA].to_numpy(float)}
for m in ("xgb", "svm"):
    d = load_pred(RES / f"test_predictions_{m}.csv")
    check(np.array_equal(d.y_true.to_numpy(int), y), f"test_predictions_{m}.csv: y_true == rf")
    pred[m], proba[m] = d.y_pred.to_numpy(int), d[PROBA].to_numpy(float)
for m in ("lgbm", "catboost", "tabnet", "ftt"):
    d = load_pred(RES / f"t21_predictions_{m}.csv")
    check(np.array_equal(d.y_true.to_numpy(int), y) and list(d.date) == list(rf.date), f"t21_predictions_{m}.csv: y_true and dates == rf")
    pred[m], proba[m] = d.y_pred.to_numpy(int), d[PROBA].to_numpy(float)
    check(np.array_equal(proba[m].argmax(1), pred[m]), f"t21_predictions_{m}.csv: argmax(proba) == y_pred on 192 days")
    check(float(np.abs(proba[m].sum(1) - 1).max()) < 1e-6, f"t21_predictions_{m}.csv: probabilities sum to 1")
for m in ("rf", "xgb", "svm", "lgbm", "catboost", "tabnet", "ftt"):
    d = load_pred(RES / f"t22_predictions_{m}.csv")
    check(np.array_equal(d.y_true.to_numpy(int), y) and list(d.date) == list(rf.date), f"t22_predictions_{m}.csv: y_true and dates == rf")
    pred[m + "_tuned"], proba[m + "_tuned"] = d.y_pred.to_numpy(int), d[PROBA].to_numpy(float)
    agree = float((proba[m + "_tuned"].argmax(1) == pred[m + "_tuned"]).mean())
    if m == "svm":
        print(f"INFO t22_predictions_svm.csv: y_pred = predict(); argmax agreement {agree:.4f} (claimed 0.9062)")
        check(round(agree, 4) == 0.9062, "svm tuned argmax agreement == 0.9062")
    else:
        check(agree == 1.0, f"t22_predictions_{m}.csv: argmax(proba) == y_pred on 192 days")

snap = pd.read_csv(ROOT / "models" / "dataset_snapshot_20260331.csv")
snap["date"] = pd.to_datetime(snap["date"])
snap = snap.sort_values("date").reset_index(drop=True)
lab = snap["nutrition_label"]
check(len(snap) == 958 and lab.notna().all(), "snapshot: 958 labelled rows")
y_all = lab.to_numpy(int)
check(np.array_equal(y_all[766:], y), "snapshot test labels (rows 766-957) == csv y_true")
check(list(snap.date.iloc[766:].dt.strftime("%Y-%m-%d")) == list(rf.date), "snapshot test dates == csv dates")
pers = np.concatenate([[y_all[765]], y[:-1]])
check(np.array_equal(pers, load_pred(RES / "test_predictions_persistence.csv").y_pred.to_numpy(int)),
      "persistence rebuilt (last train label + shifted y_true) == test_predictions_persistence.csv")
pred["persistence"] = pers
pred["majority"] = np.full(N_TEST, 1)

idx = np.random.default_rng(SEED).integers(0, N_TEST, size=(N_BOOT, N_TEST))
saved_idx = np.load(RES / "bootstrap_indices_1000x192_seed42.npy")
check(np.array_equal(idx, saved_idx), "regenerated default_rng(42) index matrix == bootstrap_indices_1000x192_seed42.npy")
n_absent = sum(any((y[r] == c).sum() == 0 for c in LABELS) for r in idx)
print(f"INFO resamples with an absent class: {n_absent} (skip rule has no effect when 0)")

# ------------------------------------------------------------------ R1  T2.1 points and CIs
print("\n--- R1. T2.1 points (F1 macro / balanced accuracy) and delta RF minus model ---")
t21 = pd.read_csv(RES / "t21_extra_models.csv")
g21 = {(r.system, r.metric): r for r in t21.itertuples()}
claims_f1 = {"rf": 0.5063, "catboost": 0.4980, "xgb": 0.4760, "svm": 0.4347, "ftt": 0.4115, "lgbm": 0.4089,
             "persistence": 0.3039, "tabnet": 0.2911, "majority": 0.2811}
points = {}
for m, exp in claims_f1.items():
    v = f1m(y, pred[m])
    points[m] = v
    check(round(v, 4) == exp and abs(v - float(g21[m, "f1_macro"].value)) < 1e-9, f"F1 macro {m:<11} {v:.4f} == claim {exp} == csv")
order = sorted(claims_f1, key=lambda k: -points[k])
check(order == ["rf", "catboost", "xgb", "svm", "ftt", "lgbm", "persistence", "tabnet", "majority"],
      "hierarchy at the point: " + " > ".join(order))
for m in ("rf", "lgbm", "catboost", "tabnet", "ftt"):
    v = bacc(y, pred[m])
    check(abs(v - float(g21[m, "balanced_accuracy"].value)) < 1e-9, f"balanced accuracy {m:<9} {v:.4f} == csv")
comp21 = pd.read_csv(RES / "t21_comparisons.csv").set_index("model")
claims_delta = {"lgbm": (0.0974, -0.0627, 0.2335), "catboost": (0.0083, -0.0841, 0.1131),
                "tabnet": (0.2152, 0.0772, 0.3345), "ftt": (0.0948, -0.0085, 0.1912)}
for m, (dp, lo, hi) in claims_delta.items():
    b = paired_boot(y, pred["rf"], pred[m], f1m, idx)
    row = comp21.loc[m]
    ok = (round(b["point"], 4), round(b["lo"], 4), round(b["hi"], 4)) == (dp, lo, hi) and b["skipped"] == 0 \
        and abs(b["lo"] - row.delta_rf_minus_model_f1_macro_ci_low) < 1e-9 and abs(b["hi"] - row.delta_rf_minus_model_f1_macro_ci_high) < 1e-9
    check(ok, f"delta RF minus {m:<9} F1 {b['point']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] prop {b['prop']:.3f} == claim / csv")
    b2 = paired_boot(y, pred["rf"], pred[m], bacc, idx)
    check(abs(b2["lo"] - row.delta_rf_minus_model_balanced_accuracy_ci_low) < 1e-9 and abs(b2["hi"] - row.delta_rf_minus_model_balanced_accuracy_ci_high) < 1e-9,
          f"delta RF minus {m:<9} balanced acc {b2['point']:+.4f} [{b2['lo']:+.4f}, {b2['hi']:+.4f}] == csv")
    b3 = paired_boot(y, pred[m], pred["persistence"], f1m, idx)
    check(abs(b3["lo"] - row.delta_model_minus_persistence_f1_macro_ci_low) < 1e-9 and abs(b3["hi"] - row.delta_model_minus_persistence_f1_macro_ci_high) < 1e-9,
          f"delta {m:<9} minus persistence F1 {b3['point']:+.4f} [{b3['lo']:+.4f}, {b3['hi']:+.4f}] == csv")

# ------------------------------------------------------------------ R2  McNemar + Holm
print("\n--- R2. McNemar model vs RF (A = model) and Holm over the family of 4 ---")
mc = {m: mcnemar(y, pred[m], pred["rf"]) for m in ("lgbm", "catboost", "tabnet", "ftt")}
t = mc["tabnet"]
check((t["n01"], t["n10"]) == (32, 67) and round(t["p_cc"], 4) == 0.0006, f"RF-TabNet n01 / n10 {t['n01']} / {t['n10']}, chi2_cc {t['chi2_cc']:.4f}, p_cc {t['p_cc']:.4f} == claim 32 / 67, 0.0006")
for m, r in mc.items():
    row = comp21.loc[m]
    check((r["n00"], r["n01"], r["n10"], r["n11"]) == (row.n00, row.n01, row.n10, row.n11) and abs(r["p_cc"] - row.p_cc) < 1e-9 and abs(r["p_exact"] - row.p_exact) < 1e-9,
          f"{m:<9} n00/n01/n10/n11 {r['n00']}/{r['n01']}/{r['n10']}/{r['n11']} p_cc {r['p_cc']:.4f} p_exact {r['p_exact']:.4f} == csv")
order4 = ["lgbm", "catboost", "tabnet", "ftt"]
h_cc = holm([mc[m]["p_cc"] for m in order4])
h_ex = holm([mc[m]["p_exact"] for m in order4])
sm_cc = multipletests([mc[m]["p_cc"] for m in order4], method="holm")[1]
for i, m in enumerate(order4):
    check(abs(h_cc[i] - comp21.loc[m].p_cc_holm) < 1e-9 and abs(h_ex[i] - comp21.loc[m].p_exact_holm) < 1e-9 and abs(h_cc[i] - sm_cc[i]) < 1e-12,
          f"Holm {m:<9} p_cc {h_cc[i]:.4f} / p_exact {h_ex[i]:.4f} == csv (manual == statsmodels)")
check(round(h_cc[2], 4) == 0.0025 and all(h_cc[i] > 0.05 for i in (0, 1, 3)), "only TabNet significant after Holm (0.0025); others > 0.05")

mc22 = pd.read_csv(RES / "t22_mcnemar.csv").set_index("model")
models7 = ["rf", "xgb", "svm", "lgbm", "catboost", "tabnet", "ftt"]
r22 = {m: mcnemar(y, pred[m + "_tuned"], pred["rf"]) for m in models7}
h22 = holm([r22[m]["p_cc"] for m in models7])
h22e = holm([r22[m]["p_exact"] for m in models7])
for i, m in enumerate(models7):
    row = mc22.loc[m]
    check((r22[m]["n01"], r22[m]["n10"]) == (row.n01, row.n10) and abs(r22[m]["p_cc"] - row.p_cc) < 1e-9 and abs(h22[i] - row.p_cc_holm) < 1e-9
          and abs(h22e[i] - row.p_exact_holm) < 1e-9,
          f"T2.2 McNemar {m:<9} tuned vs RF canonical n01/n10 {r22[m]['n01']}/{r22[m]['n10']} p_cc {r22[m]['p_cc']:.4f} Holm(7) {h22[i]:.4f} == csv")
check(all(v > 0.05 for v in h22), "T2.2: no McNemar significant after Holm over 7")

# ------------------------------------------------------------------ R3  T2.2 CV selection
print("\n--- R3. T2.2 CV: fold mean, winners, reference rank ---")
cv = pd.read_csv(RES / "t22_cv_results.csv", keep_default_na=False)
cv["eligible"] = cv["eligible"].astype(str) == "True"
check(len(cv) == 117 and int(cv.eligible.sum()) == 110 and int((~cv.eligible).sum()) == 7, f"cv rows {len(cv)} = 110 eligible + 7 reference")
mean3 = cv[["f1_fold1", "f1_fold2", "f1_fold3"]].astype(float).mean(axis=1)
check(float((mean3 - cv.f1_mean.astype(float)).abs().max()) < 1e-9, "f1_mean == simple mean of the 3 fold scores on all 117 rows")
sd3 = cv[["f1_fold1", "f1_fold2", "f1_fold3"]].astype(float).std(axis=1, ddof=1)
check(float((sd3 - cv.f1_sd.astype(float)).abs().max()) < 1e-9, "f1_sd == sample sd (ddof 1) on all 117 rows")
meta_test = json.load(open(RES / "t22_run_meta_test.json", encoding="utf-8"))
tres = pd.read_csv(RES / "t22_test_results.csv", keep_default_na=False).set_index("model")
winners = {}
for m in models7:
    sub = cv[cv.model == m]
    el = sub[sub.eligible].copy()
    el["cid"] = el.config_id.astype(int)
    el = el.sort_values(["f1_mean", "cid"], ascending=[False, True], kind="stable")
    w = el.iloc[0]
    ref = sub[~sub.eligible].iloc[0]
    rank = 1 + int((el.f1_mean.astype(float) > float(ref.f1_mean)).sum())
    n_ties = int((el.f1_mean.astype(float) == float(w.f1_mean)).sum())
    winners[m] = dict(cid=int(w.cid), f1=float(w.f1_mean), ref=float(ref.f1_mean), rank=rank, n=len(el) + 1, params=json.loads(w.params))
    jw = meta_test["winners"][m]
    same = int(jw["config_id"]) == int(w.cid) and jw["params"] == json.loads(w.params) and abs(float(jw["cv_f1_mean"]) - float(w.f1_mean)) < 1e-9
    same_t = int(tres.loc[m].best_config_id) == int(w.cid) and json.loads(tres.loc[m].best_params) == json.loads(w.params)
    check(same and same_t, f"{m:<9} winner cfg {int(w.cid):>2} mean {float(w.f1_mean):.4f} (ties {n_ties}); ref {float(ref.f1_mean):.4f} rank {rank}/{len(el) + 1}; "
          f"== t22_run_meta_test.json winners == t22_test_results.csv")
rfw = winners["rf"]
check(rfw["rank"] == 3 and rfw["n"] == 21, f"RF reference rank {rfw['rank']} / {rfw['n']} (claim 3 / 21)")
check(round(rfw["f1"] - rfw["ref"], 4) == 0.0085 and round(rfw["f1"], 4) == 0.3680 and round(rfw["ref"], 4) == 0.3595,
      f"RF winner {rfw['f1']:.4f} minus reference {rfw['ref']:.4f} = {rfw['f1'] - rfw['ref']:.4f} (claim 0.0085)")
above = cv[(cv.model == "rf") & cv.eligible & (cv.f1_mean.astype(float) > rfw["ref"])][["config_id", "f1_mean"]]
print("INFO RF eligible configurations above the reference: " + "; ".join(f"cfg {r.config_id} {float(r.f1_mean):.4f}" for r in above.itertuples()))
claim_rank = {"rf": 3, "xgb": 13, "svm": 4, "lgbm": 5, "catboost": 13, "tabnet": 6, "ftt": 2}
check(all(winners[m]["rank"] == claim_rank[m] for m in models7), "reference ranks == CIFRE_REVIZIE row T2.2 (3 / 13 / 4 / 5 / 13 / 6 / 2)")

# ------------------------------------------------------------------ R4  T2.2 tuned test points and deltas
print("\n--- R4. T2.2 tuned test F1 macro, deltas, comparison labels ---")
claims_tuned = {"lgbm": 0.5739, "rf": 0.5683, "catboost": 0.5368, "xgb": 0.5043, "svm": 0.4386, "ftt": 0.3698, "tabnet": 0.3412}
pt = {}
for m, exp in claims_tuned.items():
    v = f1m(y, pred[m + "_tuned"])
    pt[m] = v
    check(round(v, 4) == exp and abs(v - float(tres.loc[m].f1_macro)) < 1e-9, f"F1 macro {m:<9} tuned {v:.4f} == claim {exp} == t22_test_results.csv")
check(sorted(claims_tuned, key=lambda k: -pt[k]) == ["lgbm", "rf", "catboost", "xgb", "svm", "ftt", "tabnet"], "tuned hierarchy: LightGBM > RF > CatBoost > XGB > SVM > FT-T > TabNet")
comp22 = pd.read_csv(RES / "t22_comparisons.csv")
b = paired_boot(y, pred["lgbm_tuned"], pred["rf"], f1m, idx)
check((round(b["point"], 4), round(b["lo"], 4), round(b["hi"], 4)) == (0.0676, 0.0001, 0.1483), f"LightGBM tuned minus RF canonical {b['point']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] (claim +0.0676 [+0.0001, +0.1483])")
b = paired_boot(y, pred["lgbm_tuned"], pred["rf_tuned"], f1m, idx)
check((round(b["point"], 4), round(b["lo"], 4), round(b["hi"], 4)) == (0.0056, -0.0305, 0.0450), f"LightGBM tuned minus RF tuned {b['point']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] (claim +0.0056 [-0.0305, +0.0450])")
b = paired_boot(y, pred["rf_tuned"], pred["rf"], f1m, idx)
check((round(b["point"], 4), round(b["lo"], 4), round(b["hi"], 4)) == (0.0619, 0.0028, 0.1398), f"RF tuned minus RF canonical {b['point']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] (claim +0.0619 [+0.0028, +0.1398])")
# every row of t22_comparisons.csv: the 'other' vector implied by the label reproduces value_other, diff and CI
other_vec = {"rf_canonical": pred["rf"], "rf_tuned": pred["rf_tuned"], "persistence": pred["persistence"]}
for m in models7:
    other_vec[f"{m}_reference"] = pred[m]
n_rows = 0
for r in comp22.itertuples():
    fn = f1m if r.metric == "f1_macro" else bacc
    tuned = pred[f"{r.model}_tuned"]
    bb = paired_boot(y, tuned, other_vec[r.other], fn, idx)
    ok = abs(fn(y, tuned) - r.value_tuned) < 1e-9 and abs(fn(y, other_vec[r.other]) - r.value_other) < 1e-9 and abs(bb["point"] - r.diff_point) < 1e-9 \
        and abs(bb["lo"] - r.ci_low) < 1e-9 and abs(bb["hi"] - r.ci_high) < 1e-9 and abs(bb["prop"] - r.prop_gt_0) < 1e-9
    n_rows += 1
    if not ok:
        check(False, f"t22_comparisons.csv row {r.model} ({r.comparison}, {r.other}, {r.metric}) does not reproduce")
check(n_rows == 54, f"all {n_rows} rows of t22_comparisons.csv reproduce (tuned minus other; label == vector; 7 x 4 x 2 minus rf-c)")
expected_names = {"a": "rf_canonical", "c": "rf_tuned", "d": "persistence"}
lab_ok = all(set(comp22[comp22.comparison == k].other) == {v} for k, v in expected_names.items()) \
    and all(comp22[(comp22.comparison == "b") & (comp22.model == m)].other.iloc[0] == f"{m}_reference" for m in models7)
check(lab_ok, "comparison labels (a) RF canonical / (b) own reference / (c) RF tuned / (d) persistence match the 'other' column")

# ------------------------------------------------------------------ R5  search space regeneration, fold bounds
print("\n--- R5. ParameterSampler(random_state 42) regeneration and TimeSeriesSplit folds ---")
SPACES = {
    "rf": {"n_estimators": [100, 200, 500], "max_depth": [4, 6, 8, 10, None], "min_samples_leaf": [1, 3, 5, 10], "max_features": ["sqrt", 0.5, None]},
    "xgb": {"n_estimators": [100, 200, 500], "max_depth": [2, 3, 4, 6], "learning_rate": loguniform(0.01, 0.3), "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5], "reg_lambda": [1, 5, 10]},
    "svm": {"C": loguniform(0.01, 100), "gamma": loguniform(1e-4, 1)},
    "lgbm": {"n_estimators": [100, 200, 500], "num_leaves": [7, 15, 31], "max_depth": [-1, 4, 6], "learning_rate": loguniform(0.01, 0.3),
             "min_child_samples": [5, 10, 20, 40], "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0, 1, 10]},
    "catboost": {"iterations": [200, 500, 1000], "depth": [4, 6, 8], "learning_rate": loguniform(0.01, 0.3), "l2_leaf_reg": [1, 3, 10], "random_strength": [1, 5]},
}
space = json.load(open(RES / "t22_search_space.json", encoding="utf-8"))
check(sha16(RES / "t22_search_space.json") == "79c1aed12f0151e7", f"t22_search_space.json sha256 {sha16(RES / 't22_search_space.json')} == 79c1aed12f0151e7")
check(space["seed"] == 42 and space["n_iter"] == 20 and space["reduced"] is False, "search space header: seed 42, n_iter 20, not reduced")
n_cfg = 0
for m, sp in SPACES.items():
    regen = [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in c.items()} for c in ParameterSampler(sp, n_iter=20, random_state=42)]
    saved = space["models"][m]["configs"]
    same = len(saved) == 20 and all(r == s for r, s in zip(regen, saved))
    n_cfg += len(saved)
    check(same, f"{m:<9} 20 configurations regenerated == json (exact, floats included)")
check(n_cfg == 100 and all(len(space["models"][m]["configs"]) == 5 for m in ("tabnet", "ftt")), "100 sampled + 2 x 5 grid configurations")
print("INFO tabnet grid config keys in json:", sorted(space["models"]["tabnet"]["configs"][0].keys()), "| reference keys:", sorted(space["models"]["tabnet"]["reference"].keys()))
bounds = []
for tr, va in TimeSeriesSplit(n_splits=3, test_size=154).split(np.zeros(766)):
    bounds.append((int(tr[0]), int(tr[-1]), int(va[0]), int(va[-1])))
check(bounds == [(0, 303, 304, 457), (0, 457, 458, 611), (0, 611, 612, 765)], f"TimeSeriesSplit(3, test_size 154) on 766 rows: {bounds}")
act = [(int((y_all[a:b + 1] == 2).sum()), int((y_all[c:d + 1] == 2).sum())) for a, b, c, d in bounds]
check(act == [(10, 7), (17, 8), (25, 13)], f"Active in train / validation per fold from the snapshot labels: {act}")
counts_in = tuple(int((y_all[:612] == c).sum()) for c in LABELS)
print(f"INFO inner-train label counts (rows 0-611): {counts_in}; min class 25 -> SMOTE k 3; after SMOTE 3 x {max(counts_in)} = {3 * max(counts_in)} (claimed 1,356)")
counts_tr = tuple(int((y_all[:766] == c).sum()) for c in LABELS)
print(f"INFO train label counts (rows 0-765): {counts_tr}; after SMOTE 3 x {max(counts_tr)} = {3 * max(counts_tr)} (claimed 1,710)")
check(3 * max(counts_in) == 1356 and 3 * max(counts_tr) == 1710, "SMOTE row counts 1,356 (inner) and 1,710 (full train)")
print(f"INFO TabNet default batch 1024, drop_last True: batches per epoch = 1356 // 1024 = {1356 // 1024}, 1710 // 1024 = {1710 // 1024}; "
      f"rows seen per epoch 1024 ({1024 / 1356:.1%} of inner-train, {1024 / 1710:.1%} of train); best_epoch 5 -> 5 optimizer steps, 5,120 row visits; "
      f"fold 1 / 2 train after SMOTE 669 / 1008 -> {669 // 1024} / {1008 // 1024} batches")

# ------------------------------------------------------------------ R6  independent LightGBM default refit
print("\n--- R6. LightGBM default refit from the snapshot + imputer_revision.joblib + SMOTE(42, k 3) ---")
X_raw = snap[FEATURES]
imp: SimpleImputer = joblib.load(RES / "models" / "imputer_revision.joblib")
X_tr = pd.DataFrame(imp.transform(X_raw.iloc[:766]), columns=FEATURES)
X_te = pd.DataFrame(imp.transform(X_raw.iloc[766:]), columns=FEATURES)
imp2 = SimpleImputer(strategy="median").fit(X_raw.iloc[:766])
check(np.allclose(imp2.statistics_, imp.statistics_, atol=0, rtol=0), "imputer refit on the snapshot train rows == imputer_revision.joblib statistics")
X_sm, y_sm = SMOTE(random_state=SEED, k_neighbors=3).fit_resample(X_tr, y_all[:766])
check(len(y_sm) == 1710 and tuple(int((y_sm == c).sum()) for c in LABELS) == (570, 570, 570), f"SMOTE train: {len(y_sm)} rows, 570 per class")
lgb = LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1).fit(X_sm, y_sm)
p_lgb = np.asarray(lgb.predict_proba(X_te), dtype=float)
yp_lgb = p_lgb.argmax(1)
check(list(lgb.classes_) == LABELS, "LightGBM classes_ == [0, 1, 2]")
check(np.array_equal(yp_lgb, lgb.predict(X_te).astype(int)), "LightGBM argmax(proba) == predict() on 192 days")
n_diff = int((yp_lgb != pred["lgbm"]).sum())
max_dp = float(np.abs(p_lgb - proba["lgbm"]).max())
check(n_diff == 0, f"LightGBM y_pred identical to t21_predictions_lgbm.csv ({n_diff} days differ; max abs proba diff {max_dp:.2e})")
check(round(f1m(y, yp_lgb), 4) == 0.4089, f"LightGBM refit F1 macro {f1m(y, yp_lgb):.4f} (claim 0.4089)")
if n_diff:
    print("INFO differing days:", [(str(rf.date[i]), int(pred['lgbm'][i]), int(yp_lgb[i])) for i in np.where(yp_lgb != pred["lgbm"])[0]])

print("\n" + "=" * 100)
print(f"{len(fails)} FAIL" if fails else "ALL PASS")
for f in fails:
    print("  FAIL " + f)
