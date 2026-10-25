"""
recompute_f4.py  (audit slice F4, 2026-10-25)
=============================================
Independent recomputation of the T1.4 / T1.5 / T3.2 figures audited in
AUDIT_F4_provenance_shap_ablation_agreement.md. Imports numpy, pandas, scipy,
scikit-learn, joblib and shap only; nothing from revision/scripts/ and nothing
from the repository root.

Inputs (read only):
  models/dataset_snapshot_20260331.csv            958 dates, 19 lag-1 features, nutrition_label
  revision/results/models/rf_revision.joblib      canonical RF
  revision/results/models/imputer_revision.joblib canonical median imputer
  revision/results/test_predictions_rf.csv        full-RF test predictions
  revision/results/t15_predictions_logo_G5_activity_energy.csv
  revision/results/bootstrap_indices_1000x192_seed42.npy (compared with a regenerated matrix)
  data/oura_combined.csv, apple_watch_api/apple_health_daily.csv   day-t values for T3.2

Targets (CIFRE_REVIZIE.md, rows T1.4 / T1.5 / T3.2):
  SHAP top-5 mean |SHAP|: aw_hr_mean 0.0611, hrv_balance 0.0273, aw_active_calories 0.0205,
      readiness_score 0.0192, activity_score 0.0191; additivity
  LOGO G5: F1 macro 0.3780; delta full - LOGO +0.1284 [0.0164, 0.2388]
  aw_hr_mean: medians 91.95 / 86.02 / 81.32; Cliff Active-Rest -0.840 [-0.912, -0.749];
      H 182.99, epsilon2 0.1926
  aw_active_calories: Cliff Active-Rest -0.420 [-0.574, -0.246]

Run from the repo root:  python -I revision/audit/recompute_f4.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"

FEATURES = [f"{f}_lag1" for f in [
    "readiness_score", "sleep_score", "hrv_mean", "hrv_balance", "recovery_index", "rhr_lowest", "rhr_avg",
    "activity_score", "steps_oura", "respiratory_rate", "temp_deviation", "aw_hr_mean", "aw_hrv_mean", "aw_steps",
    "aw_active_calories", "aw_spo2", "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate"]]
N_TRAIN, N_TEST = 766, 192
results = []


def report(name, got, target, nd=4):
    ok = round(got, nd) == round(target, nd)
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'} {name}: recomputed {got:.6f} vs target {target:.4f} ({nd} dp)")
    return ok


# ----------------------------------------------------------------------------- 1. SHAP reference
print("=== 1. SHAP reference (rf_revision.joblib on the imputed test set) ===")
snap = pd.read_csv(ROOT / "models" / "dataset_snapshot_20260331.csv")
X = snap[FEATURES]
y = snap["nutrition_label"].astype(int).to_numpy()
assert len(snap) == 958 and list(X.columns) == FEATURES
rf = joblib.load(RES / "models" / "rf_revision.joblib")
imputer = joblib.load(RES / "models" / "imputer_revision.joblib")
X_test = pd.DataFrame(imputer.transform(X.iloc[N_TRAIN:]), columns=FEATURES)
y_test = y[N_TRAIN:]
tp = pd.read_csv(RES / "test_predictions_rf.csv")
proba = rf.predict_proba(X_test)
print(f"  test rows {len(X_test)}; NaN after imputation {int(X_test.isna().sum().sum())}; "
      f"predict_proba vs test_predictions_rf.csv max diff {np.abs(proba - tp[['p_rest', 'p_normal', 'p_active']].to_numpy()).max():.2e}; "
      f"y_true == csv {np.array_equal(y_test, tp.y_true.to_numpy())}")
ex = shap.TreeExplainer(rf)
sv = np.asarray(ex.shap_values(X_test))
print(f"  shap {shap.__version__}; shap_values shape {sv.shape}")
mean_abs = np.abs(sv).mean(axis=(0, 2))
order = np.argsort(-mean_abs, kind="stable")
print("  top-5: " + "; ".join(f"{FEATURES[i].replace('_lag1', '')} {mean_abs[i]:.4f}" for i in order[:5]))
targets = [("aw_hr_mean", 0.0611), ("hrv_balance", 0.0273), ("aw_active_calories", 0.0205),
           ("readiness_score", 0.0192), ("activity_score", 0.0191)]
for k, (name, val) in enumerate(targets):
    got_name = FEATURES[order[k]].replace("_lag1", "")
    ok_name = got_name == name
    results.append(ok_name)
    print(f"  {'PASS' if ok_name else 'FAIL'} rank {k + 1} is {name} (got {got_name})")
    report(f"mean|SHAP| {name}", float(mean_abs[order[k]]), val)
add = np.abs(sv.sum(axis=1) + np.asarray(ex.expected_value)[None, :] - proba).max()
ok = add < 1e-12
results.append(ok)
print(f"  {'PASS' if ok else 'FAIL'} additivity: max |sum_f SHAP + expected_value - predict_proba| = {add:.3e} (< 1e-12)")

# ----------------------------------------------------------------------------- 2. LOGO G5
print("\n=== 2. LOGO G5 from the stored prediction files ===")
g5 = pd.read_csv(RES / "t15_predictions_logo_G5_activity_energy.csv")
assert np.array_equal(g5.y_true.to_numpy(), tp.y_true.to_numpy()) and list(g5.date) == list(tp.date)
yt = tp.y_true.to_numpy()
pa, pb = tp.y_pred.to_numpy(), g5.y_pred.to_numpy()


def f1m(a, b):
    return float(f1_score(a, b, labels=[0, 1, 2], average="macro", zero_division=0))


f_full, f_g5 = f1m(yt, pa), f1m(yt, pb)
report("F1 macro full", f_full, 0.5063)
report("F1 macro LOGO G5", f_g5, 0.3780)
report("delta full - G5", f_full - f_g5, 0.1284)
idx = np.random.default_rng(42).integers(0, N_TEST, size=(1000, N_TEST))
stored = np.load(RES / "bootstrap_indices_1000x192_seed42.npy")
print(f"  regenerated matrix == stored npy: {np.array_equal(idx, stored)}")
diffs = []
skipped = 0
for row in idx:
    if len(np.unique(yt[row])) < 3:
        skipped += 1
        continue
    diffs.append(f1m(yt[row], pa[row]) - f1m(yt[row], pb[row]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"  resamples used {len(diffs)}, skipped {skipped}")
report("delta CI low", float(lo), 0.0164)
report("delta CI high", float(hi), 0.2388)

# ----------------------------------------------------------------------------- 3. T3.2 aw_hr_mean / aw_active_calories
print("\n=== 3. T3.2 from the daily frame (day-t values on the 958 labelled dates) ===")
oura = pd.read_csv(ROOT / "data" / "oura_combined.csv")
aw = pd.read_csv(ROOT / "apple_watch_api" / "apple_health_daily.csv")
aw["aw_active_calories"] = (aw["aw_active_energy_kj"] / 4.184).round(1)
daily = oura.merge(aw[["date", "aw_hr_mean", "aw_hrv_mean", "aw_active_calories"]], on="date", how="left").set_index("date")
frame = pd.DataFrame({"date": snap.date, "label": y})
for v in ["aw_hr_mean", "aw_hrv_mean", "aw_active_calories"]:
    frame[v] = daily[v].reindex(snap.date).to_numpy(dtype=float)
print(f"  daily frame rows {len(daily)}; labelled {len(frame)}; label distribution {np.bincount(y).tolist()}")


def cliff(x, y_):
    x = np.asarray(x, float)[:, None]
    y_ = np.asarray(y_, float)[None, :]
    return float(((x > y_).sum() - (x < y_).sum()) / (x.size * y_.size))


def kw_manual(groups):
    pooled = np.concatenate(groups)
    n = len(pooled)
    r = stats.rankdata(pooled)
    h = 0.0
    s = 0
    for g in groups:
        rg = r[s:s + len(g)]
        s += len(g)
        h += len(g) * (rg.mean() - (n + 1) / 2) ** 2
    h *= 12.0 / (n * (n + 1))
    _, cnt = np.unique(pooled, return_counts=True)
    corr = 1.0 - (cnt ** 3 - cnt).sum() / (n ** 3 - n)
    return h / corr, n


# The script draws from ONE default_rng(42) sequentially over variables (aw_hr_mean, aw_hrv_mean,
# aw_active_calories, ...) and pairs ((2,0), (2,1), (1,0)), 1000 resamples each, x-indices then
# y-indices per resample. The same consumption order is mirrored here so that the CI is exact.
rng = np.random.default_rng(42)
ci = {}
for v in ["aw_hr_mean", "aw_hrv_mean", "aw_active_calories"]:
    g = {c: frame.loc[(frame.label == c) & frame[v].notna(), v].to_numpy(float) for c in (0, 1, 2)}
    for a, b in [(2, 0), (2, 1), (1, 0)]:
        vals = np.empty(1000)
        xa, yb = g[a], g[b]
        for k in range(1000):
            vals[k] = cliff(xa[rng.integers(0, len(xa), len(xa))], yb[rng.integers(0, len(yb), len(yb))])
        ci[(v, a, b)] = tuple(np.percentile(vals, [2.5, 97.5]))
    if v == "aw_hrv_mean":
        continue
    med = {c: float(np.median(g[c])) for c in (0, 1, 2)}
    h_sc, p_sc = stats.kruskal(g[0], g[1], g[2])
    h_m, n = kw_manual([g[0], g[1], g[2]])
    cd = cliff(g[2], g[0])
    print(f"  {v}: n {len(g[0])}/{len(g[1])}/{len(g[2])}; medians R/N/A {med[0]:.2f} / {med[1]:.2f} / {med[2]:.2f}; "
          f"H scipy {h_sc:.4f} (manual tie-corrected {h_m:.4f}), p {p_sc:.2e}, eps2 {h_sc / (n - 1):.4f}; "
          f"Cliff A-R {cd:.4f} [{ci[(v, 2, 0)][0]:.4f}, {ci[(v, 2, 0)][1]:.4f}]")
    if v == "aw_hr_mean":
        report("median Rest", med[0], 91.95, 2)
        report("median Normal", med[1], 86.02, 2)
        report("median Active", med[2], 81.32, 2)
        report("H", float(h_sc), 182.99, 2)
        report("epsilon2", float(h_sc / (n - 1)), 0.1926)
        report("Cliff A-R", cd, -0.840, 3)
        report("Cliff CI low", ci[(v, 2, 0)][0], -0.912, 3)
        report("Cliff CI high", ci[(v, 2, 0)][1], -0.749, 3)
    else:
        report("Cliff A-R (aw_active_calories)", cd, -0.420, 3)
        report("Cliff CI low (aw_active_calories)", ci[(v, 2, 0)][0], -0.574, 3)
        report("Cliff CI high (aw_active_calories)", ci[(v, 2, 0)][1], -0.246, 3)
# robustness: an unrelated rng gives the CI to the second decimal
rng2 = np.random.default_rng(20261025)
g = {c: frame.loc[(frame.label == c) & frame["aw_hr_mean"].notna(), "aw_hr_mean"].to_numpy(float) for c in (0, 2)}
vals = np.array([cliff(g[2][rng2.integers(0, len(g[2]), len(g[2]))], g[0][rng2.integers(0, len(g[0]), len(g[0]))]) for _ in range(1000)])
print(f"  aw_hr_mean Cliff A-R CI with an independent rng (seed 20261025): [{np.percentile(vals, 2.5):.3f}, {np.percentile(vals, 97.5):.3f}]")
# Dunn-Holm Active-Rest for aw_hr_mean (tie-corrected z, two-sided, Holm over 3 pairs)
g3 = [frame.loc[(frame.label == c) & frame["aw_hr_mean"].notna(), "aw_hr_mean"].to_numpy(float) for c in (0, 1, 2)]
pooled = np.concatenate(g3)
N = len(pooled)
r = stats.rankdata(pooled)
_, cnt = np.unique(pooled, return_counts=True)
var = N * (N + 1) / 12 - (cnt ** 3 - cnt).sum() / (12 * (N - 1))
means = [r[sum(len(x) for x in g3[:i]):sum(len(x) for x in g3[:i + 1])].mean() for i in range(3)]
p_raw = {}
for (a, b) in [(2, 0), (2, 1), (1, 0)]:
    z = (means[a] - means[b]) / np.sqrt(var * (1 / len(g3[a]) + 1 / len(g3[b])))
    p_raw[(a, b)] = 2 * stats.norm.sf(abs(z))
ps = np.array(list(p_raw.values()))
o = np.argsort(ps)
adj = np.empty(3)
run = 0.0
for k, i in enumerate(o):
    run = max(run, min(1.0, (3 - k) * ps[i]))
    adj[i] = run
stored_t = pd.read_csv(RES / "t32_tests.csv").set_index("variable").loc["aw_hr_mean"]
print(f"  aw_hr_mean Dunn-Holm p Active-Rest {adj[0]:.3e} (stored {stored_t['dunn_p_holm_active_rest']:.3e}); "
      f"Active-Normal {adj[1]:.3e} (stored {stored_t['dunn_p_holm_active_normal']:.3e}); "
      f"Normal-Rest {adj[2]:.3e} (stored {stored_t['dunn_p_holm_normal_rest']:.3e})")

print(f"\n{'PASS' if all(results) else 'FAIL'}: {sum(results)} / {len(results)} checks")
sys.exit(0 if all(results) else 1)
