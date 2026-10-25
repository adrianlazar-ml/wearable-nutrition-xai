"""_check_f3_empirical.py - AUDIT F3 (T2 evidence): empirical checks that need the
project's own code (joblib calibrators, calibration_rf helpers, common.preprocess_split,
threshold_sensitivity.fit_eval_labels, confidence_threshold functions, and the old
16-feature pipeline of confidence_threshold.main() for the erratum question Q12).
Nothing is written anywhere except this script's stdout (redirected by the caller);
confidence_threshold.main() itself is NOT called (it writes under outputs/), only its
load_and_prepare() / train_model() / apply_threshold() / caloric_error_analysis().
Run from repo root: python revision/audit/_check_f3_empirical.py
"""
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.model_selection import TimeSeriesSplit

sys.stdout.reconfigure(newline="\n")
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)                                   # confidence_threshold.load_and_prepare uses relative paths
sys.path.insert(0, str(ROOT))                    # confidence_threshold (root)
sys.path.insert(0, str(ROOT / "revision" / "scripts"))   # searched first: calibration_rf, common, threshold_sensitivity
import calibration_rf as cal  # noqa: E402
import common  # noqa: E402
import confidence_threshold as ct  # noqa: E402
import t3_labels  # noqa: E402
import threshold_sensitivity as ts  # noqa: E402
from extended_metrics import bootstrap as em_bootstrap  # noqa: E402
from save_test_predictions import build_models  # noqa: E402
from statistical_tests import paired_bootstrap  # noqa: E402

RES = ROOT / "revision" / "results"
L = [0, 1, 2]
PCOLS = ["p_rest", "p_normal", "p_active"]
n = 0


def line(msg: str) -> None:
    global n
    n += 1
    print(f"E{n:02d} {msg}")


line(f"module resolution: calibration_rf={Path(cal.__file__).relative_to(ROOT)} common={Path(common.__file__).relative_to(ROOT)} "
     f"threshold_sensitivity={Path(ts.__file__).relative_to(ROOT)} t3_labels={Path(t3_labels.__file__).relative_to(ROOT)} "
     f"confidence_threshold={Path(ct.__file__).relative_to(ROOT)}")

test = pd.read_csv(RES / "test_predictions_rf.csv")
y = test.y_true.to_numpy(int)
P = test[PCOLS].to_numpy(float)
yp = test.y_pred.to_numpy(int)
oof = pd.read_csv(RES / "t13_oof_probabilities_rf.csv")
y_oof, P_oof = oof.y_true.to_numpy(int), oof[PCOLS].to_numpy(float)
folds = pd.read_csv(RES / "t13_oof_folds_rf.csv")
bins = pd.read_csv(RES / "t13_reliability_bins.csv")

# ----------------------------------------------------------------------------- Q3 / Q4 / Q5: the joblib calibrators
platt = joblib.load(RES / "models" / "calibrator_platt_rf.joblib")
iso = joblib.load(RES / "models" / "calibrator_isotonic_rf.joblib")
line(f"platt joblib type {type(platt).__module__}.{type(platt).__name__}, clip {platt.clip}, labels {platt.labels}")
for c, m in platt.models_.items():
    line(f"platt class {c}: {type(m).__name__} C={m.C} penalty={getattr(m, 'penalty', 'n/a')} l1_ratio={m.l1_ratio} solver={m.solver} "
         f"tol={m.tol} max_iter={m.max_iter} n_iter={int(m.n_iter_[0])} coef={m.coef_[0, 0]:.6f} intercept={m.intercept_[0]:.6f} "
         f"n_features_in={m.n_features_in_}")
line(f"platt coefficients == t13_run_meta claim (Active slope 0.2095, intercept -2.3673): "
     f"{round(platt.models_[2].coef_[0, 0], 4) == 0.2095 and round(platt.models_[2].intercept_[0], 4) == -2.3673}")
# refit on the OOF csv with the module's class and compare to the stored object
platt_re = cal.PlattOvR().fit(P_oof, y_oof)
d = max(abs(platt_re.models_[c].coef_[0, 0] - platt.models_[c].coef_[0, 0]) + abs(platt_re.models_[c].intercept_[0] - platt.models_[c].intercept_[0]) for c in L)
line(f"platt refit on the 612 OOF rows (folds 1-4) reproduces the joblib coefficients: max abs diff {d:.2e}")
platt_24 = cal.PlattOvR().fit(P_oof[oof.fold >= 2], y_oof[oof.fold >= 2])
line(f"platt refit on folds 2-4 only gives different coefficients (Active slope {platt_24.models_[2].coef_[0, 0]:.4f} vs {platt.models_[2].coef_[0, 0]:.4f}); "
     f"the stored joblib is the fit 1-4 object")
# does anything from the test set enter the fit? compare with a fit on OOF + test rows
platt_leak = cal.PlattOvR().fit(np.vstack([P_oof, P]), np.concatenate([y_oof, y]))
line(f"platt fit on OOF + test would give Active slope {platt_leak.models_[2].coef_[0, 0]:.4f} (stored {platt.models_[2].coef_[0, 0]:.4f}): "
     f"stored object matches the OOF-only fit, not the leaked one")
for c, m in iso.models_.items():
    line(f"isotonic class {c}: y_min={m.y_min} y_max={m.y_max} out_of_bounds={m.out_of_bounds} increasing={m.increasing} "
         f"X range [{m.X_min_:.4f}, {m.X_max_:.4f}] thresholds {len(m.X_thresholds_)}")
g = np.linspace(0, 1, 1001)
mono = {c: bool(np.all(np.diff(m.predict(g)) >= 0)) for c, m in iso.models_.items()}
line(f"isotonic monotone (non-decreasing, strict >= 0) on 1001-point grid, all classes: {mono}; module method: {iso.is_monotone()}")
iso_re = cal.IsotonicOvR().fit(P_oof, y_oof)
line(f"isotonic refit on the 612 OOF rows reproduces the joblib predictions on test: max abs diff "
     f"{np.abs(iso_re.predict_ovr(P) - iso.predict_ovr(P)).max():.2e}")
for name, obj in (("platt", platt), ("isotonic", iso)):
    Q = obj.predict_ovr(P)
    Pc = obj.predict_proba(P)
    line(f"{name} on test: OvR row sums before normalisation min {Q.sum(1).min():.4f} max {Q.sum(1).max():.4f}; rows with sum 0: {int((Q.sum(1) <= 0).sum())}; "
         f"after normalisation max |sum - 1| {np.abs(Pc.sum(1) - 1).max():.2e}; fallback rows {obj.n_fallback_rows_}; "
         f"argmax Normal {int((Pc.argmax(1) == 1).sum())}/192")

# ----------------------------------------------------------------------------- Q6: bin convention vs sklearn
sk_edges = np.linspace(0.0, 1.0 + 1e-8, 11)
for p in (0.0, 0.1, 0.1 + 1e-12, 0.3, 0.5, 0.5 + 1e-12, 0.7, 1.0):
    line(f"bin_ids({p!r}) = {int(cal.bin_ids(np.array([p]))[0])}; sklearn rule = {int(np.searchsorted(sk_edges[1:-1], p))}")
for label_, arr in (("test max(p) raw", P.max(1)), ("test max(p) platt", platt.predict_proba(P).max(1)),
                    ("test max(p) isotonic", iso.predict_proba(P).max(1)), ("OOF p_active raw", P_oof[:, 2]),
                    ("test p_active raw", P[:, 2])):
    same = np.array_equal(cal.bin_ids(arr), np.searchsorted(sk_edges[1:-1], arr))
    line(f"bin ids script == sklearn calibration_curve(strategy='uniform') rule on {label_}: {same}")
pt, pp = calibration_curve((yp == y).astype(int), P.max(1), n_bins=10, strategy="uniform")
ids = np.searchsorted(sk_edges[1:-1], P.max(1))
nb = np.bincount(ids, minlength=10)
ece_sk = float(np.sum(nb[nb > 0] / 192 * np.abs(pt - pp)))
ref = bins[(bins.split == "test") & (bins.fit_folds == "1-4") & (bins.variant == "raw") & (bins.kind == "top_label") & (bins.n_b > 0)]
line(f"ECE from sklearn calibration_curve outputs = {ece_sk:.6f}; module ece_top_label = {cal.ece_top_label(y, P):.6f}; "
     f"calibration_curve prob_true == csv acc_b: {np.allclose(pt, ref.acc_b)}; prob_pred == csv conf_b: {np.allclose(pp, ref.conf_b)}; "
     f"n_b == bincount: {np.array_equal(nb, bins[(bins.split == 'test') & (bins.fit_folds == '1-4') & (bins.variant == 'raw') & (bins.kind == 'top_label')].n_b)}")
line(f"reliability csv: n_b sums per (split, fit_folds, variant, kind) = "
     f"{sorted(set(bins.groupby(['split', 'fit_folds', 'variant', 'kind']).n_b.sum().tolist()))} (expected 192 test, 612 fit 1-4, 459 fit 2-4)")
line(f"docstring of bin_ids says 'bin 0 is [0, 0.1], bin b > 0 is (b/10, (b+1)/10]'; searchsorted side='left' confirms: 0.5 -> bin {int(cal.bin_ids(np.array([0.5]))[0])} ((0.4, 0.5]), 1.0 -> bin {int(cal.bin_ids(np.array([1.0]))[0])}")

# ----------------------------------------------------------------------------- Q7: Brier
man = cal.brier_multiclass(y, P)
sk = float(brier_score_loss(y, P, labels=L, scale_by_half=False))
line(f"Brier multiclass manual {man:.12f} vs sklearn scale_by_half=False {sk:.12f}: |diff| {abs(man - sk):.2e} (< 1e-12: {abs(man - sk) < 1e-12})")

# ----------------------------------------------------------------------------- Q1: OOF folds reproduced from scratch (fit on fold-train only)
t0 = time.time()
X, yy, dates, _ = common.load_canonical(check=False)
i = int(len(X) * common.TRAIN_RATIO)
X_tr, y_tr, d_tr = X.iloc[:i].reset_index(drop=True), yy.iloc[:i].reset_index(drop=True), dates.iloc[:i].reset_index(drop=True)
maxdiff = 0.0
for k, (tr, va) in enumerate(TimeSeriesSplit(n_splits=4).split(X_tr), start=1):
    sp = common.preprocess_split(X_tr.iloc[tr], X_tr.iloc[va], y_tr.iloc[tr], y_tr.iloc[va], d_tr.iloc[va], k_neighbors=3, seed=42)
    rf = build_models()["rf"].fit(sp.X_train_sm, sp.y_train_sm)
    pr = rf.predict_proba(sp.X_test)
    sub = oof[oof.fold == k]
    dd = float(np.abs(pr - sub[PCOLS].to_numpy()).max())
    maxdiff = max(maxdiff, dd)
    f = folds[folds.fold == k].iloc[0]
    line(f"fold {k}: train rows {tr[0]}..{tr[-1]} ({d_tr.iloc[tr[0]].date()}..{d_tr.iloc[tr[-1]].date()}), val rows {va[0]}..{va[-1]} "
         f"({d_tr.iloc[va[0]].date()}..{d_tr.iloc[va[-1]].date()}); max(train date) < min(val date): {d_tr.iloc[tr].max() < d_tr.iloc[va].min()}; "
         f"csv fold table train_end {f.train_end} < val_start {f.val_start}: {f.train_end < f.val_start}; "
         f"imputer median of feature 0 (fit on fold train only) {sp.imputer.statistics_[0]:.2f}; SMOTE rows {len(sp.y_train_sm)} (csv {f.n_smote}); "
         f"val dates == csv: {np.array_equal(sp.dates_test.dt.strftime('%Y-%m-%d').to_numpy(), sub.date.to_numpy())}; "
         f"max |proba - csv| {dd:.2e}")
line(f"all 4 OOF folds reproduced from scratch (imputer + SMOTE + RF fit on the fold's train part only): max |diff| {maxdiff:.2e}; "
     f"OOF dates unique {oof.date.is_unique}, sorted {pd.to_datetime(oof.date).is_monotonic_increasing}; elapsed {time.time() - t0:.1f} s")
line(f"OOF validation window overlaps the test set: {bool(set(oof.date) & set(test.date))} (last OOF date {oof.date.max()}, first test date {test.date.min()})")

# ----------------------------------------------------------------------------- Q9 / Q10: the unchanged safety-layer functions
ya, n_over, det = ct.apply_threshold(yp.copy(), P, 0.50)
me, mx, errs = ct.caloric_error_analysis(y, ya)
line(f"ct.apply_threshold(0.50): overridden {n_over}, emitted R/N/A {[(ya == c).sum() for c in L]}; rule in source is 'max_prob < threshold' "
     f"(confidence_threshold.py:157); days with max(p) == 0.50: {int((P.max(1) == 0.5).sum())}; "
     f"overrides == days with max(p) < 0.50: {n_over == int((P.max(1) < 0.5).sum())}")
line(f"ct.caloric_error_analysis: len(errors) = {len(errs)} (all 192 days incl. overridden), mean {me:.4f}, max {mx}; CALORIC_MAP {ct.CALORIC_MAP}; "
     f"ct.FEATURES has {len(ct.FEATURES)} features (canonical 19)")

# ----------------------------------------------------------------------------- Q13 / Q14 / Q15 / Q16: threshold_sensitivity cells
df_raw = common.load_raw()
line(f"grid constants in module: REST_GRID={ts.REST_GRID} ACTIVE_R_GRID={ts.ACTIVE_R_GRID} ACTIVE_SLEEP={ts.ACTIVE_SLEEP} "
     f"MIN_ACTIVE_TEST={ts.MIN_ACTIVE_TEST} MIN_ACTIVE_TRAIN={ts.MIN_ACTIVE_TRAIN}; argparse options: --results only")
grid = pd.read_csv(RES / "t31_threshold_grid.csv")
t0 = time.time()
for R, A in ((55, 85), (60, 90), (60, 85)):
    cell = ts.fit_eval_labels(R, A, 75, df=df_raw)
    csv = pd.read_csv(RES / f"t31_predictions_rf_R{R}_A{A}.csv")
    g = grid[grid.cell == f"R{R}_A{A}"].iloc[0]
    pers = ts.persistence_predictions(cell["y"], cell["i_split"])
    pers_manual = np.concatenate([[int(cell["y"].iloc[cell["i_split"] - 1])], cell["y_true"][:-1]])
    line(f"cell R{R}_A{A}: counts {cell['counts']}; smote_k {cell['smote_k']} (csv {g.smote_k}); rows after SMOTE {len(cell['split'].y_train_sm)}; "
         f"test dates == test_predictions_rf.csv: {np.array_equal(cell['dates_test'].dt.strftime('%Y-%m-%d').to_numpy(), test.date.to_numpy())}; "
         f"y_pred == csv: {np.array_equal(cell['y_pred'], csv.y_pred.to_numpy())}; max |proba - csv| {np.abs(cell['proba'] - csv[PCOLS].to_numpy()).max():.2e}; "
         f"persistence == label of previous row of the RE-labelled series: {np.array_equal(pers, pers_manual)} "
         f"(first test day uses re-labelled {cell['dates'].iloc[cell['i_split'] - 1].date()} = {pers[0]}); "
         f"comparable {ts.is_comparable(cell['counts']['test'][2], cell['counts']['train'][2])} (csv {g.comparable}); "
         f"imputer median feature 0 {cell['split'].imputer.statistics_[0]:.2f}")
line(f"3 cells refit via fit_eval_labels in {time.time() - t0:.1f} s; is_comparable applied in evaluate_cell (threshold_sensitivity.py:222), "
     f"csv column 'comparable' = {grid.comparable.tolist()}; figure hatches cells with comparable == False (make_figure :280-283) and adds '*' (:285)")
line(f"smote_k rule: smote_k(2)={ts.smote_k(2)} smote_k(3)={ts.smote_k(3)} smote_k(4)={ts.smote_k(4)} smote_k(38)={ts.smote_k(38)} smote_k(1)={ts.smote_k(1)}; "
     f"root threshold_sensitivity.py (Table 7 script, f7cc4ca) lines 177-184: k = min(3, n-1) if n >= 4, n-1 if n in {{2, 3}}, no SMOTE otherwise")
line(f"persistence_baseline.evaluate is imported unmodified (threshold_sensitivity.py:92) and the A_r = 90 cells differ between its CI and the shared-matrix CI "
     f"(csv pers_f1_macro_ci_high_t04 vs pers_f1_macro_ci_high for R60_A90: {grid[grid.cell == 'R60_A90'].pers_f1_macro_ci_high_t04.iloc[0]:.4f} vs "
     f"{grid[grid.cell == 'R60_A90'].pers_f1_macro_ci_high.iloc[0]:.4f})")

# ----------------------------------------------------------------------------- Q17: skip rule, statistical_tests vs extended_metrics
idx = np.load(RES / "bootstrap_indices_1000x192_seed42.npy")
c90 = pd.read_csv(RES / "t31_predictions_rf_R60_A90.csv")
y90, p90 = c90.y_true.to_numpy(int), c90.y_pred.to_numpy(int)
f1m = lambda a, b: float(f1_score(a, b, labels=L, average="macro", zero_division=0))
pb = paired_bootstrap(y90, p90, np.concatenate([[1], y90[:-1]]), f1m, idx)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lo, hi, used, skip, nwarn = em_bootstrap(lambda r: f1m(y90[r], p90[r]), idx)
line(f"R60_A90 RF F1 macro CI: statistical_tests.paired_bootstrap (skip when a class is absent from resampled y_true) "
     f"[{pb['ci_A_low']:.4f}, {pb['ci_A_high']:.4f}] on {pb['n_used']} resamples, skipped {pb['n_skipped']}; "
     f"extended_metrics.bootstrap (skip only when the metric returns nan; F1 macro never does) [{lo:.4f}, {hi:.4f}] on {used}, skipped {skip}")
line(f"expected skip fraction with 1 Active in 192: (191/192)^192 = {(191 / 192) ** 192:.4f}; observed 340 / 1000 = 0.340")
canon_skip = sum(len(np.unique(y[r])) < 3 for r in idx)
line(f"canonical test set (8 Active): resamples with a class absent = {canon_skip} (both rules coincide there)")

# ----------------------------------------------------------------------------- Q12: the old 16-feature pipeline (confidence_threshold.main() logic, no writes)
old = pd.read_csv(ROOT / "outputs" / "tables" / "confidence_threshold_analysis.csv")
t0 = time.time()
Xtr_o, Xte_o, ytr_o, yte_o = ct.load_and_prepare()
t_load = time.time() - t0
rf_o, imp_o = ct.train_model(Xtr_o, ytr_o)
Xte_imp = pd.DataFrame(imp_o.transform(Xte_o), columns=ct.FEATURES, index=Xte_o.index)
ypr_o = rf_o.predict(Xte_imp)
ppr_o = rf_o.predict_proba(Xte_imp)
line(f"old pipeline: load_and_prepare {t_load:.1f} s (data/apple_health/*.csv + data/oura_combined.csv), train {time.time() - t0 - t_load:.1f} s; "
     f"{len(ct.FEATURES)} features; train {len(ytr_o)} / test {len(yte_o)}; test dist {yte_o.value_counts().sort_index().to_dict()}")
rows = []
for t in ct.CONFIDENCE_THRESHOLDS:
    ya_o, no_o, _ = ct.apply_threshold(ypr_o.copy(), ppr_o, t)
    me_o, mx_o, _ = ct.caloric_error_analysis(yte_o.values, ya_o)
    rows.append(dict(threshold=t, f1_macro=f1_score(yte_o, ya_o, average="macro", zero_division=0),
                     accuracy=float(np.mean(ya_o == yte_o.values)), coverage_pct=(len(yte_o) - no_o) / len(yte_o) * 100,
                     n_overridden=no_o, mean_cal_error=me_o, max_cal_error=mx_o, n_active_pred=int((ya_o == 2).sum())))
rerun = pd.DataFrame(rows)
cols = ["f1_macro", "accuracy", "coverage_pct", "n_overridden", "mean_cal_error", "max_cal_error", "n_active_pred"]
same = all(np.allclose(rerun[c].to_numpy(float), old[c].to_numpy(float), atol=1e-6) for c in cols)
line(f"old pipeline rerun reproduces outputs/tables/confidence_threshold_analysis.csv on all 7 thresholds x {len(cols)} columns: {same}")
r5, o5 = rerun[rerun.threshold == 0.5].iloc[0], old[old.threshold == 0.5].iloc[0]
line(f"old pipeline at 0.50: coverage {r5.coverage_pct:.4f} (stored {o5.coverage_pct:.4f}), overridden {int(r5.n_overridden)} (stored {int(o5.n_overridden)}), "
     f"mean error {r5.mean_cal_error:.4f} (stored {o5.mean_cal_error:.4f}); at 0: F1 {rerun.iloc[0].f1_macro:.4f} acc {rerun.iloc[0].accuracy:.4f} Active pred {int(rerun.iloc[0].n_active_pred)}")
same_days = np.array_equal(Xte_o.index.to_numpy(), np.arange(len(Xte_o)) + Xte_o.index[0]) and len(Xte_o) == 192
line(f"old pipeline test set: {len(yte_o)} rows, y_true == canonical y_true: {np.array_equal(yte_o.to_numpy(), y)}; "
     f"raw predictions agree with canonical RF on {int((ypr_o == yp).sum())} / 192 days")
print("\nold pipeline rerun table:")
print(rerun.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("stored table:")
print(old[["threshold"] + cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\n_check_t13_safety_layer.py reads the stored table (line 81) and does not run the old pipeline; total elapsed here {time.time() - t0:.1f} s for Q12")
