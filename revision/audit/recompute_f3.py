"""recompute_f3.py - AUDIT F3 (T3): independent recomputation of T1.3 (calibration,
safety layer) and T3.1 (threshold grid, cell R55_A85) figures.
Imports only numpy, pandas, scipy, statsmodels, sklearn, imblearn. Nothing from
revision/scripts/ (the calibrator joblib files are NOT loaded here because unpickling
them needs calibration_rf.PlattOvR; the calibrators are refit from the OOF csv instead).
Inputs : revision/results/test_predictions_rf.csv (192 test days, raw RF probabilities),
         revision/results/t13_oof_probabilities_rf.csv (612 OOF rows, calibrator fit data),
         revision/results/t31_predictions_rf_R55_A85.csv (cell R55_A85 test predictions),
         revision/results/t31_threshold_grid.csv (claimed per-cell distributions),
         revision/results/bootstrap_indices_1000x192_seed42.npy (comparison only; the
         matrix used below is regenerated with np.random.default_rng(42)),
         data/oura_combined.csv (raw readiness / sleep scores: the labelling inputs;
         models/dataset_snapshot_20260331.csv holds only lag-1 columns and the canonical
         label, so the raw scores needed to regenerate labels with R = 55 come from the
         Oura file; the snapshot is used to cross-check dates and canonical labels),
         apple_watch_api/apple_health_daily.csv (scratch retraining of cell R55_A85 only).
Run from repo root: python revision/audit/recompute_f3.py
Every comparison is at 4 decimals (round(x, 4) == round(claim, 4)).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss,
                             matthews_corrcoef)

sys.stdout.reconfigure(newline="\n")
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
L = [0, 1, 2]
PCOLS = ["p_rest", "p_normal", "p_active"]
N_BOOT, N_TEST, SEED = 1000, 192, 42
CAL = {0: -200, 1: 0, 2: 200}
results: list[tuple[str, bool, str]] = []


def cmp(name: str, got: float, claim: float, nd: int = 4) -> None:
    ok = round(float(got), nd) == round(float(claim), nd)
    results.append((name, ok, f"{got:.6f} vs {claim}"))
    print(f"{'PASS' if ok else 'FAIL'} {name}: recomputed {got:.6f} claimed {claim}")


def cmp_int(name: str, got: int, claim: int) -> None:
    ok = int(got) == int(claim)
    results.append((name, ok, f"{got} vs {claim}"))
    print(f"{'PASS' if ok else 'FAIL'} {name}: recomputed {got} claimed {claim}")


# ----------------------------------------------------------------------------- inputs
test = pd.read_csv(RES / "test_predictions_rf.csv")
y = test.y_true.to_numpy(int)
P = test[PCOLS].to_numpy(float)
yp = test.y_pred.to_numpy(int)
assert len(test) == N_TEST and np.array_equal(P.argmax(1), yp)
oof = pd.read_csv(RES / "t13_oof_probabilities_rf.csv")
y_oof = oof.y_true.to_numpy(int)
P_oof = oof[PCOLS].to_numpy(float)
print(f"test rows {len(test)}, support {np.bincount(y, minlength=3).tolist()}; OOF rows {len(oof)}, "
      f"support {np.bincount(y_oof, minlength=3).tolist()}, folds {sorted(oof.fold.unique().tolist())}")
cmp_int("OOF rows", len(oof), 612)
cmp_int("OOF Active rows", int((y_oof == 2).sum()), 33)

# ----------------------------------------------------------------------------- bootstrap matrix
idx = np.random.default_rng(SEED).integers(0, N_TEST, size=(N_BOOT, N_TEST))
saved = np.load(RES / "bootstrap_indices_1000x192_seed42.npy")
print("regenerated matrix == saved npy:", np.array_equal(idx, saved), idx.dtype, idx.shape)


def ci(fn, *arrays, skip_missing_class: bool = False):
    vals = []
    n_skip = 0
    for r in idx:
        if skip_missing_class and len(np.unique(arrays[0][r])) < 3:
            n_skip += 1
            continue
        vals.append(fn(*[a[r] for a in arrays]))
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(lo), float(hi), n_skip


# ----------------------------------------------------------------------------- calibration metrics (manual)
# sklearn.calibration.calibration_curve(strategy="uniform") rule: edges linspace(0, 1 + 1e-8, 11),
# bin id = searchsorted(edges[1:-1], p) (side left): a value equal to an interior edge falls in the
# LOWER bin, i.e. bin 0 = [0, 0.1], bin b > 0 = (b/10, (b+1)/10].
EDGES_SK = np.linspace(0.0, 1.0 + 1e-8, 11)


def bin_ids_sk(p):
    return np.searchsorted(EDGES_SK[1:-1], p)


def bin_ids_left_closed(p):
    """Alternative convention [lo, hi): a value equal to an interior edge goes to the UPPER bin."""
    return np.clip(np.digitize(p, np.linspace(0.0, 1.0, 11)[1:-1], right=False), 0, 9)


def ece(conf, hit, binner=bin_ids_sk):
    b = binner(conf)
    tot = 0.0
    gaps = []
    for k in range(10):
        m = b == k
        if m.any():
            gap = abs(hit[m].mean() - conf[m].mean())
            tot += m.sum() / len(conf) * gap
            gaps.append(gap)
    return tot, max(gaps)


def ece_top(yt, Pm, binner=bin_ids_sk):
    return ece(Pm.max(1), (Pm.argmax(1) == yt).astype(float), binner)[0]


def brier_mc(yt, Pm):
    Y = np.eye(3)[yt]
    return float(((Pm - Y) ** 2).sum(1).mean())


def ll(yt, Pm):
    return float(-np.log(Pm[np.arange(len(yt)), yt]).mean())


print("\n=== T1.3 raw RF on test ===")
e_raw, mce_raw = ece(P.max(1), (yp == y).astype(float))
cmp("ECE top-label raw (sklearn-uniform bin rule)", e_raw, 0.1377)
e_alt = ece_top(y, P, bin_ids_left_closed)
print(f"     ECE with the alternative left-closed [lo, hi) bins: {e_alt:.6f} (same at 4 dp: {round(e_alt, 4) == round(e_raw, 4)})")
n_edge = int(np.isin(P.max(1), np.round(np.arange(1, 10) / 10, 10)).sum())
print(f"     test days whose max(p) equals an interior bin edge exactly: {n_edge}; "
      f"days with max(p) == 0.50: {int((P.max(1) == 0.5).sum())}")
cmp("MCE top-label raw", mce_raw, 0.8832)
lo, hi, _ = ci(lambda a, b: ece_top(a, b), y, P)
cmp("ECE raw CI low", lo, 0.0815)
cmp("ECE raw CI high", hi, 0.2074)
b_raw = brier_mc(y, P)
cmp("Brier multiclass raw (sum over classes)", b_raw, 0.5091)
b_sk = float(brier_score_loss(y, P, labels=L, scale_by_half=False))
print(f"     sklearn brier_score_loss(scale_by_half=False) = {b_sk:.12f}; |diff| = {abs(b_sk - b_raw):.2e}; "
      f"scale_by_half=True would give {b_sk / 2:.6f}")
lo, hi, _ = ci(brier_mc, y, P)
cmp("Brier raw CI low", lo, 0.4747)
cmp("Brier raw CI high", hi, 0.5489)
l_raw = ll(y, P)
cmp("log loss raw (manual -mean log p_true)", l_raw, 0.8522)
print(f"     sklearn log_loss = {log_loss(y, P, labels=L):.12f}; rows with p(true class) == 0: {int((P[np.arange(192), y] == 0).sum())}")
lo, hi, _ = ci(ll, y, P)
cmp("log loss raw CI low", lo, 0.7701)
cmp("log loss raw CI high", hi, 0.9648)
# per-class ECE / Brier Active, raw
cmp("ECE Active raw (OvR)", ece(P[:, 2], (y == 2).astype(float))[0], 0.0936)
cmp("Brier Active raw", float(((P[:, 2] - (y == 2)) ** 2).mean()), 0.0502)

# reliability of raw p_active on OOF (the "3 percent to 17 percent" sentence)
print("\n=== OOF raw p_active reliability (fit rows, 612) ===")
b = bin_ids_sk(P_oof[:, 2])
rows = []
for k in range(10):
    m = b == k
    if m.any():
        rows.append((k, int(m.sum()), float(P_oof[m, 2].mean()), float((y_oof[m] == 2).mean())))
for k, n, c, a in rows:
    print(f"     bin {k} ({k / 10:.1f}, {(k + 1) / 10:.1f}]: n = {n:3d}, mean p_active {c:.4f}, Active fraction {a:.4f}")
fr = [a for *_, a in rows]
print(f"     Active fraction over non-empty bins: min {min(fr):.4f}, max {max(fr):.4f}; "
      f"bins with 0 Active: {[k for k, n, c, a in rows if a == 0]}")
ok = min(fr) >= 0.03 - 1e-12
results.append(("OOF Active fraction >= 3 percent in every non-empty bin (manuscript sentence)", ok, f"min {min(fr):.4f}"))
print(f"{'PASS' if ok else 'FAIL'} OOF Active fraction between 3 percent and 17 percent in every bin: min {min(fr):.4f}, max {max(fr):.4f}")

# ----------------------------------------------------------------------------- calibrators refit from OOF (independent)
print("\n=== Calibrators refit on the 612 OOF rows (fit 1-4) ===")
CLIP = 1e-6


def logit(p):
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


platt = {}
for j, name, cs, ci_ in ((0, "Rest", 0.3970, -1.0593), (1, "Normal", 0.2210, 1.0260), (2, "Active", 0.2095, -2.3673)):
    m = sm.Logit((y_oof == j).astype(float), sm.add_constant(logit(P_oof[:, j]))).fit(disp=0, method="newton", tol=1e-10, maxiter=200)
    platt[j] = (float(m.params[0]), float(m.params[1]))
    cmp(f"Platt slope {name} (statsmodels Logit, no penalty)", platt[j][1], cs)
    cmp(f"Platt intercept {name}", platt[j][0], ci_)


def apply_platt(Pm):
    Q = np.column_stack([1 / (1 + np.exp(-(platt[j][0] + platt[j][1] * logit(Pm[:, j])))) for j in L])
    s = Q.sum(1)
    return Q / s[:, None], int((s <= 0).sum())


iso = {j: IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(P_oof[:, j], (y_oof == j).astype(float)) for j in L}


def apply_iso(Pm):
    Q = np.column_stack([iso[j].predict(Pm[:, j]) for j in L])
    s = Q.sum(1)
    out = np.where(s[:, None] > 0, Q / np.where(s > 0, s, 1)[:, None], 1 / 3)
    return out, int((s <= 0).sum())


grid = np.linspace(0, 1, 1001)
mono = {j: bool(np.all(np.diff(iso[j].predict(grid)) >= -1e-12)) for j in L}
print("     isotonic monotone on 1001-point grid:", mono)
for vname, fn, claims in (("platt", apply_platt, dict(ece=0.0361, lo=0.0123, hi=0.0996, brier=0.4124, ll=0.6969, pmax=0.1029, ece_act=0.0206)),
                          ("isotonic", apply_iso, dict(ece=0.0405, lo=0.0189, hi=0.1077, brier=0.4095, ll=0.7111, pmax=0.1896, ece_act=0.0340))):
    Pc, nz = fn(P)
    print(f"  -- {vname}: rows with zero sum (fallback) {nz}; max |row sum - 1| {np.abs(Pc.sum(1) - 1).max():.2e}; "
          f"argmax Normal on {int((Pc.argmax(1) == 1).sum())}/192 days; argmax changed vs raw {int((Pc.argmax(1) != yp).sum())}; "
          f"days with p_active >= 0.50: {int((Pc[:, 2] >= 0.5).sum())}")
    cmp(f"ECE top-label {vname}", ece_top(y, Pc), claims["ece"])
    lo, hi, _ = ci(lambda a, b: ece_top(a, b), y, Pc)
    cmp(f"ECE {vname} CI low", lo, claims["lo"])
    cmp(f"ECE {vname} CI high", hi, claims["hi"])
    cmp(f"Brier multiclass {vname}", brier_mc(y, Pc), claims["brier"])
    cmp(f"log loss {vname}", ll(y, Pc), claims["ll"])
    cmp(f"p_active max {vname}", Pc[:, 2].max(), claims["pmax"])
    cmp(f"ECE Active {vname}", ece(Pc[:, 2], (y == 2).astype(float))[0], claims["ece_act"])
    cmp_int(f"argmax changed vs raw {vname}", int((Pc.argmax(1) != yp).sum()), 53)
    cmp(f"F1 macro of calibrated argmax {vname} (non-canonical)", f1_score(y, Pc.argmax(1), labels=L, average="macro", zero_division=0), 0.2811)

# ----------------------------------------------------------------------------- safety layer (manual)
print("\n=== Safety layer, raw RF, manual ===")


def safety(Pm, ypred, thr):
    conf = Pm.max(1)
    out = ypred.copy()
    over = conf < thr                 # strict: max(p) < thr -> Normal
    out[over] = 1
    err = np.abs(np.vectorize(CAL.get)(out) - np.vectorize(CAL.get)(y))
    return out, int(over.sum()), float(err.mean()), int(err.max())


ya, n_over, mean_err, max_err = safety(P, yp, 0.50)
cmp("coverage at 0.50 (percent)", (192 - n_over) / 192 * 100, 65.625)
cmp_int("covered days at 0.50", 192 - n_over, 126)
cmp_int("overridden at 0.50", n_over, 66)
cmp("mean caloric error at 0.50 (all 192 days)", mean_err, 58.3333)
cmp_int("max caloric error at 0.50", max_err, 200)
cmp_int("emitted Rest at 0.50", int((ya == 0).sum()), 28)
cmp_int("emitted Normal at 0.50", int((ya == 1).sum()), 162)
cmp_int("emitted Active at 0.50", int((ya == 2).sum()), 2)
cmp("F1 macro after fallback at 0.50", f1_score(y, ya, labels=L, average="macro", zero_division=0), 0.5067)
cmp("accuracy after fallback at 0.50", accuracy_score(y, ya), 0.7083)
cmp("F1 Active after fallback at 0.50", f1_score(y, ya, labels=[2], average="macro", zero_division=0), 0.4000)
ya0, n0, me0, mx0 = safety(P, yp, 0.0)
cmp("mean caloric error at threshold 0", me0, 71.875)
cmp_int("overridden at threshold 0", n0, 0)
me_norm = float(np.abs(np.vectorize(CAL.get)(y)).mean())
cmp("mean caloric error of always Normal", me_norm, 54.1667)
print(f"     always Normal = {int((y != 1).sum())} non-Normal days x 200 / 192 = {(y != 1).sum() * 200 / 192:.4f}; "
      f"mean over covered days only at 0.50 (NOT the reported convention): "
      f"{np.abs(np.vectorize(CAL.get)(ya[P.max(1) >= 0.5]) - np.vectorize(CAL.get)(y[P.max(1) >= 0.5])).mean():.4f}")
ya_le, n_le, *_ = safety(P, yp, 0.50 + 1e-12)
print(f"     with '<= 0.50' instead of '< 0.50': overridden {n_le} (identical: {n_le == n_over}; no day has max(p) == 0.50)")
# old 16-feature table (stored) for the erratum context
old = pd.read_csv(ROOT / "outputs" / "tables" / "confidence_threshold_analysis.csv")
o5 = old[old.threshold == 0.5].iloc[0]
print(f"     stored 16-feature table at 0.50: coverage {o5.coverage_pct:.4f}, overridden {int(o5.n_overridden)}, "
      f"mean error {o5.mean_cal_error:.4f} (canonical RF: {(192 - n_over) / 192 * 100:.4f} / {n_over} / {mean_err:.4f})")

# ----------------------------------------------------------------------------- T3.1 labels regenerated from the raw scores
print("\n=== T3.1 labels regenerated from data/oura_combined.csv ===")
oura = pd.read_csv(ROOT / "data" / "oura_combined.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
r, s = oura.readiness_score.to_numpy(float), oura.sleep_score.to_numpy(float)
labelled = ~(np.isnan(r) | np.isnan(s))
print(f"     raw rows {len(oura)}, rows with both scores {int(labelled.sum())}")


def labels(R, A_r, A_s=75):
    lab = np.ones(len(oura))
    lab[(r >= A_r) & (s >= A_s)] = 2
    lab[(r < R) | (s < R)] = 0          # Rest has priority
    lab[~labelled] = np.nan
    return lab


snap = pd.read_csv(ROOT / "models" / "dataset_snapshot_20260331.csv", parse_dates=["date"])
dates958 = oura.date[labelled].reset_index(drop=True)
print("     958 labelled dates == snapshot dates:", bool((dates958.to_numpy() == snap.date.to_numpy()).all()))
canon = labels(60, 85)[labelled]
print("     canonical labels (60/85/75) == snapshot nutrition_label on 958 rows:", bool(np.array_equal(canon, snap.nutrition_label.to_numpy(float))))
i = int(958 * 0.8)
print(f"     split index int(958 * 0.8) = {i}; test dates == test_predictions_rf.csv dates:",
      bool((dates958.iloc[i:].dt.strftime('%Y-%m-%d').to_numpy() == test.date.to_numpy()).all()))
grid_csv = pd.read_csv(RES / "t31_threshold_grid.csv")
all_ok = True
for R in (55, 60, 65):
    for A in (80, 85, 90):
        lab = labels(R, A)
        n_lab = int((~np.isnan(lab)).sum())
        lab958 = lab[labelled]
        tot = np.bincount(lab958.astype(int), minlength=3)
        tr = np.bincount(lab958[:i].astype(int), minlength=3)
        te = np.bincount(lab958[i:].astype(int), minlength=3)
        g = grid_csv[grid_csv.cell == f"R{R}_A{A}"].iloc[0]
        ok = (n_lab == 958 and tot.tolist() == [g.n_rest, g.n_normal, g.n_active]
              and tr.tolist() == [g.train_rest, g.train_normal, g.train_active]
              and te.tolist() == [g.test_rest, g.test_normal, g.test_active])
        all_ok &= ok
        print(f"     R{R}_A{A}: labelled {n_lab}, 958 {tot.tolist()}, train {tr.tolist()}, test {te.tolist()}  "
              f"{'== grid csv' if ok else '!= grid csv ' + str([g.n_rest, g.n_normal, g.n_active])}")
results.append(("9 cells: 958 labelled days invariant and distributions == t31_threshold_grid.csv", all_ok, ""))
print(f"{'PASS' if all_ok else 'FAIL'} all 9 cells: 958 labelled days, 766/192 cut, distributions == grid csv")
lab55 = labels(55, 85)[labelled]
cmp_int("R55_A85 Rest on 958", int((lab55 == 0).sum()), 97)
cmp_int("R55_A85 Normal on 958", int((lab55 == 1).sum()), 815)
cmp_int("R55_A85 Active on 958", int((lab55 == 2).sum()), 46)
# Q19: A_r = 90 with A_s 75 vs A_s 80
n_viol = int(((r >= 90) & (s < 80) & labelled).sum())
print(f"     days with readiness >= 90 and sleep < 80: {n_viol} (readiness >= 90 days: {int(((r >= 90) & labelled).sum())})")
for R in (55, 60, 65):
    same = np.array_equal(labels(R, 90, 75)[labelled], labels(R, 90, 80)[labelled])
    print(f"     labels (R={R}, 90, 75) == (R={R}, 90, 80): {same}")
    results.append((f"labels (R={R}, 90, 75) == Table 7 90/80 labels", bool(same), ""))

# ----------------------------------------------------------------------------- T3.1 cell R55_A85 from its csv
print("\n=== T3.1 cell R55_A85 (t31_predictions_rf_R55_A85.csv) ===")
c55 = pd.read_csv(RES / "t31_predictions_rf_R55_A85.csv")
yt55, yp55 = c55.y_true.to_numpy(int), c55.y_pred.to_numpy(int)
print("     csv dates == test dates:", bool((c55.date.to_numpy() == test.date.to_numpy()).all()),
      "| csv y_true == regenerated R55 test labels:", bool(np.array_equal(yt55, lab55[i:].astype(int))),
      "| argmax(proba) == y_pred:", bool(np.array_equal(c55[PCOLS].to_numpy().argmax(1), yp55)))
cmp("R55_A85 F1 macro", f1_score(yt55, yp55, labels=L, average="macro", zero_division=0), 0.5053)
print(f"     F1 macro without fixed labels: {f1_score(yt55, yp55, average='macro'):.6f} (all 3 classes present in test)")
cmp("R55_A85 F1 Active", f1_score(yt55, yp55, labels=[2], average="macro", zero_division=0), 0.6154)
cmp("R55_A85 accuracy", accuracy_score(yt55, yp55), 0.7135)
cmp("R55_A85 MCC", matthews_corrcoef(yt55, yp55), 0.0379)
first = int(lab55[i - 1])
print(f"     last train day {dates958.iloc[i - 1].date()} readiness {r[labelled][i - 1]:.0f} sleep {s[labelled][i - 1]:.0f} -> "
      f"R55 label {first} (canonical label {int(canon[i - 1])})")
pers = np.concatenate([[first], yt55[:-1]])
cmp("R55_A85 persistence F1 macro", f1_score(yt55, pers, labels=L, average="macro", zero_division=0), 0.3211)
f1m = lambda a, b: f1_score(a, b, labels=L, average="macro", zero_division=0)
diffs, va, vb = [], [], []
for row in idx:
    yt = yt55[row]
    if len(np.unique(yt)) < 3:
        continue
    a, b_ = f1m(yt, yp55[row]), f1m(yt, pers[row])
    va.append(a)
    vb.append(b_)
    diffs.append(a - b_)
print(f"     paired bootstrap: used {len(diffs)} / 1000 resamples")
cmp("R55_A85 delta RF minus persistence (point)", f1m(yt55, yp55) - f1m(yt55, pers), 0.1842)
cmp("R55_A85 delta CI low", np.percentile(diffs, 2.5), 0.0281)
cmp("R55_A85 delta CI high", np.percentile(diffs, 97.5), 0.2919)
cmp("R55_A85 RF F1 macro CI low", np.percentile(va, 2.5), 0.3574)
cmp("R55_A85 RF F1 macro CI high", np.percentile(va, 97.5), 0.6084)
cmp("R55_A85 persistence CI low", np.percentile(vb, 2.5), 0.2771)
cmp("R55_A85 persistence CI high", np.percentile(vb, 97.5), 0.3679)
cmp("R55_A85 prop(delta > 0)", float(np.mean(np.asarray(diffs) > 0)), 0.988)

# ----------------------------------------------------------------------------- scratch retraining of cell R55_A85 (sklearn / imblearn only)
print("\n=== Scratch retraining of cell R55_A85 (no revision/scripts import) ===")
t0 = time.time()
BASE = ["readiness_score", "sleep_score", "hrv_mean", "hrv_balance", "recovery_index", "rhr_lowest", "rhr_avg",
        "activity_score", "steps_oura", "respiratory_rate", "temp_deviation", "aw_hr_mean", "aw_hrv_mean", "aw_steps",
        "aw_active_calories", "aw_spo2", "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate"]
aw = pd.read_csv(ROOT / "apple_watch_api" / "apple_health_daily.csv", parse_dates=["date"])
aw["aw_active_calories"] = (aw["aw_active_energy_kj"] / 4.184).round(1)
df = pd.read_csv(ROOT / "data" / "oura_combined.csv", parse_dates=["date"])
for col in BASE[11:]:
    df = df.merge(aw[["date", col]].dropna(subset=[col]), on="date", how="left")
df = df.sort_values("date").reset_index(drop=True)
assert (df.date.to_numpy() == oura.date.to_numpy()).all()
feats = [f"{f}_lag1" for f in BASE]
for f in BASE:
    df[f"{f}_lag1"] = df[f].shift(1)
df["label"] = labels(55, 85)
dm = df.dropna(subset=["label"]).sort_values("date").reset_index(drop=True)
X, yy = dm[feats], dm["label"].astype(int)
imp = SimpleImputer(strategy="median")
Xtr = pd.DataFrame(imp.fit_transform(X.iloc[:i]), columns=feats)
Xte = pd.DataFrame(imp.transform(X.iloc[i:]), columns=feats)
Xsm, ysm = SMOTE(random_state=SEED, k_neighbors=3).fit_resample(Xtr, yy.iloc[:i])
rf = RandomForestClassifier(n_estimators=100, random_state=SEED, max_depth=10, min_samples_leaf=5).fit(Xsm, ysm)
pr = rf.predict_proba(Xte)
print(f"     train after SMOTE {len(ysm)} rows ({np.bincount(ysm).tolist()}); elapsed {time.time() - t0:.1f} s")
ok = np.array_equal(pr.argmax(1), yp55)
results.append(("scratch R55_A85 y_pred == t31 csv", ok, ""))
print(f"{'PASS' if ok else 'FAIL'} scratch retrain y_pred == t31_predictions_rf_R55_A85.csv on 192 days; "
      f"max |proba diff| {np.abs(pr - c55[PCOLS].to_numpy()).max():.2e}")
cmp("scratch R55_A85 F1 macro", f1m(yt55, pr.argmax(1)), 0.5053)

# ----------------------------------------------------------------------------- verdict
print("\n=== VERDICT ===")
n_fail = sum(not ok for _, ok, _ in results)
for name, ok, detail in results:
    if not ok:
        print(f"  FAIL {name}: {detail}")
print(f"{len(results) - n_fail} / {len(results)} comparisons PASS at 4 decimals; {n_fail} FAIL")
