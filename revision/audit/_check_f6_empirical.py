"""
_check_f6_empirical.py - evidence for the T2 questions of audit slice F6 (T3.3 expanding window)
that need code execution.

Imports the revision scripts in memory (expanding_window and the modules it imports) and calls
none of their main() functions. Nothing under revision/results/ is written; run_fold() is called
with its outputs kept in memory only.

E0  sha256 of the inputs read; identity of the helper objects (paired_bootstrap, evaluate,
    preprocess_split, build_models) with the modules audited in F1 / F2 / F3.         (T0, Q6)
E1  Fold bounds from (N_TOTAL, BLOCK) through fold_bounds / all_fold_bounds; disjoint; cover
    rows 190..957; fold 1 == int(958 * 0.8).                                             (Q1)
E2  Per fold: imputer.statistics_ of preprocess_split == nan-median of rows 0..start-1 and
    != the median of rows 0..end-1 / all 958 (fold 3 printed in full); SMOTE k and rows.
                                                                                     (Q2, Q3)
E3  Persistence on the first 3 rows of folds 2 and 4 (and 3): == y[start-1], y[start],
    y[start+1] of the 958 series.                                                        (Q4)
E4  Fold 1 file vs test_predictions_rf.csv: field-by-field string identity of date, y_true,
    y_pred and the three probability columns.                                            (Q5)
E5  Bootstrap matrix == default_rng(42); skip counts per fold from the y_true of the four
    prediction files with the statistical_tests rule.                                   (Q6)
E6  Pooled: order of concatenation (fold 1..4, i.e. newest block first) read from the main()
    loop; the pooled CI recomputed in fold order (== csv) and in chronological order
    (diagnostic only, to size the dependence on the order).                             (Q7)
E7  run_fold(k) for k = 1..4 through the script's own function: rows == t33_expanding_window.csv
    at 1e-9 and SHAP top-3 == t33_shap_top3.csv (confirms csv == code at HEAD; ~30 s).
                                                                                 (Q2, Q9, T3)
E8  SHAP aggregation: mean_abs_shap is the function used by shap_stability (T1.4).       (Q9)
E9  Question 12: gap rows per block (folds 1..4) and the two "previous day" conventions
    (feature lag-1 on the 1,009-row raw frame vs previous labelled row of the 958 series);
    the labelled-day set of the six comparable T3.1 cells vs the canonical one.         (Q12)
E10 Discussion claims "33 of the 46 Active days fall on or after July 2024" and Rest
    prevalence 15% to 31% per block.                                                    (Q10)
Run:  python revision/audit/_check_f6_empirical.py > revision/audit/_check_f6_empirical_output.txt
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(newline="\n")                      # LF output on Windows
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
sys.path.insert(0, str(ROOT / "revision" / "scripts"))
sys.path.insert(0, str(ROOT / "revision"))

import common                                                     # noqa: E402
import expanding_window as ew                                     # noqa: E402
import persistence_baseline as pb                                 # noqa: E402
import save_test_predictions as stp                               # noqa: E402
import shap_stability as ss                                       # noqa: E402
import statistical_tests as st                                    # noqa: E402
import threshold_sensitivity as ts                                # noqa: E402

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


t_start = time.time()
print("--- E0. inventory ---")
head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
print("git HEAD:", head)
for rel in ["revision/scripts/expanding_window.py", "revision/scripts/test_expanding_window.py", "revision/scripts/common.py",
            "revision/scripts/threshold_sensitivity.py", "revision/scripts/persistence_baseline.py",
            "revision/scripts/statistical_tests.py", "revision/scripts/shap_stability.py", "revision/scripts/group_ablation.py",
            "revision/scripts/save_test_predictions.py", "revision/t3_labels.py",
            "revision/results/t33_expanding_window.csv", "revision/results/t33_shap_top3.csv", "revision/results/t33_run_meta.json",
            "revision/results/t33_predictions_rf_fold1.csv", "revision/results/t33_predictions_rf_fold2.csv",
            "revision/results/t33_predictions_rf_fold3.csv", "revision/results/t33_predictions_rf_fold4.csv",
            "revision/results/test_predictions_rf.csv", "revision/results/test_predictions_persistence.csv",
            "revision/results/bootstrap_indices_1000x192_seed42.npy", "models/dataset_snapshot_20260331.csv",
            "data/oura_combined.csv", "apple_watch_api/apple_health_daily.csv",
            "revision/manuscript/KBS_Manuscript_R1.docx", "revision/manuscript/KBS_Response_R1.docx"]:
    print(f"  {sha16(ROOT / rel)}  {rel}")
check(ew.paired_bootstrap is st.paired_bootstrap, "expanding_window.paired_bootstrap is statistical_tests.paired_bootstrap (same object, F2 rule)")
check(ew.t04_evaluate is pb.evaluate, "expanding_window.t04_evaluate is persistence_baseline.evaluate (unmodified T0.4)")
check(ew.preprocess_split is common.preprocess_split, "expanding_window.preprocess_split is common.preprocess_split (F1-audited)")
check(ew.build_models is stp.build_models, "expanding_window.build_models is save_test_predictions.build_models")
check(ew.mean_abs_shap is ss.mean_abs_shap and ew.rank_desc is ss.rank_desc, "SHAP helpers are shap_stability.mean_abs_shap / rank_desc (T1.4)")
check(ew.persistence_predictions is ts.persistence_predictions and ew.smote_k is ts.smote_k and ew.is_comparable is ts.is_comparable,
      "persistence_predictions / smote_k / is_comparable are the threshold_sensitivity (T3.1) objects")
print("RF params:", {k: v for k, v in stp.build_models()["rf"].get_params().items()
                     if k in ("n_estimators", "max_depth", "min_samples_leaf", "random_state", "class_weight", "max_features", "n_jobs")})

print("\n--- E1. fold bounds (Q1) ---")
print("constants:", dict(N_TOTAL=ew.N_TOTAL, BLOCK=ew.BLOCK, N_FOLDS=ew.N_FOLDS, N_BOOT=ew.N_BOOT, SEED=ew.SEED))
b = ew.all_fold_bounds()
print("all_fold_bounds():", b)
check(b == [(766, 766, 958), (574, 574, 766), (382, 382, 574), (190, 190, 382)], "bounds == 766/574/382/190 starts, 192 rows each")
check(all(te0 == ew.N_TOTAL - ew.BLOCK * k for k, (_, te0, _) in enumerate(b, 1)), "test_start(k) == N_TOTAL - BLOCK * k (computed, not hard-coded)")
check(b[0][1] == int(958 * common.TRAIN_RATIO), "fold 1 start == int(958 * 0.8) == 766")
tests = [set(range(a, c)) for _, a, c in b]
check(all(not (tests[i] & tests[j]) for i in range(4) for j in range(i + 1, 4)), "test blocks pairwise disjoint")
check(set().union(*tests) == set(range(190, 958)), "union of blocks == rows 190..957 (no gap, no overlap)")
check(all(tr_end == te0 for tr_end, te0, _ in b), "train = rows [0, test_start) for every fold (no gap between train and block)")
try:
    ew.fold_bounds(5)
    check(False, "fold 5 raises")
except ValueError as e:
    check(True, f"fold 5 raises ValueError ({e})")

print("\n--- data ---")
raw = common.load_raw()
X, y, dates = common.build_dataset(raw)
print("raw rows", len(raw), "labelled rows", len(X), "dist", ts.label_counts(y), "dates monotonic", bool(dates.is_monotonic_increasing))
check(len(X) == 958 and ts.label_counts(y) == (202, 710, 46), "958 labelled days, 202/710/46")

print("\n--- E2. imputer / SMOTE per fold (Q2, Q3) ---")
splits = {}
for k, (tr_end, te0, te1) in enumerate(b, 1):
    c_tr = ts.label_counts(y.iloc[:tr_end])
    kk = ts.smote_k(min(c_tr))
    s = common.preprocess_split(X.iloc[:tr_end], X.iloc[te0:te1], y.iloc[:tr_end], y.iloc[te0:te1], dates.iloc[te0:te1], k_neighbors=kk)
    splits[k] = s
    med_train = X.iloc[:tr_end].median().to_numpy()
    med_through_block = X.iloc[:te1].median().to_numpy()
    med_all = X.median().to_numpy()
    eq_train = bool(np.allclose(s.imputer.statistics_, med_train, rtol=0, atol=0))
    eq_block = bool(np.allclose(s.imputer.statistics_, med_through_block, rtol=0, atol=0))
    eq_all = bool(np.allclose(s.imputer.statistics_, med_all, rtol=0, atol=0))
    print(f"fold {k}: train 0..{tr_end - 1} dist {c_tr} min {min(c_tr)} smote_k {kk}; imputer == median(train) {eq_train}, "
          f"== median(0..{te1 - 1}) {eq_block}, == median(958) {eq_all}; SMOTE rows {len(s.y_train_sm)} "
          f"= {ts.label_counts(s.y_train_sm)} (3 x {max(c_tr)} = {3 * max(c_tr)}); imputer n_features_in_ {s.imputer.n_features_in_}; "
          f"test rows {len(s.X_test)}; NaN cells train {int(X.iloc[:tr_end].isna().sum().sum())} test {int(X.iloc[te0:te1].isna().sum().sum())}")
    check(eq_train and not eq_block and not eq_all, f"fold {k}: imputer statistics == median of rows 0..{tr_end - 1} only")
    check(kk == 3, f"fold {k}: smote_k == 3 (min train class {min(c_tr)})")
    check(len(s.y_train_sm) == 3 * max(c_tr) and ts.label_counts(s.y_train_sm) == (max(c_tr),) * 3, f"fold {k}: SMOTE rows == 3 x majority ({3 * max(c_tr)})")
    check(bool(np.array_equal(s.X_train_sm.iloc[:tr_end].to_numpy(), s.X_train.to_numpy())), f"fold {k}: first {tr_end} SMOTE rows == imputed train (synthetic rows appended)")
s3 = splits[3]
print("fold 3 imputer.statistics_ :", np.round(s3.imputer.statistics_, 4).tolist())
print("fold 3 median rows 0..381  :", np.round(X.iloc[:382].median().to_numpy(), 4).tolist())
print("fold 3 median rows 0..573  :", np.round(X.iloc[:574].median().to_numpy(), 4).tolist())
print("fold 3 median rows 382..573:", np.round(X.iloc[382:574].median().to_numpy(), 4).tolist())
# imputer statistics differ between folds (so each fold has its own train statistics)
diffs = [int((splits[k].imputer.statistics_ != splits[k + 1].imputer.statistics_).sum()) for k in (1, 2, 3)]
print("features whose imputer median differs fold1/2, fold2/3, fold3/4:", diffs)

print("\n--- E3. persistence first rows (Q4) ---")
y_int = y.to_numpy(dtype=int)
for k, (tr_end, te0, te1) in enumerate(b, 1):
    p = ts.persistence_predictions(y, te0)[:te1 - te0]
    exp = y_int[te0 - 1:te1 - 1]
    print(f"fold {k}: first 3 persistence preds {p[:3].tolist()} vs y[{te0 - 1}..{te0 + 1}] = {y_int[te0 - 1:te0 + 2].tolist()} "
          f"(dates {dates.iloc[te0 - 1].date()} -> {dates.iloc[te0].date()}, {dates.iloc[te0 + 1].date()}, {dates.iloc[te0 + 2].date()}); "
          f"len {len(p)}; all rows == y shifted by one row: {bool(np.array_equal(p, exp))}")
    check(bool(np.array_equal(p, exp)) and len(p) == 192, f"fold {k}: persistence == label of previous row of the 958 series, first block day uses row {te0 - 1} (last train day)")
pers_file = pd.read_csv(RES / "test_predictions_persistence.csv")
check(np.array_equal(pers_file["y_pred"].to_numpy(), ts.persistence_predictions(y, 766)), "fold 1 persistence == test_predictions_persistence.csv")

print("\n--- E4. fold 1 file vs canonical file (Q5) ---")
f1 = pd.read_csv(RES / "t33_predictions_rf_fold1.csv", dtype=str)
ref = pd.read_csv(RES / "test_predictions_rf.csv", dtype=str)
same_cols = all(f1[c].tolist() == ref[c].tolist() for c in ref.columns)
print("fold1 columns", f1.columns.tolist(), "| canonical columns", ref.columns.tolist())
check(same_cols and (f1["fold"] == "1").all() and len(f1) == 192, "fold 1 file: date, y_true, y_pred, p_rest, p_normal, p_active identical as text to test_predictions_rf.csv (plus fold column)")
raw1 = (RES / "t33_predictions_rf_fold1.csv").read_bytes()
rawr = (RES / "test_predictions_rf.csv").read_bytes()
stripped = b"\n".join(line.split(b",", 1)[1] for line in raw1.split(b"\n") if line)
cr1, crr = raw1.count(b"\r"), rawr.count(b"\r")
print(f"CR bytes: fold 1 file {cr1}, test_predictions_rf.csv {crr} (the canonical T0.2 file is CRLF, the T3.3 file LF)")
check(stripped == rawr.replace(b"\r\n", b"\n").rstrip(b"\n"), "fold 1 file with the leading 'fold,' field removed == test_predictions_rf.csv byte for byte after CRLF -> LF")
check(b"\r" not in raw1, "fold 1 file LF only")

print("\n--- E5. bootstrap matrix and skip counts (Q6) ---")
idx = np.load(RES / st.INDEX_FILE)
regen = np.random.default_rng(42).integers(0, 192, size=(1000, 192))
check(np.array_equal(idx, regen) and idx.shape == (1000, 192) and idx.min() == 0 and idx.max() == 191, f"{st.INDEX_FILE} == default_rng(42).integers(0, 192, (1000, 192)); range 0..191")
folds_pred = {k: pd.read_csv(RES / f"t33_predictions_rf_fold{k}.csv") for k in range(1, 5)}
for k in range(1, 5):
    yt = folds_pred[k]["y_true"].to_numpy(dtype=int)
    n_skip = int(sum(len(np.unique(yt[r])) < 3 for r in idx))
    absent = [int(c) for c in st.LABELS if any((yt[r] == c).sum() == 0 for r in idx)]
    print(f"fold {k}: resamples with a class absent {n_skip} (classes ever absent: {absent}); Active in block {int((yt == 2).sum())}")
    check(n_skip == [0, 0, 1, 8][k - 1], f"fold {k}: skipped == {[0, 0, 1, 8][k - 1]} with the statistical_tests rule")
# the rule is the same code path: paired_bootstrap counts n_skipped; confirm on fold 4 arrays
yt4, yp4 = folds_pred[4]["y_true"].to_numpy(int), folds_pred[4]["y_pred"].to_numpy(int)
pp4 = ts.persistence_predictions(y, 190)[:192]
bb = st.paired_bootstrap(yt4, yp4, pp4, ew.f1_macro, idx)
print("fold 4 paired_bootstrap:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in bb.items()})
check(bb["n_skipped"] == 8 and bb["n_used"] == 992, "fold 4 paired_bootstrap n_skipped 8 / n_used 992")

print("\n--- E6. pooled order (Q7) ---")
pooled_dates = pd.concat([folds_pred[k]["date"] for k in range(1, 5)], ignore_index=True)
chron = pooled_dates.is_monotonic_increasing
print("main() concatenates pooled_true/pooled_pred in the loop order k = 1..4 (expanding_window.py:325-326, :361);")
print("concatenated dates monotonic increasing:", bool(chron), "| first/last:", pooled_dates.iloc[0], pooled_dates.iloc[-1],
      "| block order by first date:", [folds_pred[k]["date"].iloc[0] for k in range(1, 5)])
yt_fold = np.concatenate([folds_pred[k]["y_true"].to_numpy(int) for k in range(1, 5)])
yp_fold = np.concatenate([folds_pred[k]["y_pred"].to_numpy(int) for k in range(1, 5)])
yt_chr = np.concatenate([folds_pred[k]["y_true"].to_numpy(int) for k in (4, 3, 2, 1)])
yp_chr = np.concatenate([folds_pred[k]["y_pred"].to_numpy(int) for k in (4, 3, 2, 1)])
pf = ew.pooled_bootstrap(yt_fold, yp_fold, np.random.default_rng(42))
pc = ew.pooled_bootstrap(yt_chr, yp_chr, np.random.default_rng(42))
t33 = pd.read_csv(RES / "t33_expanding_window.csv")
pooled_row = t33[t33.fold == "pooled"].iloc[0]
print("pooled CI fold order (1..4):      f1_macro [%.6f, %.6f] f1_active [%.6f, %.6f] acc [%.6f, %.6f] mcc [%.6f, %.6f] skipped %d" % (
    pf["f1_macro_ci_low"], pf["f1_macro_ci_high"], pf["f1_active_ci_low"], pf["f1_active_ci_high"], pf["accuracy_ci_low"], pf["accuracy_ci_high"], pf["mcc_ci_low"], pf["mcc_ci_high"], pf["n_skipped"]))
print("pooled CI chronological (4..1):   f1_macro [%.6f, %.6f] f1_active [%.6f, %.6f] acc [%.6f, %.6f] mcc [%.6f, %.6f] skipped %d  (diagnostic only)" % (
    pc["f1_macro_ci_low"], pc["f1_macro_ci_high"], pc["f1_active_ci_low"], pc["f1_active_ci_high"], pc["accuracy_ci_low"], pc["accuracy_ci_high"], pc["mcc_ci_low"], pc["mcc_ci_high"], pc["n_skipped"]))
print("csv pooled row:                   f1_macro [%.6f, %.6f] f1_active [%.6f, %.6f]" % (
    pooled_row.f1_macro_ci_low, pooled_row.f1_macro_ci_high, pooled_row.f1_active_ci_low, pooled_row.f1_active_ci_high))
check(abs(pf["f1_macro_ci_low"] - pooled_row.f1_macro_ci_low) < 1e-9 and abs(pf["f1_macro_ci_high"] - pooled_row.f1_macro_ci_high) < 1e-9,
      "pooled CI in fold order == csv (so the csv CI corresponds to the fold-1-first concatenation)")
print("pooled point metrics order-invariant:", ew.pooled_metrics(yt_fold, yp_fold) == ew.pooled_metrics(yt_chr, yp_chr))
check(pf["n_skipped"] == 0, "pooled: 0 resamples skipped")

print("\n--- E7. run_fold(k) for k = 1..4 vs t33_expanding_window.csv / t33_shap_top3.csv (Q2, Q9, T3) ---")
shap_top = pd.read_csv(RES / "t33_shap_top3.csv")
t0 = time.time()
for k in range(1, 5):
    row, arr, srows = ew.run_fold(k, X, y, dates, idx)
    csv_row = t33[t33.fold == str(k)].iloc[0]
    num_diff = []
    for c in t33.columns:
        if c in row and c not in ("fold",):
            v, w = row[c], csv_row[c]
            if isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) and not isinstance(v, str):
                try:
                    if abs(float(v) - float(w)) > 1e-9:
                        num_diff.append((c, v, w))
                except (TypeError, ValueError):
                    num_diff.append((c, v, w))
            elif str(v) != str(w):
                num_diff.append((c, v, w))
    pred_same = bool(np.array_equal(arr["y_pred"], folds_pred[k]["y_pred"].to_numpy(int)))
    max_dp = float(np.abs(arr["proba"] - folds_pred[k][["p_rest", "p_normal", "p_active"]].to_numpy()).max())
    s_csv = shap_top[shap_top.fold == k].sort_values("rank")
    s_run = sorted([r for r in srows if r["rank"] <= 3], key=lambda r: r["rank"])
    shap_same = [r["feature"] for r in s_run] == s_csv.feature.tolist() and bool(np.allclose([r["mean_abs_shap"] for r in s_run], s_csv.mean_abs_shap.to_numpy(), atol=1e-9))
    print(f"fold {k}: columns differing from csv (>1e-9): {num_diff} | y_pred == file {pred_same} | max |dproba| {max_dp:.1e} | SHAP top-3 == file {shap_same} "
          f"({[(r['feature'], round(r['mean_abs_shap'], 4)) for r in s_run]}) | elapsed {time.time() - t0:.1f}s")
    check(not num_diff, f"fold {k}: run_fold row == t33_expanding_window.csv on every shared column")
    check(pred_same and max_dp < 1e-9, f"fold {k}: run_fold predictions == t33_predictions_rf_fold{k}.csv")
    check(shap_same, f"fold {k}: run_fold SHAP top-3 == t33_shap_top3.csv")

print("\n--- E8. SHAP aggregation (Q9) ---")
import inspect  # noqa: E402
src = inspect.getsource(ss.aggregate_multiclass)
print("shap_stability.aggregate_multiclass axis rule:", "mean(axis=(0, 2))" in src, "| expanding_window SHAP input: split.X_test (imputed block), lines 201-202")
check("mean(axis=(0, 2))" in src and ew.mean_abs_shap is ss.mean_abs_shap, "SHAP = mean |SHAP| over block days x 3 classes through the T1.4 function")

print("\n--- E9. Question 12: gap rows and the two 'previous day' conventions per block / T3.1 cell ---")
raw_dates = pd.to_datetime(raw["date"]).reset_index(drop=True)
raw_set = set(raw_dates)
raw_prev = {raw_dates.iloc[i]: raw_dates.iloc[i - 1] for i in range(1, len(raw_dates))}   # previous raw row
print("raw frame rows:", len(raw), "| raw consecutive-row gaps > 1 day:", int((raw_dates.diff().dt.days > 1).sum()))
gap_tab = []
for k, (tr_end, te0, te1) in enumerate(b, 1):
    n_gap = n_diff = 0
    details = []
    for i in range(te0, te1):
        d, d_prev_lab = dates.iloc[i], dates.iloc[i - 1]
        if (d - d_prev_lab).days > 1:
            n_gap += 1
            feat_src = raw_prev[d]                     # lag-1 on the raw frame = previous raw row (common.py:67)
            if feat_src != d_prev_lab:
                n_diff += 1
            details.append((str(d.date()), str(d_prev_lab.date()), str(feat_src.date()), feat_src != d_prev_lab))
    gap_tab.append(dict(fold=k, rows=f"{te0}-{te1 - 1}", n_gap=n_gap, n_differ=n_diff))
    print(f"fold {k} (rows {te0}-{te1 - 1}, {dates.iloc[te0].date()}..{dates.iloc[te1 - 1].date()}): rows whose previous labelled row is > 1 day back: {n_gap}; "
          f"of which feature source day (previous raw row) != persistence source day (previous labelled row): {n_diff}")
    for det in details:
        print(f"    day {det[0]}: persistence uses {det[1]}, features use {det[2]} -> {'DIFFER' if det[3] else 'same'}")
# sanity: the feature values of the built frame equal the raw-frame previous-row values
chk = 0
for i in range(len(X)):
    d = dates.iloc[i]
    j = raw.index[raw_dates == d][0]
    v_raw, v_x = raw.loc[j - 1, "readiness_score"], X.iloc[i]["readiness_score_lag1"]
    if not ((pd.isna(v_raw) and pd.isna(v_x)) or v_raw == v_x):
        chk += 1
print("rows where readiness_score_lag1 != raw previous-row readiness_score:", chk)
check(chk == 0, "feature lag-1 == previous raw row (confirms the feature convention used above)")
# train-side counts per fold (rows 1..start-1)
for k, (tr_end, te0, te1) in enumerate(b, 1):
    g = dates.diff().dt.days.iloc[1:tr_end]
    print(f"fold {k}: train rows (1..{tr_end - 1}) whose previous row is > 1 day back: {int((g > 1).sum())}")
# T3.1 cells: same labelled-day set?
cells = [(55, 80), (55, 85), (60, 80), (60, 85), (65, 80), (65, 85)]
for R, A in cells:
    Xc, yc, dc = common.build_dataset(raw, rest_cutoff=R, active_readiness=A, active_sleep=75)
    f = pd.read_csv(RES / f"t31_predictions_rf_R{R}_A{A}.csv")
    same_dates = bool((dc.reset_index(drop=True) == dates.reset_index(drop=True)).all()) and len(dc) == 958
    same_test = f["date"].tolist() == dates.iloc[766:].dt.strftime("%Y-%m-%d").tolist()
    print(f"cell R{R}_A{A}: 958 labelled days identical to canonical {same_dates}; test dates of t31_predictions == canonical test dates {same_test}; "
          f"y_true in file == regenerated labels {bool(np.array_equal(f['y_true'].to_numpy(int), yc.iloc[766:].to_numpy(int)))}")
    check(same_dates and same_test, f"cell R{R}_A{A}: same 958-day series as fold 1 (gap count 1, differing 0)")
print("gap table:", gap_tab)
# E9b: the sentence of AUDIT_DECISIONS.md #2 ("persistence coincides with a function of two of the features, the previous
# day's readiness and sleep scores, on all but one test day"): per block, days where the canonical label rule applied
# to (readiness_score_lag1, sleep_score_lag1) equals the persistence prediction; NaN lag features count as "not computable".
import t3_labels  # noqa: E402
lag_df = pd.DataFrame({"readiness_score": X["readiness_score_lag1"].to_numpy(), "sleep_score": X["sleep_score_lag1"].to_numpy()})
lab_from_lag = t3_labels.assign_labels(lag_df)
for k, (tr_end, te0, te1) in enumerate(b, 1):
    p = ts.persistence_predictions(y, te0)[:te1 - te0]
    f = lab_from_lag.iloc[te0:te1].to_numpy()
    n_nan = int(np.isnan(f).sum())
    n_eq = int(np.sum(~np.isnan(f) & (f == p)))
    n_ne = int(np.sum(~np.isnan(f) & (f != p)))
    print(f"fold {k}: label rule on the lag-1 readiness / sleep features vs persistence: equal {n_eq}, different {n_ne}, lag features NaN (not computable) {n_nan} of 192")

print("\n--- E10. Discussion claims (Q10) ---")
n_act_after = int(((y == 2) & (dates >= pd.Timestamp("2024-07-01"))).sum())
n_act_after_block = int(((y == 2) & (dates >= dates.iloc[382])).sum())
print(f"Active days on or after 2024-07-01: {n_act_after} of {int((y == 2).sum())} | on or after fold-3 block start {dates.iloc[382].date()}: {n_act_after_block}")
check(n_act_after == 33, "'33 of the 46 Active days fall on or after July 2024' (calendar July 1)")
rest_prev = [int((folds_pred[k]["y_true"] == 0).sum()) / 192 for k in range(1, 5)]
print("Rest prevalence per block (fold 1..4):", [round(100 * v, 1) for v in rest_prev], "%")
check(round(100 * min(rest_prev)) == 15 and round(100 * max(rest_prev)) == 31, "Rest prevalence 15% to 31%")
act_train = [int((y.iloc[:tr_end] == 2).sum()) for tr_end, _, _ in b]
print("Active in train per fold (1..4):", act_train)
check(min(act_train[1:]) == 9 and max(act_train[1:]) == 22, "'few Active days (9 to 22)' in the earlier training sets")

print("\n" + "=" * 100)
print(f"elapsed {time.time() - t_start:.1f}s")
print(f"{len(fails)} FAIL" if fails else "ALL PASS")
for f in fails:
    print("  FAIL " + f)
