"""
recompute_f6.py - independent recomputation for audit slice F6 (T3.3 expanding window).

Imports: numpy, pandas, scikit-learn, imbalanced-learn (plus hashlib, json, re, zipfile from the
standard library). Nothing from revision/scripts/ or the repo root is imported.

Inputs (read only):
  revision/results/t33_predictions_rf_fold{1,2,3,4}.csv, t33_expanding_window.csv, t33_shap_top3.csv,
  t33_run_meta.json, test_predictions_rf.csv, bootstrap_indices_1000x192_seed42.npy (compared with a
  regenerated matrix), models/dataset_snapshot_20260331.csv (dates, 19 lag-1 features, labels),
  revision/manuscript/KBS_Manuscript_R1.docx (Table 11 cells, read through zipfile).

Sections: R1 block bounds from N = 958 and 192, fold files vs snapshot; R2 per-fold point metrics and
the fold-2 confusion matrix; R3 persistence per block from the snapshot; R4 per-fold CIs and the
paired delta on the regenerated matrix with the absent-class skip rule; R5 pooled point metrics and
CI with default_rng(42) on 768 days; R6 independent refits of folds 4, 3 and 2 from the snapshot
(median imputer, SMOTE(42, k 3), RF 100 / 10 / 5 / seed 42) vs the prediction files; R7 Table 11
and the Section 4.8 / Discussion numbers at the printed precision.
Claims are the figures of CIFRE_REVIZIE.md rows T3.3 (run 2026-10-19) as quoted in the audit brief.
Run:  python -I revision/audit/recompute_f6.py > revision/audit/recompute_f6_output.txt
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef

sys.stdout.reconfigure(newline="\n")                      # LF output on Windows
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
LABELS = [0, 1, 2]
PROBA = ["p_rest", "p_normal", "p_active"]
N_TOTAL, BLOCK, N_FOLDS, N_BOOT, SEED = 958, 192, 4, 1000, 42
FEATURES = [f + "_lag1" for f in ["readiness_score", "sleep_score", "hrv_mean", "hrv_balance", "recovery_index", "rhr_lowest",
                                  "rhr_avg", "activity_score", "steps_oura", "respiratory_rate", "temp_deviation", "aw_hr_mean",
                                  "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2", "aw_sleep_deep_hours",
                                  "aw_sleep_rem_hours", "aw_resp_rate"]]
fails: list[str] = []
n_checks = 0


def check(cond: bool, msg: str) -> None:
    global n_checks
    n_checks += 1
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def f1m(y, p) -> float:
    return float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0))


def f1a(y, p) -> float:
    return float(f1_score(y, p, labels=[2], average="macro", zero_division=0))


def r4(x) -> float:
    return round(float(x), 4)


def point(y, p) -> dict:
    cm = confusion_matrix(y, p, labels=LABELS)
    return dict(f1_macro=f1m(y, p), f1_active=f1a(y, p), accuracy=float(accuracy_score(y, p)), mcc=float(matthews_corrcoef(y, p)),
                balanced_accuracy=float(balanced_accuracy_score(y, p)),
                f1_rest=float(f1_score(y, p, labels=[0], average="macro", zero_division=0)),
                f1_normal=float(f1_score(y, p, labels=[1], average="macro", zero_division=0)), cm=cm)


def boot_paired(y, a, b, fn, idx):
    """fn(A) - fn(B) per resample; resamples with a class absent from y[r] are skipped (statistical_tests rule)."""
    d, va, vb, skipped = [], [], [], 0
    for r in idx:
        yr = y[r]
        if any((yr == c).sum() == 0 for c in LABELS):
            skipped += 1
            continue
        ma, mb = fn(yr, a[r]), fn(yr, b[r])
        va.append(ma)
        vb.append(mb)
        d.append(ma - mb)
    d = np.asarray(d)
    return dict(point=fn(y, a) - fn(y, b), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                a_lo=float(np.percentile(va, 2.5)), a_hi=float(np.percentile(va, 97.5)),
                b_lo=float(np.percentile(vb, 2.5)), b_hi=float(np.percentile(vb, 97.5)),
                prop=float((d > 0).mean()), skipped=skipped, used=len(d))


# claims (CIFRE_REVIZIE.md rows T3.3 / audit brief), 4 decimals
CLAIM = {
    1: dict(train=(158, 570, 38), test=(44, 140, 8), f1_macro=0.5063, f1_active=0.4444, accuracy=0.6406, mcc=0.1416,
            ci=(0.3784, 0.6130), ci_a=(0.1176, 0.7000), pers=0.3039, pers_ci=(0.2612, 0.3501), delta=0.2024, dci=(0.0642, 0.3173),
            skipped=0, cm=None, maj=0.2811),
    2: dict(train=(129, 423, 22), test=(29, 147, 16), f1_macro=0.3124, f1_active=0.1026, accuracy=0.5990, mcc=-0.0291,
            ci=(0.2561, 0.3749), ci_a=(0.0000, 0.2424), pers=0.3070, pers_ci=(0.2559, 0.3682), delta=0.0054, dci=(-0.0768, 0.0930),
            skipped=0, cm=dict(rr=2, rn=22, ra=5, nr=20, nn=111, na=16, ar=1, an=13, aa=2), maj=0.2891),
    3: dict(train=(96, 272, 14), test=(33, 151, 8), f1_macro=0.3771, f1_active=0.0741, accuracy=0.5938, mcc=0.1112,
            ci=(0.3127, 0.4456), ci_a=(0.0000, 0.2222), pers=0.4120, pers_ci=(0.3253, 0.5136), delta=-0.0349, dci=(-0.1600, 0.0841),
            skipped=1, cm=dict(rr=14, rn=17, ra=2, nr=36, nn=99, na=16, ar=2, an=5, aa=1), maj=0.2935),
    4: dict(train=(36, 145, 9), test=(60, 127, 5), f1_macro=0.3605, f1_active=0.0000, accuracy=0.5781, mcc=0.1046,
            ci=(0.3095, 0.4069), ci_a=(0.0000, 0.0000), pers=0.3161, pers_ci=(0.2735, 0.3627), delta=0.0444, dci=(-0.0317, 0.1168),
            skipped=8, cm=dict(rr=21, rn=34, ra=5, nr=30, nn=90, na=7, ar=0, an=5, aa=0), maj=0.2654),
}
POOLED = dict(n=768, dist=(166, 565, 37), f1_macro=0.3932, ci=(0.3523, 0.4349), f1_active=0.1386, ci_a=(0.0522, 0.2316),
              accuracy=0.6029, acc_ci=(0.5716, 0.6354), mcc=0.0894, mcc_ci=(0.0281, 0.1474), skipped=0)

print("--- inputs ---")
for rel in ["revision/results/t33_predictions_rf_fold1.csv", "revision/results/t33_predictions_rf_fold2.csv",
            "revision/results/t33_predictions_rf_fold3.csv", "revision/results/t33_predictions_rf_fold4.csv",
            "revision/results/t33_expanding_window.csv", "revision/results/t33_shap_top3.csv", "revision/results/t33_run_meta.json",
            "revision/results/test_predictions_rf.csv", "revision/results/bootstrap_indices_1000x192_seed42.npy",
            "models/dataset_snapshot_20260331.csv", "revision/manuscript/KBS_Manuscript_R1.docx"]:
    print(f"  {sha16(ROOT / rel)}  {rel}")

snap = pd.read_csv(ROOT / "models" / "dataset_snapshot_20260331.csv")
snap["date"] = pd.to_datetime(snap["date"])
y_all = snap["nutrition_label"].to_numpy(dtype=int)
check(len(snap) == N_TOTAL and list(snap.columns[1:20]) == FEATURES and bool(snap["date"].is_monotonic_increasing),
      "snapshot: 958 rows, 19 lag-1 feature columns, dates increasing")
check(tuple(int((y_all == c).sum()) for c in LABELS) == (202, 710, 46), "snapshot labels 202 / 710 / 46")
fold = {k: pd.read_csv(RES / f"t33_predictions_rf_fold{k}.csv") for k in range(1, N_FOLDS + 1)}
t33 = pd.read_csv(RES / "t33_expanding_window.csv")
meta = json.loads((RES / "t33_run_meta.json").read_text(encoding="utf-8"))

print("\n--- R1. block bounds from N = 958 and 192; fold files vs snapshot ---")
bounds = {k: (N_TOTAL - BLOCK * k, N_TOTAL - BLOCK * k + BLOCK) for k in range(1, N_FOLDS + 1)}
print("bounds (start, end-exclusive):", bounds)
check([bounds[k][0] for k in range(1, 5)] == [766, 574, 382, 190], "block starts 766 / 574 / 382 / 190 from N - 192 k")
check(all(meta["fold_bounds"][f"fold{k}"]["test_rows"] == [bounds[k][0], bounds[k][1] - 1]
          and meta["fold_bounds"][f"fold{k}"]["train_rows"] == [0, bounds[k][0] - 1] for k in range(1, 5)), "t33_run_meta.json fold_bounds == recomputed")
check(sorted(sum((list(range(*bounds[k])) for k in range(1, 5)), [])) == list(range(190, 958)), "blocks disjoint and cover rows 190..957")
for k in range(1, 5):
    s, e = bounds[k]
    f = fold[k]
    same_dates = f["date"].tolist() == snap["date"].iloc[s:e].dt.strftime("%Y-%m-%d").tolist()
    same_y = np.array_equal(f["y_true"].to_numpy(int), y_all[s:e])
    p = f[PROBA].to_numpy()
    check(len(f) == BLOCK and same_dates and same_y and (f["fold"] == k).all(), f"fold {k}: 192 rows, dates == snapshot rows {s}..{e - 1}, y_true == snapshot labels")
    check(bool(np.allclose(p.sum(1), 1.0, atol=1e-12)) and np.array_equal(p.argmax(1), f["y_pred"].to_numpy(int)), f"fold {k}: probabilities sum to 1, argmax == y_pred")
    tr = tuple(int((y_all[:s] == c).sum()) for c in LABELS)
    te = tuple(int((y_all[s:e] == c).sum()) for c in LABELS)
    check(tr == CLAIM[k]["train"] and te == CLAIM[k]["test"], f"fold {k}: train {tr} / test {te} == claim {CLAIM[k]['train']} / {CLAIM[k]['test']}")
ref = pd.read_csv(RES / "test_predictions_rf.csv")
check(np.array_equal(ref["y_pred"].to_numpy(int), fold[1]["y_pred"].to_numpy(int)) and float(np.abs(ref[PROBA].to_numpy() - fold[1][PROBA].to_numpy()).max()) == 0.0
      and ref["date"].tolist() == fold[1]["date"].tolist(), "fold 1 file == test_predictions_rf.csv (dates, y_pred, probabilities exact)")

print("\n--- R2. per-fold point metrics (from the prediction files) ---")
pts = {}
for k in range(1, 5):
    y, p = fold[k]["y_true"].to_numpy(int), fold[k]["y_pred"].to_numpy(int)
    m = point(y, p)
    pts[k] = m
    c = CLAIM[k]
    print(f"fold {k}: F1 macro {m['f1_macro']:.6f} F1 Active {m['f1_active']:.6f} acc {m['accuracy']:.6f} MCC {m['mcc']:.6f} "
          f"bal acc {m['balanced_accuracy']:.4f} F1 Rest {m['f1_rest']:.4f} F1 Normal {m['f1_normal']:.4f}; cm rows=true r/n/a: {m['cm'].tolist()}")
    for key in ("f1_macro", "f1_active", "accuracy", "mcc"):
        check(r4(m[key]) == c[key], f"fold {k}: {key} {m[key]:.6f} == claim {c[key]}")
    row = t33[t33.fold == str(k)].iloc[0]
    check(abs(float(row.f1_macro) - m["f1_macro"]) < 1e-9 and abs(float(row.mcc) - m["mcc"]) < 1e-9 and abs(float(row.balanced_accuracy) - m["balanced_accuracy"]) < 1e-9
          and abs(float(row.f1_rest) - m["f1_rest"]) < 1e-9 and abs(float(row.f1_normal) - m["f1_normal"]) < 1e-9,
          f"fold {k}: csv row (f1_macro, mcc, balanced_accuracy, f1_rest, f1_normal) == recomputed at 1e-9")
    if c["cm"]:
        names = "rna"
        got = {f"{names[i]}{names[j]}": int(m["cm"][i, j]) for i in range(3) for j in range(3)}
        check(got == c["cm"], f"fold {k}: confusion matrix {got} == claim")
        csv_cm = {f"{names[i]}{names[j]}": int(row[f"cm_{names[i]}{names[j]}"]) for i in range(3) for j in range(3)}
        check(csv_cm == got, f"fold {k}: csv cm_* columns == recomputed")
    n_pred_active = int((p == 2).sum())
    print(f"        Active predicted {n_pred_active}, correct {int(((p == 2) & (y == 2)).sum())} of {int((y == 2).sum())}")

print("\n--- R3. persistence and majority per block from the snapshot ---")
pers = {}
for k in range(1, 5):
    s, e = bounds[k]
    pp = y_all[s - 1:e - 1]                           # label of the previous row; first block day uses row s-1 (last train day)
    pers[k] = pp
    y = y_all[s:e]
    m = point(y, pp)
    print(f"fold {k}: first 3 persistence preds {pp[:3].tolist()} (rows {s - 1}..{s + 1} labels {y_all[s - 1:s + 2].tolist()}); "
          f"F1 macro {m['f1_macro']:.6f} F1 Active {m['f1_active']:.6f} acc {m['accuracy']:.6f} MCC {m['mcc']:.6f}")
    check(r4(m["f1_macro"]) == CLAIM[k]["pers"], f"fold {k}: persistence F1 macro {m['f1_macro']:.6f} == claim {CLAIM[k]['pers']}")
    row = t33[t33.fold == str(k)].iloc[0]
    check(abs(float(row.pers_f1_macro) - m["f1_macro"]) < 1e-9 and abs(float(row.pers_f1_active) - m["f1_active"]) < 1e-9
          and abs(float(row.pers_accuracy) - m["accuracy"]) < 1e-9 and abs(float(row.pers_mcc) - m["mcc"]) < 1e-9, f"fold {k}: csv pers_* columns == recomputed")
    maj = int(np.bincount(y_all[:s]).argmax())
    pm = np.full(len(y), maj)
    check(maj == 1 and r4(f1m(y, pm)) == CLAIM[k]["maj"], f"fold {k}: majority class {maj} (Normal), F1 macro {f1m(y, pm):.6f} == claim {CLAIM[k]['maj']}")
    check(abs(float(row.maj_accuracy) - float((pm == y).mean())) < 1e-9, f"fold {k}: csv maj_accuracy == recomputed")
check(np.array_equal(pers[1], pd.read_csv(RES / "test_predictions_persistence.csv")["y_pred"].to_numpy(int)), "fold 1 persistence == test_predictions_persistence.csv")

print("\n--- R4. per-fold CIs and paired delta on the regenerated 1000 x 192 matrix (absent-class skip rule) ---")
idx = np.random.default_rng(SEED).integers(0, BLOCK, size=(N_BOOT, BLOCK))
idx_file = np.load(RES / "bootstrap_indices_1000x192_seed42.npy")
check(np.array_equal(idx, idx_file), "regenerated matrix == bootstrap_indices_1000x192_seed42.npy")
for k in range(1, 5):
    y, p = fold[k]["y_true"].to_numpy(int), fold[k]["y_pred"].to_numpy(int)
    bm = boot_paired(y, p, pers[k], f1m, idx)
    ba = boot_paired(y, p, pers[k], f1a, idx)
    c = CLAIM[k]
    print(f"fold {k}: RF F1 macro CI [{bm['a_lo']:.6f}, {bm['a_hi']:.6f}] F1 Active CI [{ba['a_lo']:.6f}, {ba['a_hi']:.6f}] "
          f"persistence CI [{bm['b_lo']:.6f}, {bm['b_hi']:.6f}] delta {bm['point']:+.6f} [{bm['lo']:+.6f}, {bm['hi']:+.6f}] "
          f"prop>0 {bm['prop']:.4f} skipped {bm['skipped']} used {bm['used']}")
    check((r4(bm["a_lo"]), r4(bm["a_hi"])) == c["ci"], f"fold {k}: RF F1 macro CI == claim {c['ci']}")
    check((r4(ba["a_lo"]), r4(ba["a_hi"])) == c["ci_a"], f"fold {k}: RF F1 Active CI == claim {c['ci_a']}")
    check((r4(bm["b_lo"]), r4(bm["b_hi"])) == c["pers_ci"], f"fold {k}: persistence F1 macro CI == claim {c['pers_ci']}")
    check(r4(bm["point"]) == c["delta"] and (r4(bm["lo"]), r4(bm["hi"])) == c["dci"], f"fold {k}: delta {bm['point']:+.4f} [{bm['lo']:+.4f}, {bm['hi']:+.4f}] == claim {c['delta']:+} {c['dci']}")
    check(bm["skipped"] == c["skipped"] and ba["skipped"] == c["skipped"], f"fold {k}: skipped {bm['skipped']} == claim {c['skipped']}")
    row = t33[t33.fold == str(k)].iloc[0]
    check(abs(float(row.delta_f1_macro_prop_gt_0) - bm["prop"]) < 1e-9 and int(row.n_used) == bm["used"], f"fold {k}: csv prop_gt_0 {float(row.delta_f1_macro_prop_gt_0):.6f} and n_used {int(row.n_used)} == recomputed")
    check(bool(row.delta_f1_macro_include_zero) == (bm["lo"] <= 0.0 <= bm["hi"]), f"fold {k}: include_zero flag {bool(row.delta_f1_macro_include_zero)} consistent")

print("\n--- R5. pooled (fold order 1..4 as in the script) ---")
yt = np.concatenate([fold[k]["y_true"].to_numpy(int) for k in range(1, 5)])
yp = np.concatenate([fold[k]["y_pred"].to_numpy(int) for k in range(1, 5)])
m = point(yt, yp)
dist = tuple(int((yt == c).sum()) for c in LABELS)
print(f"pooled n {len(yt)} dist {dist}: F1 macro {m['f1_macro']:.6f} F1 Active {m['f1_active']:.6f} acc {m['accuracy']:.6f} MCC {m['mcc']:.6f} "
      f"bal acc {m['balanced_accuracy']:.4f} F1 Rest {m['f1_rest']:.4f} F1 Normal {m['f1_normal']:.4f}")
check(len(yt) == POOLED["n"] and dist == POOLED["dist"], "pooled 768 days, 166 / 565 / 37")
for key in ("f1_macro", "f1_active", "accuracy", "mcc"):
    check(r4(m[key]) == POOLED[key], f"pooled {key} {m[key]:.6f} == claim {POOLED[key]}")
rng = np.random.default_rng(SEED)
vals = {k: [] for k in ("f1_macro", "f1_active", "accuracy", "mcc")}
skipped = 0
for _ in range(N_BOOT):
    r = rng.integers(0, len(yt), len(yt))
    yr, pr = yt[r], yp[r]
    if any((yr == c).sum() == 0 for c in LABELS):
        skipped += 1
        continue
    vals["f1_macro"].append(f1m(yr, pr))
    vals["f1_active"].append(f1a(yr, pr))
    vals["accuracy"].append(float(accuracy_score(yr, pr)))
    vals["mcc"].append(float(matthews_corrcoef(yr, pr)))
ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}
print("pooled CI:", {k: (round(a, 6), round(b, 6)) for k, (a, b) in ci.items()}, "skipped", skipped)
check((r4(ci["f1_macro"][0]), r4(ci["f1_macro"][1])) == POOLED["ci"], f"pooled F1 macro CI == claim {POOLED['ci']}")
check((r4(ci["f1_active"][0]), r4(ci["f1_active"][1])) == POOLED["ci_a"], f"pooled F1 Active CI == claim {POOLED['ci_a']}")
check((r4(ci["accuracy"][0]), r4(ci["accuracy"][1])) == POOLED["acc_ci"], f"pooled accuracy CI == claim {POOLED['acc_ci']}")
check((r4(ci["mcc"][0]), r4(ci["mcc"][1])) == POOLED["mcc_ci"], f"pooled MCC CI == claim {POOLED['mcc_ci']}")
check(skipped == 0, "pooled: 0 resamples skipped")
pr_ = t33[t33.fold == "pooled"].iloc[0]
check(abs(float(pr_.f1_macro_ci_low) - ci["f1_macro"][0]) < 1e-9 and abs(float(pr_.mcc_ci_high) - ci["mcc"][1]) < 1e-9, "pooled csv CI columns == recomputed at 1e-9")

print("\n--- R6. independent refits from the snapshot: folds 4, 3, 2 (imputer median, SMOTE(42, k 3), RF 100 / 10 / 5 / 42) ---")
X_all = snap[FEATURES]
for k in (4, 3, 2):
    s, e = bounds[k]
    imp = SimpleImputer(strategy="median").fit(X_all.iloc[:s])
    X_tr = pd.DataFrame(imp.transform(X_all.iloc[:s]), columns=FEATURES)
    X_te = pd.DataFrame(imp.transform(X_all.iloc[s:e]), columns=FEATURES)
    check(bool(np.allclose(imp.statistics_, X_all.iloc[:s].median().to_numpy(), rtol=0, atol=0)), f"fold {k}: imputer statistics == nan-median of snapshot rows 0..{s - 1}")
    X_sm, y_sm = SMOTE(random_state=SEED, k_neighbors=3).fit_resample(X_tr, y_all[:s])
    n_maj = int(np.bincount(y_all[:s]).max())
    check(len(y_sm) == 3 * n_maj and all(int((y_sm == c).sum()) == n_maj for c in LABELS), f"fold {k}: SMOTE {len(y_sm)} rows == 3 x {n_maj}")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=SEED).fit(X_sm, y_sm)
    check(list(rf.classes_) == LABELS, f"fold {k}: rf.classes_ == [0, 1, 2]")
    yp_k = rf.predict(X_te).astype(int)
    pp_k = rf.predict_proba(X_te)
    n_diff = int((yp_k != fold[k]["y_pred"].to_numpy(int)).sum())
    max_dp = float(np.abs(pp_k - fold[k][PROBA].to_numpy()).max())
    check(n_diff == 0, f"fold {k}: refit y_pred identical to t33_predictions_rf_fold{k}.csv ({n_diff} days differ)")
    check(max_dp < 1e-9, f"fold {k}: refit probabilities == file (max abs diff {max_dp:.1e})")
    check(r4(f1m(y_all[s:e], yp_k)) == CLAIM[k]["f1_macro"], f"fold {k}: refit F1 macro {f1m(y_all[s:e], yp_k):.6f} == claim {CLAIM[k]['f1_macro']}")

print("\n--- R7. Table 11 and the Section 4.8 / Discussion numbers (KBS_Manuscript_R1.docx) ---")
xml = zipfile.ZipFile(ROOT / "revision" / "manuscript" / "KBS_Manuscript_R1.docx").read("word/document.xml").decode("utf8")
txt = re.sub(r"<[^>]+>", "", re.sub(r"</w:tc>", " | ", re.sub(r"</w:p>", "\n", xml)))
i0, i1 = txt.index("Table 11. Expanding-window evaluation"), txt.index("Figure 4. Expanding-window evaluation")
cells = [c.strip() for c in txt[i0:i1].replace("\n", " ").split("|")]
cells = [c for c in cells if c]
print("Table 11 cells:", cells[9:])


def ci3(v, lo, hi, signed=False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"


exp_rows = {}
for k in range(1, 5):
    y, p = fold[k]["y_true"].to_numpy(int), fold[k]["y_pred"].to_numpy(int)
    bm = boot_paired(y, p, pers[k], f1m, idx)
    ba = boot_paired(y, p, pers[k], f1a, idx)
    pm = point(y, pers[k])
    exp_rows[k] = [ci3(pts[k]["f1_macro"], bm["a_lo"], bm["a_hi"]), ci3(pts[k]["f1_active"], ba["a_lo"], ba["a_hi"]),
                   ci3(pm["f1_macro"], bm["b_lo"], bm["b_hi"]), ci3(bm["point"], bm["lo"], bm["hi"], signed=True)]
exp_rows["pooled"] = [ci3(m["f1_macro"], *ci["f1_macro"]), ci3(m["f1_active"], *ci["f1_active"])]
for k, cells_k in exp_rows.items():
    for s_ in cells_k:
        present = s_ in cells
        if not present:
            near = [c for c in cells if c[:5] == s_[:5] or (s_.startswith("+") and c[:6] == s_[:6])]
            print(f"  Table 11 fold {k}: expected cell '{s_}' NOT found; nearest printed cell(s): {near}")
        check(present, f"Table 11 fold {k}: cell '{s_}' present (values rounded once from the full-precision csv)")
# the two double-rounded cells, if present, are what the register's 4-decimal figures give when rounded again
dbl = {f"{0.3095:.3f}": "0.310", f"{0.2735:.3f}": "0.274"}
print("rounding 0.3095 -> %.3f (python, binary half-up effect) ; 0.2735 -> %.3f" % (0.3095, 0.2735))
print("csv fold 4 f1_macro_ci_low %.6f -> %.3f ; pers_f1_macro_ci_low %.6f -> %.3f" % (
    float(t33[t33.fold == "4"].iloc[0].f1_macro_ci_low), float(t33[t33.fold == "4"].iloc[0].f1_macro_ci_low),
    float(t33[t33.fold == "4"].iloc[0].pers_f1_macro_ci_low), float(t33[t33.fold == "4"].iloc[0].pers_f1_macro_ci_low)))
# train / test columns of Table 11
for k, lab in zip(range(1, 5), ["766 (38)", "574 (22)", "382 (14)", "190 (9)"]):
    check(lab in cells and f"{CLAIM[k]['test'][0]}/{CLAIM[k]['test'][1]}/{CLAIM[k]['test'][2]}" in cells, f"Table 11 fold {k}: train '{lab}' and test R/N/A cells present")
check("166/565/37" in cells, "Table 11 pooled: 166/565/37 present")
top = pd.read_csv(RES / "t33_shap_top3.csv")
top1 = {int(r.fold): r.feature for _, r in top[top["rank"] == 1].iterrows()}
check(top1 == {1: "aw_hr_mean", 2: "oura_readiness_score", 3: "oura_readiness_score", 4: "aw_hr_mean"}, f"SHAP rank 1 per fold {top1} (Table 11: mean heart rate / readiness / readiness / mean heart rate)")
# Section 4.8 / Discussion / Response sentences
s48 = [f"{pts[4]['f1_macro']:.3f}", f"{pts[3]['f1_macro']:.3f}", f"{pts[2]['f1_macro']:.3f}", f"{pts[1]['f1_macro']:.3f}",
       f"{pts[4]['f1_active']:.3f}", f"{pts[3]['f1_active']:.3f}", f"{pts[2]['f1_active']:.3f}", f"{pts[1]['f1_active']:.3f}"]
print("4.8 numbers from recomputation (macro F1 folds 4,3,2,1; F1 Active folds 4,3,2,1):", s48)
check(s48 == ["0.361", "0.377", "0.312", "0.506", "0.000", "0.074", "0.103", "0.444"], "4.8: 'Macro F1 is 0.361, 0.377 and 0.312 ... against 0.506' and 'F1 Active is 0.000, 0.074 and 0.103 against 0.444'")
pers3 = point(fold[3]["y_true"].to_numpy(int), pers[3])["f1_macro"]
check(f"{pts[3]['f1_macro']:.3f}" == "0.377" and f"{pers3:.3f}" == "0.412" and pts[3]["f1_macro"] < pers3, "4.8: 'below persistence (0.377 against 0.412)' on the Jul 2024 to Feb 2025 block (fold 3)")
excl = [k for k in range(1, 5) if not (boot_paired(fold[k]["y_true"].to_numpy(int), fold[k]["y_pred"].to_numpy(int), pers[k], f1m, idx)["lo"] <= 0)]
neg = [k for k in range(1, 5) if pts[k]["f1_macro"] < point(fold[k]["y_true"].to_numpy(int), pers[k])["f1_macro"]]
check(excl == [1] and neg == [3], f"4.8 / Discussion / Response: delta CI excludes zero only on fold {excl}; RF below persistence on fold {neg} (one block)")
check(sum(top1[k] != "aw_hr_mean" for k in range(1, 5)) == 2, "Discussion / Response: 'the leading SHAP feature changes on two blocks' (folds 2 and 3)")
rng_macro = (min(pts[k]["f1_macro"] for k in (2, 3, 4)), max(pts[k]["f1_macro"] for k in (2, 3, 4)))
rng_act = (min(pts[k]["f1_active"] for k in (2, 3, 4)), max(pts[k]["f1_active"] for k in (2, 3, 4)))
check((f"{rng_macro[0]:.2f}", f"{rng_macro[1]:.2f}") == ("0.31", "0.38") and (f"{rng_act[0]:.2f}", f"{rng_act[1]:.2f}") == ("0.00", "0.10"),
      f"Discussion: 'macro F1 is 0.31 to 0.38 and F1 Active 0.00 to 0.10' on the earlier blocks ({rng_macro[0]:.4f}-{rng_macro[1]:.4f}, {rng_act[0]:.4f}-{rng_act[1]:.4f})")
check(f"{rng_act[0]:.3f}" == "0.000" and f"{rng_act[1]:.3f}" == "0.103", "Response: 'F1 for the Active class 0.000 to 0.103'")
check(f"{m['f1_macro']:.3f}" == "0.393" and ci3(m["f1_macro"], *ci["f1_macro"]) == "0.393 [0.352, 0.435]" and ci3(m["f1_active"], *ci["f1_active"]) == "0.139 [0.052, 0.232]",
      "4.8: pooled 'macro F1 is 0.393 [0.352, 0.435] and F1 Active 0.139 [0.052, 0.232]'")
best = max(range(1, 5), key=lambda k: pts[k]["f1_macro"])
check(best == 1 and max(range(1, 5), key=lambda k: pts[k]["f1_active"]) == 1, "4.8: 'the final block is the most favorable of the four' (fold 1 highest on macro F1 and F1 Active)")
rest_prev = [int((fold[k]["y_true"] == 0).sum()) / BLOCK * 100 for k in range(1, 5)]
check(round(min(rest_prev)) == 15 and round(max(rest_prev)) == 31, f"Discussion: Rest prevalence 15% to 31% ({[round(v, 1) for v in rest_prev]})")
n_act_jul = int(((y_all == 2) & (snap['date'] >= pd.Timestamp('2024-07-01'))).sum())
check(n_act_jul == 33, f"Discussion: '33 of the 46 Active days fall on or after July 2024' ({n_act_jul})")
mono = [pts[k]["f1_macro"] for k in (4, 3, 2, 1)]
check(not all(mono[i] <= mono[i + 1] for i in range(3)), f"4.8: 'does not increase monotonically with training size' (train 190 -> 766: {[round(v, 4) for v in mono]})")

print("\n" + "=" * 100)
print(f"{n_checks} checks; " + (f"{len(fails)} FAIL" if fails else "ALL PASS"))
for f in fails:
    print("  FAIL " + f)
