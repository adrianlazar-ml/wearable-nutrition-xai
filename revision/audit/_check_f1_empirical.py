"""_check_f1_empirical.py - AUDIT F1 (T2/T4): empirical checks on the split chain.
Imports revision/scripts/common.py (allowed for T2/T4). No retraining, no file writes.
Run from repo root: python revision/audit/_check_f1_empirical.py
"""
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "scripts"))
import common  # noqa: E402
from common import LAG_FEATURES, FEATURES, build_dataset, canonical_split, load_raw  # noqa: E402
import t3_labels  # noqa: E402  (resolved via common's sys.path insert -> revision/t3_labels.py)

print("t3_labels module file:", t3_labels.__file__)
raw = load_raw()
oura = pd.read_csv(ROOT / "data" / "oura_combined.csv")
print("oura rows", len(oura), "unique dates", oura["date"].nunique())
print("raw rows", len(raw), raw["date"].min().date(), "->", raw["date"].max().date(),
      "| duplicate dates:", int(raw["date"].duplicated().sum()), "| sorted:", raw["date"].is_monotonic_increasing)
print("raw first 3 rows readiness/sleep:", raw[["date", "readiness_score", "sleep_score"]].head(3).to_dict("records"))

# --- Q4 drop criteria
lab = t3_labels.assign_labels(raw)
print("label NaN:", int(lab.isna().sum()), "| readiness NaN:", int(raw["readiness_score"].isna().sum()),
      "| sleep NaN:", int(raw["sleep_score"].isna().sum()),
      "| either NaN:", int((raw["readiness_score"].isna() | raw["sleep_score"].isna()).sum()))
X, y, dates = build_dataset(raw)
print("built n:", len(X), "dist:", y.value_counts().sort_index().to_dict(),
      dates.min().date(), "->", dates.max().date(), "| sorted:", dates.is_monotonic_increasing)
print("dropped rows whose 19 lag features are all NaN:", int(raw.loc[lab.isna(), LAG_FEATURES].isna().all(axis=1).sum()))
print("dropped rows, dates:", [str(d.date()) for d in raw.loc[lab.isna(), "date"]])

# --- Q1 lag direction (code: df[f].shift(1) on raw frame sorted by date)
raw2 = raw.copy()
for f in LAG_FEATURES:
    raw2[f"{f}_lag1"] = raw2[f].shift(1)
for d in ["2024-01-15", "2025-09-19", "2026-03-30"]:
    i = raw2.index[raw2["date"] == d][0]
    print(f"Q1 {d}: readiness_score_lag1={raw2.loc[i, 'readiness_score_lag1']} | prev raw row date="
          f"{raw2.loc[i-1, 'date'].date()} readiness={raw2.loc[i-1, 'readiness_score']} | same-day readiness="
          f"{raw2.loc[i, 'readiness_score']} label={lab.loc[i]}")
bad = []
for f in LAG_FEATURES:
    a = raw2[f"{f}_lag1"].iloc[1:].to_numpy(dtype=float)
    b = raw2[f].iloc[:-1].to_numpy(dtype=float)
    if not np.array_equal(a, b, equal_nan=True):
        bad.append(f)
print("Q1 identity lag1(row t) == raw(row t-1) for all 19 over all raw rows; mismatching features:", bad)
Xr = raw2.set_index("date")
for d in ["2024-01-15", "2025-09-19", "2026-03-30"]:
    k = dates.index[dates == d][0]
    print(f"Q1 built-frame {d}: X.readiness_score_lag1={X.iloc[k]['readiness_score_lag1']} ; raw-frame lag1 at same date={Xr.loc[pd.Timestamp(d), 'readiness_score_lag1']}")

# --- Q3 gaps
g_raw = raw["date"].diff().dt.days
print("Q3 raw consecutive-row gaps >1 day:", int((g_raw > 1).sum()), "| max gap:", g_raw.max())
g = dates.diff().dt.days
print("Q3 958-frame consecutive-row gaps >1 day:", int((g > 1).sum()), "| gap sizes:", g[g > 1].value_counts().sort_index().to_dict())
print("Q3 958-frame gap positions (date of row after gap):", [str(dates.iloc[k].date()) for k in g[g > 1].index][:60])
mism = prev_missing = 0
for k in range(len(X)):
    d = dates.iloc[k]
    pd1 = d - pd.Timedelta(days=1)
    if pd1 in Xr.index:
        v = Xr.loc[pd1, "readiness_score"]
        l = X.iloc[k]["readiness_score_lag1"]
        if not ((pd.isna(v) and pd.isna(l)) or v == l):
            mism += 1
    else:
        prev_missing += 1
print("Q3 lag1 vs calendar-previous-day readiness: mismatches", mism, "| rows whose calendar prev day absent from raw:", prev_missing)

# --- Q5 split
s = canonical_split(X, y, dates)
i = int(len(X) * common.TRAIN_RATIO)
print("Q5 split idx", i, "| train end", dates.iloc[i-1].date(), "| test start", dates.iloc[i].date(), "| test end", dates.iloc[-1].date())
print("Q5 train dist", y.iloc[:i].value_counts().sort_index().to_dict(), "| test dist", s.y_test.value_counts().sort_index().to_dict(), "| n_test", len(s.y_test))
g_test = dates.diff().dt.days.iloc[i:]
print("Q3 test rows whose previous row is >1 day back:", int((g_test > 1).sum()), [str(dates.iloc[k].date()) for k in g_test[g_test > 1].index])
print("Q3 train rows whose previous row is >1 day back:", int((dates.diff().dt.days.iloc[1:i] > 1).sum()))
print("Q3 gap between train end and test start (days):", (dates.iloc[i] - dates.iloc[i-1]).days)

# --- Q6 imputer / Q8 scaler
med_train = X.iloc[:i].median().to_numpy()
med_all = X.median().to_numpy()
print("Q6 imputer.statistics_ == median(train 766):", bool(np.allclose(s.imputer.statistics_, med_train)),
      "| == median(all 958):", bool(np.allclose(s.imputer.statistics_, med_all)))
print("Q6 NaN cells: train", int(X.iloc[:i].isna().sum().sum()), "| test", int(X.iloc[i:].isna().sum().sum()))
imp_rev = joblib.load(ROOT / "revision/results/models/imputer_revision.joblib")
imp_fro = joblib.load(ROOT / "models/imputer_frozen_20260331.joblib")
print("Q6 imputer_revision == recomputed:", bool(np.array_equal(imp_rev.statistics_, s.imputer.statistics_)),
      "| imputer_frozen == recomputed:", bool(np.array_equal(imp_fro.statistics_, s.imputer.statistics_)))
print("Q8 scaler.mean_ == mean(imputed train 766):", bool(np.allclose(s.scaler.mean_, s.X_train.mean().to_numpy())),
      "| scaler.n_samples_seen_:", s.scaler.n_samples_seen_)
sc_rev = joblib.load(ROOT / "revision/results/models/scaler_revision.joblib")
print("Q8 scaler_revision == recomputed:", bool(np.allclose(sc_rev.mean_, s.scaler.mean_) and np.allclose(sc_rev.scale_, s.scaler.scale_)))

# --- Q7 SMOTE
print("Q7 rows after SMOTE (trees):", len(s.X_train_sm), s.y_train_sm.value_counts().sort_index().to_dict(),
      "| rows after SMOTE (scaled, SVM):", len(s.X_train_scaled_sm), "| X_test rows:", len(s.X_test), "| X_test_scaled rows:", len(s.X_test_scaled))
print("Q7 first 766 rows of SMOTE output == imputed train (SMOTE appends synthetic rows):",
      bool(np.array_equal(s.X_train_sm.iloc[:i].to_numpy(), s.X_train.to_numpy())))

# --- Q15 RF params + prediction identity without retraining
rf = joblib.load(ROOT / "revision/results/models/rf_revision.joblib")
p = rf.get_params()
print("Q15 rf_revision params:", {k: p[k] for k in ["n_estimators", "max_depth", "min_samples_leaf", "random_state", "class_weight", "max_features", "criterion", "bootstrap"]},
      "| n_features_in_", rf.n_features_in_, "| classes_", rf.classes_.tolist())
rf_fro = joblib.load(ROOT / "models/rf_frozen_20260331.joblib")
print("Q15 rf_frozen params same:", rf_fro.get_params() == p)
csv = pd.read_csv(ROOT / "revision/results/test_predictions_rf.csv")
pred = rf.predict(s.X_test)
print("Q15 rf_revision.predict(recomputed X_test) == csv y_pred:", bool(np.array_equal(pred, csv["y_pred"].to_numpy())),
      "| y_true == csv:", bool(np.array_equal(s.y_test.to_numpy(), csv["y_true"].to_numpy())),
      "| dates == csv:", bool((pd.to_datetime(csv["date"]).reset_index(drop=True) == s.dates_test).all()))

# --- snapshot vs rebuilt, Q14 hash
snap = pd.read_csv(ROOT / "models/dataset_snapshot_20260331.csv")
snap["date"] = pd.to_datetime(snap["date"])
print("Q14 snapshot rows", len(snap), "| labels == rebuilt:", bool(np.array_equal(snap["nutrition_label"].astype(int).to_numpy(), y.to_numpy())),
      "| dates == rebuilt:", bool((snap["date"].to_numpy() == dates.to_numpy()).all()),
      "| features == rebuilt (nan-aware):", bool(np.array_equal(snap[FEATURES].to_numpy(dtype=float), X.to_numpy(dtype=float), equal_nan=True)))
snap_sorted = snap.sort_values("date").reset_index(drop=True)
content = snap_sorted[sorted(snap.columns.tolist())].to_csv(index=False, lineterminator="\n").encode("utf-8")
print("Q14 hash(snapshot normalised, as sanity_check_frozen.py:46-48):", hashlib.sha256(content).hexdigest()[:16])
print("Q14 hash(snapshot raw file bytes):", hashlib.sha256((ROOT / "models/dataset_snapshot_20260331.csv").read_bytes()).hexdigest()[:16])
reb = pd.concat([dates, X, y.rename("nutrition_label").astype(float)], axis=1)
print("Q14 hash(rebuilt frame from raw csvs, same normalisation):", hashlib.sha256(reb[sorted(reb.columns)].to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()[:16])

# --- Q13 bootstrap
idx = np.random.default_rng(42).integers(0, 192, size=(1000, 192))
saved = np.load(ROOT / "revision/results/bootstrap_indices_1000x192_seed42.npy")
print("Q13 regenerated == npy:", bool(np.array_equal(idx, saved)), "| dtypes", idx.dtype, saved.dtype, "| bytes equal:", idx.tobytes() == saved.tobytes(), "| shape", saved.shape)
rng = np.random.default_rng(42)
seq = np.stack([rng.integers(0, 192, 192) for _ in range(1000)])
print("Q13 sequential per-resample draws (persistence_baseline.bootstrap_ci:94-98) == matrix:", bool(np.array_equal(seq, idx)))
yt = csv["y_true"].to_numpy()
absent = [k for k, row in enumerate(idx) if len(np.unique(yt[row])) < 3]
print("Q13 resamples with a class absent from y_true:", len(absent), "| min Active count in a resample:", min(int((yt[row] == 2).sum()) for row in idx))

# --- Q10/Q11 persistence, majority
pers = pd.read_csv(ROOT / "revision/results/test_predictions_persistence.csv")
last_train = int(y.iloc[i-1])
recon = np.concatenate([[last_train], yt[:-1]])
print("Q10 persistence csv y_pred == [last_train_label] + y_true[:-1]:", bool(np.array_equal(pers["y_pred"].to_numpy(), recon)),
      "| last train label (", dates.iloc[i-1].date(), ") =", last_train,
      "| first 5 y_pred", pers["y_pred"][:5].tolist(), "| first 5 y_true", pers["y_true"][:5].tolist())
print("Q10 persistence == shift(1) over the 958-labelled series (row-based):", bool(np.array_equal(pers["y_pred"].to_numpy(), y.shift(1).iloc[i:].to_numpy())))
maj = pd.read_csv(ROOT / "revision/results/test_predictions_majority.csv")
print("Q11 majority csv unique y_pred:", maj["y_pred"].unique().tolist(), "| train mode:", int(y.iloc[:i].mode()[0]),
      "| all-958 mode:", int(y.mode()[0]), "| test mode:", int(y.iloc[i:].mode()[0]))

# --- Q9 label rule
r, sl = raw["readiness_score"], raw["sleep_score"]
alt = pd.Series(1.0, index=raw.index)
alt[(r >= 85) & (sl >= 75)] = 2.0
alt[(r < 60) | (sl < 60)] = 0.0
alt[r.isna() | sl.isna()] = np.nan
print("Q9 vectorised assign_labels == row rule from cap5 (lines 110-117):", bool(np.array_equal(alt.to_numpy(), lab.to_numpy(), equal_nan=True)))
print("Q9 boundary counts in raw: readiness==60:", int((r == 60).sum()), "| sleep==60:", int((sl == 60).sum()),
      "| readiness==85:", int((r == 85).sum()), "| sleep==75:", int((sl == 75).sum()))
print("Q9 rows meeting both the Rest and the Active condition:", int((((r < 60) | (sl < 60)) & ((r >= 85) & (sl >= 75))).sum()))
