"""
xml_vs_json_accuracy.py -- RF accuracy pe 16 caracteristici, sursa XML vs JSON.

READ-ONLY: incarca dataseturile 16f EXISTENTE de pe disc, NU le reconstruieste
si NU suprascrie nimic. Reproduce exact _train_eval_core din cap5_pipeline.py:
  - split temporal 80/20 cronologic (iloc, ordinea randurilor e deja cronologica)
  - SimpleImputer(strategy="median"), fit pe train, transform pe test
  - SMOTE(random_state=42, k_neighbors=3) doar pe train
  - RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5,
    random_state=42)

NU atinge dataset_combinat_json_extended.csv (canonic, hash 63bc3a2915bae864)
si nici modelul frozen. Verificabil cu sanity_check_frozen.py inainte/dupa.

Output: outputs/tables/xml_vs_json_accuracy.csv
"""
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# Cele 16 caracteristici lag-1 (identice cu LAG_FEATURES din cap5_pipeline.py)
FEATURES = [
    "readiness_score_lag1", "sleep_score_lag1", "hrv_mean_lag1", "hrv_balance_lag1",
    "recovery_index_lag1", "rhr_lowest_lag1", "rhr_avg_lag1", "activity_score_lag1",
    "steps_oura_lag1", "respiratory_rate_lag1", "temp_deviation_lag1",
    "aw_hr_mean_lag1", "aw_hrv_mean_lag1", "aw_steps_lag1",
    "aw_active_calories_lag1", "aw_spo2_lag1",
]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def rf_temporal(df):
    """Reproduce _train_eval_core pe split temporal 80/20."""
    X = df[FEATURES]
    y = df["nutrition_label"].astype(int)
    split_idx = int(len(df) * 0.8)
    X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr, y_te = y.iloc[:split_idx], y.iloc[split_idx:]

    imputer = SimpleImputer(strategy="median")
    X_tr_i = pd.DataFrame(imputer.fit_transform(X_tr), columns=FEATURES, index=X_tr.index)
    X_te_i = pd.DataFrame(imputer.transform(X_te), columns=FEATURES, index=X_te.index)

    X_tr_s, y_tr_s = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_tr_i, y_tr)

    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, max_depth=10, min_samples_leaf=5
    )
    rf.fit(X_tr_s, y_tr_s)
    y_pred = rf.predict(X_te_i)

    return {
        "n_rows": len(df),
        "n_train": split_idx,
        "n_test": len(df) - split_idx,
        "rf_accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "rf_f1_macro": round(float(f1_score(y_te, y_pred, average="macro", zero_division=0)), 4),
    }


def steps_mae():
    """MAE absoluta a pasilor XML (toate sursele, sum/zi) vs JSON (aw_steps), pe zile aliniate.
    Replica agregarea din compare_apple_xml_vs_json.py (read-only)."""
    xml = pd.read_csv("data/apple_health/steps.csv", low_memory=False)
    xml["date"] = pd.to_datetime(xml["startDate"], format="mixed", errors="coerce").dt.date
    xml["_val"] = pd.to_numeric(xml["value"], errors="coerce")
    xml = xml.dropna(subset=["_val", "date"])
    xml_daily = xml.groupby("date")["_val"].sum().reset_index()
    xml_daily.columns = ["date", "xml_steps"]

    js = pd.read_csv("apple_watch_api/apple_health_daily.csv")
    js["date"] = pd.to_datetime(js["date"]).dt.date
    js = js[["date", "aw_steps"]]

    m = pd.merge(xml_daily, js, on="date", how="inner").dropna(subset=["xml_steps", "aw_steps"])
    diff = m["xml_steps"] - m["aw_steps"]
    return {
        "metric": "Steps (XML all sources vs JSON)",
        "n_days": int(len(m)),
        "xml_mean": round(float(m["xml_steps"].mean()), 2),
        "json_mean": round(float(m["aw_steps"].mean()), 2),
        "mean_diff_signed": round(float(diff.mean()), 2),
        "mae_abs": round(float(diff.abs().mean()), 2),
        "median_abs_diff": round(float(diff.abs().median()), 2),
        "corr": round(float(m["xml_steps"].corr(m["aw_steps"])), 4),
    }


CONFIGS = [
    ("XML (16f nativ)", "data/dataset_combinat.csv", "cap5_pipeline.py", None),
    ("JSON (16f nativ)", "data/dataset_combinat_json.csv", "cap5_pipeline_json.py", None),
    ("JSON (19f-extended drop 3)", "data/dataset_combinat_json_extended.csv",
     "compare_16f_vs_19f: 19f minus 3 feat AW", "drop3"),
]

rows = []
print("=" * 78)
print("  RF Accuracy 16 caracteristici -- XML vs JSON (read-only, fara rebuild)")
print("=" * 78)
for source, path, origin, mode in CONFIGS:
    df = pd.read_csv(path)
    if mode == "drop3":
        df = df[FEATURES + ["nutrition_label"]]
    res = rf_temporal(df)
    res.update({
        "source": source, "dataset_file": path, "built_by": origin,
        "n_features": 16, "input_sha256_16": file_hash(path),
    })
    rows.append(res)
    print(f"  {source:30s} acc={res['rf_accuracy']:.4f}  f1_macro={res['rf_f1_macro']:.4f}"
          f"  (n={res['n_rows']}, test={res['n_test']})  file={path}")

out = pd.DataFrame(rows)[[
    "source", "dataset_file", "built_by", "n_features",
    "n_rows", "n_train", "n_test", "rf_accuracy", "rf_f1_macro", "input_sha256_16",
]]

out_path = Path("outputs/tables/xml_vs_json_accuracy.csv")
with open(out_path, "w", encoding="utf-8", newline="") as f:
    f.write("# xml_vs_json_accuracy.csv -- RF accuracy 16 caracteristici, sursa XML vs JSON\n")
    f.write("# generated_by: scripts/canonical/xml_vs_json_accuracy.py\n")
    f.write("# conditii: split temporal 80/20, SMOTE k=3, random_state=42, RF(100, max_depth=10, min_samples_leaf=5)\n")
    f.write("# read-only: dataseturile NU au fost reconstruite; canonic 63bc3a2915bae864 neatins\n")
    out.to_csv(f, index=False, lineterminator="\n")

print(f"\n  Scris: {out_path}")

# ── Steps MAE absoluta (XML all vs JSON) ─────────────────────────────
print("\n  --- Steps MAE absoluta (XML all sources vs JSON) ---")
s = steps_mae()
for k, v in s.items():
    print(f"    {k}: {v}")

steps_path = Path("outputs/tables/steps_xml_vs_json_mae.csv")
with open(steps_path, "w", encoding="utf-8", newline="") as f:
    f.write("# steps_xml_vs_json_mae.csv -- MAE absoluta pasi XML (toate sursele) vs JSON, pe zi\n")
    f.write("# generated_by: scripts/canonical/xml_vs_json_accuracy.py\n")
    f.write("# read-only: agregare din data/apple_health/steps.csv + apple_watch_api/apple_health_daily.csv\n")
    f.write("# mae_abs = mean(|xml_steps - json_steps|) ; mean_diff_signed = mean(xml - json)\n")
    pd.DataFrame([s]).to_csv(f, index=False, lineterminator="\n")
print(f"  Scris: {steps_path}")
