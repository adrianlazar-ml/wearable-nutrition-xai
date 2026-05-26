"""
Cap.5 - Pipeline ML (v3 - JSON + Extended Features)
=====================================================
Bazat pe cap5_pipeline_json.py, cu 3 features NOI din JSON:
- aw_sleep_deep_hours (sleep stage deep, 99% coverage)
- aw_sleep_rem_hours (sleep stage REM, 99% coverage)
- aw_resp_rate (Apple Watch respiratory rate, 99% coverage)

Total: 19 features (16 originale + 3 noi) vs 16 features.
Scopul: testăm dacă sleep stages îmbunătățesc clasificarea.

Fix v3.1: rhr_avg < rhr_lowest → NaN (valori fiziologic imposibile
din bug agregare API sleep_metrics.csv, 60 zile afectate).

Autor: Adrian Lazar | Disertație UPB 2026
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# 1. LOAD DATE OURA COMBINED
# ============================================================
def load_oura():
    oura = pd.read_csv("data/oura_combined.csv")
    oura["date"] = pd.to_datetime(oura["date"])
    print(f"Oura: {len(oura)} zile ({oura['date'].min().date()} -> {oura['date'].max().date()})")
    print(f"Gen3: {(oura['oura_device']=='Gen3').sum()} | Gen4: {(oura['oura_device']=='Gen4').sum()}")
    return oura


# ============================================================
# 2. LOAD DATE APPLE WATCH — din JSON (Health Auto Export)
# ============================================================
def load_apple_watch():
    """
    Citeste datele Apple Watch din apple_health_daily.csv
    (generat de parse_health_auto_export.py din JSON-urile Health Auto Export).

    Diferente vs XML:
    - Steps: deduplicate (JSON) vs dublu-contorizate (XML)
    - Active energy: kJ in JSON, convertim in kcal
    - SpO2: % in JSON (93.6), fractie in XML (0.93)
    """
    csv_path = "apple_watch_api/apple_health_daily.csv"
    aw = pd.read_csv(csv_path)
    aw["date"] = pd.to_datetime(aw["date"])

    print(f"Apple Watch (JSON): {len(aw)} zile ({aw['date'].min().date()} -> {aw['date'].max().date()})")
    print(f"Columns available: {len(aw.columns)}")

    # Convertim active energy din kJ in kcal
    aw["aw_active_calories"] = (aw["aw_active_energy_kj"] / 4.184).round(1)

    # SpO2: JSON vine ca % (93.6), pipeline-ul original folosea fractie (0.93)
    # Pastram ca % — modelul invata pe scala noua, nu comparam coeficienti direct
    aw["aw_spo2"] = aw["aw_spo2"]

    # Selectam coloanele necesare (echivalente cu XML + NOILE features)
    hr_aw = aw[["date", "aw_hr_mean"]].dropna(subset=["aw_hr_mean"])
    hrv_aw = aw[["date", "aw_hrv_mean"]].dropna(subset=["aw_hrv_mean"])
    steps_aw = aw[["date", "aw_steps"]].dropna(subset=["aw_steps"])
    calories_aw = aw[["date", "aw_active_calories"]].dropna(subset=["aw_active_calories"])
    spo2_aw = aw[["date", "aw_spo2"]].dropna(subset=["aw_spo2"])

    # NOI: sleep stages + resp rate
    sleep_deep_aw = aw[["date", "aw_sleep_deep_hours"]].dropna(subset=["aw_sleep_deep_hours"])
    sleep_rem_aw = aw[["date", "aw_sleep_rem_hours"]].dropna(subset=["aw_sleep_rem_hours"])
    resp_aw = aw[["date", "aw_resp_rate"]].dropna(subset=["aw_resp_rate"])

    print(f"HR Apple Watch: {len(hr_aw)} zile")
    print(f"HRV Apple Watch: {len(hrv_aw)} zile")
    print(f"Steps Apple Watch: {len(steps_aw)} zile (deduplicated)")
    print(f"Calories Apple Watch: {len(calories_aw)} zile (kJ→kcal)")
    print(f"SpO2 Apple Watch: {len(spo2_aw)} zile (as %)")
    print(f"Sleep Deep: {len(sleep_deep_aw)} zile [NOU]")
    print(f"Sleep REM:  {len(sleep_rem_aw)} zile [NOU]")
    print(f"Resp Rate AW: {len(resp_aw)} zile [NOU]")

    return hr_aw, hrv_aw, steps_aw, calories_aw, spo2_aw, sleep_deep_aw, sleep_rem_aw, resp_aw


# ============================================================
# 3. MERGE PE DATA + DATA QUALITY FIX
# ============================================================
def merge_datasets(oura, hr_aw, hrv_aw, steps_aw, calories_aw, spo2_aw,
                   sleep_deep_aw, sleep_rem_aw, resp_aw):
    df = oura.merge(hr_aw, on="date", how="left")
    df = df.merge(hrv_aw, on="date", how="left")
    df = df.merge(steps_aw, on="date", how="left")
    df = df.merge(calories_aw, on="date", how="left")
    df = df.merge(spo2_aw, on="date", how="left")
    df = df.merge(sleep_deep_aw, on="date", how="left")
    df = df.merge(sleep_rem_aw, on="date", how="left")
    df = df.merge(resp_aw, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    # ── DATA QUALITY FIX: rhr_avg ──────────────────────────────
    # Bug identificat în agregarea API (sleep_metrics.csv): pe ~60 zile
    # rhr_avg < rhr_lowest, ceea ce este fiziologic imposibil (media
    # ritmului cardiac nu poate fi mai mică decât minimul).
    # Cauza: sleep_metrics.csv agregă rhr_avg peste toate sesiunile
    # de somn (long_sleep + naps + rest), incluzând sesiuni parțiale
    # cu HR foarte scăzut sau incomplet.
    # Fix: setăm valorile imposibile la NaN → imputare cu mediană.
    # Afectează exclusiv perioada API v2 (iul 2025+), toate pe Gen4.
    invalid_rhr = df["rhr_avg"] < df["rhr_lowest"]
    n_invalid = invalid_rhr.sum()
    if n_invalid > 0:
        df.loc[invalid_rhr, "rhr_avg"] = np.nan
        print(f"  rhr_avg fix: {n_invalid} zile cu rhr_avg < rhr_lowest → NaN")
    # ───────────────────────────────────────────────────────────

    print(f"Dataset combinat: {len(df)} zile, {df.shape[1]} coloane")
    return df


# ============================================================
# 4. FEATURE ENGINEERING - LAGGED FEATURES
# ============================================================
LAG_FEATURES = [
    "readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
    "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
    "steps_oura", "respiratory_rate", "temp_deviation",
    "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2",
    # NOI din JSON (sleep stages + AW resp rate)
    "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate",
]

FEATURES = [f"{feat}_lag1" for feat in LAG_FEATURES]


def assign_nutrition_label(row):
    if pd.isna(row["readiness_score"]) or pd.isna(row["sleep_score"]):
        return np.nan
    if row["readiness_score"] < 60 or row["sleep_score"] < 60:
        return 0
    elif row["readiness_score"] >= 85 and row["sleep_score"] >= 75:
        return 2
    else:
        return 1


def engineer_features(df):
    for feat in LAG_FEATURES:
        df[f"{feat}_lag1"] = df[feat].shift(1)

    df["nutrition_label"] = df.apply(assign_nutrition_label, axis=1)

    print("Distributie labels:")
    print(df["nutrition_label"].value_counts().sort_index())

    df_model = df[["date"] + FEATURES + ["nutrition_label"]].dropna(subset=["nutrition_label"])
    df_model = df_model.sort_values("date").reset_index(drop=True)
    print(f"Randuri pentru model (inainte de dropna features): {len(df_model)}")

    return df_model


# ============================================================
# 5. TRAIN / EVALUATE
# ============================================================
def _train_eval_core(X_train, X_test, y_train, y_test, split_name=""):
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=FEATURES, index=X_test.index
    )

    print(f"  Train inainte SMOTE: {y_train.value_counts().sort_index().to_dict()}")

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

    print(f"  Train dupa SMOTE:    {pd.Series(y_train_sm).value_counts().sort_index().to_dict()}")

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_leaf=5,
    )
    rf.fit(X_train_sm, y_train_sm)

    y_pred = rf.predict(X_test_imp)
    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n  Accuracy:  {acc:.2%}")
    print(f"  F1 Macro:  {f1_mac:.2%}")
    print(f"\n  Classification Report ({split_name}):")
    print(classification_report(
        y_test, y_pred,
        target_names=["Rest", "Normal", "Active"],
        zero_division=0
    ))

    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return rf, imputer, fi, acc, f1_mac


def train_and_evaluate(df_model):
    X = df_model[FEATURES]
    y = df_model["nutrition_label"].astype(int)
    dates = df_model["date"]

    # A) SPLIT TEMPORAL
    print("\n--- A) SPLIT TEMPORAL (80/20 cronologic) ---")
    split_idx = int(len(df_model) * 0.8)

    X_train_t = X.iloc[:split_idx]
    X_test_t = X.iloc[split_idx:]
    y_train_t = y.iloc[:split_idx]
    y_test_t = y.iloc[split_idx:]

    train_end = dates.iloc[split_idx - 1].strftime("%Y-%m-%d")
    test_start = dates.iloc[split_idx].strftime("%Y-%m-%d")
    test_end = dates.iloc[-1].strftime("%Y-%m-%d")
    print(f"  Train: {len(X_train_t)} zile (-> {train_end})")
    print(f"  Test:  {len(X_test_t)} zile ({test_start} -> {test_end})")
    print(f"  Test distributie: {y_test_t.value_counts().sort_index().to_dict()}")

    rf_t, imputer_t, fi_t, acc_t, f1_t = _train_eval_core(
        X_train_t, X_test_t, y_train_t, y_test_t, "temporal"
    )

    cv_pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("rf", RandomForestClassifier(
            n_estimators=100, random_state=42,
            max_depth=10, min_samples_leaf=5
        ))
    ])
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores_t = cross_val_score(cv_pipeline, X, y, cv=tscv, scoring="accuracy")
    f1_cv_t = cross_val_score(cv_pipeline, X, y, cv=tscv, scoring="f1_macro")
    print(f"  TimeSeriesSplit CV (5-fold): acc={cv_scores_t.mean():.2%} (+/- {cv_scores_t.std():.2%})")
    print(f"  TimeSeriesSplit CV (5-fold): f1_macro={f1_cv_t.mean():.2%} (+/- {f1_cv_t.std():.2%})")

    # B) SPLIT RANDOM
    print("\n--- B) SPLIT RANDOM STRATIFICAT (baseline comparativ) ---")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train_r)} | Test: {len(X_test_r)}")

    _, _, _, acc_r, f1_r = _train_eval_core(
        X_train_r, X_test_r, y_train_r, y_test_r, "random"
    )

    cv_scores_r = cross_val_score(cv_pipeline, X, y, cv=5, scoring="accuracy")
    f1_cv_r = cross_val_score(cv_pipeline, X, y, cv=5, scoring="f1_macro")
    print(f"  StratifiedKFold CV (5-fold): acc={cv_scores_r.mean():.2%} (+/- {cv_scores_r.std():.2%})")
    print(f"  StratifiedKFold CV (5-fold): f1_macro={f1_cv_r.mean():.2%} (+/- {f1_cv_r.std():.2%})")

    # C) SUMAR
    print("\n" + "=" * 60)
    print("  SUMAR: TEMPORAL vs RANDOM SPLIT")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Temporal':>12} {'Random':>12}")
    print(f"  {'-'*49}")
    print(f"  {'Test Accuracy':<25} {acc_t:>11.2%} {acc_r:>11.2%}")
    print(f"  {'Test F1 Macro':<25} {f1_t:>11.2%} {f1_r:>11.2%}")
    print(f"  {'CV Accuracy':<25} {cv_scores_t.mean():>11.2%} {cv_scores_r.mean():>11.2%}")
    print(f"  {'CV F1 Macro':<25} {f1_cv_t.mean():>11.2%} {f1_cv_r.mean():>11.2%}")

    print("\n  Feature Importance (top 10, model temporal):")
    print(fi_t.head(10))

    return rf_t, imputer_t, fi_t, acc_t, f1_t, cv_scores_t, f1_cv_t


# ============================================================
# 6. MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  CAP.5 - PIPELINE ML (JSON + EXTENDED FEATURES)")
    print("  Oura Ring + Apple Watch via Health Auto Export")
    print("  19 features (16 originale + 3 sleep/resp NOI)")
    print("  + rhr_avg data quality fix (v3.1)")
    print("=" * 60)

    print("\n=== LOAD DATE OURA ===")
    oura = load_oura()

    print("\n=== LOAD DATE APPLE WATCH (JSON + EXTENDED) ===")
    hr_aw, hrv_aw, steps_aw, calories_aw, spo2_aw, sleep_deep_aw, sleep_rem_aw, resp_aw = load_apple_watch()

    print("\n=== MERGE DATE ===")
    df = merge_datasets(oura, hr_aw, hrv_aw, steps_aw, calories_aw, spo2_aw,
                        sleep_deep_aw, sleep_rem_aw, resp_aw)

    print("\n=== FEATURE ENGINEERING (lag t-1) ===")
    df_model = engineer_features(df)

    print("\n=== TRAIN & EVALUATE (temporal + random) ===")
    rf, imputer, fi, acc_t, f1_t, cv_acc, cv_f1 = train_and_evaluate(df_model)

    # Salvare dataset JSON
    df_model[FEATURES + ["nutrition_label"]].to_csv(
        "data/dataset_combinat_json_extended.csv", index=False
    )
    print(f"\nDataset salvat: {len(df_model)} randuri -> data/dataset_combinat_json_extended.csv")

    return rf, imputer, fi


if __name__ == "__main__":
    rf, imputer, fi = main()