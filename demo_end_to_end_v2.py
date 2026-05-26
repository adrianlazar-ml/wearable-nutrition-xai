"""
DEMO END-TO-END V2: Sistem AI pentru recomandari nutritionale personalizate
===========================================================================
V2 — Schimbari fata de V1:
  - Sursa Apple Watch: JSON (Health Auto Export) in loc de XML
  - 19 features (16 originale + sleep_deep, sleep_rem, resp_rate)
  - Consistent cu cap5_pipeline_json_extended.py

Flux complet:
1. Incarca datele combinate Oura + Apple Watch (JSON)
2. Pregateste features (lag t-1, 19 features)
3. Antreneaza modelul pe primele 80% zile (split temporal)
4. Prezice clasificarea pentru cea mai recenta zi (nevazuta de model)
5. Genereaza recomandare nutritionala personalizata

Rulare: python demo_end_to_end_v2.py

Autor: Adrian Lazar | Disertatie UPB 2026
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from nutrition_recommender import get_nutrition_recommendation, print_recommendation, USER_PROFILE

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "models/rf_model.joblib"
IMPUTER_PATH = "models/imputer.joblib"

LAG_FEATURES = [
    "readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
    "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
    "steps_oura", "respiratory_rate", "temp_deviation",
    "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2",
    # NOI din JSON (sleep stages + AW resp rate)
    "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate",
]
FEATURES = [f"{f}_lag1" for f in LAG_FEATURES]

LABEL_NAMES = {0: "Rest", 1: "Normal", 2: "Active"}

TRAIN_RATIO = 0.8

# Safety layer: predicții cu confidență scăzută → fallback Normal
CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# 1. LOAD SI PREGATIRE DATE (JSON source)
# ============================================================
def load_full_dataset():
    """Incarca si combina datele din Oura + Apple Watch (JSON)."""
    print("  Incarc date Oura...")
    oura = pd.read_csv("data/oura_combined.csv")
    oura["date"] = pd.to_datetime(oura["date"])

    print("  Incarc date Apple Watch (JSON — Health Auto Export)...")
    aw = pd.read_csv("apple_watch_api/apple_health_daily.csv")
    aw["date"] = pd.to_datetime(aw["date"])

    # Convertim active energy din kJ in kcal
    aw["aw_active_calories"] = (aw["aw_active_energy_kj"] / 4.184).round(1)

    # Merge toate coloanele Apple Watch necesare
    df = oura
    for col in ["aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories",
                "aw_spo2", "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate"]:
        merge_df = aw[["date", col]].dropna(subset=[col])
        df = df.merge(merge_df, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)

    # Lag features
    for feat in LAG_FEATURES:
        df[f"{feat}_lag1"] = df[feat].shift(1)

    # Target
    def assign_label(row):
        if pd.isna(row["readiness_score"]) or pd.isna(row["sleep_score"]):
            return np.nan
        if row["readiness_score"] < 60 or row["sleep_score"] < 60:
            return 0
        elif row["readiness_score"] >= 85 and row["sleep_score"] >= 75:
            return 2
        return 1

    df["nutrition_label"] = df.apply(assign_label, axis=1)

    return df


# ============================================================
# 2. TRAIN SI SALVARE MODEL (SPLIT TEMPORAL)
# ============================================================
def train_and_save(df):
    """Antreneaza RF pe primele 80% zile (split temporal)."""
    print("  Pregatesc dataset pentru training...")
    df_model = df[["date"] + FEATURES + ["nutrition_label"]].dropna(subset=["nutrition_label"])
    df_model = df_model.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df_model) * TRAIN_RATIO)

    X_train = df_model[FEATURES].iloc[:split_idx]
    y_train = df_model["nutrition_label"].astype(int).iloc[:split_idx]

    train_end = df_model["date"].iloc[split_idx - 1].strftime("%Y-%m-%d")
    print(f"  Training pe primele {split_idx} zile (-> {train_end})")
    print(f"  Distributie train: {y_train.value_counts().sort_index().to_dict()}")
    print(f"  Features: {len(FEATURES)} (19 = 16 originale + 3 sleep/resp)")

    # Imputer fit pe train
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURES
    )

    # SMOTE pe train
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)
    X_train_sm = pd.DataFrame(X_train_sm, columns=FEATURES)

    # Train
    print("  Antrenez Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42,
        max_depth=10, min_samples_leaf=5
    )
    rf.fit(X_train_sm, y_train_sm)

    # Salvare
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    print(f"  Model salvat: {MODEL_PATH}")
    print(f"  Imputer salvat: {IMPUTER_PATH}")
    print(f"  Model n_features_in_: {rf.n_features_in_}")

    return rf, imputer


def load_model():
    """Incarca model si imputer din disk."""
    rf = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return rf, imputer


# ============================================================
# 3. PREDICTIE PENTRU CEA MAI RECENTA ZI
# ============================================================
def predict_today(df, rf, imputer):
    """Prezice clasificarea pentru cea mai recenta zi din dataset."""
    df_model = df[["date"] + FEATURES + ["nutrition_label"]].dropna(subset=["nutrition_label"])
    df_model = df_model.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df_model) * TRAIN_RATIO)

    # Ultima zi cu date complete (folosim iloc pentru robustețe)
    df_valid = df.dropna(subset=["nutrition_label"]).sort_values("date")
    last_row = df_valid.iloc[-1]

    # Verificăm că ziua e în afara training set-ului
    pred_date_val = last_row["date"]
    train_end_date = df_model["date"].iloc[split_idx - 1]
    if pred_date_val <= train_end_date:
        print(f"  WARN: Ziua {pred_date_val} e in training set!")

    # Features lag — pastram ca DataFrame
    features = pd.DataFrame(
        [last_row[FEATURES].values],
        columns=FEATURES
    )
    features_imp = pd.DataFrame(
        imputer.transform(features), columns=FEATURES
    )

    # Predictie
    label = rf.predict(features_imp)[0]
    proba = rf.predict_proba(features_imp)[0]

    today_data = {
        "date": last_row["date"],
        "readiness_score": last_row["readiness_score"],
        "sleep_score": last_row["sleep_score"],
        "actual_label": int(last_row["nutrition_label"]),
    }

    return int(label), proba, today_data


# ============================================================
# 4. MAIN - DEMO END-TO-END
# ============================================================
def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 60)
    print("  SISTEM AI - RECOMANDARI NUTRITIONALE PERSONALIZATE")
    print("  V2: JSON Extended (19 features)")
    print("  Oura Ring + Apple Watch -> ML -> Recomandare zilnica")
    print(f"  Rulat la: {now}")
    print("=" * 60)

    # Step 1: Load date
    print("\n[1/4] INCARCARE DATE (JSON source)")
    df = load_full_dataset()
    last_date = df["date"].max().strftime("%Y-%m-%d")
    print(f"  Ultima zi cu date: {last_date}")
    print(f"  Total zile: {len(df)}")

    # Step 2: Train model (sterge modelul vechi si re-antreneaza)
    print("\n[2/4] MODEL ML (split temporal {:.0%}/{:.0%}, 19 features)".format(
        TRAIN_RATIO, 1 - TRAIN_RATIO))

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if os.path.exists(IMPUTER_PATH):
        os.remove(IMPUTER_PATH)

    print("  Antrenez model nou (split temporal, 19 features)...")
    rf, imputer = train_and_save(df)

    # Step 3: Predictie
    print("\n[3/4] PREDICTIE (zi nevazuta de model)")
    label, proba, today = predict_today(df, rf, imputer)

    print(f"  Data:            {today['date'].strftime('%Y-%m-%d') if hasattr(today['date'], 'strftime') else today['date']}")
    print(f"  Readiness Score: {today['readiness_score']}")
    print(f"  Sleep Score:     {today['sleep_score']}")
    print(f"  Label real:      {LABEL_NAMES[today['actual_label']]}")
    print(f"  Label prezis:    {LABEL_NAMES[label]}")
    print(f"  Probabilitati:   Rest={proba[0]:.1%}  Normal={proba[1]:.1%}  Active={proba[2]:.1%}")

    # Safety layer: confidence threshold
    max_conf = max(proba)
    original_label = label
    if max_conf < CONFIDENCE_THRESHOLD:
        label = 1  # Fallback Normal
        print(f"  ⚠ Confidence {max_conf:.1%} < threshold {CONFIDENCE_THRESHOLD:.0%}")
        print(f"    → Fallback: {LABEL_NAMES[original_label]} → Normal (safety layer)")
    else:
        print(f"  ✓ Confidence {max_conf:.1%} >= threshold {CONFIDENCE_THRESHOLD:.0%}")

    match = "CORECT" if label == today["actual_label"] else "INCORECT"
    print(f"  Predictie:       {match}")

    # Step 4: Recomandare nutritionala
    print("\n[4/4] RECOMANDARE NUTRITIONALA")
    rec = get_nutrition_recommendation(label, USER_PROFILE)
    print_recommendation(rec)

    # Sumar final
    print(f"\n{'='*60}")
    print("  SUMAR SISTEM END-TO-END V2")
    print(f"{'='*60}")
    oura_dev = df['oura_device'].iloc[-1] if 'oura_device' in df.columns else "Gen3/Gen4"
    aw_dev = df['aw_device'].iloc[-1] if 'aw_device' in df.columns else "Ultra/Ultra3"
    print(f"  Surse date:     Oura Ring ({oura_dev}) + Apple Watch ({aw_dev})")
    print(f"  Apple Watch:    JSON (Health Auto Export via iCloud)")
    print(f"  Dataset:        {len(df)} zile")
    print(f"  Features:       {len(FEATURES)} (16 originale + 3 sleep/resp)")
    print(f"  Model:          Random Forest (100 trees, max_depth=10)")
    print(f"  Training:       Primele {TRAIN_RATIO:.0%} zile (split temporal)")
    print(f"  Safety layer:   Confidence threshold = {CONFIDENCE_THRESHOLD:.0%}")
    print(f"  Clasificare:    {LABEL_NAMES[label]} (confidence: {max(proba):.1%})"
          f"{' [fallback]' if label != original_label else ''}")
    print(f"  Calorii target: {rec['calories_target']} kcal")
    print(f"  Proteine:       {rec['protein_g']}g | Carbs: {rec['carbs_g']}g | "
          f"Fat: {rec['fat_g']}g")


if __name__ == "__main__":
    main()