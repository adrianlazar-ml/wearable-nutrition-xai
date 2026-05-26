"""
DEMO END-TO-END: Sistem AI pentru recomandari nutritionale personalizate
=======================================================================
Flux complet:
1. Incarca datele combinate Oura + Apple Watch
2. Pregateste features (lag t-1)
3. Antreneaza modelul pe primele 80% zile (split temporal)
4. Prezice clasificarea pentru cea mai recenta zi (nevazuta de model)
5. Genereaza recomandare nutritionala personalizata

v2 fix-uri:
- Training pe primele 80% zile (consistent cu cap5_pipeline.py v3)
- Modelul NU vede zilele recente - predictia e onesta
- Feature names pastrate ca DataFrame (elimina sklearn warnings)
- Sterge modelul vechi la re-antrenare (forteaza consistenta)

v3 fix:
- SMOTE output convertit inapoi la DataFrame cu feature names
  (elimina warning "X does not have valid feature names")
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
    "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2"
]
FEATURES = [f"{f}_lag1" for f in LAG_FEATURES]

LABEL_NAMES = {0: "Rest", 1: "Normal", 2: "Active"}

# Proportia de date folosite pentru training (consistent cu pipeline)
TRAIN_RATIO = 0.8


# ============================================================
# 1. LOAD SI PREGATIRE DATE
# ============================================================
def load_full_dataset():
    """Incarca si combina datele din Oura + Apple Watch."""
    print("  Incarc date Oura...")
    oura = pd.read_csv("data/oura_combined.csv")
    oura["date"] = pd.to_datetime(oura["date"])

    def load_aw(filename, agg="mean"):
        df = pd.read_csv(f"data/apple_health/{filename}")
        df["date"] = pd.to_datetime(df["startDate"]).dt.date
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.groupby("date")["value"].agg(agg).reset_index()

    print("  Incarc date Apple Watch...")
    hr = load_aw("heart_rate.csv", "mean").rename(columns={"value": "aw_hr_mean"})
    hrv = load_aw("hrv.csv", "mean").rename(columns={"value": "aw_hrv_mean"})
    steps = load_aw("steps.csv", "sum").rename(columns={"value": "aw_steps"})
    cals = load_aw("active_calories.csv", "sum").rename(columns={"value": "aw_active_calories"})
    spo2 = load_aw("spo2.csv", "mean").rename(columns={"value": "aw_spo2"})

    df = oura
    for d in [hr, hrv, steps, cals, spo2]:
        df = df.merge(d, on="date", how="left")
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
    """
    Antreneaza RF pe primele 80% zile (split temporal).
    Consistent cu metodologia din cap5_pipeline.py v3.
    """
    print("  Pregatesc dataset pentru training...")
    df_model = df[["date"] + FEATURES + ["nutrition_label"]].dropna(subset=["nutrition_label"])
    df_model = df_model.sort_values("date").reset_index(drop=True)

    # Split temporal: train pe primele 80%
    split_idx = int(len(df_model) * TRAIN_RATIO)

    X_train = df_model[FEATURES].iloc[:split_idx]
    y_train = df_model["nutrition_label"].astype(int).iloc[:split_idx]

    train_end = df_model["date"].iloc[split_idx - 1].strftime("%Y-%m-%d")
    print(f"  Training pe primele {split_idx} zile (-> {train_end})")
    print(f"  Distributie train: {y_train.value_counts().sort_index().to_dict()}")

    # Imputer fit pe train
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURES
    )

    # SMOTE pe train
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)
    X_train_sm = pd.DataFrame(X_train_sm, columns=FEATURES)  # FIX: păstrăm feature names

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
    """
    Prezice clasificarea pentru cea mai recenta zi din dataset.
    Foloseste datele de ieri (lag features) pentru a prezice azi.
    Valideaza ca ziua prezisa NU a fost vazuta la antrenare.
    """
    # Construim df_model consistent cu training
    df_model = df[["date"] + FEATURES + ["nutrition_label"]].dropna(subset=["nutrition_label"])
    df_model = df_model.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df_model) * TRAIN_RATIO)

    # Ultima zi cu date complete
    last_rows = df.dropna(subset=["nutrition_label"]).tail(5)
    last_row = last_rows.iloc[-1]
    prediction_date = last_row.name

    # Verificam ca ziua e in afara training set-ului
    pred_date_val = df.loc[prediction_date, "date"]
    train_end_date = df_model["date"].iloc[split_idx - 1]
    if pred_date_val <= train_end_date:
        print(f"  WARN: Ziua {pred_date_val} e in training set (train -> {train_end_date})!")
        print(f"  Predictia NU e pe date nevazute. Rezultatul poate fi optimist.")

    # Features lag (datele de ieri) - pastram ca DataFrame cu coloane
    features = pd.DataFrame(
        [df.loc[prediction_date, FEATURES].values],
        columns=FEATURES
    )
    features_imp = pd.DataFrame(
        imputer.transform(features), columns=FEATURES
    )

    # Predictie
    label = rf.predict(features_imp)[0]
    proba = rf.predict_proba(features_imp)[0]

    # Date reale ale zilei (readiness, sleep)
    today_data = {
        "date": df.loc[prediction_date, "date"],
        "readiness_score": df.loc[prediction_date, "readiness_score"],
        "sleep_score": df.loc[prediction_date, "sleep_score"],
        "actual_label": int(df.loc[prediction_date, "nutrition_label"]),
    }

    return int(label), proba, today_data


# ============================================================
# 4. MAIN - DEMO END-TO-END
# ============================================================
def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 60)
    print("  SISTEM AI - RECOMANDARI NUTRITIONALE PERSONALIZATE")
    print("  Oura Ring + Apple Watch -> ML -> Recomandare zilnica")
    print(f"  Rulat la: {now}")
    print("=" * 60)

    # Step 1: Load date
    print("\n[1/4] INCARCARE DATE")
    df = load_full_dataset()
    last_date = df["date"].max().strftime("%Y-%m-%d")
    print(f"  Ultima zi cu date: {last_date}")
    print(f"  Total zile: {len(df)}")

    # Step 2: Train sau load model
    print("\n[2/4] MODEL ML (split temporal {:.0%}/{:.0%})".format(
        TRAIN_RATIO, 1 - TRAIN_RATIO))

    # Sterge modelul vechi si re-antreneaza (asigura consistenta)
    # Daca vrei sa refolosesti un model existent, comenteaza urmatoarele 3 linii
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if os.path.exists(IMPUTER_PATH):
        os.remove(IMPUTER_PATH)

    if os.path.exists(MODEL_PATH) and os.path.exists(IMPUTER_PATH):
        print("  Model gasit pe disk, il incarc...")
        rf, imputer = load_model()
        print("  Model incarcat.")
    else:
        print("  Antrenez model nou (split temporal)...")
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

    match = "CORECT" if label == today["actual_label"] else "INCORECT"
    print(f"  Predictie:       {match}")

    # Step 4: Recomandare nutritionala
    print("\n[4/4] RECOMANDARE NUTRITIONALA")
    rec = get_nutrition_recommendation(label, USER_PROFILE)
    print_recommendation(rec)

    # Sumar final
    print(f"\n{'='*60}")
    print("  SUMAR SISTEM END-TO-END")
    print(f"{'='*60}")
    print(f"  Surse date:     Oura Ring ({df['oura_device'].iloc[-1]}) + "
          f"Apple Watch ({df['aw_device'].iloc[-1]})")
    print(f"  Dataset:        {len(df)} zile")
    print(f"  Model:          Random Forest (100 trees, max_depth=10)")
    print(f"  Training:       Primele {TRAIN_RATIO:.0%} zile (split temporal)")
    print(f"  Clasificare:    {LABEL_NAMES[label]} (confidence: {max(proba):.1%})")
    print(f"  Calorii target: {rec['calories_target']} kcal")
    print(f"  Proteine:       {rec['protein_g']}g | Carbs: {rec['carbs_g']}g | "
          f"Fat: {rec['fat_g']}g")


if __name__ == "__main__":
    main()