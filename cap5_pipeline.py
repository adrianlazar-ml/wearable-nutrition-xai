"""
Cap.5 - Pipeline ML pentru clasificarea starii fiziologice zilnice
Surse date: Oura Ring (Gen3/Gen4) + Apple Watch (Ultra/Ultra3)
Model: Random Forest cu SMOTE pentru class imbalance
Target: Rest (0) / Normal (1) / Active (2)

Fix-uri aplicate (v3):
- v2: Imputer fit doar pe train, SMOTE fara class_weight, steps sum()
- v3: Split TEMPORAL (80/20 cronologic) in loc de random
      TimeSeriesSplit pentru cross-validation
      Pastram si split random pentru comparatie in Cap.5
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
# 2. LOAD DATE APPLE WATCH - agregare zilnica
# ============================================================
def load_aw_metric(filename, agg_func="mean"):
    """
    Incarca o metrica Apple Watch si o agraga zilnic.

    agg_func: "mean" pentru HR, HRV, SpO2 (media e corecta)
              "sum" pentru steps, active_calories (se aduna pe zi)
    """
    df = pd.read_csv(f"data/apple_health/{filename}")
    df["date"] = pd.to_datetime(df["startDate"]).dt.date
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.groupby("date")["value"].agg(agg_func).reset_index()


def load_apple_watch():
    hr_aw = load_aw_metric("heart_rate.csv", "mean").rename(columns={"value": "aw_hr_mean"})
    hrv_aw = load_aw_metric("hrv.csv", "mean").rename(columns={"value": "aw_hrv_mean"})
    steps_aw = load_aw_metric("steps.csv", "sum").rename(columns={"value": "aw_steps"})
    calories_aw = load_aw_metric("active_calories.csv", "sum").rename(columns={"value": "aw_active_calories"})
    spo2_aw = load_aw_metric("spo2.csv", "mean").rename(columns={"value": "aw_spo2"})

    print(f"HR Apple Watch: {len(hr_aw)} zile")
    print(f"HRV Apple Watch: {len(hrv_aw)} zile")
    print(f"Steps Apple Watch: {len(steps_aw)} zile (sum/zi)")
    print(f"Calories Apple Watch: {len(calories_aw)} zile (sum/zi)")

    return hr_aw, hrv_aw, steps_aw, calories_aw, spo2_aw


# ============================================================
# 3. MERGE PE DATA
# ============================================================
def merge_datasets(oura, hr_aw, hrv_aw, steps_aw, calories_aw, spo2_aw):
    df = oura.merge(hr_aw, on="date", how="left")
    df = df.merge(hrv_aw, on="date", how="left")
    df = df.merge(steps_aw, on="date", how="left")
    df = df.merge(calories_aw, on="date", how="left")
    df = df.merge(spo2_aw, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Dataset combinat: {len(df)} zile, {df.shape[1]} coloane")
    return df


# ============================================================
# 4. FEATURE ENGINEERING - LAGGED FEATURES (t-1)
# ============================================================
LAG_FEATURES = [
    "readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
    "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
    "steps_oura", "respiratory_rate", "temp_deviation",
    "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2"
]

FEATURES = [f"{feat}_lag1" for feat in LAG_FEATURES]


def assign_nutrition_label(row):
    """
    Clasificare stare fiziologica zilnica bazata pe scoruri Oura.
    Praguri bazate pe documentatia Oura Ring:
    - Readiness < 60 sau Sleep < 60 = recuperare necesara
    - Readiness >= 85 si Sleep >= 75 = capacitate maxima
    """
    if pd.isna(row["readiness_score"]) or pd.isna(row["sleep_score"]):
        return np.nan
    if row["readiness_score"] < 60 or row["sleep_score"] < 60:
        return 0  # Rest day
    elif row["readiness_score"] >= 85 and row["sleep_score"] >= 75:
        return 2  # Active day
    else:
        return 1  # Normal day


def engineer_features(df):
    # Lag features: datele de ieri prezic recomandarea de azi
    for feat in LAG_FEATURES:
        df[f"{feat}_lag1"] = df[feat].shift(1)

    # Target pe ziua curenta
    df["nutrition_label"] = df.apply(assign_nutrition_label, axis=1)

    print("Distributie labels:")
    print(df["nutrition_label"].value_counts().sort_index())

    # IMPORTANT: Pastram coloana 'date' pentru split temporal
    df_model = df[["date"] + FEATURES + ["nutrition_label"]].dropna(subset=["nutrition_label"])
    # Sortam cronologic (ar trebui sa fie deja, dar asiguram)
    df_model = df_model.sort_values("date").reset_index(drop=True)
    print(f"Randuri pentru model (inainte de dropna features): {len(df_model)}")

    return df_model


# ============================================================
# 5. TRAIN / EVALUATE - SPLIT TEMPORAL
# ============================================================
def _train_eval_core(X_train, X_test, y_train, y_test, split_name=""):
    """Core training logic refolosit pentru ambele split-uri."""
    # Imputer fit doar pe train
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=FEATURES, index=X_test.index
    )

    print(f"  Train inainte SMOTE: {y_train.value_counts().sort_index().to_dict()}")

    # SMOTE doar pe training
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

    # Feature importance
    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    return rf, imputer, fi, acc, f1_mac


def train_and_evaluate(df_model):
    X = df_model[FEATURES]
    y = df_model["nutrition_label"].astype(int)
    dates = df_model["date"]

    # ===========================================================
    # A) SPLIT TEMPORAL (PRIMARY) - 80% primele zile, 20% ultimele
    # ===========================================================
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

    # TimeSeriesSplit cross-validation
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

    # ===========================================================
    # B) SPLIT RANDOM STRATIFICAT (pentru comparatie in Cap.5)
    # ===========================================================
    print("\n--- B) SPLIT RANDOM STRATIFICAT (baseline comparativ) ---")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train_r)} | Test: {len(X_test_r)}")

    _, _, _, acc_r, f1_r = _train_eval_core(
        X_train_r, X_test_r, y_train_r, y_test_r, "random"
    )

    # StratifiedKFold CV (pentru comparatie)
    cv_scores_r = cross_val_score(cv_pipeline, X, y, cv=5, scoring="accuracy")
    f1_cv_r = cross_val_score(cv_pipeline, X, y, cv=5, scoring="f1_macro")
    print(f"  StratifiedKFold CV (5-fold): acc={cv_scores_r.mean():.2%} (+/- {cv_scores_r.std():.2%})")
    print(f"  StratifiedKFold CV (5-fold): f1_macro={f1_cv_r.mean():.2%} (+/- {f1_cv_r.std():.2%})")

    # ===========================================================
    # C) SUMAR COMPARATIV
    # ===========================================================
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

    # Returnam modelul TEMPORAL (cel corect metodologic)
    return rf_t, imputer_t, fi_t


# ============================================================
# 6. MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  CAP.5 - PIPELINE ML NUTRITIE PERSONALIZATA")
    print("  Oura Ring (Gen3/Gen4) + Apple Watch (Ultra/Ultra3)")
    print("=" * 60)

    print("\n=== LOAD DATE OURA ===")
    oura = load_oura()

    print("\n=== LOAD DATE APPLE WATCH ===")
    hr_aw, hrv_aw, steps_aw, calories_aw, spo2_aw = load_apple_watch()

    print("\n=== MERGE DATE ===")
    df = merge_datasets(oura, hr_aw, hrv_aw, steps_aw, calories_aw, spo2_aw)

    print("\n=== FEATURE ENGINEERING (lag t-1) ===")
    df_model = engineer_features(df)

    print("\n=== TRAIN & EVALUATE (temporal + random) ===")
    rf, imputer, fi = train_and_evaluate(df_model)

    # Salvare dataset (fara coloana date - modelul nu o foloseste)
    df_model[FEATURES + ["nutrition_label"]].to_csv("data/dataset_combinat.csv", index=False)
    print(f"\nDataset salvat: {len(df_model)} randuri -> data/dataset_combinat.csv")

    return rf, imputer, fi


if __name__ == "__main__":
    rf, imputer, fi = main()
