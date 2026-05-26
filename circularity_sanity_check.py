"""
Cap.6 - Experiment sanity check: CIRCULARITATE PSEUDO-LABELS
==============================================================

Problema:
- Label generation foloseste `readiness_score[T]` si `sleep_score[T]`
- Features pentru predictia T includ `readiness_score_lag1` si `sleep_score_lag1`
- Autocorrelation puternica intre zile consecutive => modelul "vede" labelul

Scop: masuram empiric cat din F1=0.506 vine din circularitate vs
din informatie fiziologica reala.

Abordare: trei variante de features progresiv mai restrictive.

Variant A (BASELINE):       19 features (toate) - asteptare F1 ~0.50
Variant B (SEMI-RESTRICT):  17 features (fara readiness_lag1, sleep_score_lag1)
Variant C (STRICT):          11 features (fara NICIUN scor compozit Oura)

Autor: Adrian Lazar | Disertatie UPB 2026
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# 1. LOAD DATE (reproduce pipeline-ul din cap5_pipeline_json_extended)
# ============================================================
def load_data():
    oura = pd.read_csv("data/oura_combined.csv")
    oura["date"] = pd.to_datetime(oura["date"])

    aw = pd.read_csv("apple_watch_api/apple_health_daily.csv")
    aw["date"] = pd.to_datetime(aw["date"])
    aw["aw_active_calories"] = (aw["aw_active_energy_kj"] / 4.184).round(1)

    aw_cols = ["date", "aw_hr_mean", "aw_hrv_mean", "aw_steps",
               "aw_active_calories", "aw_spo2",
               "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate"]
    aw_subset = aw[aw_cols]

    df = oura.merge(aw_subset, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    # Fix rhr_avg imposibile (v3.1)
    invalid = df["rhr_avg"] < df["rhr_lowest"]
    df.loc[invalid, "rhr_avg"] = np.nan

    return df


# ============================================================
# 2. AUTOCORRELATION ANALYSIS
# ============================================================
def analyze_autocorrelation(df):
    """
    Masuram cat de puternic coreleaza variabilele care fac labelul
    cu versiunile lor lag-1. Autocorrelation mare = circularitate
    potentiala mare.
    """
    print("\n" + "=" * 60)
    print("  AUTOCORRELATION ANALYSIS (zi T vs zi T-1)")
    print("=" * 60)

    key_vars = ["readiness_score", "sleep_score", "hrv_mean",
                "hrv_balance", "recovery_index", "activity_score"]

    for var in key_vars:
        series = df[var].dropna()
        lag1 = series.shift(1)
        aligned = pd.concat([series, lag1], axis=1).dropna()
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        print(f"  {var:<25} autocorr(lag-1) = {corr:.3f}")

    print("\n  Interpretare:")
    print("  > 0.7 = autocorrelation foarte puternica (risc circularitate mare)")
    print("  0.4-0.7 = autocorrelation moderata (risc mediu)")
    print("  < 0.4 = autocorrelation slaba (risc mic)")


# ============================================================
# 3. LABEL GENERATION (copy din cap5_pipeline_json_extended)
# ============================================================
def assign_label(row):
    if pd.isna(row["readiness_score"]) or pd.isna(row["sleep_score"]):
        return np.nan
    if row["readiness_score"] < 60 or row["sleep_score"] < 60:
        return 0  # Rest
    elif row["readiness_score"] >= 85 and row["sleep_score"] >= 75:
        return 2  # Active
    else:
        return 1  # Normal


def prepare_features_labels(df):
    """
    Genereaza features lag-1 si labels. Returneaza df-ul complet
    cu toate features disponibile, pentru a putea face sub-seturi.
    """
    all_features = [
        "readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
        "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
        "steps_oura", "respiratory_rate", "temp_deviation",
        "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2",
        "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate",
    ]

    for feat in all_features:
        df[f"{feat}_lag1"] = df[feat].shift(1)

    df["nutrition_label"] = df.apply(assign_label, axis=1)
    df = df.dropna(subset=["nutrition_label"]).reset_index(drop=True)

    return df, all_features


# ============================================================
# 4. TRAIN & EVALUATE pe subset de features
# ============================================================
def train_variant(df, feature_set, variant_name):
    """
    Antrenam Random Forest cu acelasi setup ca in pipeline-ul
    principal, dar cu un subset de features diferit.

    Setup: temporal split 80/20 + SMOTE k=3 + median imputer.
    """
    FEATURES = [f"{f}_lag1" for f in feature_set]

    X = df[FEATURES]
    y = df["nutrition_label"].astype(int)

    # Temporal split
    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Pipeline: impute -> SMOTE -> RF
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

    rf = RandomForestClassifier(
        n_estimators=100, random_state=42,
        max_depth=10, min_samples_leaf=5
    )
    rf.fit(X_train_sm, y_train_sm)
    y_pred = rf.predict(X_test_imp)

    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    # TimeSeriesSplit CV pentru robustete
    cv_pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("rf", RandomForestClassifier(
            n_estimators=100, random_state=42,
            max_depth=10, min_samples_leaf=5
        ))
    ])
    tscv = TimeSeriesSplit(n_splits=5)
    cv_f1 = cross_val_score(cv_pipeline, X_train, y_train, cv=tscv,
                            scoring="f1_macro", n_jobs=-1)

    # Feature importance top 5
    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

    print(f"\n--- VARIANTA {variant_name}: {len(FEATURES)} features ---")
    print(f"  Accuracy (test):  {acc:.3f}")
    print(f"  F1 Macro (test):  {f1_mac:.3f}")
    print(f"  F1 per class:     Rest={f1_per_class[0]:.3f}, "
          f"Normal={f1_per_class[1]:.3f}, Active={f1_per_class[2]:.3f}")
    print(f"  F1 Macro (CV 5f): {cv_f1.mean():.3f} (+/- {cv_f1.std():.3f})")
    print(f"  Top 5 features:")
    for feat, imp in fi.head(5).items():
        print(f"    {feat:<35} {imp:.4f}")

    return {
        "variant": variant_name,
        "n_features": len(FEATURES),
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_rest": f1_per_class[0],
        "f1_normal": f1_per_class[1],
        "f1_active": f1_per_class[2],
        "f1_cv_mean": cv_f1.mean(),
        "f1_cv_std": cv_f1.std(),
        "top_features": fi.head(5).to_dict(),
    }


# ============================================================
# 5. DEFINE FEATURE SUBSETS
# ============================================================

# Features "COMPOZITE ORUA" (scoruri derivate proprietar, direct legate de label)
OURA_COMPOSITE_SCORES = [
    "readiness_score",   # DIRECT folosit in label
    "sleep_score",       # DIRECT folosit in label
    "hrv_balance",       # Scor compozit Oura (rolling avg)
    "recovery_index",    # Scor compozit derivat
    "activity_score",    # Scor compozit
]

# Features "FIZIOLOGICE BRUTE" (nu sunt scoruri derivate Oura)
PHYSIOLOGICAL_RAW = [
    "hrv_mean",          # HRV brut (nu balance)
    "rhr_lowest",        # RHR brut
    "rhr_avg",           # RHR brut
    "steps_oura",        # Pasi brut
    "respiratory_rate",  # Rata respiratorie bruta
    "temp_deviation",    # Deviatie temperatura
    "aw_hr_mean",        # Apple Watch HR brut
    "aw_hrv_mean",       # Apple Watch HRV brut
    "aw_steps",          # Apple Watch pasi
    "aw_active_calories",
    "aw_spo2",
    "aw_sleep_deep_hours",
    "aw_sleep_rem_hours",
    "aw_resp_rate",
]


# ============================================================
# 6. MAIN - RULEAZA CELE 3 VARIANTE
# ============================================================
def main():
    print("=" * 60)
    print("  CIRCULARITY SANITY CHECK")
    print("  Disertatie Adrian Lazar - UPB 2026")
    print("=" * 60)

    df = load_data()
    print(f"\nDataset complet: {len(df)} zile")

    # Autocorrelation analysis
    analyze_autocorrelation(df)

    # Prepare features
    df_model, all_features = prepare_features_labels(df)
    print(f"\nDupa label generation si lag: {len(df_model)} zile cu label valid")
    print(f"Distributie labels: {df_model['nutrition_label'].value_counts().sort_index().to_dict()}")

    # Drop randuri cu toate features NaN (primul rand dupa shift)
    FEATURES_ALL = [f"{f}_lag1" for f in all_features]
    df_model = df_model.dropna(subset=FEATURES_ALL, how="all").reset_index(drop=True)
    print(f"Dupa dropna randuri complet goale: {len(df_model)} zile")

    # ============================================================
    # VARIANT A: BASELINE - toate 19 features (reproducere cap5)
    # ============================================================
    result_A = train_variant(df_model, all_features, "A (BASELINE: toate 19 features)")

    # ============================================================
    # VARIANT B: SEMI-RESTRICT - fara readiness_lag1 + sleep_score_lag1
    # (eliminam DOAR features folosite direct in label, 17 features)
    # ============================================================
    features_B = [f for f in all_features if f not in ["readiness_score", "sleep_score"]]
    result_B = train_variant(df_model, features_B, "B (SEMI: fara readiness+sleep_score)")

    # ============================================================
    # VARIANT C: STRICT - DOAR features fiziologice brute (14 features)
    # (eliminam TOATE scorurile compozite Oura)
    # ============================================================
    result_C = train_variant(df_model, PHYSIOLOGICAL_RAW,
                             "C (STRICT: doar fiziologice brute)")

    # ============================================================
    # SUMAR FINAL
    # ============================================================
    print("\n" + "=" * 60)
    print("  SUMAR: IMPACTUL ELIMINARII FEATURES CIRCULARE")
    print("=" * 60)

    results = [result_A, result_B, result_C]
    print(f"\n  {'Varianta':<45} {'nFeat':>6} {'F1':>7} {'F1 CV':>8}")
    print(f"  {'-'*67}")
    for r in results:
        print(f"  {r['variant']:<45} {r['n_features']:>6} "
              f"{r['f1_macro']:>7.3f} {r['f1_cv_mean']:>8.3f}")

    # Interpretare
    f1_baseline = result_A["f1_macro"]
    f1_strict = result_C["f1_macro"]
    drop_pp = (f1_baseline - f1_strict) * 100

    print(f"\n  IMPACT CIRCULARITATE:")
    print(f"    F1 Baseline (A):  {f1_baseline:.3f}")
    print(f"    F1 Strict (C):    {f1_strict:.3f}")
    print(f"    Drop total:       {drop_pp:+.1f} puncte procentuale")

    if drop_pp > 15:
        verdict = "CIRCULARITATE MAJORA - modelul depinde mult de scoruri compozite"
    elif drop_pp > 8:
        verdict = "CIRCULARITATE MODERATA - scoruri compozite contribuie semnificativ"
    else:
        verdict = "CIRCULARITATE MICA - modelul invata si din features brute"
    print(f"    Verdict:          {verdict}")

    # Salvare rezultate
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "top_features"}
        for r in results
    ])
    results_df.to_csv("outputs/tables/circularity_sanity_check.csv", index=False)
    print(f"\n  Rezultate salvate: outputs/tables/circularity_sanity_check.csv")

    return results


if __name__ == "__main__":
    main()
