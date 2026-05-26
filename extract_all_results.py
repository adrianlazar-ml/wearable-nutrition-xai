"""
EXTRAGERE VALORI NUMERICE — Referinta pentru scriere disertatie
================================================================
Ruleaza pipeline-ul principal + citeste toate CSV-urile din outputs/tables/
si genereaza un singur fisier de referinta cu TOATE valorile numerice
necesare pentru actualizarea lucrarii.

Output:
  outputs/reports/valori_disertatie.txt  (referinta text)
  outputs/tables/valori_disertatie.csv   (tabel master)

Rulare: python extract_stats.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, confusion_matrix
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

TBL_DIR = "outputs/tables"
RPT_DIR = "outputs/reports"
os.makedirs(TBL_DIR, exist_ok=True)
os.makedirs(RPT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATE — identic cu pipeline-ul principal
# ============================================================
def load_and_prepare():
    oura = pd.read_csv("data/oura_combined.csv")
    oura["date"] = pd.to_datetime(oura["date"])

    aw = pd.read_csv("apple_watch_api/apple_health_daily.csv")
    aw["date"] = pd.to_datetime(aw["date"])
    aw["aw_active_calories"] = (aw["aw_active_energy_kj"] / 4.184).round(1)

    hr = aw[["date", "aw_hr_mean"]].dropna(subset=["aw_hr_mean"])
    hrv = aw[["date", "aw_hrv_mean"]].dropna(subset=["aw_hrv_mean"])
    steps = aw[["date", "aw_steps"]].dropna(subset=["aw_steps"])
    cals = aw[["date", "aw_active_calories"]].dropna(subset=["aw_active_calories"])
    spo2 = aw[["date", "aw_spo2"]].dropna(subset=["aw_spo2"])
    sleep_deep = aw[["date", "aw_sleep_deep_hours"]].dropna(subset=["aw_sleep_deep_hours"])
    sleep_rem = aw[["date", "aw_sleep_rem_hours"]].dropna(subset=["aw_sleep_rem_hours"])
    resp = aw[["date", "aw_resp_rate"]].dropna(subset=["aw_resp_rate"])

    df = oura
    for d in [hr, hrv, steps, cals, spo2, sleep_deep, sleep_rem, resp]:
        df = df.merge(d, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    LAG_FEATURES = [
        "readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
        "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
        "steps_oura", "respiratory_rate", "temp_deviation",
        "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2",
        "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate",
    ]
    FEATURES = [f"{f}_lag1" for f in LAG_FEATURES]
    for feat in LAG_FEATURES:
        df[f"{feat}_lag1"] = df[feat].shift(1)

    def assign_label(row):
        if pd.isna(row["readiness_score"]) or pd.isna(row["sleep_score"]):
            return np.nan
        if row["readiness_score"] < 60 or row["sleep_score"] < 60:
            return 0
        elif row["readiness_score"] >= 85 and row["sleep_score"] >= 75:
            return 2
        return 1

    df["label"] = df.apply(assign_label, axis=1)
    df_model = df[["date"] + FEATURES + ["label"]].dropna(subset=["label"])
    df_model = df_model.sort_values("date").reset_index(drop=True)

    X = df_model[FEATURES]
    y = df_model["label"].astype(int)
    dates = df_model["date"]

    return X, y, FEATURES, dates, df


# ============================================================
# 2. PIPELINE PRINCIPAL — extrage metrici core
# ============================================================
def run_core_pipeline(X, y, FEATURES, dates):
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURES)

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_sm), columns=FEATURES)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=FEATURES)

    tscv = TimeSeriesSplit(n_splits=5)

    # Baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_imp, y_train)
    y_dummy = dummy.predict(X_test_imp)

    models = {
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, max_depth=10,
                                   min_samples_leaf=5, random_state=42),
            X_train_sm, X_test_imp
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                          random_state=42, eval_metric="mlogloss", verbosity=0),
            X_train_sm, X_test_imp
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42,
                probability=True),
            X_train_scaled, X_test_scaled
        ),
    }

    results = {}
    results["Baseline"] = {
        "accuracy": accuracy_score(y_test, y_dummy),
        "f1_macro": f1_score(y_test, y_dummy, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_dummy, average="weighted", zero_division=0),
    }

    for name, (model, X_tr, X_te) in models.items():
        model.fit(X_tr, y_train_sm)
        y_pred = model.predict(X_te)

        report = classification_report(
            y_test, y_pred,
            target_names=["Rest", "Normal", "Active"],
            zero_division=0, output_dict=True
        )

        # CV
        if "SVM" in name:
            cv_pipe = ImbPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("smote", SMOTE(random_state=42, k_neighbors=3)),
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
            ])
        elif "XGBoost" in name:
            cv_pipe = ImbPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("smote", SMOTE(random_state=42, k_neighbors=3)),
                ("model", XGBClassifier(n_estimators=100, max_depth=6,
                    learning_rate=0.1, random_state=42,
                    eval_metric="mlogloss", verbosity=0)),
            ])
        else:
            cv_pipe = ImbPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("smote", SMOTE(random_state=42, k_neighbors=3)),
                ("model", RandomForestClassifier(n_estimators=100, max_depth=10,
                    min_samples_leaf=5, random_state=42)),
            ])

        cv_acc = cross_val_score(cv_pipe, X, y, cv=tscv, scoring="accuracy")
        cv_f1 = cross_val_score(cv_pipe, X, y, cv=tscv, scoring="f1_macro")

        cm = confusion_matrix(y_test, y_pred)

        # Feature importance (RF si XGB)
        top3_features = []
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=FEATURES)
            top3 = fi.sort_values(ascending=False).head(3)
            top3_features = [(f.replace("_lag1", ""), v) for f, v in top3.items()]

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "cv_acc_mean": cv_acc.mean(),
            "cv_acc_std": cv_acc.std(),
            "cv_f1_mean": cv_f1.mean(),
            "cv_f1_std": cv_f1.std(),
            "f1_rest": report["Rest"]["f1-score"],
            "f1_normal": report["Normal"]["f1-score"],
            "f1_active": report["Active"]["f1-score"],
            "precision_rest": report["Rest"]["precision"],
            "recall_rest": report["Rest"]["recall"],
            "precision_normal": report["Normal"]["precision"],
            "recall_normal": report["Normal"]["recall"],
            "precision_active": report["Active"]["precision"],
            "recall_active": report["Active"]["recall"],
            "support_rest": report["Rest"]["support"],
            "support_normal": report["Normal"]["support"],
            "support_active": report["Active"]["support"],
            "confusion_matrix": cm.tolist(),
            "top3_features": top3_features,
        }

    meta = {
        "total_zile_oura": len(pd.read_csv("data/oura_combined.csv")),
        "total_zile_aw": len(pd.read_csv("apple_watch_api/apple_health_daily.csv")),
        "zile_utile": len(X),
        "n_features": len(FEATURES),
        "clase": dict(y.value_counts().sort_index()),
        "train_size": split_idx,
        "test_size": len(X) - split_idx,
        "perioada_start": dates.iloc[0].strftime("%Y-%m-%d"),
        "perioada_end": dates.iloc[-1].strftime("%Y-%m-%d"),
        "train_end": dates.iloc[split_idx - 1].strftime("%Y-%m-%d"),
        "test_start": dates.iloc[split_idx].strftime("%Y-%m-%d"),
    }

    return results, meta


# ============================================================
# 3. CITESTE CSV-URI DIN EXPERIMENTE ANTERIOARE
# ============================================================
def load_experiment_tables():
    tables = {}

    # CSV-uri cu model names in prima coloana (index)
    indexed_csvs = {
        "new_models": "new_models_comparison.csv",
        "calibration": "calibration_brier_scores.csv",
        "rolling": "rolling_features_comparison.csv",
    }
    for key, fname in indexed_csvs.items():
        path = os.path.join(TBL_DIR, fname)
        if os.path.exists(path):
            tables[key] = pd.read_csv(path, index_col=0)
        else:
            print(f"  WARN: Nu gasesc {path}")
            tables[key] = None

    # CSV-uri fara index special
    normal_csvs = {
        "tuning": "tuning_comparison.csv",
        "tuning_params": "tuning_best_params.csv",
        "ablation": "ablation_study_table.csv",
        "sensitivity": "sensitivity_threshold_table.csv",
        "confidence": "confidence_threshold_analysis.csv",
        "smote": "smote_comparison_table.csv",
    }
    for key, fname in normal_csvs.items():
        path = os.path.join(TBL_DIR, fname)
        if os.path.exists(path):
            tables[key] = pd.read_csv(path)
        else:
            print(f"  WARN: Nu gasesc {path}")
            tables[key] = None

    return tables


# ============================================================
# 4. GENEREAZA FISIERUL DE REFERINTA
# ============================================================
def generate_reference(results, meta, tables):
    lines = []
    L = lines.append

    L("=" * 70)
    L(f"  VALORI NUMERICE DISERTATIE — Referinta pentru scriere")
    L(f"  Generat: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L(f"  Date: post-fix rhr_avg, curate")
    L("=" * 70)

    # --- DATASET ---
    L("")
    L("-" * 70)
    L("  DATASET")
    L("-" * 70)
    L(f"  Zile totale Oura:        {meta['total_zile_oura']}")
    L(f"  Zile totale Apple Watch:  {meta['total_zile_aw']}")
    L(f"  Zile utile (dupa lag+dropna): {meta['zile_utile']}")
    L(f"  Features:                 {meta['n_features']}")
    L(f"  Perioada:                 {meta['perioada_start']} -> {meta['perioada_end']}")
    L(f"  Clase:                    Rest={meta['clase'][0]}, Normal={meta['clase'][1]}, Active={meta['clase'][2]}")
    L(f"  Train:                    {meta['train_size']} zile (-> {meta['train_end']})")
    L(f"  Test:                     {meta['test_size']} zile ({meta['test_start']} ->)")

    # --- PIPELINE PRINCIPAL ---
    L("")
    L("-" * 70)
    L("  PIPELINE PRINCIPAL (19 features, JSON, SMOTE k=3)")
    L("-" * 70)
    L(f"  {'Metrica':<22} {'Baseline':>10} {'RF':>10} {'XGBoost':>10} {'SVM':>10}")
    L(f"  {'-'*62}")

    metrics = [
        ("Accuracy", "accuracy", "{:.4f}"),
        ("F1 Macro", "f1_macro", "{:.4f}"),
        ("F1 Weighted", "f1_weighted", "{:.4f}"),
        ("CV Accuracy", "cv_acc_mean", "{:.4f}"),
        ("CV F1 Macro", "cv_f1_mean", "{:.4f}"),
        ("CV Acc std", "cv_acc_std", "{:.4f}"),
        ("CV F1 std", "cv_f1_std", "{:.4f}"),
        ("F1 Rest", "f1_rest", "{:.4f}"),
        ("F1 Normal", "f1_normal", "{:.4f}"),
        ("F1 Active", "f1_active", "{:.4f}"),
        ("Precision Rest", "precision_rest", "{:.4f}"),
        ("Recall Rest", "recall_rest", "{:.4f}"),
        ("Precision Normal", "precision_normal", "{:.4f}"),
        ("Recall Normal", "recall_normal", "{:.4f}"),
        ("Precision Active", "precision_active", "{:.4f}"),
        ("Recall Active", "recall_active", "{:.4f}"),
    ]

    model_names = ["Random Forest", "XGBoost", "SVM (RBF)"]
    for label, key, fmt in metrics:
        bval = fmt.format(results["Baseline"].get(key, 0)) if key in results["Baseline"] else "—"
        vals = []
        for mn in model_names:
            if key in results.get(mn, {}):
                vals.append(fmt.format(results[mn][key]))
            else:
                vals.append("—")
        L(f"  {label:<22} {bval:>10} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    # Confusion matrices
    for mn in model_names:
        if "confusion_matrix" in results.get(mn, {}):
            cm = results[mn]["confusion_matrix"]
            L(f"\n  Confusion matrix {mn}:")
            L(f"               Pred Rest  Pred Norm  Pred Act")
            L(f"    Real Rest    {cm[0][0]:>6}     {cm[0][1]:>6}     {cm[0][2]:>6}")
            L(f"    Real Norm    {cm[1][0]:>6}     {cm[1][1]:>6}     {cm[1][2]:>6}")
            L(f"    Real Act     {cm[2][0]:>6}     {cm[2][1]:>6}     {cm[2][2]:>6}")

    # Top features
    for mn in model_names:
        if results.get(mn, {}).get("top3_features"):
            L(f"\n  Top 3 features {mn}:")
            for feat, imp in results[mn]["top3_features"]:
                L(f"    {feat:<30} {imp:.4f}")

    # --- EXPERIMENTE DIN CSV-URI ---

    # Tuning
    if tables.get("tuning") is not None:
        L("")
        L("-" * 70)
        L("  HYPERPARAMETER TUNING (RandomizedSearchCV)")
        L("-" * 70)
        df = tables["tuning"]
        for _, row in df.iterrows():
            L(f"  {row['Model']:<10} Default: {row['Default_F1_Macro']:.4f}  "
              f"Tuned: {row['Tuned_F1_Macro']:.4f}  "
              f"Delta: {row['Delta_pp']:+.2f}pp")

    if tables.get("tuning_params") is not None:
        L("\n  Best parameters:")
        df = tables["tuning_params"]
        for _, row in df.iterrows():
            params = {k: v for k, v in row.items() if k != "Model" and pd.notna(v)}
            L(f"  {row['Model']}: {params}")

    # New models
    if tables.get("new_models") is not None:
        L("")
        L("-" * 70)
        L("  MODELE NOI (LR + LightGBM)")
        L("-" * 70)
        df = tables["new_models"]
        L(f"  {'Model':<25} {'F1 Macro':>10} {'CV F1':>10} {'F1 Rest':>8} "
          f"{'F1 Norm':>8} {'F1 Act':>8}")
        L(f"  {'-'*69}")
        for model_name, row in df.iterrows():
            L(f"  {model_name:<25} {row['f1_macro']:>10.4f} {row['cv_f1_mean']:>10.4f} "
              f"{row['f1_rest']:>8.4f} {row['f1_normal']:>8.4f} "
              f"{row['f1_active']:>8.4f}")

    # Calibration
    if tables.get("calibration") is not None:
        L("")
        L("-" * 70)
        L("  CALIBRATION (Brier scores)")
        L("-" * 70)
        df = tables["calibration"]
        for model_name, row in df.iterrows():
            L(f"  {model_name:<20} Rest: {row['Rest']:.4f}  Normal: {row['Normal']:.4f}  "
              f"Active: {row['Active']:.4f}  Avg: {row['Average']:.4f}")

    # Ablation
    if tables.get("ablation") is not None:
        L("")
        L("-" * 70)
        L("  ABLATION STUDY")
        L("-" * 70)
        df = tables["ablation"]
        # Afisam tot tabelul formatat
        L(f"  {df.to_string(index=False)}")

    # Sensitivity
    if tables.get("sensitivity") is not None:
        L("")
        L("-" * 70)
        L("  THRESHOLD SENSITIVITY")
        L("-" * 70)
        df = tables["sensitivity"]
        L(f"  {df.to_string(index=False)}")

    # SMOTE
    if tables.get("smote") is not None:
        L("")
        L("-" * 70)
        L("  SMOTE COMPARISON")
        L("-" * 70)
        df = tables["smote"]
        L(f"  {df.to_string(index=False)}")

    # Confidence
    if tables.get("confidence") is not None:
        L("")
        L("-" * 70)
        L("  CONFIDENCE THRESHOLD")
        L("-" * 70)
        df = tables["confidence"]
        L(f"  {df.to_string(index=False)}")

    # Rolling
    if tables.get("rolling") is not None:
        L("")
        L("-" * 70)
        L("  ROLLING FEATURES (19 vs 36)")
        L("-" * 70)
        df = tables["rolling"]
        L(f"  {'Config':<25} {'N feat':>7} {'F1 test':>9} {'CV F1':>9} "
          f"{'F1 Rest':>8} {'F1 Norm':>8} {'F1 Act':>8}")
        L(f"  {'-'*74}")
        for config_name, row in df.iterrows():
            n_feat = int(row['n_features']) if 'n_features' in row else '—'
            L(f"  {config_name:<25} {n_feat:>7} {row.get('f1_macro_test', 0):>9.4f} "
              f"{row.get('cv_f1_mean', 0):>9.4f} {row.get('f1_rest', 0):>8.4f} "
              f"{row.get('f1_normal', 0):>8.4f} {row.get('f1_active', 0):>8.4f}")

    L("")
    L("=" * 70)
    L("  SFARSIT REFERINTA")
    L("=" * 70)

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  EXTRAGERE VALORI NUMERICE — Referinta disertatie")
    print("=" * 60)

    print("\n  Loading date si rulare pipeline...")
    X, y, FEATURES, dates, df = load_and_prepare()
    results, meta = run_core_pipeline(X, y, FEATURES, dates)

    print("  Loading CSV-uri din experimente anterioare...")
    tables = load_experiment_tables()

    print("  Generez fisierul de referinta...")
    reference = generate_reference(results, meta, tables)

    # Salvare text
    txt_path = f"{RPT_DIR}/valori_disertatie.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(reference)

    # Salvare CSV master (metrici core)
    csv_rows = []
    for name in ["Baseline", "Random Forest", "XGBoost", "SVM (RBF)"]:
        row = {"Model": name}
        row.update({k: v for k, v in results[name].items()
                   if k not in ["confusion_matrix", "top3_features"]})
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(f"{TBL_DIR}/valori_disertatie.csv", index=False)

    # Print in consola
    print(reference)

    print(f"\n  -> {txt_path}")
    print(f"  -> {TBL_DIR}/valori_disertatie.csv")


if __name__ == "__main__":
    main()