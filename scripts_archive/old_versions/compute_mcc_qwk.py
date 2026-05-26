"""
compute_mcc_qwk.py — Calcul MCC + QWK pentru Cap.6
====================================================
Replică exact pipeline-ul din cap5_compare_algorithms_json_ext.py:
  - 19 features, sursa JSON, SMOTE k=3, split temporal 80/20
  - RF, XGBoost, SVM cu aceiași hiperparametri
  - Adaugă: MCC (Matthews Correlation Coefficient) + QWK (Quadratic Weighted Kappa)

Utilizare:
  python compute_mcc_qwk.py

Output:
  - Print rezultate în consolă (format tabel)
  - Salvează outputs/tables/mcc_qwk_results.csv
  
Autor: Adrian Lazar, UPB ACS 2026
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, accuracy_score, matthews_corrcoef,
    cohen_kappa_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURARE CĂII — ADAPTEAZĂ DACĂ E NEVOIE
# ============================================================
# Calea către directorul rădăcină al proiectului
# Script-ul caută automat fișierele relativ la locația sa
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Dacă rulezi din rădăcina proiectului, setează PROJECT_ROOT = "."
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")

OURA_CSV = os.path.join(PROJECT_ROOT, "data", "oura_combined.csv")
AW_CSV = os.path.join(PROJECT_ROOT, "apple_watch_api", "apple_health_daily.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. FEATURES ȘI PRAGURI (identice cu pipeline-ul existent)
# ============================================================
OURA_FEATURES = [
    'readiness_score', 'sleep_score', 'hrv_mean', 'hrv_balance',
    'recovery_index', 'rhr_lowest', 'rhr_avg', 'activity_score',
    'steps_oura', 'respiratory_rate', 'temp_deviation'
]

AW_FEATURES = [
    'aw_hr_mean', 'aw_hrv_mean', 'aw_steps', 'aw_active_energy_kj',
    'aw_spo2', 'aw_sleep_deep_hours', 'aw_sleep_rem_hours', 'aw_resp_rate'
]

ALL_FEATURES = OURA_FEATURES + AW_FEATURES  # 19 features

# Praguri T3
def label_day(row):
    r, s = row['readiness_score'], row['sleep_score']
    if r < 60 or s < 60:
        return 'Rest'
    elif r >= 85 and s >= 75:
        return 'Active'
    else:
        return 'Normal'

# ============================================================
# 3. LOAD & PREPROCESS
# ============================================================
def load_and_merge():
    """Încarcă și unifică datele Oura + Apple Watch."""
    oura = pd.read_csv(OURA_CSV, parse_dates=['date'])
    aw = pd.read_csv(AW_CSV, parse_dates=['date'])
    
    # Fix rhr_avg bug (Gen4: rhr_avg < rhr_lowest)
    mask = oura['rhr_avg'] < oura['rhr_lowest']
    oura.loc[mask, 'rhr_avg'] = np.nan
    if mask.sum() > 0:
        print(f"  [fix] rhr_avg < rhr_lowest corectate: {mask.sum()} zile")
    
    # Merge pe dată
    df = pd.merge(oura, aw, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Zile merged: {len(df)}")
    
    # Etichetare T3
    df['label'] = df.apply(label_day, axis=1)
    
    # Lag-1: folosim features din ziua anterioară pentru predicția zilei curente
    for feat in ALL_FEATURES:
        df[f'{feat}_lag1'] = df[feat].shift(1)
    
    # Prima zi nu are lag-1
    df = df.iloc[1:].reset_index(drop=True)
    
    lag_features = [f'{f}_lag1' for f in ALL_FEATURES]
    
    # Drop rows cu orice lag feature NaN (consistent cu pipeline-ul existent)
    df = df.dropna(subset=lag_features, how='any').reset_index(drop=True)
    
    print(f"  Zile după lag-1 + dropna: {len(df)}")
    print(f"  Distribuție clase: {df['label'].value_counts().to_dict()}")
    
    return df, lag_features

# ============================================================
# 4. SPLIT TEMPORAL 80/20
# ============================================================
def temporal_split(df, lag_features, train_ratio=0.8):
    """Split temporal: primele 80% train, ultimele 20% test."""
    n_train = int(len(df) * train_ratio)
    
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    
    X_train = train_df[lag_features].values
    y_train = train_df['label'].values
    X_test = test_df[lag_features].values
    y_test = test_df['label'].values
    
    print(f"  Train: {len(X_train)} zile, Test: {len(X_test)} zile")
    print(f"  Train distribs: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"  Test distribs:  {pd.Series(y_test).value_counts().to_dict()}")
    
    return X_train, y_train, X_test, y_test

# ============================================================
# 5. PIPELINE: IMPUTE + SMOTE + TRAIN + EVALUATE
# ============================================================
def build_and_evaluate(X_train, y_train, X_test, y_test):
    """Antrenează RF, XGB, SVM și calculează toate metricile."""
    
    # Imputer (fit pe train)
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # SMOTE k=3 (doar pe train)
    smote = SMOTE(k_neighbors=3, random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)
    print(f"  Post-SMOTE train: {pd.Series(y_train_sm).value_counts().to_dict()}")
    
    # Scaler pentru SVM (fit pe train SMOTE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sm)
    X_test_scaled = scaler.transform(X_test_imp)
    
    # Encoding labels: XGBoost necesită int, RF/SVM acceptă string
    label_map = {'Rest': 0, 'Normal': 1, 'Active': 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    
    y_train_enc = np.array([label_map[l] for l in y_train_sm])
    y_test_enc = np.array([label_map[l] for l in y_test])
    
    # Definire modele (aceiași hiperparametri ca în pipeline-ul existent)
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='mlogloss',
            use_label_encoder=False
        ),
        'SVM (RBF)': SVC(
            C=1.0, gamma='scale', kernel='rbf',
            probability=True, random_state=42
        ),
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n  === {name} ===")
        
        if 'SVM' in name:
            model.fit(X_train_scaled, y_train_sm)
            y_pred = model.predict(X_test_scaled)
        elif 'XGBoost' in name:
            # XGBoost necesită labels numerice
            model.fit(X_train_sm, y_train_enc)
            y_pred_enc = model.predict(X_test_imp)
            y_pred = np.array([inv_label_map[int(p)] for p in y_pred_enc])
        else:
            model.fit(X_train_sm, y_train_sm)
            y_pred = model.predict(X_test_imp)
        
        # --- Metrici existente ---
        acc = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average='macro')
        f1_per_class = f1_score(y_test, y_pred, average=None,
                                labels=['Rest', 'Normal', 'Active'])
        
        # --- Metrici NOI ---
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # QWK: encoding ordinal (Rest=0, Normal=1, Active=2)
        y_test_ord = np.array([label_map[l] for l in y_test])
        y_pred_ord = np.array([label_map[l] for l in y_pred])
        qwk = cohen_kappa_score(y_test_ord, y_pred_ord, weights='quadratic')
        
        # Cohen's Kappa (neponderat, pentru comparație)
        kappa_unw = cohen_kappa_score(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=['Rest', 'Normal', 'Active'])
        
        # Print
        print(f"  Accuracy:   {acc:.4f}")
        print(f"  F1 Macro:   {f1_mac:.4f}")
        print(f"  F1 Rest:    {f1_per_class[0]:.4f}")
        print(f"  F1 Normal:  {f1_per_class[1]:.4f}")
        print(f"  F1 Active:  {f1_per_class[2]:.4f}")
        print(f"  MCC:        {mcc:.4f}")
        print(f"  QWK:        {qwk:.4f}")
        print(f"  Kappa (unw):{kappa_unw:.4f}")
        print(f"  Confusion matrix:")
        print(f"  {cm}")
        
        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'F1 Macro': round(f1_mac, 4),
            'F1 Rest': round(f1_per_class[0], 4),
            'F1 Normal': round(f1_per_class[1], 4),
            'F1 Active': round(f1_per_class[2], 4),
            'MCC': round(mcc, 4),
            'QWK': round(qwk, 4),
            'Kappa (unweighted)': round(kappa_unw, 4),
        })
    
    return pd.DataFrame(results)

# ============================================================
# 6. MAIN
# ============================================================
def main():
    print("=" * 60)
    print("CALCUL MCC + QWK — Pipeline 19 features, JSON, SMOTE k=3")
    print("=" * 60)
    
    # Verificare fișiere
    for path, desc in [(OURA_CSV, "Oura CSV"), (AW_CSV, "Apple Watch CSV")]:
        if not os.path.exists(path):
            print(f"\n[EROARE] Nu găsesc {desc}: {path}")
            print("Setează PROJECT_ROOT sau rulează din directorul proiectului.")
            print("Exemplu: PROJECT_ROOT=/cale/catre/proiect python compute_mcc_qwk.py")
            return
    
    print("\n[1/3] Încărcare date...")
    df, lag_features = load_and_merge()
    
    print("\n[2/3] Split temporal 80/20...")
    X_train, y_train, X_test, y_test = temporal_split(df, lag_features)
    
    print("\n[3/3] Antrenare modele + calcul metrici...")
    results_df = build_and_evaluate(X_train, y_train, X_test, y_test)
    
    # Salvare
    output_path = os.path.join(OUTPUT_DIR, "mcc_qwk_results.csv")
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("REZULTATE FINALE")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nSalvat: {output_path}")
    
    # Interpretare scurtă pentru Cap.6
    print("\n" + "-" * 60)
    print("INTERPRETARE PENTRU CAP.6:")
    print("-" * 60)
    best_mcc = results_df.loc[results_df['MCC'].idxmax()]
    best_qwk = results_df.loc[results_df['QWK'].idxmax()]
    print(f"Best MCC:  {best_mcc['Model']} ({best_mcc['MCC']:.4f})")
    print(f"Best QWK:  {best_qwk['Model']} ({best_qwk['QWK']:.4f})")
    print()
    print("MCC range:  [-1, +1], 0 = random, >0.3 = fair, >0.5 = good")
    print("QWK range:  [-1, +1], 0 = random, >0.4 = moderate, >0.6 = substantial")
    print("MCC ia în calcul toate cele 4 cadrane ale confusion matrix → ")
    print("mai informativ decât accuracy pe date dezechilibrate.")
    print("QWK penalizează erorile proporțional cu distanța ordinală →")
    print("confuzia Rest↔Active e penalizată mai mult decât Rest↔Normal.")

if __name__ == '__main__':
    main()