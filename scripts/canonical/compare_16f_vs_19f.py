"""
compare_16f_vs_19f.py - Deltas canonice 16 features vs 19 features
==================================================================
FND-SVM-DELTA-001.

Context: Tabel 5.4 din disertatie are coloana "Delta F1 Macro vs. 16f"
cu valori stale (RF +3,6pp / XGB +0,5pp / SVM -3,1pp), calculate cand
SVM 19f era 0,482 (stale). Canonic curent: SVM 19f = 0,4347. In plus,
fix-ul rhr_avg (rhr_avg < rhr_lowest -> NaN) ar putea fi schimbat si
valorile 16f. Acest script recalculeaza deltas pe sursa JSON canonica.

Metodologie (identica cu pipeline-ul canonic cap5_compare_algorithms_json_ext):
- Sursa: JSON (apple_watch_api/apple_health_daily.csv) + Oura combined
- Dataset: 958 zile (hash 63bc3a2915bae864), split temporal 80/20 = 766/192
- 19f = setul canonic complet (FEATURES din cap5_pipeline_json_extended)
- 16f = 19f minus {aw_sleep_deep_hours, aw_sleep_rem_hours, aw_resp_rate}_lag1
- RF: hiperparametri din frozen_meta_20260331.json (model de productie)
- XGB / SVM: defaults canonice (identice cu cap5_compare)
- SVM: imputer -> StandardScaler -> SMOTE (fix FND-105)
- CV: TimeSeriesSplit 5-fold, ImbPipeline complet (fix leakage FND-101)

Toate componentele stochastice cu random_state=42 explicit.

Output: outputs/tables/compare_16f_vs_19f.csv (cu bloc metadata #-comment).

Sanity (avort daca pica, vs CIFRE_CANONICE.md sectiunea 3):
- RF  19f F1 Macro == 0,5063 +/- 1e-3
- XGB 19f F1 Macro == 0,4760 +/- 1e-3
- SVM 19f F1 Macro == 0,4347 +/- 1e-3

Rulare (Windows): PYTHONUTF8=1 python scripts/canonical/compare_16f_vs_19f.py

Autor: Adrian Lazar | Disertatie UPB 2026
"""

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root in sys.path + CWD, ca importurile si caile relative la date sa mearga
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import sklearn
import imblearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# Reutilizam logica canonica de incarcare/feature engineering (NU duplicam)
from cap5_pipeline_json_extended import (
    FEATURES,
    LAG_FEATURES,
    assign_nutrition_label,
    load_apple_watch,
    load_oura,
    merge_datasets,
)

# ============================================================
# CONSTANTE CANONICE
# ============================================================
RANDOM_STATE = 42
CUTOFF_DATE = "2026-03-31"
EXPECTED_HASH = "63bc3a2915bae864"
N_TRAIN_EXPECTED = 766
N_TEST_EXPECTED = 192

# 16f = 19f minus cele 3 features noi (sleep stages + AW resp rate)
DROP_FOR_16F = [
    "aw_sleep_deep_hours_lag1",
    "aw_sleep_rem_hours_lag1",
    "aw_resp_rate_lag1",
]
FEATURES_19 = list(FEATURES)
FEATURES_16 = [f for f in FEATURES if f not in DROP_FOR_16F]

# Sanity vs CIFRE_CANONICE.md sectiunea 3 (test F1 Macro 19f)
SANITY_19F = {
    "Random Forest": 0.5063,
    "XGBoost": 0.4760,
    "SVM (RBF)": 0.4347,
}
SANITY_TOL = 1e-3

OUTPUT_CSV = Path("outputs/tables/compare_16f_vs_19f.csv")
REPORT_MD = Path("audit/SPRINT_4_DELTA_RESULTS.md")  # generat separat de agent


# ============================================================
# 1. DATASET CANONIC (replica freeze_production_model.build_dataset_until)
# ============================================================
def build_dataset(cutoff=CUTOFF_DATE):
    print("=" * 60)
    print("  BUILD DATASET CANONIC (sursa JSON)")
    print("=" * 60)
    oura = load_oura()
    aw_tuple = load_apple_watch()
    df = merge_datasets(oura, *aw_tuple)

    df = df[df["date"] <= pd.Timestamp(cutoff)].copy()

    for feat in LAG_FEATURES:
        df[f"{feat}_lag1"] = df[feat].shift(1)

    df["nutrition_label"] = df.apply(assign_nutrition_label, axis=1)

    df_model = df[["date"] + FEATURES_19 + ["nutrition_label"]].dropna(
        subset=["nutrition_label"]
    )
    df_model = df_model.sort_values("date").reset_index(drop=True)
    print(f"  Dataset final: {len(df_model)} zile, {len(FEATURES_19)} features")
    return df_model


def hash_dataset(df):
    """Hash identic cu freeze_production_model.hash_dataset (sorted cols, LF)."""
    cols_sorted = sorted(df.columns.tolist())
    content = df[cols_sorted].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


# ============================================================
# 2. MODELE CANONICE
# ============================================================
def make_models():
    """Hiperparametri: RF din frozen_meta, XGB/SVM defaults canonice."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", verbosity=0, random_state=RANDOM_STATE,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=RANDOM_STATE,
        ),
    }


def make_cv_pipe(name):
    """ImbPipeline complet pentru CV de-leaked (fix FND-101). random_state explicit."""
    if "SVM" in name:
        return ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)),
        ])
    if "XGBoost" in name:
        return ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("model", XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                eval_metric="mlogloss", verbosity=0, random_state=RANDOM_STATE,
            )),
        ])
    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
        ("model", RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE,
        )),
    ])


# ============================================================
# 3. EVALUARE PE O CONFIGURATIE (16f sau 19f)
# ============================================================
def eval_config(df_model, feats, config_name):
    print("\n" + "=" * 60)
    print(f"  CONFIG: {config_name} ({len(feats)} features)")
    print("=" * 60)

    X = df_model[feats]
    y = df_model["nutrition_label"].astype(int)

    split_idx = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # Imputer fit pe train (median)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feats, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feats, index=X_test.index)

    # Cale tree-based (RF/XGB): SMOTE pe date imputate nescalate
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

    # Cale SVM: imputer -> scaler -> SMOTE (FND-105)
    scaler = StandardScaler()
    X_train_imp_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=feats, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=feats, index=X_test.index)
    smote_svm = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_train_scaled_sm, y_train_scaled_sm = smote_svm.fit_resample(X_train_imp_scaled, y_train)

    tscv = TimeSeriesSplit(n_splits=5)
    models = make_models()
    out = {}

    for name, model in models.items():
        if "SVM" in name:
            model.fit(X_train_scaled_sm, y_train_scaled_sm)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train_sm, y_train_sm)
            y_pred = model.predict(X_test_imp)

        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_per = f1_score(y_test, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # CV F1 Macro de-leaked (pipeline complet pe X_train brut)
        cv_pipe = make_cv_pipe(name)
        cv_f1 = cross_val_score(cv_pipe, X_train, y_train, cv=tscv, scoring="f1_macro")

        out[name] = {
            "f1_macro": float(f1_mac),
            "f1_rest": float(f1_per[0]),
            "f1_normal": float(f1_per[1]),
            "f1_active": float(f1_per[2]),
            "accuracy": float(acc),
            "mcc": float(mcc),
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_f1_std": float(cv_f1.std()),
        }
        print(f"  {name:<16} F1={f1_mac:.4f}  Acc={acc:.4f}  "
              f"F1act={f1_per[2]:.4f}  CV={cv_f1.mean():.4f}(+/-{cv_f1.std():.4f})")

    return out


# ============================================================
# 4. MAIN
# ============================================================
def main():
    df_model = build_dataset()

    # Verificare hash dataset
    computed_hash = hash_dataset(df_model)
    print(f"\n  Dataset hash calculat: {computed_hash}")
    print(f"  Dataset hash canonic:  {EXPECTED_HASH}")
    if computed_hash != EXPECTED_HASH:
        print(f"\n  [FAIL] HASH MISMATCH - dataset diverg de canonic. ABORT.")
        sys.exit(1)
    print(f"  [OK] HASH MATCH")

    # Verificare split
    split_idx = int(len(df_model) * 0.8)
    n_train, n_test = split_idx, len(df_model) - split_idx
    if (n_train, n_test) != (N_TRAIN_EXPECTED, N_TEST_EXPECTED):
        print(f"\n  [FAIL] SPLIT MISMATCH: {n_train}/{n_test} vs "
              f"{N_TRAIN_EXPECTED}/{N_TEST_EXPECTED}. ABORT.")
        sys.exit(1)
    print(f"  [OK] SPLIT {n_train}/{n_test}")

    # Evaluare ambele configuratii
    res_19 = eval_config(df_model, FEATURES_19, "19f (canonic)")
    res_16 = eval_config(df_model, FEATURES_16, "16f (fara 3 sleep/resp)")

    # Sanity vs canonic (19f)
    print("\n" + "=" * 60)
    print("  SANITY CHECK 19f vs CIFRE_CANONICE.md sectiunea 3")
    print("=" * 60)
    sanity_ok = True
    for name, ref in SANITY_19F.items():
        got = res_19[name]["f1_macro"]
        ok = abs(got - ref) < SANITY_TOL
        sanity_ok = sanity_ok and ok
        print(f"  {name:<16} got={got:.4f}  canonic={ref:.4f}  "
              f"{'[OK]' if ok else '[FAIL]'}")

    # Construire tabel deltas (delta = 19f - 16f, in puncte procentuale)
    rows = []
    model_order = ["Random Forest", "XGBoost", "SVM (RBF)"]
    for name in model_order:
        r16, r19 = res_16[name], res_19[name]
        rows.append({
            "Model": name,
            "F1_Macro_16f": round(r16["f1_macro"], 4),
            "F1_Macro_19f": round(r19["f1_macro"], 4),
            "Delta_F1_pp": round((r19["f1_macro"] - r16["f1_macro"]) * 100, 2),
            "F1_Active_16f": round(r16["f1_active"], 4),
            "F1_Active_19f": round(r19["f1_active"], 4),
            "Delta_F1_Active_pp": round((r19["f1_active"] - r16["f1_active"]) * 100, 2),
            "Accuracy_16f": round(r16["accuracy"], 4),
            "Accuracy_19f": round(r19["accuracy"], 4),
            "MCC_16f": round(r16["mcc"], 4),
            "MCC_19f": round(r19["mcc"], 4),
            "CV_F1_16f": round(r16["cv_f1_mean"], 4),
            "CV_F1_19f": round(r19["cv_f1_mean"], 4),
        })
    df_out = pd.DataFrame(rows)

    # Bloc metadata (#-comment), pattern frozen_meta_20260331.json
    meta_lines = [
        "# compare_16f_vs_19f.csv -- deltas canonice 16f vs 19f (FND-SVM-DELTA-001)",
        "# generated_by: scripts/canonical/compare_16f_vs_19f.py",
        f"# timestamp_run: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
        f"# dataset_hash: {computed_hash} (verified == canonic {EXPECTED_HASH})",
        f"# random_state: {RANDOM_STATE}",
        f"# n_train: {n_train}",
        f"# n_test: {n_test}",
        f"# cutoff_date: {CUTOFF_DATE}",
        f"# features_19f: {len(FEATURES_19)}",
        f"# features_16f: {len(FEATURES_16)} (dropped: {','.join(DROP_FOR_16F)})",
        f"# sanity_19f_vs_canonic: {'PASS' if sanity_ok else 'FAIL'}",
        f"# python_version: {sys.version.split()[0]}",
        f"# sklearn_version: {sklearn.__version__}",
        f"# imblearn_version: {imblearn.__version__}",
        f"# xgboost_version: {xgboost.__version__}",
        f"# numpy_version: {np.__version__}",
        f"# pandas_version: {pd.__version__}",
        "# delta_convention: Delta = 19f - 16f (puncte procentuale)",
    ]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        for ln in meta_lines:
            f.write(ln + "\n")
        df_out.to_csv(f, index=False, lineterminator="\n")

    print("\n" + "=" * 60)
    print("  TABEL DELTAS 16f vs 19f (canonic)")
    print("=" * 60)
    print(df_out.to_string(index=False))
    print(f"\n  Salvat: {OUTPUT_CSV}")

    if not sanity_ok:
        print("\n" + "!" * 60)
        print("  [FAIL] SANITY 19f NU trece. CSV scris pentru inspectie,")
        print("  dar cifrele NU sunt canonice. NU comita. Raporteaza Adrian.")
        print("!" * 60)
        sys.exit(1)

    print("\n  [OK] Sanity 19f PASS. Deltas canonice generate.")


if __name__ == "__main__":
    main()
