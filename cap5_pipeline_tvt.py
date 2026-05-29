"""
Cap.5 - Pipeline ML (B3 - Train / Validation / Test split temporal)
===================================================================
Sprint B3.0. Script DERIVAT din cap5_pipeline_json_extended.py.

Scop: introducem un set de VALIDARE dedicat (val) decupat cronologic
din partea de train, in loc de CV pe train. Split temporal 70/10/20
pozitional, fara shuffle:
  - train = primele 70% zile
  - val   = urmatoarele 10% zile
  - test  = ultimele 20% zile

NU rescriem load/merge/feature engineering. Importam functiile canonice
din cap5_pipeline_json_extended ca sa garantam ca cele 19 features si
fix-ul rhr_avg sunt bit-identice cu pipeline-ul canonic. Nimic canonic
nu este modificat: NU scriem in data/, models/, outputs/tables/,
outputs/figures/. Singurul output este outputs/b3_tvt/.

Nota (verificat la rulare): N=958 (canonic). Cu 70/10/20 pozitional rezulta
train=670, val=95, test=193. Test-ul B3 incepe la iloc[765:], deci NU este
identic cu test-ul canonic frozen (iloc[766:], 192 randuri, {0:44,1:140,2:8}):
are o zi Normal in plus la granita -> 193 randuri, {0:44,1:141,2:8}. In plus
modelul e re-antrenat pe 670 zile (nu 766), iar val (95 zile, vara 2025) este
decupat exact din intervalul cel mai apropiat temporal de test, deci metricile
pe test scad fata de frozen. Comparatia val vs test ramane pe acelasi model.

Rulare (Windows): PYTHONUTF8=1 python cap5_pipeline_tvt.py
(functiile importate printeaza caractere Unicode, ex. kJ->kcal).

Autor: Adrian Lazar | Disertatie UPB 2026
"""

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Refolosim pipeline-ul canonic. NU rescriem aceste functii: orice
# diferenta de features ar invalida comparatia cu modelul frozen.
from cap5_pipeline_json_extended import (
    load_oura,
    load_apple_watch,
    merge_datasets,
    engineer_features,
    FEATURES,
)

# ============================================================
# CONFIG
# ============================================================
SEED = 42

# k_neighbors pentru SMOTE pe train. Parametrizabil: daca o clasa
# minoritara din train are < SMOTE_K + 1 esantioane, SMOTE arunca
# ValueError -> scade SMOTE_K (sau 2, sau 1).
SMOTE_K = 3

OUTPUT_DIR = "outputs/b3_tvt"
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "metrics_val_vs_test.csv")

CLASS_NAMES = ["Rest", "Normal", "Active"]  # 0, 1, 2
CLASS_LABELS = [0, 1, 2]


# ============================================================
# SPLIT TEMPORAL 70 / 10 / 20 (pozitional, fara shuffle)
# ============================================================
def _split_info(df_split):
    """nr zile, interval date (prima -> ultima), distributie clase {0,1,2}."""
    y = df_split["nutrition_label"].astype(int)
    dist = y.value_counts().reindex(CLASS_LABELS, fill_value=0).to_dict()
    date_start = pd.Timestamp(df_split["date"].iloc[0]).date()
    date_end = pd.Timestamp(df_split["date"].iloc[-1]).date()
    return len(df_split), date_start, date_end, dist


def train_val_test_split_temporal(df_model):
    """
    Split POZITIONAL 70/10/20 pe df_model (deja sortat cronologic).
    Fara shuffle. Returneaza (train, val, test) ca DataFrame-uri.
    """
    n = len(df_model)
    n_train = int(n * 0.70)
    n_val = int(n * 0.10)

    train = df_model.iloc[:n_train]
    val = df_model.iloc[n_train:n_train + n_val]
    test = df_model.iloc[n_train + n_val:]

    print("\n--- SPLIT TEMPORAL 70/10/20 (pozitional, fara shuffle) ---")
    print(f"  Total: {n} zile")
    for name, part in (("train", train), ("val", val), ("test", test)):
        n_days, d0, d1, dist = _split_info(part)
        print(f"  {name:<6}: {n_days:>4} zile  ({d0} -> {d1})  dist={dist}")

    return train, val, test


# ============================================================
# TRAIN / EVAL
# ============================================================
def _evaluate_split(rf, imputer, df_split, split_name):
    """
    Aplica imputer-ul (fit pe train) pe df_split, prezice cu rf si afiseaza
    Accuracy, F1 Macro, MCC, classification_report cu F1 per clasa.
    Returneaza un dict cu metricile.
    """
    X = df_split[FEATURES]
    y = df_split["nutrition_label"].astype(int)

    X_imp = pd.DataFrame(
        imputer.transform(X), columns=FEATURES, index=X.index
    )
    y_pred = rf.predict(X_imp)

    acc = accuracy_score(y, y_pred)
    f1_mac = f1_score(y, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)
    f1_per = f1_score(
        y, y_pred, labels=CLASS_LABELS, average=None, zero_division=0
    )

    print(f"\n  === EVAL pe {split_name.upper()} ===")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Macro:  {f1_mac:.4f}")
    print(f"  MCC:       {mcc:.4f}")
    print(f"  Classification Report ({split_name}):")
    print(classification_report(
        y, y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    n_days, d0, d1, dist = _split_info(df_split)
    return {
        "split": split_name,
        "n_days": n_days,
        "date_start": d0,
        "date_end": d1,
        "n_class0": dist[0],
        "n_class1": dist[1],
        "n_class2": dist[2],
        "accuracy": round(acc, 6),
        "f1_macro": round(f1_mac, 6),
        "mcc": round(mcc, 6),
        "f1_rest": round(float(f1_per[0]), 6),
        "f1_normal": round(float(f1_per[1]), 6),
        "f1_active": round(float(f1_per[2]), 6),
        "smote_k": SMOTE_K,
        "seed": SEED,
    }


def train_eval_tvt(train, val, test):
    """
    Imputer median fit DOAR pe train (transform pe val/test).
    SMOTE doar pe train (k=SMOTE_K). RF identic cu _extended.
    Evalueaza pe VAL si pe TEST separat. Returneaza (val_metrics, test_metrics).
    """
    X_train = train[FEATURES]
    y_train = train["nutrition_label"].astype(int)

    # Imputer fit DOAR pe train (de-leaked) -> transform pe train.
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )

    print("\n--- TRAIN (imputer median fit pe train, SMOTE pe train) ---")
    print(f"  Train inainte SMOTE: {y_train.value_counts().sort_index().to_dict()}")

    # SMOTE doar pe train.
    smote = SMOTE(random_state=SEED, k_neighbors=SMOTE_K)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

    print(f"  Train dupa SMOTE:    {pd.Series(y_train_sm).value_counts().sort_index().to_dict()}")

    # RF identic cu cap5_pipeline_json_extended.
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=SEED,
        max_depth=10,
        min_samples_leaf=5,
    )
    rf.fit(X_train_sm, y_train_sm)

    val_metrics = _evaluate_split(rf, imputer, val, "val")
    test_metrics = _evaluate_split(rf, imputer, test, "test")

    return val_metrics, test_metrics


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  CAP.5 - PIPELINE ML (B3 TRAIN/VAL/TEST TEMPORAL)")
    print("  Derivat din cap5_pipeline_json_extended (19 features)")
    print(f"  Split 70/10/20 pozitional | SMOTE_K={SMOTE_K} | seed={SEED}")
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

    print("\n=== SPLIT TRAIN / VAL / TEST ===")
    train, val, test = train_val_test_split_temporal(df_model)

    print("\n=== TRAIN & EVALUATE (val + test) ===")
    val_metrics, test_metrics = train_eval_tvt(train, val, test)

    # Output DOAR in outputs/b3_tvt/.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = pd.DataFrame([val_metrics, test_metrics])
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 60)
    print("  SUMAR: VAL vs TEST")
    print("=" * 60)
    print(f"  {'Metric':<14} {'Val':>10} {'Test':>10}")
    print(f"  {'-'*36}")
    print(f"  {'Accuracy':<14} {val_metrics['accuracy']:>10.4f} {test_metrics['accuracy']:>10.4f}")
    print(f"  {'F1 Macro':<14} {val_metrics['f1_macro']:>10.4f} {test_metrics['f1_macro']:>10.4f}")
    print(f"  {'MCC':<14} {val_metrics['mcc']:>10.4f} {test_metrics['mcc']:>10.4f}")
    print(f"\nSumar salvat -> {SUMMARY_CSV}")

    return val_metrics, test_metrics


if __name__ == "__main__":
    main()
