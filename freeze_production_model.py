"""
Freeze Production Model — pentru Prospective Validation
========================================================
Rulat O SINGURA DATA pentru a inregistra modelul de productie
la un moment bine definit in timp (train_end_date).

Outputul acestui script este un snapshot reproducibil:
- modelul RF antrenat
- imputer-ul fit-uit
- metadata completa (date, versiuni, hash dataset)

Dupa inghetare, modelul NU se mai atinge. Peste 6-8 saptamani,
prospective_validation.py aplica acest snapshot pe date noi.

Importanta metadata: e dovada pentru comisie ca separarea
temporala train/prospective e stricta (anti-leakage by construction).

Autor: Adrian Lazar | Disertatie UPB 2026
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Import pipeline-ul existent ca sa nu duplicam logica
from cap5_pipeline_json_extended import (
    FEATURES,
    LAG_FEATURES,
    assign_nutrition_label,
    load_apple_watch,
    load_oura,
    merge_datasets,
)


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def build_dataset_until(cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """
    Construieste dataset-ul folosind pipeline-ul existent,
    dar taie strict la cutoff_date (inclusiv).
    """
    print(f"\n=== CONSTRUIRE DATASET PANA LA {cutoff_date.date()} ===")

    oura = load_oura()
    aw_tuple = load_apple_watch()
    df = merge_datasets(oura, *aw_tuple)

    # Taiere stricta: pastram doar zilele <= cutoff_date
    n_before = len(df)
    df = df[df["date"] <= cutoff_date].copy()
    print(f"Taiat la cutoff: {n_before} -> {len(df)} zile")

    # Feature engineering: lag-1 + label T3
    # (replicam logica din engineer_features fara re-sort la final)
    for feat in LAG_FEATURES:
        df[f"{feat}_lag1"] = df[feat].shift(1)

    df["nutrition_label"] = df.apply(assign_nutrition_label, axis=1)

    df_model = df[["date"] + FEATURES + ["nutrition_label"]].dropna(
        subset=["nutrition_label"]
    )
    df_model = df_model.sort_values("date").reset_index(drop=True)

    print(f"Dataset final pentru training: {len(df_model)} zile")
    print(f"Distributie clase: {df_model['nutrition_label'].value_counts().sort_index().to_dict()}")
    return df_model


def hash_dataset(df: pd.DataFrame) -> str:
    """
    Hash SHA256 pe continutul dataset-ului, pentru dovada reproducibilitatii.
    Forteaza lineterminator='\\n' ca hash-ul sa fie invariant cross-platform
    (altfel Windows produce \\r\\n si hash diferit fata de Linux).
    """
    # Sortam coloanele ca sa fie deterministic
    cols_sorted = sorted(df.columns.tolist())
    content = df[cols_sorted].to_csv(
        index=False,
        lineterminator="\n",  # forteaza LF, nu CRLF
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def freeze(train_end_date: str, tag: str) -> None:
    cutoff = pd.Timestamp(train_end_date)

    # 1. Construim dataset-ul
    df_model = build_dataset_until(cutoff)
    dataset_hash = hash_dataset(df_model)

    X_full = df_model[FEATURES]
    y_full = df_model["nutrition_label"].astype(int)
    dates = df_model["date"]

    # 2. Split INTERN 80/20 cronologic - pentru metrica de referinta
    #    (test intern = ultimele 20% ale perioadei pre-cutoff)
    split_idx = int(len(df_model) * 0.8)
    X_train, X_test_internal = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_train, y_test_internal = y_full.iloc[:split_idx], y_full.iloc[split_idx:]
    train_end_effective = dates.iloc[split_idx - 1]
    test_internal_start = dates.iloc[split_idx]

    print(f"\n=== SPLIT INTERN (pentru metrica de referinta) ===")
    print(f"Train: {len(X_train)} zile ({dates.iloc[0].date()} -> {train_end_effective.date()})")
    print(f"Test intern: {len(X_test_internal)} zile ({test_internal_start.date()} -> {dates.iloc[-1].date()})")

    # 3. Fit imputer + SMOTE + RF pe train intern (acelasi config ca pipeline-ul final)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_leaf=5,
    )
    rf.fit(X_train_sm, y_train_sm)

    # 4. Metrica de referinta pe test intern (va fi comparata cu prospective)
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, matthews_corrcoef,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_internal), columns=FEATURES, index=X_test_internal.index
    )
    y_pred = rf.predict(X_test_imp)
    acc = accuracy_score(y_test_internal, y_pred)
    f1_mac = f1_score(y_test_internal, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test_internal, y_pred)

    print(f"\n=== METRICA DE REFERINTA (test intern) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Macro:  {f1_mac:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(classification_report(
        y_test_internal, y_pred, target_names=["Rest", "Normal", "Active"], zero_division=0
    ))

    # 5. Salvare artefacte
    model_path = MODELS_DIR / f"rf_frozen_{tag}.joblib"
    imputer_path = MODELS_DIR / f"imputer_frozen_{tag}.joblib"
    meta_path = MODELS_DIR / f"frozen_meta_{tag}.json"

    joblib.dump(rf, model_path)
    joblib.dump(imputer, imputer_path)

    # 6. Metadata (dovada reproducibilitatii pentru comisie)
    import sklearn
    import imblearn

    meta = {
        "tag": tag,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cutoff_date": train_end_date,  # ultima zi <= care luam date (intrare user)
        "last_day_in_dataset": str(dates.iloc[-1].date()),  # ultima zi efectiv gasita in date
        "train_internal_end": str(train_end_effective.date()),  # ultima zi a train-ului intern (80%)
        "test_internal_range": [str(test_internal_start.date()), str(dates.iloc[-1].date())],
        "n_total_days": int(len(df_model)),
        "n_train": int(len(X_train)),
        "n_test_internal": int(len(X_test_internal)),
        "class_distribution_full": {
            str(k): int(v) for k, v in y_full.value_counts().sort_index().items()
        },
        "class_distribution_train_pre_smote": {
            str(k): int(v) for k, v in y_train.value_counts().sort_index().items()
        },
        "class_distribution_train_post_smote": {
            str(k): int(v) for k, v in pd.Series(y_train_sm).value_counts().sort_index().items()
        },
        "class_distribution_test_internal": {
            str(k): int(v) for k, v in y_test_internal.value_counts().sort_index().items()
        },
        "features": FEATURES,
        "label_scheme": "T3: Rest if readiness<60 OR sleep<60; Active if readiness>=85 AND sleep>=75; Normal otherwise",
        "class_mapping": {"0": "Rest", "1": "Normal", "2": "Active"},
        "model_config": {
            "algorithm": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
        },
        "preprocessing": {
            "imputer": "SimpleImputer(strategy='median')",
            "smote": "SMOTE(random_state=42, k_neighbors=3)",
            "applied_on": "train only, after imputation",
        },
        "reference_metrics_internal_test": {
            "accuracy": float(acc),
            "f1_macro": float(f1_mac),
            "mcc": float(mcc),
        },
        "dataset_hash_sha256_16": dataset_hash,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
            "imblearn": imblearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "artifacts": {
            "model": str(model_path),
            "imputer": str(imputer_path),
            "meta": f"models/frozen_meta_{tag}.json",
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Salvam si dataset-ul pentru reproducibilitate (fara PII, doar feature values)
    snapshot_path = MODELS_DIR / f"dataset_snapshot_{tag}.csv"
    df_model.to_csv(snapshot_path, index=False)

    print(f"\n=== ARTEFACTE SALVATE ===")
    print(f"  Model:    {model_path}")
    print(f"  Imputer:  {imputer_path}")
    print(f"  Metadata: {meta_path}")
    print(f"  Snapshot: {snapshot_path}")
    print(f"  Dataset hash (primele 16 char SHA256): {dataset_hash}")
    print(f"\n  Aceste artefacte sunt imutabile. Peste 6-8 saptamani,")
    print(f"  prospective_validation.py le va incarca si va rula pe date noi.")


def main():
    parser = argparse.ArgumentParser(description="Freeze production model pentru prospective validation.")
    parser.add_argument(
        "--train_end_date",
        type=str,
        default="2026-03-31",
        help="Data ultimei zile incluse in training (inclusiv). Default: 2026-03-31.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag pentru artefacte (ex: '31032026'). Default: derivat din train_end_date.",
    )
    args = parser.parse_args()

    tag = args.tag or args.train_end_date.replace("-", "")

    print("=" * 60)
    print("  FREEZE PRODUCTION MODEL")
    print(f"  Tag: {tag}")
    print(f"  Train end date: {args.train_end_date}")
    print("=" * 60)

    freeze(args.train_end_date, tag)


if __name__ == "__main__":
    main()
