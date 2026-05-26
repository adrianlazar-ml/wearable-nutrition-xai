"""
Patch utilitar D10 (FND-116, FND-139): adauga f1_per_class, precision/recall
per clasa si confusion_matrix pentru test intern in frozen_meta_20260331.json.
NU reantreneaza, NU atinge modelul .joblib (hash neschimbat). Cifrele sunt
deterministe (verificate sanity_check Pas 5). Idempotent.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

MODELS_DIR = Path("models")
TAG = "20260331"
CLASS_NAMES = ["Rest", "Normal", "Active"]

meta_path = MODELS_DIR / f"frozen_meta_{TAG}.json"
with open(meta_path, encoding="utf-8") as f:
    meta = json.load(f)

rf = joblib.load(MODELS_DIR / f"rf_frozen_{TAG}.joblib")
imputer = joblib.load(MODELS_DIR / f"imputer_frozen_{TAG}.joblib")

df = pd.read_csv(MODELS_DIR / f"dataset_snapshot_{TAG}.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

FEATURES = meta["features"]
split_idx = int(len(df) * 0.8)
X_test = df.iloc[split_idx:][FEATURES]
y_test = df.iloc[split_idx:]["nutrition_label"].astype(int)

X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURES)
y_pred = rf.predict(X_test_imp)

report = classification_report(
    y_test, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
    output_dict=True, zero_division=0,
)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()

# Injectare aditiva (modelul .joblib NU se atinge, cheile vechi raman intacte)
ref = meta["reference_metrics_internal_test"]
ref["f1_per_class"] = {n: report[n]["f1-score"] for n in CLASS_NAMES}
ref["precision_per_class"] = {n: report[n]["precision"] for n in CLASS_NAMES}
ref["recall_per_class"] = {n: report[n]["recall"] for n in CLASS_NAMES}
meta["confusion_matrix_internal_test"] = {
    "labels": CLASS_NAMES,
    "rows_true_cols_pred": cm,
}

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("[OK] frozen_meta patch aplicat:")
print(f"  f1_per_class: {ref['f1_per_class']}")
print(f"  confusion_matrix: {cm}")
