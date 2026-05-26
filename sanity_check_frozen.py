"""
Sanity check Pas 5 AUDIT_NEW Faza 1.
Verifica reproducibilitatea modelului inghetat fara re-antrenare.
"""
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef,
)

MODELS_DIR = Path("models")
TAG = "20260331"

# ============================================================
# 1. LOAD ARTEFACTE INGHETATE
# ============================================================
print("=" * 60)
print("SANITY CHECK PAS 5 - Reproducibilitate model inghetat")
print("=" * 60)

rf = joblib.load(MODELS_DIR / f"rf_frozen_{TAG}.joblib")
imputer = joblib.load(MODELS_DIR / f"imputer_frozen_{TAG}.joblib")

with open(MODELS_DIR / f"frozen_meta_{TAG}.json", encoding="utf-8") as f:
    meta = json.load(f)

print(f"\n[OK] Model incarcat: rf_frozen_{TAG}.joblib")
print(f"[OK] Imputer incarcat: imputer_frozen_{TAG}.joblib")
print(f"[OK] Meta incarcat: frozen_meta_{TAG}.json")
print(f"     Dataset hash canonic: {meta['dataset_hash_sha256_16']}")

# ============================================================
# 2. LOAD SNAPSHOT + VERIFICARE HASH
# ============================================================
snapshot_path = MODELS_DIR / f"dataset_snapshot_{TAG}.csv"
df = pd.read_csv(snapshot_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Hash DataFrame normalizat (replica exact din freeze_production_model.py)
cols_sorted = sorted(df.columns.tolist())
content = df[cols_sorted].to_csv(index=False, lineterminator="\n").encode("utf-8")
computed_hash = hashlib.sha256(content).hexdigest()[:16]

print(f"\n[INFO] Snapshot incarcat: {len(df)} zile")
print(f"[INFO] Hash calculat: {computed_hash}")
print(f"[INFO] Hash canonic:  {meta['dataset_hash_sha256_16']}")
if computed_hash == meta['dataset_hash_sha256_16']:
    print(f"[OK] HASH MATCH")
else:
    print(f"[FAIL] HASH DIFFER - dataset modificat post-freeze")

# ============================================================
# 3. SPLIT 80/20 CRONOLOGIC (replica freeze)
# ============================================================
FEATURES = meta["features"]
split_idx = int(len(df) * 0.8)
X_test = df.iloc[split_idx:][FEATURES]
y_test = df.iloc[split_idx:]["nutrition_label"].astype(int)

print(f"\n[INFO] Test intern: {len(X_test)} zile")
print(f"[INFO] Distributie test:")
print(f"       Calculat: {y_test.value_counts().sort_index().to_dict()}")
print(f"       Canonic:  {meta['class_distribution_test_internal']}")

# ============================================================
# 4. PREDICT + METRICS
# ============================================================
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURES)
y_pred = rf.predict(X_test_imp)

acc = accuracy_score(y_test, y_pred)
f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"\n{'='*60}")
print("METRICI REPRODUSE vs CANONIC")
print("=" * 60)
print(f"{'Metric':<15} {'Reprodus':>12} {'Canonic':>12} {'Match':>6}")
print("-" * 60)

canonic = meta["reference_metrics_internal_test"]
for name, calc, ref in [
    ("Accuracy", acc, canonic["accuracy"]),
    ("F1 Macro", f1_mac, canonic["f1_macro"]),
    ("MCC", mcc, canonic["mcc"]),
]:
    match = "OK" if abs(calc - ref) < 1e-6 else "FAIL"
    print(f"{name:<15} {calc:>12.6f} {ref:>12.6f} {match:>6}")

print(f"\nClassification report:")
print(classification_report(
    y_test, y_pred, target_names=["Rest", "Normal", "Active"],
    zero_division=0, digits=4
))

print(f"Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test, y_pred)
print(f"        Rest  Normal  Active")
for i, name in enumerate(["Rest  ", "Normal", "Active"]):
    print(f"{name}  {cm[i, 0]:>6} {cm[i, 1]:>7} {cm[i, 2]:>7}")