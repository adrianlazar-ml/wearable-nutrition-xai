"""
Bootstrap CI pentru modelul frozen pe test set intern (Sprint 4)
================================================================
Calculeaza intervale de incredere 95% pentru metricile modelului frozen
20260331, prin bootstrap resampling pe cele 192 zile de test intern.

Motivatie (vezi audit/DECIZIE_B3_DEVICE_SHIFT_2026-05-29.md, limitarea 3):
clasa Active are doar n=8 esantioane in test; metrica per-clasa este
instabila la acest n. Bootstrap CI cuantifica aceasta incertitudine fara
a re-antrena sau a atinge nimic canonic.

READ-ONLY pe tot ce e canonic:
  - incarca rf_frozen_20260331.joblib + imputer_frozen_20260331.joblib
  - incarca models/dataset_snapshot_20260331.csv
  - reconstruieste test set-ul EXACT ca in sanity_check_frozen.py
    (split 80/20 temporal, ultimele 192 zile)
  - NU re-antreneaza, NU re-ingheata, NU scrie in models/ sau data/

Metodologie bootstrap:
  - resampling cu inlocuire pe perechile (y_true, y_pred) ale celor 192 zile;
    y_pred este calculat O SINGURA DATA pe test-ul complet (nu re-predict per
    resample), deci modelul frozen ramane neatins. Este bootstrap pe predictii.
  - 2000 de resamples, seed 42 (numpy default_rng).
  - per resample: F1 Macro, MCC, Accuracy, F1 per clasa (Rest/Normal/Active).
  - CI 95% = percentilele 2.5 si 97.5 din distributia bootstrap.

Output DOAR in outputs/bootstrap_ci/ (folder nou). NU scrie in
outputs/tables/, models/, data/.

Rulare (Windows): PYTHONUTF8=1 python bootstrap_ci_frozen.py

Autor: Adrian Lazar | Disertatie UPB 2026
"""

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# ============================================================
# CONFIG
# ============================================================
SEED = 42
N_BOOTSTRAP = 2000
CI_LOWER_PCT = 2.5
CI_UPPER_PCT = 97.5

MODELS_DIR = Path("models")
TAG = "20260331"

OUTPUT_DIR = Path("outputs/bootstrap_ci")
OUTPUT_CSV = OUTPUT_DIR / "bootstrap_ci_frozen_test.csv"

CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["Rest", "Normal", "Active"]

# Toleranta pentru sanity-assert point estimate vs canonic.
TOL = 1e-6


# ============================================================
# METRICI (sursa unica, folosita atat pentru point estimate cat
# si per resample, ca sa fie garantat consistente)
# ============================================================
def compute_metrics(y_true, y_pred):
    """
    Returneaza dict cu accuracy, f1_macro, mcc, f1_rest, f1_normal, f1_active.
    f1 per clasa cu labels=[0,1,2] -> intotdeauna 3 valori in ordine fixa,
    chiar daca o clasa lipseste din resample (zero_division=0).
    """
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1_per = f1_score(
        y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0
    )
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_mac),
        "mcc": float(mcc),
        "f1_rest": float(f1_per[0]),
        "f1_normal": float(f1_per[1]),
        "f1_active": float(f1_per[2]),
    }


# ============================================================
# 1. LOAD ARTEFACTE FROZEN (read-only)
# ============================================================
def load_frozen():
    rf = joblib.load(MODELS_DIR / f"rf_frozen_{TAG}.joblib")
    imputer = joblib.load(MODELS_DIR / f"imputer_frozen_{TAG}.joblib")
    with open(MODELS_DIR / f"frozen_meta_{TAG}.json", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[OK] Model frozen incarcat: rf_frozen_{TAG}.joblib")
    print(f"[OK] Imputer frozen incarcat: imputer_frozen_{TAG}.joblib")
    print(f"[OK] Meta incarcat: frozen_meta_{TAG}.json")
    return rf, imputer, meta


# ============================================================
# 2. RECONSTRUIESTE TEST SET (replica EXACTA sanity_check_frozen.py)
# ============================================================
def reconstruct_test(meta):
    df = pd.read_csv(MODELS_DIR / f"dataset_snapshot_{TAG}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    features = meta["features"]
    split_idx = int(len(df) * 0.8)
    X_test = df.iloc[split_idx:][features]
    y_test = df.iloc[split_idx:]["nutrition_label"].astype(int)

    print(f"\n[INFO] Snapshot: {len(df)} zile | split_idx={split_idx}")
    print(f"[INFO] Test intern: {len(X_test)} zile")
    print(f"[INFO] Distributie test: {y_test.value_counts().sort_index().to_dict()}")
    return X_test, y_test, features


# ============================================================
# 3. POINT ESTIMATE + SANITY vs CANONIC
# ============================================================
def point_estimate_and_sanity(rf, imputer, X_test, y_test, features, meta):
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=features)
    y_pred = rf.predict(X_test_imp)

    point = compute_metrics(y_test.to_numpy(), y_pred)

    ref = meta["reference_metrics_internal_test"]
    canonic = {
        "accuracy": ref["accuracy"],
        "f1_macro": ref["f1_macro"],
        "mcc": ref["mcc"],
        "f1_rest": ref["f1_per_class"]["Rest"],
        "f1_normal": ref["f1_per_class"]["Normal"],
        "f1_active": ref["f1_per_class"]["Active"],
    }

    print(f"\n{'='*62}")
    print("  SANITY: point estimate vs canonic (frozen_meta)")
    print(f"{'='*62}")
    print(f"  {'Metric':<12} {'Reprodus':>12} {'Canonic':>12} {'Match':>7}")
    print(f"  {'-'*46}")
    all_ok = True
    for k in ["accuracy", "f1_macro", "mcc", "f1_rest", "f1_normal", "f1_active"]:
        ok = abs(point[k] - canonic[k]) < TOL
        all_ok = all_ok and ok
        print(f"  {k:<12} {point[k]:>12.6f} {canonic[k]:>12.6f} {('OK' if ok else 'FAIL'):>7}")

    if not all_ok:
        raise SystemExit(
            "\n[FAIL] Point estimate NU reproduce canonicul. "
            "Bootstrap-ul ar fi pe o baza gresita -> ABORT. "
            "Verifica dataset_snapshot / artefacte frozen."
        )
    print("\n[OK] Point estimate reproduce canonicul. Continui cu bootstrap.")
    return point, y_pred


# ============================================================
# 4. BOOTSTRAP RESAMPLING
# ============================================================
def bootstrap(y_true, y_pred):
    """
    2000 resamples cu inlocuire pe perechile (y_true, y_pred).
    Returneaza dict {metric: np.array de N_BOOTSTRAP valori}.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    rng = np.random.default_rng(SEED)
    keys = ["accuracy", "f1_macro", "mcc", "f1_rest", "f1_normal", "f1_active"]
    dist = {k: np.empty(N_BOOTSTRAP, dtype=float) for k in keys}

    n_degenerate = 0  # resamples cu o singura clasa in y_true (MCC nedefinit)

    print(f"\n[INFO] Bootstrap: {N_BOOTSTRAP} resamples pe {n} zile, seed={SEED}")
    # MCC poate fi nedefinit (0) cand y_true resamplat are o singura clasa;
    # f1 cu zero_division=0 este deja gestionat. Suprimam doar zgomotul.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)  # resample cu inlocuire
            yt = y_true[idx]
            yp = y_pred[idx]
            if len(np.unique(yt)) < 2:
                n_degenerate += 1
            m = compute_metrics(yt, yp)
            for k in keys:
                dist[k][i] = m[k]

    if n_degenerate:
        print(f"[INFO] {n_degenerate} resamples cu o singura clasa in y_true "
              f"(MCC tratat ca 0 de sklearn).")
    return dist


# ============================================================
# 5. CI + RAPORT + SALVARE
# ============================================================
def summarize(point, dist):
    rows = []
    label = {
        "accuracy": "Accuracy",
        "f1_macro": "F1 Macro",
        "mcc": "MCC",
        "f1_rest": "F1 Rest (cls0)",
        "f1_normal": "F1 Normal (cls1)",
        "f1_active": "F1 Active (cls2)",
    }
    for k in ["accuracy", "f1_macro", "mcc", "f1_rest", "f1_normal", "f1_active"]:
        lo = float(np.percentile(dist[k], CI_LOWER_PCT))
        hi = float(np.percentile(dist[k], CI_UPPER_PCT))
        rows.append({
            "metric": label[k],
            "point_estimate": round(point[k], 6),
            "ci_lower": round(lo, 6),
            "ci_upper": round(hi, 6),
            "ci_width": round(hi - lo, 6),
            "n_resamples": N_BOOTSTRAP,
            "seed": SEED,
        })

    print(f"\n{'='*72}")
    print(f"  BOOTSTRAP CI 95% (percentile {CI_LOWER_PCT}-{CI_UPPER_PCT}), "
          f"frozen test n=192")
    print(f"{'='*72}")
    print(f"  {'Metric':<18} {'Point':>9} {'CI 95% lower':>14} {'CI 95% upper':>14} {'Latime':>9}")
    print(f"  {'-'*66}")
    for r in rows:
        print(f"  {r['metric']:<18} {r['point_estimate']:>9.4f} "
              f"{r['ci_lower']:>14.4f} {r['ci_upper']:>14.4f} {r['ci_width']:>9.4f}")

    print(f"\n  ATENTIE F1 Active: clasa Active are n=8 in test. CI-ul ei este")
    print(f"  cel mai larg si lower bound-ul poate atinge 0 (resamples fara")
    print(f"  Active corect clasificat). Interpretati F1 Active 0.4444 cu rezerva.")
    return rows


def main():
    print("=" * 62)
    print("  BOOTSTRAP CI - MODEL FROZEN 20260331 (test intern n=192)")
    print(f"  {N_BOOTSTRAP} resamples | seed={SEED} | CI 95%")
    print("=" * 62)

    rf, imputer, meta = load_frozen()
    X_test, y_test, features = reconstruct_test(meta)
    point, y_pred = point_estimate_and_sanity(rf, imputer, X_test, y_test, features, meta)
    dist = bootstrap(y_test.to_numpy(), y_pred)
    rows = summarize(point, dist)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\nCI salvat -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
