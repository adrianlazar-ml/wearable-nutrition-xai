"""
ABLATION STUDY — Oura Ring vs Apple Watch Feature Contribution
================================================================
Testează contribuția fiecărui dispozitiv la performanța modelului ML.

Configurații testate:
  A. Doar features Oura Ring (11 features lag-1)
  B. Doar features Apple Watch (8 features lag-1)
  C. Ambele dispozitive (19 features lag-1)

IMPORTANT: Folosește aceeași configurație ca pipeline-ul principal
(cap5_compare_algorithms_json_ext.py):
  - Random Forest (n_estimators=100, max_depth=10, min_samples_leaf=5)
  - Sursa JSON (Health Auto Export)
  - 19 features (inclusiv sleep_deep, sleep_rem, resp_rate)
  - SMOTE k_neighbors=3
  - Split temporal 80/20

Output:
  - data/ablation_study_table.csv
  - data/ablation_study_comparison.png
  - Tabel consolă pentru Cap.5

Autor: Adrian Lazar | Disertatie UPB 2026
Rulare: python ablation_study.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================
# Features per dispozitiv (19 total = 11 Oura + 8 Apple Watch)
OURA_FEATURES = [
    "readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
    "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
    "steps_oura", "respiratory_rate", "temp_deviation",
]

AW_FEATURES = [
    "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2",
    "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate",
]

ALL_FEATURES = OURA_FEATURES + AW_FEATURES

TRAIN_RATIO = 0.8
RANDOM_STATE = 42
LABEL_NAMES = {0: "Rest", 1: "Normal", 2: "Active"}

# Configurațiile de ablation
ABLATION_CONFIGS = [
    ("Doar Oura (11 feat)", OURA_FEATURES),
    ("Doar Apple Watch (8 feat)", AW_FEATURES),
    ("Ambele dispozitive (19 feat)", ALL_FEATURES),
]


# ============================================================
# LOAD DATE — sursa JSON (identic cu cap5_compare_json_ext.py)
# ============================================================
def load_data():
    print("  Sursa: JSON (Health Auto Export)")
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

    print(f"  Total zile după merge: {len(df)}")
    return df


# ============================================================
# PIPELINE ML PER CONFIGURAȚIE — RF (identic cu pipeline principal)
# ============================================================
def run_ablation(df, config_name, base_features):
    """Rulează pipeline-ul complet pentru un subset de features."""

    lag_features = [f"{f}_lag1" for f in base_features]

    # Lag features
    df_work = df.copy()
    for feat in base_features:
        col = f"{feat}_lag1"
        if col not in df_work.columns:
            df_work[col] = df_work[feat].shift(1)

    # Labels (T3, identice pentru toate configurațiile)
    def assign_label(row):
        if pd.isna(row["readiness_score"]) or pd.isna(row["sleep_score"]):
            return np.nan
        if row["readiness_score"] < 60 or row["sleep_score"] < 60:
            return 0
        elif row["readiness_score"] >= 85 and row["sleep_score"] >= 75:
            return 2
        return 1

    df_work["label"] = df_work.apply(assign_label, axis=1)
    df_model = df_work[lag_features + ["label"]].dropna(subset=["label"])

    X = df_model[lag_features]
    y = df_model["label"].astype(int)

    # Temporal split
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Imputer — fit pe train (pre-SMOTE)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=lag_features, index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=lag_features, index=X_test.index
    )

    # SMOTE
    min_count = y_train.value_counts().min()
    k = min(3, min_count - 1) if min_count >= 2 else 0
    if k >= 1:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)
        X_train_sm = pd.DataFrame(X_train_sm, columns=lag_features)
    else:
        X_train_sm, y_train_sm = X_train_imp, y_train

    # Random Forest — identic cu pipeline-ul principal
    rf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE,
        max_depth=10, min_samples_leaf=5
    )
    rf.fit(X_train_sm, y_train_sm)
    y_pred = rf.predict(X_test_imp)

    # Metrici
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_per = {}
    for lbl in [0, 1, 2]:
        if lbl in y_test.unique():
            f1_per[lbl] = f1_score(y_test, y_pred, labels=[lbl], average="macro", zero_division=0)
        else:
            f1_per[lbl] = np.nan

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    return {
        "config": config_name,
        "n_features": len(base_features),
        "f1_macro": f1_macro,
        "accuracy": acc,
        "f1_rest": f1_per.get(0, np.nan),
        "f1_normal": f1_per.get(1, np.nan),
        "f1_active": f1_per.get(2, np.nan),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "confusion_matrix": cm,
    }


# ============================================================
# VIZUALIZĂRI
# ============================================================
def plot_comparison(results):
    """Bar chart comparativ F1 Macro + F1 per clasă."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = [r["config"] for r in results]
    x = np.arange(len(names))

    # Subplot 1: F1 Macro + Accuracy
    ax1 = axes[0]
    width = 0.35
    bars1 = ax1.bar(x - width/2, [r["f1_macro"] for r in results],
                    width, label="F1 Macro", color="#e74c3c", alpha=0.85)
    bars2 = ax1.bar(x + width/2, [r["accuracy"] for r in results],
                    width, label="Accuracy", color="#3498db", alpha=0.85)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=10)

    ax1.set_ylabel("Score")
    ax1.set_title("F1 Macro & Accuracy per Configurație", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.legend()
    ax1.set_ylim(0, 1.0)

    # Subplot 2: F1 per clasă
    ax2 = axes[1]
    width = 0.25
    f1_rest = [r["f1_rest"] for r in results]
    f1_norm = [r["f1_normal"] for r in results]
    f1_actv = [r.get("f1_active", 0) for r in results]

    ax2.bar(x - width, f1_rest, width, label="F1 Rest", color="#e74c3c", alpha=0.7)
    ax2.bar(x, f1_norm, width, label="F1 Normal", color="#f39c12", alpha=0.7)
    ax2.bar(x + width, f1_actv, width, label="F1 Active", color="#27ae60", alpha=0.7)

    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 per Clasă (Rest / Normal / Active)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.legend()
    ax2.set_ylim(0, 1.0)

    plt.suptitle(
        "Ablation Study: Contribuția Oura Ring vs Apple Watch\n"
        "Random Forest | 19 features | JSON | Temporal split 80/20 | SMOTE k=3",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/experiments/ablation_study_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Salvat: data/ablation_study_comparison.png")


def plot_confusion_matrices(results):
    """Matrici de confuzie side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    class_labels = ["Rest", "Normal", "Active"]

    for ax, r in zip(axes, results):
        cm = r["confusion_matrix"]
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_yticklabels(class_labels, fontsize=9)
        ax.set_xlabel("Prezis")
        ax.set_ylabel("Real")
        ax.set_title(r["config"], fontsize=10, fontweight="bold")

        for i in range(3):
            for j in range(3):
                color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                       fontsize=12, fontweight="bold", color=color)

    plt.suptitle(
        "Matrici de Confuzie — Ablation Study (RF, JSON, 19 feat)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/experiments/ablation_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Salvat: data/ablation_confusion_matrices.png")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("  ABLATION STUDY — OURA RING vs APPLE WATCH")
    print("  Random Forest | JSON | 19 features | Temporal 80/20")
    print("=" * 65)

    # Load
    print("\n=== LOAD DATE ===")
    df = load_data()

    # Pre-compute lag features pentru toate
    for feat in ALL_FEATURES:
        df[f"{feat}_lag1"] = df[feat].shift(1)

    # Run ablation
    print(f"\n=== RULARE {len(ABLATION_CONFIGS)} CONFIGURAȚII ===\n")
    results = []
    for name, features in ABLATION_CONFIGS:
        print(f"  [{name}]")
        print(f"    Features: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
        res = run_ablation(df, name, features)
        results.append(res)
        print(f"    F1 Macro: {res['f1_macro']:.3f} | Accuracy: {res['accuracy']:.3f}")
        print(f"    F1: Rest={res['f1_rest']:.3f}, Normal={res['f1_normal']:.3f}, "
              f"Active={res['f1_active']:.3f}")
        print()

    # Salvare CSV
    rows = []
    for r in results:
        rows.append({
            "config": r["config"],
            "n_features": r["n_features"],
            "f1_macro": r["f1_macro"],
            "accuracy": r["accuracy"],
            "f1_rest": r["f1_rest"],
            "f1_normal": r["f1_normal"],
            "f1_active": r["f1_active"],
        })
    df_results = pd.DataFrame(rows)
    df_results.to_csv("outputs/tables/ablation_study_table.csv", index=False)
    print("  Salvat: data/ablation_study_table.csv")

    # Grafice
    print("\n=== GENERARE GRAFICE ===")
    plot_comparison(results)
    plot_confusion_matrices(results)

    # Tabel final
    print(f"\n{'='*75}")
    print("  TABEL REZUMAT — ABLATION STUDY (RF, JSON, 19 features)")
    print(f"{'='*75}")
    print(f"{'Configurație':<32} {'#Feat':>5} {'F1 Mac':>7} {'Acc':>7} "
          f"{'F1_R':>6} {'F1_N':>6} {'F1_A':>6}")
    print("-" * 75)
    for r in results:
        print(f"{r['config']:<32} {r['n_features']:>5} {r['f1_macro']:>7.3f} "
              f"{r['accuracy']:>7.3f} {r['f1_rest']:>6.3f} {r['f1_normal']:>6.3f} "
              f"{r['f1_active']:>6.3f}")

    # Analiză complementaritate
    oura_f1 = results[0]["f1_macro"]
    aw_f1 = results[1]["f1_macro"]
    both_f1 = results[2]["f1_macro"]

    print(f"\n{'='*75}")
    print("  ANALIZĂ COMPLEMENTARITATE")
    print(f"{'='*75}")
    print(f"  Doar Oura:        F1 Macro = {oura_f1:.3f}")
    print(f"  Doar Apple Watch: F1 Macro = {aw_f1:.3f}")
    print(f"  Ambele (19 feat): F1 Macro = {both_f1:.3f}")

    gain_vs_oura = both_f1 - oura_f1
    gain_vs_aw = both_f1 - aw_f1

    print(f"\n  Gain combinat vs Oura Only: +{gain_vs_oura:.3f} ({gain_vs_oura*100:.1f}pp)")
    print(f"  Gain combinat vs AW Only:   +{gain_vs_aw:.3f} ({gain_vs_aw*100:.1f}pp)")

    if both_f1 > max(oura_f1, aw_f1):
        print(f"\n  COMPLEMENTARITATE CONFIRMATA: modelul combinat depaseste")
        print(f"    ambele modele single-device.")

    print(f"\n{'='*65}")
    print(f"  GATA — Grafice salvate in data/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()