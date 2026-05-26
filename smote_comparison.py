"""
COMPARAȚIE SMOTE vs ALTERNATIVE — Class Imbalance Strategies
=============================================================
Testează 5 strategii de gestionare a imbalance-ului de clase
pe același pipeline ML.

IMPORTANT: Folosește aceeași configurație ca pipeline-ul principal
(cap5_compare_algorithms_json_ext.py):
  - Random Forest (n_estimators=100, max_depth=10, min_samples_leaf=5)
  - Sursa JSON (Health Auto Export)
  - 19 features (inclusiv sleep_deep, sleep_rem, resp_rate)
  - Split temporal 80/20

Configurații:
  1. Fără resampling, fără class_weight (baseline naiv)
  2. Cost-sensitive (RF cu class_weight="balanced")
  3. RandomUnderSampler pe clasa majoritară
  4. SMOTE k_neighbors=3 (curent)
  5. SMOTE k_neighbors=5

Output:
  - data/smote_comparison_table.csv
  - data/smote_comparison_chart.png
  - Tabel consolă pentru Cap.5

Autor: Adrian Lazar | Disertatie UPB 2026
Rulare: python smote_comparison.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================
LAG_FEATURES = [
    "readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
    "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
    "steps_oura", "respiratory_rate", "temp_deviation",
    "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2",
    "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate",
]
FEATURES = [f"{f}_lag1" for f in LAG_FEATURES]

TRAIN_RATIO = 0.8
RANDOM_STATE = 42


# ============================================================
# LOAD + PREPARE — sursa JSON (identic cu cap5_compare_json_ext.py)
# ============================================================
def load_and_prepare():
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
    df_model = df[FEATURES + ["label"]].dropna(subset=["label"])

    X = df_model[FEATURES]
    y = df_model["label"].astype(int)

    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Imputer fit pe train (pre-SMOTE)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=FEATURES, index=X_test.index
    )

    return X_train_imp, X_test_imp, y_train, y_test


# ============================================================
# STRATEGIE — RF (identic cu pipeline principal)
# ============================================================
def run_strategy(name, X_train, X_test, y_train, y_test, class_weight=None):
    """Antrenează RF pe datele primite (deja resampled dacă e cazul)."""

    rf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE,
        max_depth=10, min_samples_leaf=5,
        class_weight=class_weight,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    f1_per = {}
    for lbl in [0, 1, 2]:
        if lbl in y_test.unique():
            f1_per[lbl] = f1_score(y_test, y_pred, labels=[lbl],
                                   average="macro", zero_division=0)
        else:
            f1_per[lbl] = np.nan

    train_dist = pd.Series(y_train).value_counts().sort_index().to_dict()

    return {
        "strategy": name,
        "f1_macro": f1_macro,
        "accuracy": acc,
        "f1_rest": f1_per.get(0, np.nan),
        "f1_normal": f1_per.get(1, np.nan),
        "f1_active": f1_per.get(2, np.nan),
        "train_rest": train_dist.get(0, 0),
        "train_normal": train_dist.get(1, 0),
        "train_active": train_dist.get(2, 0),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("  COMPARATIE SMOTE vs ALTERNATIVE")
    print("  Random Forest | JSON | 19 features | Temporal 80/20")
    print("=" * 65)

    print("\n=== LOAD + PREPARE ===")
    X_train, X_test, y_train, y_test = load_and_prepare()
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"  Distributie train originala: {y_train.value_counts().sort_index().to_dict()}")
    min_class = y_train.value_counts().min()
    print(f"  Clasa minima in train: {min_class} instante")

    results = []

    # 1. Baseline naiv (fără resampling, fără class_weight)
    print("\n  [1/5] Fara resampling, fara class_weight")
    res = run_strategy("Baseline (nimic)", X_train, X_test, y_train, y_test)
    results.append(res)
    print(f"    F1 Macro: {res['f1_macro']:.3f} | F1 Active: {res['f1_active']:.3f}")

    # 2. Cost-sensitive (class_weight="balanced")
    print("\n  [2/5] class_weight='balanced'")
    res = run_strategy("Cost-sensitive", X_train, X_test, y_train, y_test,
                       class_weight="balanced")
    results.append(res)
    print(f"    F1 Macro: {res['f1_macro']:.3f} | F1 Active: {res['f1_active']:.3f}")

    # 3. Random UnderSampler
    print("\n  [3/5] RandomUnderSampler")
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    X_rus = pd.DataFrame(X_rus, columns=FEATURES)
    res = run_strategy("UnderSampler", X_rus, X_test, y_rus, y_test)
    results.append(res)
    print(f"    Train dupa RUS: {pd.Series(y_rus).value_counts().sort_index().to_dict()}")
    print(f"    F1 Macro: {res['f1_macro']:.3f} | F1 Active: {res['f1_active']:.3f}")

    # 4. SMOTE k=3 (curent)
    print("\n  [4/5] SMOTE k_neighbors=3 (CURENT)")
    smote3 = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_sm3, y_sm3 = smote3.fit_resample(X_train, y_train)
    X_sm3 = pd.DataFrame(X_sm3, columns=FEATURES)
    res = run_strategy("SMOTE k=3 *", X_sm3, X_test, y_sm3, y_test)
    results.append(res)
    print(f"    Train dupa SMOTE: {pd.Series(y_sm3).value_counts().sort_index().to_dict()}")
    print(f"    F1 Macro: {res['f1_macro']:.3f} | F1 Active: {res['f1_active']:.3f}")

    # 5. SMOTE k=5
    print("\n  [5/5] SMOTE k_neighbors=5")
    smote5 = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_sm5, y_sm5 = smote5.fit_resample(X_train, y_train)
    X_sm5 = pd.DataFrame(X_sm5, columns=FEATURES)
    res = run_strategy("SMOTE k=5", X_sm5, X_test, y_sm5, y_test)
    results.append(res)
    print(f"    Train dupa SMOTE: {pd.Series(y_sm5).value_counts().sort_index().to_dict()}")
    print(f"    F1 Macro: {res['f1_macro']:.3f} | F1 Active: {res['f1_active']:.3f}")

    # ============================================================
    # REZULTATE
    # ============================================================
    df_results = pd.DataFrame(results)
    df_results.to_csv("outputs/tables/smote_comparison_table.csv", index=False)
    print("\n  Salvat: data/smote_comparison_table.csv")

    # Grafic
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [r["strategy"] for r in results]
    x = np.arange(len(names))

    # F1 Macro + Accuracy
    ax1 = axes[0]
    w = 0.35
    colors_f1 = ["#95a5a6" if "*" not in n else "#e74c3c" for n in names]
    colors_acc = ["#bdc3c7" if "*" not in n else "#3498db" for n in names]

    bars1 = ax1.bar(x - w/2, df_results["f1_macro"], w, color=colors_f1, label="F1 Macro")
    bars2 = ax1.bar(x + w/2, df_results["accuracy"], w, color=colors_acc, label="Accuracy", alpha=0.7)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")

    ax1.set_ylabel("Score")
    ax1.set_title("F1 Macro & Accuracy per Strategie", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax1.legend()
    ax1.set_ylim(0, 1.0)

    # F1 per clasă
    ax2 = axes[1]
    w = 0.25
    ax2.bar(x - w, df_results["f1_rest"], w, label="F1 Rest", color="#e74c3c", alpha=0.7)
    ax2.bar(x, df_results["f1_normal"], w, label="F1 Normal", color="#f39c12", alpha=0.7)
    ax2.bar(x + w, df_results["f1_active"], w, label="F1 Active", color="#27ae60", alpha=0.7)

    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 per Clasa", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax2.legend()
    ax2.set_ylim(0, 1.0)

    plt.suptitle(
        "Comparatie Strategii Class Imbalance\n"
        "Random Forest | 19 features | JSON | Temporal split 80/20",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/experiments/smote_comparison_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Salvat: data/smote_comparison_chart.png")

    # Tabel consolă
    print(f"\n{'='*80}")
    print("  TABEL REZUMAT — COMPARATIE CLASS IMBALANCE (RF, JSON, 19 feat)")
    print(f"{'='*80}")
    print(f"{'Strategie':<22} {'F1 Mac':>7} {'Acc':>7} {'F1_R':>6} {'F1_N':>6} "
          f"{'F1_A':>6} {'Train R/N/A':>14}")
    print("-" * 80)
    for r in results:
        tag = " <-" if "*" in r["strategy"] else ""
        print(f"{r['strategy']:<22} {r['f1_macro']:>7.3f} {r['accuracy']:>7.3f} "
              f"{r['f1_rest']:>6.3f} {r['f1_normal']:>6.3f} {r['f1_active']:>6.3f} "
              f"{r['train_rest']:>4}/{r['train_normal']:>3}/{r['train_active']:>3}{tag}")

    # Concluzie
    best = max(results, key=lambda r: r["f1_macro"])
    smote3_res = [r for r in results if "k=3" in r["strategy"]][0]

    print(f"\n  Best F1 Macro: {best['strategy']} ({best['f1_macro']:.3f})")
    print(f"  SMOTE k=3:     {smote3_res['f1_macro']:.3f}")

    if best["strategy"] == smote3_res["strategy"]:
        print(f"\n  SMOTE k=3 confirma cel mai bun compromis F1 Macro / F1 Active.")
    else:
        diff = smote3_res["f1_macro"] - best["f1_macro"]
        print(f"\n  SMOTE k=3 e la {abs(diff)*100:.1f}pp de best ({best['strategy']}).")

    print(f"\n{'='*65}")
    print(f"  GATA")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()