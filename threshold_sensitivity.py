"""
ANALIZA DE SENSIBILITATE — Praguri de clasificare Readiness/Sleep
=================================================================
Testează robustețea rezultatelor ML la variații ale pragurilor
de clasificare (Rest/Normal/Active).

IMPORTANT: Folosește aceeași configurație ca pipeline-ul principal
(cap5_compare_algorithms_json_ext.py):
  - Random Forest (n_estimators=100, max_depth=10, min_samples_leaf=5)
  - Sursa JSON (Health Auto Export)
  - 19 features (inclusiv sleep_deep, sleep_rem, resp_rate)
  - SMOTE k_neighbors=3
  - Split temporal 80/20

Praguri testate:
  T1-T6: seturi manuale (inclusiv T3 = curent)
  T7: bazat pe quantile (Q25/Q75 din date)

Output:
  - data/sensitivity_threshold_table.csv
  - data/sensitivity_threshold_heatmap.png
  - data/sensitivity_threshold_distributions.png
  - data/sensitivity_f1_macro_comparison.png
  - Tabel console pentru copiere directă în Cap.5

Autor: Adrian Lazar | Disertatie UPB 2026
Rulare: python threshold_sensitivity.py
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

TRAIN_RATIO = 0.8
RANDOM_STATE = 42
LABEL_NAMES = {0: "Rest", 1: "Normal", 2: "Active"}


# ============================================================
# DEFINIRE SETURI DE PRAGURI
# ============================================================
THRESHOLD_SETS = [
    ("T1: 55/55-80/70",    55, 55, 80, 70),
    ("T2: 55/55-85/75",    55, 55, 85, 75),
    ("T3: 60/60-85/75 *",  60, 60, 85, 75),  # * = curent
    ("T4: 60/60-90/80",    60, 60, 90, 80),
    ("T5: 65/65-85/75",    65, 65, 85, 75),
    ("T6: 65/65-90/80",    65, 65, 90, 80),
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

    print(f"  Total zile dupa merge: {len(df)}")
    return df


# ============================================================
# ASSIGN LABELS CU PRAGURI PARAMETRIZATE
# ============================================================
def assign_labels(df, r_rest, s_rest, r_active, s_active):
    def _label(row):
        if pd.isna(row["readiness_score"]) or pd.isna(row["sleep_score"]):
            return np.nan
        if row["readiness_score"] < r_rest or row["sleep_score"] < s_rest:
            return 0  # Rest
        elif row["readiness_score"] >= r_active and row["sleep_score"] >= s_active:
            return 2  # Active
        return 1  # Normal

    return df.apply(_label, axis=1)


# ============================================================
# COMPUTE QUANTILE THRESHOLDS
# ============================================================
def compute_quantile_thresholds(df, q_low=0.25, q_high=0.75):
    r = df["readiness_score"].dropna()
    s = df["sleep_score"].dropna()
    return (
        int(np.round(r.quantile(q_low))),
        int(np.round(s.quantile(q_low))),
        int(np.round(r.quantile(q_high))),
        int(np.round(s.quantile(q_high))),
    )


# ============================================================
# PIPELINE ML — RF (identic cu pipeline principal)
# ============================================================
def run_pipeline(df, r_rest, s_rest, r_active, s_active, lag_features):
    features = [f"{f}_lag1" for f in lag_features]

    # Lag features
    df_work = df.copy()
    for feat in lag_features:
        df_work[f"{feat}_lag1"] = df_work[feat].shift(1)

    # Labels
    df_work["label"] = assign_labels(df_work, r_rest, s_rest, r_active, s_active)
    df_model = df_work[features + ["label"]].dropna(subset=["label"])

    X = df_model[features]
    y = df_model["label"].astype(int)

    # Verificare: minimum 2 clase cu >0 instanțe
    class_counts = y.value_counts().sort_index()
    n_classes = (class_counts > 0).sum()
    if n_classes < 2:
        return None

    # Temporal split
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Imputer — fit pe train (pre-SMOTE)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=features, index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=features, index=X_test.index
    )

    # SMOTE — doar dacă clasa minimă are >= 2 instanțe
    train_counts = y_train.value_counts()
    min_class_count = train_counts.min()

    if min_class_count >= 4:
        k = min(3, min_class_count - 1)
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)
        X_train_sm = pd.DataFrame(X_train_sm, columns=features)
    elif min_class_count >= 2:
        k = min_class_count - 1
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)
        X_train_sm = pd.DataFrame(X_train_sm, columns=features)
    else:
        X_train_sm = X_train_imp
        y_train_sm = y_train

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

    # F1 per clasă
    labels_present = sorted(y.unique())
    f1_per_class = {}
    for lbl in [0, 1, 2]:
        if lbl in labels_present and lbl in y_test.unique():
            f1_per_class[lbl] = f1_score(
                y_test, y_pred, labels=[lbl], average="macro", zero_division=0
            )
        else:
            f1_per_class[lbl] = np.nan

    # Distribuție
    dist = y.value_counts().sort_index().to_dict()

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_rest": f1_per_class.get(0, np.nan),
        "f1_normal": f1_per_class.get(1, np.nan),
        "f1_active": f1_per_class.get(2, np.nan),
        "n_total": len(y),
        "n_rest": dist.get(0, 0),
        "n_normal": dist.get(1, 0),
        "n_active": dist.get(2, 0),
        "n_test": len(y_test),
    }


# ============================================================
# VIZUALIZĂRI
# ============================================================
def plot_heatmap(results_df):
    fig, ax = plt.subplots(figsize=(12, 5))

    metrics = ["f1_macro", "f1_rest", "f1_normal", "f1_active", "accuracy"]
    labels = ["F1 Macro", "F1 Rest", "F1 Normal", "F1 Active", "Accuracy"]

    data = results_df[metrics].values.T
    names = results_df["threshold_set"].values

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(labels)):
        for j in range(len(names)):
            val = data[i, j]
            text = "N/A" if np.isnan(val) else f"{val:.3f}"
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(
        "Analiza de Sensibilitate — Praguri de Clasificare\n"
        "Random Forest | 19 features | JSON | Temporal split 80/20",
        fontsize=12, fontweight="bold"
    )

    current_idx = None
    for idx, name in enumerate(names):
        if "*" in name:
            current_idx = idx
            break
    if current_idx is not None:
        ax.axvline(x=current_idx - 0.5, color="blue", linewidth=2, linestyle="--", alpha=0.7)
        ax.axvline(x=current_idx + 0.5, color="blue", linewidth=2, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("outputs/figures/experiments/sensitivity_threshold_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Salvat: data/sensitivity_threshold_heatmap.png")


def plot_distributions(results_df):
    fig, ax = plt.subplots(figsize=(12, 5))

    names = results_df["threshold_set"].values
    x = np.arange(len(names))
    width = 0.25

    ax.bar(x - width, results_df["n_rest"], width, label="Rest", color="#e74c3c", alpha=0.8)
    ax.bar(x, results_df["n_normal"], width, label="Normal", color="#f39c12", alpha=0.8)
    ax.bar(x + width, results_df["n_active"], width, label="Active", color="#27ae60", alpha=0.8)

    ax.set_xlabel("Set de praguri")
    ax.set_ylabel("Numar zile")
    ax.set_title(
        "Distributia claselor per set de praguri\n"
        "Impact al pragurilor asupra imbalance-ului de clase",
        fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.legend()

    for i, row in results_df.iterrows():
        total = row["n_rest"] + row["n_normal"] + row["n_active"]
        ax.text(i, max(row["n_rest"], row["n_normal"], row["n_active"]) + 15,
                f"N={int(total)}", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig("outputs/figures/experiments/sensitivity_threshold_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Salvat: data/sensitivity_threshold_distributions.png")


def plot_f1_macro_comparison(results_df):
    fig, ax = plt.subplots(figsize=(10, 5))

    names = results_df["threshold_set"].values
    f1_values = results_df["f1_macro"].values

    colors = ["#3498db"] * len(names)
    for i, name in enumerate(names):
        if "*" in name:
            colors[i] = "#e74c3c"

    bars = ax.bar(range(len(names)), f1_values, color=colors, alpha=0.85, edgecolor="gray")

    for bar, val in zip(bars, f1_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Macro", fontsize=11)
    ax.set_title(
        "Robustetea F1 Macro la variatii ale pragurilor de clasificare\n"
        "Random Forest | 19 features | JSON | Temporal split 80/20 | SMOTE k=3",
        fontsize=12, fontweight="bold"
    )

    mean_f1 = np.nanmean(f1_values)
    ax.axhline(y=mean_f1, color="gray", linestyle="--", alpha=0.5)
    ax.text(len(names) - 0.5, mean_f1 + 0.005, f"Media: {mean_f1:.3f}",
            ha="right", fontsize=9, color="gray")

    ax.set_ylim(0, max(f1_values) * 1.15)
    ax.legend(["Prag curent (T3)", "Alte praguri"], loc="upper left",
              handles=[
                  plt.Rectangle((0, 0), 1, 1, color="#e74c3c", alpha=0.85),
                  plt.Rectangle((0, 0), 1, 1, color="#3498db", alpha=0.85),
              ])

    plt.tight_layout()
    plt.savefig("outputs/figures/experiments/sensitivity_f1_macro_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Salvat: data/sensitivity_f1_macro_comparison.png")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  ANALIZA DE SENSIBILITATE — PRAGURI DE CLASIFICARE")
    print("  Random Forest | JSON | 19 features | Temporal 80/20")
    print("=" * 60)

    # Load date
    print("\n=== LOAD DATE ===")
    df = load_data()

    # Calculare praguri quantile
    r_q, s_q, ra_q, sa_q = compute_quantile_thresholds(df)
    print(f"\n=== PRAGURI QUANTILE (Q25/Q75) ===")
    print(f"  Rest: readiness < {r_q}, sleep < {s_q}")
    print(f"  Active: readiness >= {ra_q}, sleep >= {sa_q}")

    # Adaugă quantile la lista de praguri
    all_thresholds = THRESHOLD_SETS + [
        (f"T7: Q25/Q75 ({r_q}/{s_q}-{ra_q}/{sa_q})", r_q, s_q, ra_q, sa_q),
    ]

    # Rulare pipeline per set de praguri
    print(f"\n=== TESTARE {len(all_thresholds)} SETURI DE PRAGURI ===\n")

    results = []
    for name, r_rest, s_rest, r_active, s_active in all_thresholds:
        print(f"  [{name}]")
        print(f"    Rest: readiness<{r_rest} OR sleep<{s_rest}")
        print(f"    Active: readiness>={r_active} AND sleep>={s_active}")

        res = run_pipeline(df, r_rest, s_rest, r_active, s_active, LAG_FEATURES)

        if res is None:
            print(f"    ! SKIP — prea putine clase")
            continue

        res["threshold_set"] = name
        res["r_rest"] = r_rest
        res["s_rest"] = s_rest
        res["r_active"] = r_active
        res["s_active"] = s_active
        results.append(res)

        print(f"    Distributie: Rest={res['n_rest']}, Normal={res['n_normal']}, Active={res['n_active']}")
        print(f"    F1 Macro={res['f1_macro']:.3f}, Acc={res['accuracy']:.3f}")
        f1_a_str = f"{res['f1_active']:.3f}" if not np.isnan(res['f1_active']) else "N/A"
        print(f"    F1: Rest={res['f1_rest']:.3f}, Normal={res['f1_normal']:.3f}, Active={f1_a_str}")
        print()

    if not results:
        print("EROARE: Niciun set de praguri nu a produs rezultate valide.")
        return

    results_df = pd.DataFrame(results)

    col_order = [
        "threshold_set", "r_rest", "s_rest", "r_active", "s_active",
        "n_total", "n_rest", "n_normal", "n_active",
        "f1_macro", "accuracy", "f1_rest", "f1_normal", "f1_active", "n_test"
    ]
    results_df = results_df[col_order]

    # Salvare CSV
    results_df.to_csv("outputs/tables/sensitivity_threshold_table.csv", index=False)
    print("  Salvat: data/sensitivity_threshold_table.csv")

    # Vizualizări
    print("\n=== GENERARE GRAFICE ===")
    plot_heatmap(results_df)
    plot_distributions(results_df)
    plot_f1_macro_comparison(results_df)

    # Tabel final
    print("\n" + "=" * 90)
    print("  TABEL REZUMAT — ANALIZA DE SENSIBILITATE (RF, JSON, 19 features)")
    print("=" * 90)
    print(f"{'Prag':<28} {'N':>5} {'Rest':>5} {'Norm':>5} {'Act':>4} "
          f"{'F1 Mac':>7} {'Acc':>7} {'F1_R':>6} {'F1_N':>6} {'F1_A':>6}")
    print("-" * 90)
    for _, row in results_df.iterrows():
        f1_a = f"{row['f1_active']:.3f}" if not np.isnan(row['f1_active']) else "  N/A"
        print(f"{row['threshold_set']:<28} {int(row['n_total']):>5} "
              f"{int(row['n_rest']):>5} {int(row['n_normal']):>5} {int(row['n_active']):>4} "
              f"{row['f1_macro']:>7.3f} {row['accuracy']:>7.3f} "
              f"{row['f1_rest']:>6.3f} {row['f1_normal']:>6.3f} {f1_a:>6}")

    # Analiză stabilitate
    f1_values = results_df["f1_macro"].values
    f1_mean = np.nanmean(f1_values)
    f1_std = np.nanstd(f1_values)
    f1_range = np.nanmax(f1_values) - np.nanmin(f1_values)

    print(f"\n{'STABILITATE':=^90}")
    print(f"  F1 Macro: media = {f1_mean:.3f}, std = {f1_std:.3f}, "
          f"range = {f1_range:.3f} ({f1_range*100:.1f} pp)")

    print(f"\n{'='*60}")
    print(f"  GATA — Grafice salvate in data/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()