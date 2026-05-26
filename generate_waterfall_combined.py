"""
Generează un combined waterfall plot curat, folosind matplotlib pur.
Înlocuiește shap_waterfall_all_xgboost.png care avea text suprapus.

Rulare: python generate_waterfall_combined.py
Cerință: Rulează DUPĂ cap5_compare_algorithms.py (folosește modelul existent).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import joblib
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATE (identic cu cap5_compare_algorithms.py)
# ============================================================
def load_data():
    oura = pd.read_csv("data/oura_combined.csv")
    oura["date"] = pd.to_datetime(oura["date"])

    def load_aw(filename, agg="mean"):
        df = pd.read_csv(f"data/apple_health/{filename}")
        df["date"] = pd.to_datetime(df["startDate"]).dt.date
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.groupby("date")["value"].agg(agg).reset_index()

    hr = load_aw("heart_rate.csv", "mean").rename(columns={"value": "aw_hr_mean"})
    hrv = load_aw("hrv.csv", "mean").rename(columns={"value": "aw_hrv_mean"})
    steps = load_aw("steps.csv", "sum").rename(columns={"value": "aw_steps"})
    cals = load_aw("active_calories.csv", "sum").rename(columns={"value": "aw_active_calories"})
    spo2 = load_aw("spo2.csv", "mean").rename(columns={"value": "aw_spo2"})

    df = oura
    for d in [hr, hrv, steps, cals, spo2]:
        df = df.merge(d, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    LAG_FEATURES = [
        "readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
        "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
        "steps_oura", "respiratory_rate", "temp_deviation",
        "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2"
    ]
    FEATURES = [f"{f}_lag1" for f in LAG_FEATURES]

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

    df["nutrition_label"] = df.apply(assign_label, axis=1)
    df_model = df[["date"] + FEATURES + ["nutrition_label"]].dropna(subset=["nutrition_label"])
    df_model = df_model.sort_values("date").reset_index(drop=True)

    SHORT_NAMES = [f.replace("_lag1", "") for f in FEATURES]
    SHORT_NAMES = [n.replace("aw_", "AW:") for n in SHORT_NAMES]

    return df_model, FEATURES, SHORT_NAMES


# ============================================================
# 2. TRAIN XGBOOST + SHAP (rapid, doar ce trebuie)
# ============================================================
def get_shap_data(df_model, FEATURES):
    X = df_model[FEATURES]
    y = df_model["nutrition_label"].astype(int)

    split_idx = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURES)

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

    xgb = XGBClassifier(
        n_estimators=100, random_state=42, max_depth=6,
        learning_rate=0.1, eval_metric="mlogloss", verbosity=0
    )
    xgb.fit(X_train_sm, y_train_sm)

    explainer = shap.TreeExplainer(xgb)

    # Găsim câte o instanță reprezentativă per clasă
    class_names = {0: "Rest", 1: "Normal", 2: "Active"}
    example_indices = {}
    for cls_id in [0, 1, 2]:
        mask = y_test == cls_id
        if mask.any():
            example_indices[cls_id] = mask[mask].index[0] - split_idx

    waterfall_data = {}
    for cls_id, test_idx in example_indices.items():
        single = X_test_imp.iloc[[test_idx]]
        sv = explainer.shap_values(single)
        sv_array = np.array(sv)

        if sv_array.ndim == 3:
            if sv_array.shape[0] == 1:
                sv_cls = sv_array[0, :, cls_id]
            else:
                sv_cls = sv_array[cls_id, 0, :]
            base_val = explainer.expected_value[cls_id]
        elif isinstance(sv, list):
            sv_cls = sv[cls_id][0]
            base_val = explainer.expected_value[cls_id]
        else:
            sv_cls = sv[0]
            base_val = float(explainer.expected_value)

        waterfall_data[cls_id] = {
            "shap_values": sv_cls,
            "feature_values": single.values[0],
            "base_value": base_val,
            "f_x": base_val + sv_cls.sum(),
        }

    return waterfall_data, class_names


# ============================================================
# 3. CUSTOM WATERFALL PLOT (matplotlib pur)
# ============================================================
def draw_waterfall(ax, shap_vals, feat_vals, feat_names, base_value, f_x,
                   title, max_display=10, color_pos="#FF0051", color_neg="#008BFB"):
    """
    Desenează un waterfall plot curat pe un ax matplotlib.
    """
    n_features = len(shap_vals)

    # Sortăm după |SHAP value| descrescător
    order = np.argsort(np.abs(shap_vals))[::-1]

    # Top N features + "other"
    if n_features > max_display:
        top_idx = order[:max_display - 1]
        other_val = sum(shap_vals[i] for i in order[max_display - 1:])
        other_count = n_features - max_display + 1

        display_shap = [shap_vals[i] for i in top_idx] + [other_val]
        display_labels = [
            f"{feat_vals[i]:.4g} = {feat_names[i]}" for i in top_idx
        ] + [f"{other_count} other features"]
    else:
        top_idx = order
        display_shap = [shap_vals[i] for i in top_idx]
        display_labels = [f"{feat_vals[i]:.4g} = {feat_names[i]}" for i in top_idx]

    n_bars = len(display_shap)
    y_pos = np.arange(n_bars)[::-1]  # de sus în jos

    # Bare orizontale
    colors = [color_pos if v > 0 else color_neg for v in display_shap]

    bars = ax.barh(y_pos, display_shap, height=0.6, color=colors, edgecolor="none", alpha=0.85)

    # Labels pe bare
    for i, (bar, val) in enumerate(zip(bars, display_shap)):
        sign = "+" if val > 0 else ""
        x_text = val + (0.02 * max(abs(v) for v in display_shap)) * (1 if val > 0 else -1)
        ax.text(
            x_text, y_pos[i], f"{sign}{val:.2f}",
            va="center",
            ha="left" if val > 0 else "right",
            fontsize=9, fontweight="bold",
            color=colors[i],
        )

    # Y labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels, fontsize=10)

    # Titlu și annotations
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axvline(0, color="black", linewidth=0.5)

    # f(x) și E[f(X)]
    ax.text(
        0.98, 0.97, f"f(x) = {f_x:.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color="#555555",
    )
    ax.text(
        0.98, 0.03, f"E[f(X)] = {base_value:.3f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#888888",
    )

    ax.set_xlabel("SHAP value", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    print("Loading data and training XGBoost...")
    df_model, FEATURES, SHORT_NAMES = load_data()
    waterfall_data, class_names = get_shap_data(df_model, FEATURES)
    print("SHAP values computed.")

    # Generare plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    class_colors = {
        0: ("#008BFB", "#FF0051"),   # Rest: blue dominant
        1: ("#FF0051", "#008BFB"),   # Normal: red dominant
        2: ("#008BFB", "#FF0051"),   # Active: blue dominant
    }

    for i, cls_id in enumerate([0, 1, 2]):
        data = waterfall_data[cls_id]
        draw_waterfall(
            ax=axes[i],
            shap_vals=data["shap_values"],
            feat_vals=data["feature_values"],
            feat_names=SHORT_NAMES,
            base_value=data["base_value"],
            f_x=data["f_x"],
            title=f"Zi {class_names[cls_id]}",
            max_display=12,
            color_pos="#FF0051",
            color_neg="#008BFB",
        )

    plt.suptitle(
        "SHAP Waterfall – De ce modelul clasifică fiecare zi (XGBoost)",
        fontsize=16, fontweight="bold", y=0.995,
    )
    plt.tight_layout(h_pad=3.0)
    plt.subplots_adjust(top=0.96)

    output = "outputs/figures/xml_pipeline/shap_waterfall_all_xgboost.png"
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Salvat: {output}")


if __name__ == "__main__":
    main()
