"""
generate_waterfall_combined_json.py — Waterfall SHAP curat (JSON source)
=========================================================================
Generează waterfall plots lizibile cu matplotlib pur (fără shap.plots).
3 subplots verticale: Rest / Normal / Active, pe XGBoost (JSON source).

Rulare: python generate_waterfall_combined_json.py

Autor: Adrian Lazar | Disertație UPB 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings("ignore")


def load_and_train():
    """Încarcă date JSON, antrenează XGBoost, returnează SHAP."""
    print("Loading data...")
    oura = pd.read_csv("data/oura_combined.csv")
    oura["date"] = pd.to_datetime(oura["date"])

    aw = pd.read_csv("apple_watch_api/apple_health_daily.csv")
    aw["date"] = pd.to_datetime(aw["date"])
    aw["aw_active_calories"] = (aw["aw_active_energy_kj"] / 4.184).round(1)

    df = oura
    for col, src_col in [
        ("aw_hr_mean", "aw_hr_mean"),
        ("aw_hrv_mean", "aw_hrv_mean"),
        ("aw_steps", "aw_steps"),
        ("aw_active_calories", "aw_active_calories"),
        ("aw_spo2", "aw_spo2"),
    ]:
        merge_df = aw[["date", src_col]].dropna(subset=[src_col])
        df = df.merge(merge_df, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)

    LAG_FEATURES = [
        "readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
        "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
        "steps_oura", "respiratory_rate", "temp_deviation",
        "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2"
    ]
    FEATURES = [f"{f}_lag1" for f in LAG_FEATURES]
    SHORT_NAMES = [f.replace("_lag1", "").replace("aw_", "AW:") for f in FEATURES]

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
    df_model = df[["date"] + FEATURES + ["label"]].dropna(subset=["label"])
    df_model = df_model.sort_values("date").reset_index(drop=True)

    X = df_model[FEATURES]
    y = df_model["label"].astype(int)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURES)

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=100, random_state=42, max_depth=6,
        learning_rate=0.1, eval_metric="mlogloss", verbosity=0
    )
    model.fit(X_train_sm, y_train_sm)

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_imp)

    return model, explainer, shap_values, X_test_imp, y_test, FEATURES, SHORT_NAMES


def draw_waterfall(ax, shap_vals, feature_vals, feature_names, base_value,
                   title, max_display=12, color_pos="#FF0051", color_neg="#008BFB"):
    """Desenează un waterfall plot cu matplotlib pur."""
    n_features = len(shap_vals)
    abs_vals = np.abs(shap_vals)
    sorted_idx = np.argsort(abs_vals)[::-1]

    if n_features > max_display:
        top_idx = sorted_idx[:max_display - 1]
        other_val = np.sum(shap_vals[sorted_idx[max_display - 1:]])
        display_vals = list(shap_vals[top_idx]) + [other_val]
        display_feat_vals = list(feature_vals[top_idx]) + [None]
        display_names = [feature_names[i] for i in top_idx] + [f"{n_features - max_display + 1} other features"]
    else:
        top_idx = sorted_idx
        display_vals = list(shap_vals[top_idx])
        display_feat_vals = list(feature_vals[top_idx])
        display_names = [feature_names[i] for i in top_idx]

    display_vals.reverse()
    display_feat_vals.reverse()
    display_names.reverse()

    n = len(display_vals)
    y_pos = np.arange(n)

    cumulative = base_value
    lefts = []
    for v in reversed(display_vals):
        cumulative += v
    running = base_value
    starts = []
    for v in display_vals:
        starts.append(running)
        running += v

    colors = [color_pos if v > 0 else color_neg for v in display_vals]

    for i in range(n):
        ax.barh(y_pos[i], display_vals[i], left=starts[i], height=0.6,
                color=colors[i], edgecolor="none", alpha=0.85)
        val_text = f"+{display_vals[i]:.2f}" if display_vals[i] > 0 else f"{display_vals[i]:.2f}"
        text_x = starts[i] + display_vals[i]
        ha = "left" if display_vals[i] > 0 else "right"
        offset = 0.02 * (max(abs(v) for v in display_vals))
        ax.text(text_x + (offset if display_vals[i] > 0 else -offset),
                y_pos[i], val_text, va="center", ha=ha, fontsize=8, fontweight="bold",
                color=colors[i])

    labels = []
    for j in range(n):
        if display_feat_vals[j] is not None:
            fv = display_feat_vals[j]
            fv_str = f"{fv:.1f}" if abs(fv) < 100 else f"{fv:.0f}"
            labels.append(f"{fv_str} = {display_names[j]}")
        else:
            labels.append(display_names[j])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axvline(x=base_value, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(f"E[f(X)] = {base_value:.3f}", fontsize=9, color="gray")

    f_x = base_value + sum(display_vals)
    ax.text(0.98, 0.97, f"f(x) = {f_x:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, fontstyle="italic", color="gray")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    model, explainer, shap_values, X_test_imp, y_test, FEATURES, SHORT_NAMES = load_and_train()

    sv_array = np.array(shap_values)
    class_names = {0: "Rest", 1: "Normal", 2: "Active"}
    class_colors = {
        0: ("#FF0051", "#008BFB"),
        1: ("#FF0051", "#008BFB"),
        2: ("#FF0051", "#008BFB"),
    }

    examples = {}
    for cls_id, cls_name in class_names.items():
        mask = y_test.values == cls_id
        if mask.sum() > 0:
            examples[cls_id] = np.where(mask)[0][0]
            print(f"  {cls_name}: test[{examples[cls_id]}]")

    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 7 * len(examples)))
    if len(examples) == 1:
        axes = [axes]

    for i, (cls_id, test_idx) in enumerate(examples.items()):
        cls_name = class_names[cls_id]

        if sv_array.ndim == 3:
            if sv_array.shape[0] == X_test_imp.shape[0]:
                sv_for_class = sv_array[test_idx, :, cls_id]
            else:
                sv_for_class = sv_array[cls_id, test_idx, :]
        elif isinstance(shap_values, list):
            sv_for_class = shap_values[cls_id][test_idx]
        else:
            sv_for_class = shap_values[test_idx]

        base_val = explainer.expected_value
        if hasattr(base_val, "__len__"):
            base_val = base_val[cls_id]

        feature_vals = X_test_imp.iloc[test_idx].values

        draw_waterfall(
            axes[i], sv_for_class, feature_vals, SHORT_NAMES, base_val,
            title=f"Zi {cls_name} — de ce modelul clasifică test[{test_idx}] ca {cls_name}",
            max_display=12,
        )

    plt.suptitle(
        "SHAP Waterfall — XGBoost pe sursa JSON (Health Auto Export)",
        fontsize=15, fontweight="bold", y=1.005
    )
    plt.tight_layout(h_pad=3.0)

    out_path = "outputs/figures/json_pipeline/json_shap_waterfall_combined_clean.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Salvat: {out_path}")


if __name__ == "__main__":
    main()
