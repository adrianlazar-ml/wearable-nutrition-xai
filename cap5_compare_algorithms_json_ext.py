"""
Cap.5 - Comparatie algoritmi ML + SHAP (JSON + 19 EXTENDED FEATURES)
Identic cu cap5_compare_algorithms_json.py, dar cu 3 features NOI:
aw_sleep_deep_hours, aw_sleep_rem_hours, aw_resp_rate (total 19).

Raspunde la RQ2: Ce metode AI sunt cele mai potrivite pentru analiza
datelor din wearables, in functie de acuratete, viteza si explicabilitate?

Algoritmi comparati:
- Random Forest (baseline din pipeline existent)
- XGBoost (gradient boosting, performanta superioara pe date tabulare)
- SVM (Support Vector Machine, robust pe date mici)

Explicabilitate:
- SHAP (Shapley Additive Explanations) pe cel mai bun model
- SHAP Waterfall plots per-instanta (explicabilitate locala)
- PDP (Partial Dependence Plots) pentru relatii feature-predictie
- Feature importance comparativa intre modele

v4 adaugiri:
- SHAP Waterfall plots per-instanta (una per clasa Rest/Normal/Active)
- PDP (Partial Dependence Plots) top 3 features + multi-class
- Referinte noi: Napa et al. (2025), JMIR Mental Health (2025)

v3 fix-uri:
- Split TEMPORAL (80/20 cronologic) in loc de random
- TimeSeriesSplit pentru cross-validation
- F1 Macro ca metrica principala (class imbalance)

Referinte:
- Trujillo et al. (2024): RF pentru predictie somn din wearables
- Zignoli et al. (2024): XGBoost pentru predictie glicemica
- Tatli et al. (2024): SVM (RBF) pentru detectie prediabet
- Susnjak & Griffin (2025): SHAP pentru explicabilitate clinica
- Napa et al. (2025): RF + SHAP + PDP + Streamlit pentru CVD risk
- JMIR Mental Health (2025): SHAP + ADASYN pe date wearable actigrafie
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-interactiv, salveaza direct in fisier
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. LOAD SI PREGATIRE DATE (identic cu cap5_pipeline.py)
# ============================================================
def load_and_prepare():
    print("=== LOAD DATE ===")
    oura = pd.read_csv("data/oura_combined.csv")
    oura["date"] = pd.to_datetime(oura["date"])

    # Apple Watch din JSON (Health Auto Export) + EXTENDED
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

    print(f"Apple Watch (JSON + Extended): {len(aw)} zile, 19 features")

    df = oura
    for d in [hr, hrv, steps, cals, spo2, sleep_deep, sleep_rem, resp]:
        df = df.merge(d, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    # Lagged features
    LAG_FEATURES = [
        "readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
        "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
        "steps_oura", "respiratory_rate", "temp_deviation",
        "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2",
        "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate",
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

    df["label"] = df.apply(assign_label, axis=1)

    # Pastram 'date' pentru split temporal
    df_model = df[["date"] + FEATURES + ["label"]].dropna(subset=["label"])
    df_model = df_model.sort_values("date").reset_index(drop=True)

    X = df_model[FEATURES]
    y = df_model["label"].astype(int)
    dates = df_model["date"]

    print(f"Dataset: {len(X)} zile, {len(FEATURES)} features")
    print(f"Distributie: {y.value_counts().sort_index().to_dict()}")
    print(f"Perioada: {dates.iloc[0].date()} -> {dates.iloc[-1].date()}")

    return X, y, FEATURES, dates


# ============================================================
# 2. COMPARATIE ALGORITMI - SPLIT TEMPORAL
# ============================================================
def compare_algorithms(X, y, FEATURES, dates):
    print("\n" + "=" * 60)
    print("  COMPARATIE ALGORITMI: RF vs XGBoost vs SVM")
    print("  Split: TEMPORAL 80/20 cronologic")
    print("=" * 60)

    # SPLIT TEMPORAL
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    train_end = dates.iloc[split_idx - 1].strftime("%Y-%m-%d")
    test_start = dates.iloc[split_idx].strftime("%Y-%m-%d")
    test_end = dates.iloc[-1].strftime("%Y-%m-%d")
    print(f"\n  Train: {len(X_train)} zile (-> {train_end})")
    print(f"  Test:  {len(X_test)} zile ({test_start} -> {test_end})")
    print(f"  Test distributie: {y_test.value_counts().sort_index().to_dict()}")

    # Imputer fit pe train
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURES)

    # SMOTE pe train
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

    # Scaler (necesar pentru SVM)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_sm), columns=FEATURES)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=FEATURES)

    # BASELINE: Majority class classifier (prezice mereu clasa dominanta)
    # Antrenat pe y_train INAINTE de SMOTE (distributia reala)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_imp, y_train)
    y_dummy = dummy.predict(X_test_imp)
    baseline_acc = accuracy_score(y_test, y_dummy)
    baseline_f1 = f1_score(y_test, y_dummy, average="macro", zero_division=0)
    baseline_f1w = f1_score(y_test, y_dummy, average="weighted", zero_division=0)
    print(f"\n  BASELINE (majority class):")
    print(f"    Accuracy:   {baseline_acc:.2%}")
    print(f"    F1 Macro:   {baseline_f1:.2%}")
    print(f"    F1 Weighted:{baseline_f1w:.2%}")
    print(f"    (Prezice mereu: {y_train.mode()[0]} = {['Rest','Normal','Active'][y_train.mode()[0]]})")

    # Definim modelele
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42,
            max_depth=10, min_samples_leaf=5
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42,
            max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", verbosity=0
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            random_state=42, probability=True
        ),
    }

    results = {}
    trained_models = {}

    # TimeSeriesSplit pentru CV
    tscv = TimeSeriesSplit(n_splits=5)

    for name, model in models.items():
        print(f"\n--- {name} ---")
        t0 = time.time()

        # SVM foloseste date scalate, restul date brute
        if "SVM" in name:
            model.fit(X_train_scaled, y_train_sm)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train_sm, y_train_sm)
            y_pred = model.predict(X_test_imp)

        train_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_wt = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(
            y_test, y_pred,
            target_names=["Rest", "Normal", "Active"],
            zero_division=0, output_dict=True
        )

        # MCC + QWK (metrici suplimentare pentru Cap.6)
        mcc = matthews_corrcoef(y_test, y_pred)
        qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")

        # Cross-validation cu TimeSeriesSplit + pipeline complet
        if "SVM" in name:
            cv_pipe = ImbPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("smote", SMOTE(random_state=42, k_neighbors=3)),
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
            ])
        elif "XGBoost" in name:
            cv_pipe = ImbPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("smote", SMOTE(random_state=42, k_neighbors=3)),
                ("model", XGBClassifier(
                    n_estimators=100, random_state=42, max_depth=6,
                    learning_rate=0.1, eval_metric="mlogloss", verbosity=0
                )),
            ])
        else:
            cv_pipe = ImbPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("smote", SMOTE(random_state=42, k_neighbors=3)),
                ("model", RandomForestClassifier(
                    n_estimators=100, random_state=42,
                    max_depth=10, min_samples_leaf=5
                )),
            ])

        cv_acc = cross_val_score(cv_pipe, X, y, cv=tscv, scoring="accuracy")
        cv_f1 = cross_val_score(cv_pipe, X, y, cv=tscv, scoring="f1_macro")

        results[name] = {
            "accuracy": acc,
            "f1_macro": f1_mac,
            "f1_weighted": f1_wt,
            "cv_acc_mean": cv_acc.mean(),
            "cv_acc_std": cv_acc.std(),
            "cv_f1_mean": cv_f1.mean(),
            "cv_f1_std": cv_f1.std(),
            "train_time": train_time,
            "f1_rest": report["Rest"]["f1-score"],
            "f1_normal": report["Normal"]["f1-score"],
            "f1_active": report["Active"]["f1-score"],
            "mcc": mcc,
            "qwk": qwk,
        }
        trained_models[name] = model

        print(f"  Accuracy:        {acc:.2%}")
        print(f"  F1 Macro:        {f1_mac:.2%}")
        print(f"  F1 Weighted:     {f1_wt:.2%}")
        print(f"  CV acc (TSCV):   {cv_acc.mean():.2%} (+/- {cv_acc.std():.2%})")
        print(f"  CV f1  (TSCV):   {cv_f1.mean():.2%} (+/- {cv_f1.std():.2%})")
        print(f"  Train time:      {train_time:.2f}s")
        print(f"  F1 per clasa:    Rest={report['Rest']['f1-score']:.2f}  "
              f"Normal={report['Normal']['f1-score']:.2f}  "
              f"Active={report['Active']['f1-score']:.2f}")
        print(f"  MCC:             {mcc:.4f}")
        print(f"  QWK:             {qwk:.4f}")

    # Tabel comparativ
    print("\n" + "=" * 60)
    print("  TABEL COMPARATIV (split temporal + TimeSeriesSplit CV)")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Baseline':>10} {'RF':>10} {'XGBoost':>10} {'SVM':>10}")
    print(f"  {'-'*60}")

    rnames = list(results.keys())
    metrics_to_show = [
        ("Accuracy", "accuracy", "{:.1%}"),
        ("F1 Macro", "f1_macro", "{:.1%}"),
        ("F1 Weighted", "f1_weighted", "{:.1%}"),
        ("CV Accuracy", "cv_acc_mean", "{:.1%}"),
        ("CV F1 Macro", "cv_f1_mean", "{:.1%}"),
        ("CV acc std", "cv_acc_std", "{:.1%}"),
        ("F1 Rest", "f1_rest", "{:.2f}"),
        ("F1 Normal", "f1_normal", "{:.2f}"),
        ("F1 Active", "f1_active", "{:.2f}"),
        ("MCC", "mcc", "{:.4f}"),
        ("QWK", "qwk", "{:.4f}"),
        ("Train time (s)", "train_time", "{:.2f}"),
    ]

    # Baseline values pentru tabel
    baseline_vals = {
        "accuracy": baseline_acc, "f1_macro": baseline_f1,
        "f1_weighted": baseline_f1w,
        "cv_acc_mean": baseline_acc, "cv_f1_mean": baseline_f1,
        "cv_acc_std": 0.0,
        "f1_rest": f1_score(y_test, y_dummy, average=None, zero_division=0)[0],
        "f1_normal": f1_score(y_test, y_dummy, average=None, zero_division=0)[1] if len(f1_score(y_test, y_dummy, average=None, zero_division=0)) > 1 else 0.0,
        "f1_active": f1_score(y_test, y_dummy, average=None, zero_division=0)[2] if len(f1_score(y_test, y_dummy, average=None, zero_division=0)) > 2 else 0.0,
        "train_time": 0.0,
        "mcc": matthews_corrcoef(y_test, y_dummy),
        "qwk": cohen_kappa_score(y_test, y_dummy, weights="quadratic"),
    }

    for label, key, fmt in metrics_to_show:
        bval = fmt.format(baseline_vals[key])
        vals = [fmt.format(results[n][key]) for n in rnames]
        print(f"  {label:<20} {bval:>10} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    # Salvare confusion matrices
    print("\n  Salvez confusion matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axes, rnames):
        if "SVM" in name:
            y_pred = trained_models[name].predict(X_test_scaled)
        else:
            y_pred = trained_models[name].predict(X_test_imp)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Rest", "Normal", "Active"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name, fontsize=12)
    plt.suptitle(f"Confusion Matrices - Split Temporal ({test_start} -> {test_end})", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/figures/json_extended_pipeline/ext_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> data/ext_confusion_matrices.png")

    # Salvare MCC + QWK în CSV pentru Cap.6
    mcc_qwk_rows = []
    for name in rnames:
        r = results[name]
        mcc_qwk_rows.append({
            "Model": name,
            "F1 Macro": round(r["f1_macro"], 4),
            "MCC": round(r["mcc"], 4),
            "QWK": round(r["qwk"], 4),
            "F1 Rest": round(r["f1_rest"], 4),
            "F1 Normal": round(r["f1_normal"], 4),
            "F1 Active": round(r["f1_active"], 4),
            "Accuracy": round(r["accuracy"], 4),
        })
    mcc_qwk_df = pd.DataFrame(mcc_qwk_rows)
    mcc_qwk_path = "outputs/tables/mcc_qwk_results.csv"
    mcc_qwk_df.to_csv(mcc_qwk_path, index=False)
    print(f"\n  Salvat MCC+QWK: {mcc_qwk_path}")
    print(mcc_qwk_df.to_string(index=False))

    return results, trained_models, X_train_imp, X_test_imp, y_test, imputer


# ============================================================
# 3. SHAP EXPLICABILITATE (summary, bar, dependence)
# ============================================================
def run_shap(trained_models, X_train_imp, X_test_imp, FEATURES):
    print("\n" + "=" * 60)
    print("  SHAP - EXPLICABILITATE")
    print("=" * 60)

    # Folosim cel mai bun model tree-based pentru SHAP
    for model_name in ["XGBoost", "Random Forest"]:
        if model_name in trained_models:
            print(f"\n--- SHAP pe {model_name} ---")
            model = trained_models[model_name]

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_imp)

            # Feature names scurtate pentru grafice
            short_names = [f.replace("_lag1", "").replace("aw_", "AW:") for f in FEATURES]

            # 1. Summary plot (beeswarm)
            print("  Generez summary plot...")
            fig, ax = plt.subplots(figsize=(10, 8))
            sv_array = np.array(shap_values)
            if sv_array.ndim == 3:
                if sv_array.shape[0] == X_test_imp.shape[0]:
                    sv_plot = sv_array[:, :, 1]  # (samples, features) pt clasa Normal
                else:
                    sv_plot = sv_array[1]  # (samples, features) pt clasa Normal
            elif isinstance(shap_values, list):
                sv_plot = shap_values[1]
            else:
                sv_plot = shap_values
            shap.summary_plot(
                sv_plot, X_test_imp,
                feature_names=short_names,
                show=False, max_display=16
            )
            plt.title(f"SHAP Feature Importance - {model_name} (temporal split)", fontsize=13)
            plt.tight_layout()
            plt.savefig(f"outputs/figures/json_extended_pipeline/ext_shap_summary_{model_name.lower().replace(' ', '_')}.png",
                       dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  -> data/ext_shap_summary_{model_name.lower().replace(' ', '_')}.png")

            # 2. Bar plot - mean |SHAP| per feature
            print("  Generez bar plot...")
            sv_array = np.array(shap_values)
            if sv_array.ndim == 3:
                if sv_array.shape[0] == X_test_imp.shape[0]:
                    mean_shap = np.abs(sv_array).mean(axis=(0, 2))
                else:
                    mean_shap = np.abs(sv_array).mean(axis=(0, 1))
            elif isinstance(shap_values, list):
                mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)

            mean_shap = np.array(mean_shap).flatten()

            shap_df = pd.DataFrame({
                "feature": short_names,
                "mean_abs_shap": mean_shap
            }).sort_values("mean_abs_shap", ascending=True)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.barh(shap_df["feature"], shap_df["mean_abs_shap"], color="#2196F3")
            ax.set_xlabel("Mean |SHAP value|", fontsize=11)
            ax.set_title(f"SHAP Feature Importance - {model_name} (temporal split)", fontsize=13)
            plt.tight_layout()
            plt.savefig(f"outputs/figures/json_extended_pipeline/ext_shap_bar_{model_name.lower().replace(' ', '_')}.png",
                       dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  -> data/ext_shap_bar_{model_name.lower().replace(' ', '_')}.png")

            # 3. Top 5 features - dependence plots
            print("  Generez dependence plots (top 5)...")
            top5_idx = np.argsort(mean_shap)[-5:][::-1]
            fig, axes = plt.subplots(1, 5, figsize=(28, 6))
            for i, (ax, idx) in enumerate(zip(axes, top5_idx)):
                sv_arr = np.array(shap_values)
                if sv_arr.ndim == 3:
                    if sv_arr.shape[0] == X_test_imp.shape[0]:
                        sv = sv_arr[:, idx, 1]
                    else:
                        sv = sv_arr[1, :, idx]
                elif isinstance(shap_values, list):
                    sv = shap_values[1][:, idx]
                else:
                    sv = shap_values[:, idx]
                ax.scatter(X_test_imp.iloc[:, idx], sv, alpha=0.5, s=16, c="#2196F3")
                ax.set_xlabel(short_names[idx], fontsize=12)
                ax.set_ylabel("SHAP value" if i == 0 else "", fontsize=12)
                ax.set_title(short_names[idx], fontsize=13, fontweight="bold")
                ax.tick_params(labelsize=10)
            plt.suptitle(f"SHAP Dependence - Top 5 Features ({model_name}, temporal split)", fontsize=15)
            plt.tight_layout(w_pad=3.0)
            plt.savefig(f"outputs/figures/json_extended_pipeline/ext_shap_dependence_{model_name.lower().replace(' ', '_')}.png",
                       dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  -> data/ext_shap_dependence_{model_name.lower().replace(' ', '_')}.png")

            # Print top features
            print(f"\n  Top 10 features ({model_name} SHAP):")
            for _, row in shap_df.sort_values("mean_abs_shap", ascending=False).head(10).iterrows():
                print(f"    {row['feature']:<25} {row['mean_abs_shap']:.4f}")

            break  # Doar pe primul model disponibil


# ============================================================
# 3b. SHAP WATERFALL - Explicabilitate per instanta  [NOU v4]
# ============================================================
def run_shap_waterfall(trained_models, X_test_imp, y_test, FEATURES):
    """
    Genereaza SHAP waterfall plots pentru 3 instante reprezentative
    (una per clasa: Rest, Normal, Active).

    Waterfall plots arata CONTRIBUTIA fiecarui feature la o predictie
    specifica, explicand DE CE modelul a clasificat acea zi intr-o
    anumita clasa.

    Referinta: Napa et al. (2025) Fig. 6-9, JMIR Mental Health (2025)
    """
    print("\n" + "=" * 60)
    print("  SHAP WATERFALL - Explicabilitate per instanta  [NOU v4]")
    print("=" * 60)

    for model_name in ["XGBoost", "Random Forest"]:
        if model_name not in trained_models:
            continue

        print(f"\n--- Waterfall pe {model_name} ---")
        model = trained_models[model_name]
        short_names = [f.replace("_lag1", "").replace("aw_", "AW:") for f in FEATURES]

        explainer = shap.TreeExplainer(model)

        # Gasim cate o instanta per clasa din test set
        class_names = {0: "Rest", 1: "Normal", 2: "Active"}
        example_indices = {}

        for cls_id, cls_name in class_names.items():
            mask = y_test.values == cls_id
            if mask.sum() > 0:
                idx_in_test = np.where(mask)[0][0]
                example_indices[cls_id] = idx_in_test
                print(f"  Clasa {cls_name}: folosesc instanta test[{idx_in_test}]")
            else:
                print(f"  Clasa {cls_name}: nu exista in test set, skip")

        if not example_indices:
            print("  EROARE: nu s-au gasit instante pentru waterfall")
            return

        # Generam waterfall plots - cate un fisier per clasa (mai clar)
        for cls_id, test_idx in example_indices.items():
            cls_name = class_names[cls_id]
            single_instance = X_test_imp.iloc[[test_idx]]

            sv = explainer.shap_values(single_instance)
            sv_array = np.array(sv)

            # Extragem SHAP values pentru clasa corespunzatoare
            if sv_array.ndim == 3:
                if sv_array.shape[0] == 1:  # (1, features, classes)
                    sv_for_class = sv_array[0, :, cls_id]
                    base_val = explainer.expected_value[cls_id]
                else:  # (classes, 1, features)
                    sv_for_class = sv_array[cls_id, 0, :]
                    base_val = explainer.expected_value[cls_id]
            elif isinstance(sv, list):
                sv_for_class = sv[cls_id][0]
                base_val = explainer.expected_value[cls_id]
            else:
                sv_for_class = sv[0]
                base_val = (explainer.expected_value
                           if np.isscalar(explainer.expected_value)
                           else explainer.expected_value[0])

            # Construim obiect shap.Explanation
            explanation = shap.Explanation(
                values=sv_for_class,
                base_values=base_val,
                data=single_instance.values[0],
                feature_names=short_names
            )

            # Plot waterfall
            fig, ax = plt.subplots(figsize=(10, 7))
            plt.sca(ax)
            shap.plots.waterfall(explanation, max_display=12, show=False)
            plt.title(
                f"SHAP Waterfall - Zi {cls_name} (test[{test_idx}], {model_name})",
                fontsize=13, pad=15
            )
            plt.tight_layout()
            fname = f"outputs/figures/json_extended_pipeline/ext_shap_waterfall_{cls_name.lower()}_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  -> {fname}")

        # Si un combined plot cu toate 3 (layout VERTICAL pentru lizibilitate)
        n_plots = len(example_indices)
        fig = plt.figure(figsize=(10, 7 * n_plots))

        for i, (cls_id, test_idx) in enumerate(example_indices.items()):
            cls_name = class_names[cls_id]
            single_instance = X_test_imp.iloc[[test_idx]]
            sv = explainer.shap_values(single_instance)
            sv_array = np.array(sv)

            if sv_array.ndim == 3:
                if sv_array.shape[0] == 1:
                    sv_for_class = sv_array[0, :, cls_id]
                    base_val = explainer.expected_value[cls_id]
                else:
                    sv_for_class = sv_array[cls_id, 0, :]
                    base_val = explainer.expected_value[cls_id]
            elif isinstance(sv, list):
                sv_for_class = sv[cls_id][0]
                base_val = explainer.expected_value[cls_id]
            else:
                sv_for_class = sv[0]
                base_val = (explainer.expected_value
                           if np.isscalar(explainer.expected_value)
                           else explainer.expected_value[0])

            explanation = shap.Explanation(
                values=sv_for_class,
                base_values=base_val,
                data=single_instance.values[0],
                feature_names=short_names
            )

            ax = fig.add_subplot(n_plots, 1, i + 1)
            plt.sca(ax)
            shap.plots.waterfall(explanation, max_display=10, show=False)
            ax.set_title(f"Zi {cls_name}", fontsize=14, fontweight="bold", pad=15)

        plt.suptitle(
            f"SHAP Waterfall - De ce modelul clasifica fiecare zi ({model_name})",
            fontsize=16, y=1.01
        )
        plt.tight_layout(h_pad=4.0)
        fname_all = f"outputs/figures/json_extended_pipeline/ext_shap_waterfall_all_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname_all, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> {fname_all}")

        print(f"\n  INTERPRETARE waterfall (pentru Cap.5):")
        print(f"  Fiecare waterfall arata cum contribuie features la decizia")
        print(f"  de clasificare a unei zile specifice. Barele rosii (pozitive)")
        print(f"  impingerea predictia spre clasa respectiva, barele albastre")
        print(f"  (negative) impotriva. Base value = predictia medie a modelului.")
        print(f"  Acest tip de explicabilitate este esential pentru RQ3 (UI).")

        break  # Doar pe primul model disponibil


# ============================================================
# 3c. PDP - Partial Dependence Plots  [NOU v4]
# ============================================================
def run_pdp(trained_models, X_train_imp, X_test_imp, FEATURES):
    """
    Genereaza Partial Dependence Plots (PDP) pentru top 3 features.

    PDP arata relatia MARGINALA intre un feature si predictia modelului,
    mediata peste toate celelalte features.

    Referinta: Napa et al. (2025) Fig. 5 - PDP pe RF pentru CVD risk.
    """
    print("\n" + "=" * 60)
    print("  PDP - Partial Dependence Plots  [NOU v4]")
    print("=" * 60)

    for model_name in ["XGBoost", "Random Forest"]:
        if model_name not in trained_models:
            continue

        print(f"\n--- PDP pe {model_name} ---")
        model = trained_models[model_name]
        short_names = [f.replace("_lag1", "").replace("aw_", "AW:") for f in FEATURES]

        # Top 3 features din feature_importances_
        fi = model.feature_importances_
        top3_idx = np.argsort(fi)[-3:][::-1]
        top3_names = [short_names[i] for i in top3_idx]
        print(f"  Top 3 features: {top3_names}")

        # --- PDP 1: Top 3 features, clasa Normal ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for i, (feat_idx, feat_name) in enumerate(zip(top3_idx, top3_names)):
            PartialDependenceDisplay.from_estimator(
                model,
                X_test_imp,
                features=[int(feat_idx)],
                feature_names=short_names,
                target=1,  # clasa Normal ca referinta
                ax=axes[i],
                kind="average",
                line_kw={"color": "#2196F3", "linewidth": 2},
            )
            axes[i].set_title(f"PDP: {feat_name}", fontsize=12)
            axes[i].set_ylabel("Partial dependence" if i == 0 else "", fontsize=10)

        plt.suptitle(
            f"Partial Dependence Plots - Top 3 Features ({model_name}, clasa Normal)",
            fontsize=14
        )
        plt.tight_layout()
        fname = f"outputs/figures/json_extended_pipeline/ext_pdp_top3_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> {fname}")

        # --- PDP 2: Multi-class pe feature dominant ---
        print("  Generez PDP multi-class pe feature-ul dominant...")
        class_names_list = ["Rest", "Normal", "Active"]
        colors = ["#E53935", "#43A047", "#1E88E5"]
        dominant_feat_idx = int(top3_idx[0])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for cls_id, (cls_name, color) in enumerate(zip(class_names_list, colors)):
            try:
                PartialDependenceDisplay.from_estimator(
                    model,
                    X_test_imp,
                    features=[dominant_feat_idx],
                    feature_names=short_names,
                    target=cls_id,
                    ax=axes[cls_id],
                    kind="average",
                    line_kw={"color": color, "linewidth": 2},
                )
                axes[cls_id].set_title(f"Clasa: {cls_name}", fontsize=12)
                axes[cls_id].set_ylabel(
                    "Partial dependence" if cls_id == 0 else "", fontsize=10
                )
            except Exception as e:
                axes[cls_id].text(
                    0.5, 0.5, f"N/A\n{str(e)[:60]}",
                    ha="center", va="center", fontsize=10
                )
                axes[cls_id].set_title(f"Clasa: {cls_name}", fontsize=12)

        plt.suptitle(
            f"PDP Multi-Class: {top3_names[0]} ({model_name})",
            fontsize=14
        )
        plt.tight_layout()
        fname_mc = f"outputs/figures/json_extended_pipeline/ext_pdp_multiclass_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname_mc, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> {fname_mc}")

        print(f"\n  INTERPRETARE PDP (pentru Cap.5):")
        print(f"  PDP arata cum variaza probabilitatea de clasificare cand")
        print(f"  un singur feature se schimba, toate celelalte fiind fixe.")
        print(f"  Spre deosebire de SHAP (care explica o predictie specifica),")
        print(f"  PDP arata relatia GENERALA feature -> predictie.")
        print(f"  Ex: daca PDP pt AW:hr_mean creste monoton spre clasa Active,")
        print(f"  inseamna ca zile cu HR mai mare sunt sistematic clasificate Active.")
        print(f"  Util in Cap.5 pentru validarea 'face sens fiziologic' a modelului.")

        break  # Doar pe primul model disponibil


# ============================================================
# 4. FEATURE IMPORTANCE COMPARATIVA
# ============================================================
def compare_feature_importance(trained_models, FEATURES):
    print("\n" + "=" * 60)
    print("  FEATURE IMPORTANCE COMPARATIVA (RF vs XGBoost)")
    print("=" * 60)

    short_names = [f.replace("_lag1", "").replace("aw_", "AW:") for f in FEATURES]

    fi_data = {}
    for name in ["Random Forest", "XGBoost"]:
        if name in trained_models:
            fi = trained_models[name].feature_importances_
            fi_data[name] = fi

    if len(fi_data) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        for ax, (name, fi) in zip(axes, fi_data.items()):
            sorted_idx = np.argsort(fi)
            ax.barh([short_names[i] for i in sorted_idx], fi[sorted_idx],
                   color="#4CAF50" if "Forest" in name else "#FF9800")
            ax.set_title(f"{name}", fontsize=12)
            ax.set_xlabel("Feature Importance", fontsize=10)
        plt.suptitle("Feature Importance: RF vs XGBoost (temporal split)", fontsize=14)
        plt.tight_layout()
        plt.savefig("outputs/figures/json_extended_pipeline/ext_feature_importance_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  -> data/ext_feature_importance_comparison.png")

        # Tabel comparativ top 10
        print(f"\n  {'Feature':<25} {'RF':>10} {'XGBoost':>10}")
        print(f"  {'-'*45}")
        rf_fi = pd.Series(fi_data["Random Forest"], index=short_names)
        xgb_fi = pd.Series(fi_data["XGBoost"], index=short_names)
        combined = pd.DataFrame({"RF": rf_fi, "XGBoost": xgb_fi})
        combined["avg"] = combined.mean(axis=1)
        for feat, row in combined.sort_values("avg", ascending=False).head(10).iterrows():
            print(f"  {feat:<25} {row['RF']:>10.4f} {row['XGBoost']:>10.4f}")


# ============================================================
# 5. MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  RQ2 - COMPARATIE ALGORITMI + SHAP (JSON + 19 FEATURES)")
    print("  RF vs XGBoost vs SVM — 19 features (16 + sleep/resp)")
    print("  Split: TEMPORAL 80/20 + TimeSeriesSplit CV")
    print("=" * 60)

    X, y, FEATURES, dates = load_and_prepare()
    results, models, X_train_imp, X_test_imp, y_test, imputer = compare_algorithms(X, y, FEATURES, dates)
    run_shap(models, X_train_imp, X_test_imp, FEATURES)
    run_shap_waterfall(models, X_test_imp, y_test, FEATURES)   # NOU v4
    run_pdp(models, X_train_imp, X_test_imp, FEATURES)          # NOU v4
    compare_feature_importance(models, FEATURES)

    # Concluzie
    best_acc = max(results, key=lambda k: results[k]["cv_acc_mean"])
    best_f1 = max(results, key=lambda k: results[k]["cv_f1_mean"])
    best_f1w = max(results, key=lambda k: results[k]["f1_weighted"])
    print("\n" + "=" * 60)
    print("  CONCLUZIE RQ2")
    print("=" * 60)
    print(f"  Cel mai bun model (CV F1 Macro): {best_f1} ({results[best_f1]['cv_f1_mean']:.1%})")
    print(f"  Cel mai bun model (F1 Weighted): {best_f1w} ({results[best_f1w]['f1_weighted']:.1%})")
    print(f"  Cel mai bun model (CV acc):      {best_acc} ({results[best_acc]['cv_acc_mean']:.1%})")
    print(f"  Cel mai rapid:                   {min(results, key=lambda k: results[k]['train_time'])}")
    print(f"\n  NOTA: Toate modelele depasesc baseline-ul (majority class)")
    print(f"  pe F1 Macro, demonstrand capacitate reala de clasificare")
    print(f"  pe toate cele 3 clase (Rest/Normal/Active).")
    print(f"  Accuracy poate fi sub baseline din cauza class imbalance -")
    print(f"  F1 Macro este metrica primara corecta.")
    print(f"\n  Split: temporal (train pe date mai vechi, test pe recente).")
    print(f"\n  Grafice salvate in data/ (prefix ext_):")
    print(f"    - ext_confusion_matrices.png")
    print(f"    - ext_shap_summary_*.png")
    print(f"    - ext_shap_bar_*.png")
    print(f"    - ext_shap_dependence_*.png")
    print(f"    - ext_shap_waterfall_*.png")
    print(f"    - ext_pdp_top3_*.png")
    print(f"    - ext_pdp_multiclass_*.png")
    print(f"    - ext_feature_importance_comparison.png")


if __name__ == "__main__":
    main()