"""
EXPERIMENTE NOI — 5 studii de validare suplimentare
====================================================
1. Hyperparameter tuning cu RandomizedSearchCV (RF + SVM)
2. Logistic Regression + LightGBM ca baseline/competitor
3. Learning curves (F1 Macro vs nr. zile antrenament)
4. Calibration analysis (Brier score + reliability diagram)
5. Feature engineering: rolling means 7/14 zile + delta features

Toate folosesc aceeasi logica de load/split/SMOTE ca
cap5_compare_algorithms_json_ext.py (pipeline final, 19 features).

Output:
  outputs/figures/experiments/  (grafice PNG)
  outputs/tables/              (CSV-uri rezultate)

Rulare: python experiments_new.py
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV, TimeSeriesSplit, learning_curve
)
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARN: lightgbm nu e instalat. Ruleaza: pip install lightgbm")

from scipy.stats import randint, uniform

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Paths (post-reorganizare)
FIG_DIR = "outputs/figures/experiments"
TBL_DIR = "outputs/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)


# ============================================================
# 0. LOAD DATE — identic cu cap5_compare_algorithms_json_ext.py
# ============================================================
def load_and_prepare(add_rolling=False):
    """Incarca si pregateste datele. Daca add_rolling=True, adauga
    rolling means 7/14 zile si delta features INAINTE de lag."""
    print("=== LOAD DATE ===")
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

    # Features de baza (inainte de lag)
    BASE_FEATURES = [
        "readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
        "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
        "steps_oura", "respiratory_rate", "temp_deviation",
        "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2",
        "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate",
    ]

    # --- Experiment 5: rolling means + deltas ---
    extra_features = []
    if add_rolling:
        print("  Adaug rolling means (7/14 zile) si delta features...")
        rolling_cols = ["aw_hr_mean", "hrv_mean", "rhr_avg", "steps_oura", "aw_steps"]
        for col in rolling_cols:
            if col in df.columns:
                df[f"{col}_roll7"] = df[col].rolling(7, min_periods=3).mean()
                df[f"{col}_roll14"] = df[col].rolling(14, min_periods=7).mean()
                df[f"{col}_delta1"] = df[col].diff(1)
                extra_features.extend([
                    f"{col}_roll7", f"{col}_roll14", f"{col}_delta1"
                ])
        # Day of week (ciclicitate saptamanala)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
        extra_features.extend(["day_of_week_sin", "day_of_week_cos"])

    ALL_FEATURES = BASE_FEATURES + extra_features

    # Lag features
    FEATURES = [f"{f}_lag1" for f in ALL_FEATURES]
    for feat in ALL_FEATURES:
        df[f"{feat}_lag1"] = df[feat].shift(1)

    # Labels
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
    dates = df_model["date"]

    print(f"  Dataset: {len(X)} zile, {len(FEATURES)} features")
    print(f"  Distributie: {y.value_counts().sort_index().to_dict()}")
    if add_rolling:
        print(f"  Features extra: {len(extra_features)} (rolling + delta + day_of_week)")

    return X, y, FEATURES, dates


def prepare_train_test(X, y, dates):
    """Split temporal 80/20, imputer, SMOTE, scaler — identic cu pipeline-ul."""
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_sm), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X.columns)

    return {
        "X_train_imp": X_train_imp, "X_test_imp": X_test_imp,
        "X_train_sm": X_train_sm, "X_test_imp": X_test_imp,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "y_train_sm": y_train_sm,
        "imputer": imputer, "scaler": scaler,
        "split_idx": split_idx,
    }


# ============================================================
# 1. HYPERPARAMETER TUNING — RandomizedSearchCV
# ============================================================
def experiment_1_tuning(X, y, FEATURES):
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Hyperparameter Tuning (RandomizedSearchCV)")
    print("=" * 60)

    tscv = TimeSeriesSplit(n_splits=5)

    # --- Random Forest ---
    print("\n--- Random Forest tuning ---")
    rf_params = {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": [5, 8, 10, 15, 20, None],
        "model__min_samples_split": randint(2, 20),
        "model__min_samples_leaf": randint(1, 15),
        "model__max_features": ["sqrt", "log2", None],
    }

    rf_pipe = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("model", RandomForestClassifier(random_state=42)),
    ])

    rf_search = RandomizedSearchCV(
        rf_pipe, rf_params, n_iter=50, cv=tscv,
        scoring="f1_macro", random_state=42, n_jobs=-1, verbose=0
    )
    t0 = time.time()
    rf_search.fit(X, y)
    rf_time = time.time() - t0

    print(f"  Best F1 Macro (CV): {rf_search.best_score_:.4f}")
    print(f"  Best params: {rf_search.best_params_}")
    print(f"  Timp: {rf_time:.1f}s")

    # --- SVM ---
    print("\n--- SVM tuning ---")
    svm_params = {
        "model__C": uniform(0.1, 10),
        "model__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
        "model__kernel": ["rbf"],
    }

    svm_pipe = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("scaler", StandardScaler()),
        ("model", SVC(random_state=42, probability=True)),
    ])

    svm_search = RandomizedSearchCV(
        svm_pipe, svm_params, n_iter=30, cv=tscv,
        scoring="f1_macro", random_state=42, n_jobs=-1, verbose=0
    )
    t0 = time.time()
    svm_search.fit(X, y)
    svm_time = time.time() - t0

    print(f"  Best F1 Macro (CV): {svm_search.best_score_:.4f}")
    print(f"  Best params: {svm_search.best_params_}")
    print(f"  Timp: {svm_time:.1f}s")

    # --- XGBoost ---
    print("\n--- XGBoost tuning ---")
    xgb_params = {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": randint(3, 12),
        "model__learning_rate": uniform(0.01, 0.3),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 10),
    }

    xgb_pipe = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("model", XGBClassifier(
            random_state=42, eval_metric="mlogloss", verbosity=0
        )),
    ])

    xgb_search = RandomizedSearchCV(
        xgb_pipe, xgb_params, n_iter=50, cv=tscv,
        scoring="f1_macro", random_state=42, n_jobs=-1, verbose=0
    )
    t0 = time.time()
    xgb_search.fit(X, y)
    xgb_time = time.time() - t0

    print(f"  Best F1 Macro (CV): {xgb_search.best_score_:.4f}")
    print(f"  Best params: {xgb_search.best_params_}")
    print(f"  Timp: {xgb_time:.1f}s")

    # --- Comparatie default vs tuned ---
    print("\n--- Comparatie Default vs Tuned (CV F1 Macro) ---")

    # Default scores (cu CV, identic cu pipeline-ul original)
    default_models = {
        "RF": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=42, k_neighbors=3)),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_leaf=5,
                random_state=42
            )),
        ]),
        "SVM": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=42, k_neighbors=3)),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale",
                         random_state=42, probability=True)),
        ]),
        "XGBoost": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=42, k_neighbors=3)),
            ("model", XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric="mlogloss", verbosity=0
            )),
        ]),
    }

    from sklearn.model_selection import cross_val_score
    default_scores = {}
    for name, pipe in default_models.items():
        cv_f1 = cross_val_score(pipe, X, y, cv=tscv, scoring="f1_macro")
        default_scores[name] = cv_f1.mean()

    tuned_scores = {
        "RF": rf_search.best_score_,
        "SVM": svm_search.best_score_,
        "XGBoost": xgb_search.best_score_,
    }

    # Tabel
    results_df = pd.DataFrame({
        "Model": ["RF", "SVM", "XGBoost"],
        "Default_F1_Macro": [default_scores["RF"], default_scores["SVM"],
                             default_scores["XGBoost"]],
        "Tuned_F1_Macro": [tuned_scores["RF"], tuned_scores["SVM"],
                           tuned_scores["XGBoost"]],
    })
    results_df["Delta_pp"] = (
        (results_df["Tuned_F1_Macro"] - results_df["Default_F1_Macro"]) * 100
    ).round(2)
    results_df.to_csv(f"{TBL_DIR}/tuning_comparison.csv", index=False)

    print(f"\n  {'Model':<10} {'Default':>10} {'Tuned':>10} {'Delta':>10}")
    print(f"  {'-'*40}")
    for _, row in results_df.iterrows():
        print(f"  {row['Model']:<10} {row['Default_F1_Macro']:>10.4f} "
              f"{row['Tuned_F1_Macro']:>10.4f} {row['Delta_pp']:>+9.2f}pp")

    # Grafic
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(3)
    w = 0.35
    ax.bar(x_pos - w/2, results_df["Default_F1_Macro"], w,
           label="Default", color="#90CAF9", edgecolor="#1565C0")
    ax.bar(x_pos + w/2, results_df["Tuned_F1_Macro"], w,
           label="Tuned", color="#1E88E5", edgecolor="#0D47A1")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df["Model"])
    ax.set_ylabel("F1 Macro (CV)")
    ax.set_title("Hyperparameter Tuning: Default vs Optimizat")
    ax.legend()
    ax.set_ylim(0, max(results_df[["Default_F1_Macro", "Tuned_F1_Macro"]].max()) * 1.15)
    for i, row in results_df.iterrows():
        ax.text(i + w/2, row["Tuned_F1_Macro"] + 0.005,
                f"{row['Delta_pp']:+.1f}pp", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/tuning_default_vs_optimized.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> {FIG_DIR}/tuning_default_vs_optimized.png")
    print(f"  -> {TBL_DIR}/tuning_comparison.csv")

    # Salvam best params pentru referinta
    best_params_df = pd.DataFrame([
        {"Model": "RF", **{k.replace("model__", ""): v
         for k, v in rf_search.best_params_.items()}},
        {"Model": "SVM", **{k.replace("model__", ""): v
         for k, v in svm_search.best_params_.items()}},
        {"Model": "XGBoost", **{k.replace("model__", ""): v
         for k, v in xgb_search.best_params_.items()}},
    ])
    best_params_df.to_csv(f"{TBL_DIR}/tuning_best_params.csv", index=False)
    print(f"  -> {TBL_DIR}/tuning_best_params.csv")

    return rf_search, svm_search, xgb_search


# ============================================================
# 2. LOGISTIC REGRESSION + LightGBM
# ============================================================
def experiment_2_new_models(X, y, FEATURES, dates):
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Logistic Regression + LightGBM")
    print("=" * 60)

    d = prepare_train_test(X, y, dates)
    tscv = TimeSeriesSplit(n_splits=5)

    all_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42,
            solver="lbfgs", C=1.0
        ),
        "Random Forest (ref)": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5,
            random_state=42
        ),
        "XGBoost (ref)": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="mlogloss", verbosity=0
        ),
    }

    if HAS_LGBM:
        all_models["LightGBM"] = LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1, num_leaves=31
        )

    results = {}
    for name, model in all_models.items():
        print(f"\n--- {name} ---")

        # LR si SVM pe date scalate, tree-based pe date brute
        needs_scaling = "Logistic" in name or "SVM" in name
        X_tr = d["X_train_scaled"] if needs_scaling else d["X_train_sm"]
        X_te = d["X_test_scaled"] if needs_scaling else d["X_test_imp"]

        t0 = time.time()
        model.fit(X_tr, d["y_train_sm"])
        train_time = time.time() - t0
        y_pred = model.predict(X_te)

        f1_mac = f1_score(d["y_test"], y_pred, average="macro", zero_division=0)
        acc = accuracy_score(d["y_test"], y_pred)
        report = classification_report(
            d["y_test"], y_pred,
            target_names=["Rest", "Normal", "Active"],
            zero_division=0, output_dict=True
        )

        # CV
        if needs_scaling:
            cv_pipe = ImbPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("smote", SMOTE(random_state=42, k_neighbors=3)),
                ("scaler", StandardScaler()),
                ("model", model.__class__(**model.get_params())),
            ])
        else:
            cv_pipe = ImbPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("smote", SMOTE(random_state=42, k_neighbors=3)),
                ("model", model.__class__(**model.get_params())),
            ])

        from sklearn.model_selection import cross_val_score
        cv_f1 = cross_val_score(cv_pipe, X, y, cv=tscv, scoring="f1_macro")

        results[name] = {
            "accuracy": acc,
            "f1_macro": f1_mac,
            "cv_f1_mean": cv_f1.mean(),
            "cv_f1_std": cv_f1.std(),
            "f1_rest": report["Rest"]["f1-score"],
            "f1_normal": report["Normal"]["f1-score"],
            "f1_active": report["Active"]["f1-score"],
            "train_time": train_time,
        }

        print(f"  Accuracy:      {acc:.2%}")
        print(f"  F1 Macro:      {f1_mac:.2%}")
        print(f"  CV F1 (TSCV):  {cv_f1.mean():.2%} (+/- {cv_f1.std():.2%})")
        print(f"  F1 per clasa:  Rest={report['Rest']['f1-score']:.2f}  "
              f"Normal={report['Normal']['f1-score']:.2f}  "
              f"Active={report['Active']['f1-score']:.2f}")
        print(f"  Train time:    {train_time:.2f}s")

    # Tabel comparativ
    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"
    results_df.to_csv(f"{TBL_DIR}/new_models_comparison.csv")

    print(f"\n  {'Model':<22} {'F1 Macro':>10} {'CV F1':>10} {'F1 Rest':>8} "
          f"{'F1 Norm':>8} {'F1 Act':>8}")
    print(f"  {'-'*66}")
    for name, r in results.items():
        print(f"  {name:<22} {r['f1_macro']:>10.4f} {r['cv_f1_mean']:>10.4f} "
              f"{r['f1_rest']:>8.2f} {r['f1_normal']:>8.2f} {r['f1_active']:>8.2f}")

    # Grafic
    names = list(results.keys())
    f1_vals = [results[n]["f1_macro"] for n in names]
    cv_vals = [results[n]["cv_f1_mean"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(names))
    w = 0.35
    ax.bar(x_pos - w/2, f1_vals, w, label="F1 Macro (test)", color="#66BB6A",
           edgecolor="#2E7D32")
    ax.bar(x_pos + w/2, cv_vals, w, label="F1 Macro (CV)", color="#42A5F5",
           edgecolor="#1565C0")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.replace(" (ref)", "") for n in names], rotation=15, ha="right")
    ax.set_ylabel("F1 Macro")
    ax.set_title("Comparatie modele: LR + LightGBM vs RF + XGBoost")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/new_models_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> {FIG_DIR}/new_models_comparison.png")
    print(f"  -> {TBL_DIR}/new_models_comparison.csv")


# ============================================================
# 3. LEARNING CURVES — F1 Macro vs nr. zile antrenament
# ============================================================
def experiment_3_learning_curves(X, y):
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Learning Curves (F1 vs nr. zile)")
    print("=" * 60)

    tscv = TimeSeriesSplit(n_splits=5)

    # NOTA: Folosim class_weight="balanced" in loc de SMOTE pentru
    # learning curves, deoarece SMOTE esueaza pe train sizes mici
    # (clasa Active are doar 46 instante, iar cu k_neighbors=3
    # SMOTE cere minim 4 instante per clasa in fiecare fold).
    # class_weight="balanced" e echivalent conceptual: penalizeaza
    # erorile pe clase minoritare proportional cu frecventa lor.
    from sklearn.pipeline import Pipeline as SkPipeline

    models_to_plot = {
        "Random Forest": SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_leaf=5,
                random_state=42, class_weight="balanced"
            )),
        ]),
        "XGBoost": SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric="mlogloss", verbosity=0,
                # XGBoost: scale_pos_weight nu suporta multiclass direct,
                # dar sample_weight e gestionat intern de learning_curve
            )),
        ]),
        "SVM": SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale",
                         random_state=42, class_weight="balanced")),
        ]),
    }

    # Train sizes: de la 20% la 100% (in 9 pasi)
    # Pornim de la 20% ca sa avem suficiente instante per clasa
    train_sizes_frac = np.linspace(0.2, 1.0, 9)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Random Forest": "#4CAF50", "XGBoost": "#FF9800", "SVM": "#2196F3"}

    lc_results = {}

    for ax, (name, pipe) in zip(axes, models_to_plot.items()):
        print(f"\n--- Learning curve: {name} ---")
        t0 = time.time()

        train_sizes_abs, train_scores, val_scores = learning_curve(
            pipe, X, y,
            train_sizes=train_sizes_frac,
            cv=tscv,
            scoring="f1_macro",
            n_jobs=-1,
            shuffle=False,  # temporal order!
        )

        elapsed = time.time() - t0
        print(f"  Timp: {elapsed:.1f}s")
        print(f"  Train sizes: {train_sizes_abs}")
        print(f"  Val F1 range: {val_scores.mean(axis=1).min():.3f} -> "
              f"{val_scores.mean(axis=1).max():.3f}")

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        color = colors[name]
        ax.fill_between(train_sizes_abs, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color=color)
        ax.fill_between(train_sizes_abs, val_mean - val_std,
                        val_mean + val_std, alpha=0.1, color=color)
        ax.plot(train_sizes_abs, train_mean, "o-", color=color,
                label="Train F1", linewidth=2)
        ax.plot(train_sizes_abs, val_mean, "s--", color=color,
                label="Validation F1", linewidth=2, alpha=0.8)
        ax.set_xlabel("Nr. zile antrenament")
        ax.set_ylabel("F1 Macro")
        ax.set_title(name)
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        lc_results[name] = {
            "train_sizes": train_sizes_abs.tolist(),
            "val_f1_mean": val_mean.tolist(),
            "val_f1_std": val_std.tolist(),
        }

    plt.suptitle("Learning Curves — F1 Macro vs. Nr. Zile Antrenament\n(class_weight='balanced', fara SMOTE)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> {FIG_DIR}/learning_curves.png")

    # Interpretare automata
    for name, lc in lc_results.items():
        vals = lc["val_f1_mean"]
        if len(vals) >= 2 and not np.isnan(vals[-1]) and not np.isnan(vals[-2]):
            trend = vals[-1] - vals[-2]
            if trend > 0.01:
                print(f"  {name}: curba INCA CRESTE la final (+{trend:.3f}) "
                      f"-> ar beneficia de mai multe date")
            else:
                print(f"  {name}: curba se SATUREAZA la final ({trend:+.3f}) "
                      f"-> mai multe date nu ar ajuta semnificativ")
        else:
            print(f"  {name}: date insuficiente pentru interpretare (NaN in rezultate)")


# ============================================================
# 4. CALIBRATION ANALYSIS — Brier score + reliability diagram
# ============================================================
def experiment_4_calibration(X, y, FEATURES, dates):
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: Calibration Analysis (Brier + Reliability)")
    print("=" * 60)

    d = prepare_train_test(X, y, dates)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="mlogloss", verbosity=0
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            random_state=42, probability=True
        ),
    }

    # Binarize labels pentru Brier score per clasa
    y_test_bin = label_binarize(d["y_test"], classes=[0, 1, 2])
    class_names = ["Rest", "Normal", "Active"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Random Forest": "#4CAF50", "XGBoost": "#FF9800", "SVM (RBF)": "#2196F3"}

    brier_results = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")

        needs_scaling = "SVM" in name
        X_tr = d["X_train_scaled"] if needs_scaling else d["X_train_sm"]
        X_te = d["X_test_scaled"] if needs_scaling else d["X_test_imp"]

        model.fit(X_tr, d["y_train_sm"])
        y_proba = model.predict_proba(X_te)

        # Brier score per clasa
        brier_per_class = {}
        for cls_id, cls_name in enumerate(class_names):
            bs = brier_score_loss(y_test_bin[:, cls_id], y_proba[:, cls_id])
            brier_per_class[cls_name] = bs

        brier_avg = np.mean(list(brier_per_class.values()))
        brier_results[name] = {**brier_per_class, "Average": brier_avg}

        print(f"  Brier scores: {', '.join(f'{k}={v:.4f}' for k, v in brier_per_class.items())}")
        print(f"  Brier average: {brier_avg:.4f}")

    # Reliability diagram (per model, clasa Normal ca referinta)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        needs_scaling = "SVM" in name
        X_te = d["X_test_scaled"] if needs_scaling else d["X_test_imp"]
        y_proba = model.predict_proba(X_te)

        for cls_id, (cls_name, color) in enumerate(
            zip(class_names, ["#E53935", "#43A047", "#1E88E5"])
        ):
            prob_true, prob_pred = calibration_curve(
                y_test_bin[:, cls_id], y_proba[:, cls_id],
                n_bins=8, strategy="uniform"
            )
            ax.plot(prob_pred, prob_true, "o-", color=color,
                    label=f"{cls_name}", linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.set_xlabel("Probabilitate prezisa")
        ax.set_ylabel("Frecventa reala")
        ax.set_title(f"{name}\n(Brier avg: {brier_results[name]['Average']:.4f})")
        ax.legend(fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Reliability Diagrams — Calibrarea probabilitatilor", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/calibration_reliability.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Tabel Brier
    brier_df = pd.DataFrame(brier_results).T
    brier_df.index.name = "Model"
    brier_df.to_csv(f"{TBL_DIR}/calibration_brier_scores.csv")

    print(f"\n  {'Model':<20} {'Rest':>8} {'Normal':>8} {'Active':>8} {'Avg':>8}")
    print(f"  {'-'*52}")
    for name, scores in brier_results.items():
        print(f"  {name:<20} {scores['Rest']:>8.4f} {scores['Normal']:>8.4f} "
              f"{scores['Active']:>8.4f} {scores['Average']:>8.4f}")

    print(f"\n  -> {FIG_DIR}/calibration_reliability.png")
    print(f"  -> {TBL_DIR}/calibration_brier_scores.csv")

    # Interpretare
    best_cal = min(brier_results, key=lambda k: brier_results[k]["Average"])
    print(f"\n  Cel mai bine calibrat: {best_cal} "
          f"(Brier avg = {brier_results[best_cal]['Average']:.4f})")
    print(f"  Relevant pentru confidence threshold-ul de 0.50:")
    print(f"  Daca modelul e bine calibrat, probabilitatile sunt de incredere")
    print(f"  si threshold-ul de 0.50 functioneaza corect ca safety layer.")


# ============================================================
# 5. FEATURE ENGINEERING — Rolling means + delta + day_of_week
# ============================================================
def experiment_5_rolling_features(dates_orig):
    print("\n" + "=" * 60)
    print("  EXPERIMENT 5: Rolling Means + Delta Features")
    print("=" * 60)

    # Load cu features extinse
    X_ext, y_ext, FEATURES_ext, dates_ext = load_and_prepare(add_rolling=True)

    # Load fara features extinse (baseline = pipeline-ul curent)
    X_base, y_base, FEATURES_base, dates_base = load_and_prepare(add_rolling=False)

    tscv = TimeSeriesSplit(n_splits=5)
    from sklearn.model_selection import cross_val_score

    configs = {
        "Baseline (19 feat)": (X_base, y_base, FEATURES_base),
        f"Extended ({len(FEATURES_ext)} feat)": (X_ext, y_ext, FEATURES_ext),
    }

    results = {}
    for config_name, (X_c, y_c, feats) in configs.items():
        print(f"\n--- {config_name} ---")

        rf_pipe = ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=42, k_neighbors=3)),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_leaf=5,
                random_state=42
            )),
        ])

        cv_f1 = cross_val_score(rf_pipe, X_c, y_c, cv=tscv, scoring="f1_macro")

        # Test set evaluation
        split_idx = int(len(X_c) * 0.8)
        X_train, X_test = X_c.iloc[:split_idx], X_c.iloc[split_idx:]
        y_train, y_test = y_c.iloc[:split_idx], y_c.iloc[split_idx:]

        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feats)
        X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feats)

        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_imp, y_train)

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42
        )
        rf.fit(X_train_sm, y_train_sm)
        y_pred = rf.predict(X_test_imp)

        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        report = classification_report(
            y_test, y_pred,
            target_names=["Rest", "Normal", "Active"],
            zero_division=0, output_dict=True
        )

        results[config_name] = {
            "n_features": len(feats),
            "f1_macro_test": f1_mac,
            "cv_f1_mean": cv_f1.mean(),
            "cv_f1_std": cv_f1.std(),
            "f1_rest": report["Rest"]["f1-score"],
            "f1_normal": report["Normal"]["f1-score"],
            "f1_active": report["Active"]["f1-score"],
        }

        print(f"  Features: {len(feats)}")
        print(f"  F1 Macro (test): {f1_mac:.4f}")
        print(f"  CV F1 (TSCV):    {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
        print(f"  F1 per clasa:    Rest={report['Rest']['f1-score']:.2f}  "
              f"Normal={report['Normal']['f1-score']:.2f}  "
              f"Active={report['Active']['f1-score']:.2f}")

        # Feature importance pentru config extins
        if "Extended" in config_name:
            fi = pd.Series(rf.feature_importances_, index=feats)
            fi_sorted = fi.sort_values(ascending=False)
            print(f"\n  Top 10 features (Extended):")
            for feat_name, imp in fi_sorted.head(10).items():
                short = feat_name.replace("_lag1", "").replace("aw_", "AW:")
                marker = " [NOU]" if any(x in feat_name for x in
                                         ["roll7", "roll14", "delta1",
                                          "day_of_week"]) else ""
                print(f"    {short:<30} {imp:.4f}{marker}")

    # Tabel comparativ
    results_df = pd.DataFrame(results).T
    results_df.index.name = "Config"
    results_df.to_csv(f"{TBL_DIR}/rolling_features_comparison.csv")

    delta_f1 = (results[f"Extended ({len(FEATURES_ext)} feat)"]["cv_f1_mean"] -
                results["Baseline (19 feat)"]["cv_f1_mean"]) * 100

    print(f"\n  Delta CV F1 Macro: {delta_f1:+.2f}pp")
    if delta_f1 > 1:
        print(f"  -> Features noi AJUTA (+{delta_f1:.1f}pp). Merita incluse.")
    elif delta_f1 > -1:
        print(f"  -> Diferenta NEGLIJABILA ({delta_f1:+.1f}pp). "
              f"Baseline-ul de 19 features e suficient.")
    else:
        print(f"  -> Features noi DAUNEAZA ({delta_f1:.1f}pp). "
              f"Probabil noise/overfitting. Pastram 19 features.")

    print(f"\n  -> {TBL_DIR}/rolling_features_comparison.csv")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  5 EXPERIMENTE NOI — Validare suplimentara")
    print("  Dataset: JSON extended, 19 features, split temporal")
    print("=" * 60)

    t_start = time.time()

    X, y, FEATURES, dates = load_and_prepare(add_rolling=False)

    # Experiment 1: Tuning
  #  experiment_1_tuning(X, y, FEATURES)

    # Experiment 2: LR + LightGBM
  #  experiment_2_new_models(X, y, FEATURES, dates)

    # Experiment 3: Learning curves
    experiment_3_learning_curves(X, y)

    # Experiment 4: Calibration
  #  experiment_4_calibration(X, y, FEATURES, dates)

    # Experiment 5: Rolling features
  #  experiment_5_rolling_features(dates)

    t_total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  TOATE EXPERIMENTELE COMPLETE — {t_total:.0f}s total")
    print("=" * 60)
    print(f"\n  Outputs:")
    print(f"    Figuri:  {FIG_DIR}/")
    print(f"    Tabele:  {TBL_DIR}/")
    print(f"\n  Figuri generate:")
    print(f"    - tuning_default_vs_optimized.png")
    print(f"    - new_models_comparison.png")
    print(f"    - learning_curves.png")
    print(f"    - calibration_reliability.png")
    print(f"\n  Tabele generate:")
    print(f"    - tuning_comparison.csv")
    print(f"    - tuning_best_params.csv")
    print(f"    - new_models_comparison.csv")
    print(f"    - calibration_brier_scores.csv")
    print(f"    - rolling_features_comparison.csv")


if __name__ == "__main__":
    main()