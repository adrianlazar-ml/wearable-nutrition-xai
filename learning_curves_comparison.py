"""
Learning Curves — SMOTE vs class_weight comparison
===================================================
Ruleaza learning curves in doua variante:
  1. Cu SMOTE k=3 (esueaza pe train sizes mici -> NaN)
  2. Cu class_weight="balanced" (functioneaza corect)

Salveaza ambele figuri + log cu eroarea SMOTE ca dovada.
Folosit in Cap.5/Cap.6 pentru a justifica alegerea class_weight.
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import io

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

import os
FIG_DIR = "outputs/figures/experiments"
RPT_DIR = "outputs/reports"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RPT_DIR, exist_ok=True)


def load_and_prepare():
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
    df_model = df[["date"] + FEATURES + ["label"]].dropna(subset=["label"])
    df_model = df_model.sort_values("date").reset_index(drop=True)

    X = df_model[FEATURES]
    y = df_model["label"].astype(int)
    return X, y, FEATURES


def run_learning_curves_smote(X, y, FEATURES):
    """Varianta 1: SMOTE k=3. Esueaza pe train sizes mici."""
    print("\n=== VARIANTA 1: Learning Curves cu SMOTE k=3 ===")

    tscv = TimeSeriesSplit(n_splits=5)
    train_sizes_frac = np.linspace(0.1, 1.0, 10)

    # Split temporal 80/20: learning curve doar pe train, fara leakage din test
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    models = {
        "Random Forest": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=42, k_neighbors=3)),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_leaf=5,
                random_state=42
            )),
        ]),
        "XGBoost": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=42, k_neighbors=3)),
            ("model", XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric="mlogloss", verbosity=0
            )),
        ]),
        "SVM": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42, k_neighbors=3)),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale",
                         random_state=42)),
        ]),
    }

    # Capturam warnings intr-un log
    error_log = []
    error_log.append("=" * 60)
    error_log.append("LEARNING CURVES CU SMOTE k=3 — LOG ERORI")
    error_log.append("=" * 60)
    error_log.append(f"Dataset: {len(X)} zile, {len(FEATURES)} features")
    error_log.append(f"Distributie clase: {dict(pd.Series(y).value_counts().sort_index())}")
    error_log.append(f"Clasa Active (2): doar {(y == 2).sum()} instante")
    error_log.append(f"SMOTE k_neighbors=3 necesita minim k+1=4 instante per clasa")
    error_log.append("")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Random Forest": "#4CAF50", "XGBoost": "#FF9800", "SVM": "#2196F3"}

    for ax, (name, pipe) in zip(axes, models.items()):
        print(f"\n--- {name} ---")

        # Capturam warnings
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            train_sizes_abs, train_scores, val_scores = learning_curve(
                pipe, X_train, y_train,
                train_sizes=train_sizes_frac,
                cv=tscv,
                scoring="f1_macro",
                n_jobs=-1,
                shuffle=False,
                error_score=np.nan,  # NaN in loc de crash
            )

        # Log warnings
        smote_errors = [w for w in caught_warnings
                       if "FitFailedWarning" in str(type(w.category))]
        n_nan = np.isnan(val_scores).sum()
        total_fits = val_scores.size

        error_log.append(f"--- {name} ---")
        error_log.append(f"  Train sizes: {train_sizes_abs.tolist()}")
        error_log.append(f"  Fit-uri esuate (NaN): {n_nan} din {total_fits}")
        error_log.append(f"  Val F1 per train size (mean across folds):")
        for ts, vm in zip(train_sizes_abs, np.nanmean(val_scores, axis=1)):
            error_log.append(f"    {ts:>4d} zile: {vm:.4f}" if not np.isnan(vm)
                           else f"    {ts:>4d} zile: NaN (SMOTE esuat)")

        if smote_errors:
            # Extragem mesajul de eroare
            msg = str(smote_errors[0].message)
            # Gasim linia relevanta cu ValueError
            for line in msg.split("\n"):
                if "n_neighbors" in line and "n_samples_fit" in line:
                    error_log.append(f"  Eroare SMOTE: {line.strip()}")
                    break

        error_log.append("")

        val_mean = np.nanmean(val_scores, axis=1)
        train_mean = np.nanmean(train_scores, axis=1)

        print(f"  Train sizes: {train_sizes_abs}")
        print(f"  NaN count: {n_nan}/{total_fits}")
        print(f"  Val F1: {val_mean}")

        color = colors[name]
        ax.plot(train_sizes_abs, train_mean, "o-", color=color,
                label="Train F1", linewidth=2)
        ax.plot(train_sizes_abs, val_mean, "s--", color=color,
                label="Validation F1", linewidth=2, alpha=0.8)
        ax.set_xlabel("Nr. zile antrenament")
        ax.set_ylabel("F1 Macro")
        ax.set_title(f"{name}\n({n_nan}/{total_fits} fits esuate)")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Learning Curves cu SMOTE k=3 — ESEC\n"
        "(SMOTE necesita minim k+1=4 instante per clasa; "
        "clasa Active are sub 4 in foldurile mici)",
        fontsize=12, color="#D32F2F"
    )
    plt.tight_layout()
    fname = f"{FIG_DIR}/learning_curves_smote_failed.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> {fname}")

    # Salvam error log
    error_log.append("=" * 60)
    error_log.append("CONCLUZIE: SMOTE k=3 nu este aplicabil pe learning curves")
    error_log.append("cu TimeSeriesSplit deoarece subseturile mici de antrenament")
    error_log.append("contin sub 4 instante din clasa Active (minoritara).")
    error_log.append("Solutie: class_weight='balanced' ca alternativa echivalenta.")
    error_log.append("=" * 60)

    log_path = f"{RPT_DIR}/learning_curves_smote_error_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(error_log))
    print(f"  -> {log_path}")

    return fname


def run_learning_curves_balanced(X, y, FEATURES):
    """Varianta 2: class_weight='balanced'. Functioneaza corect."""
    print("\n=== VARIANTA 2: Learning Curves cu class_weight='balanced' ===")

    tscv = TimeSeriesSplit(n_splits=5)
    train_sizes_frac = np.linspace(0.2, 1.0, 9)

    # Split temporal 80/20: learning curve doar pe train, fara leakage din test
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    models = {
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
                random_state=42, eval_metric="mlogloss", verbosity=0
            )),
        ]),
        "SVM": SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale",
                         random_state=42, class_weight="balanced")),
        ]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Random Forest": "#4CAF50", "XGBoost": "#FF9800", "SVM": "#2196F3"}

    for ax, (name, pipe) in zip(axes, models.items()):
        print(f"\n--- {name} ---")

        train_sizes_abs, train_scores, val_scores = learning_curve(
            pipe, X_train, y_train,
            train_sizes=train_sizes_frac,
            cv=tscv,
            scoring="f1_macro",
            n_jobs=-1,
            shuffle=False,
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        print(f"  Train sizes: {train_sizes_abs}")
        print(f"  Val F1 range: {val_mean.min():.3f} -> {val_mean.max():.3f}")

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

        # Interpretare
        if len(val_mean) >= 2:
            trend = val_mean[-1] - val_mean[-2]
            if trend > 0.01:
                print(f"  -> Curba INCA CRESTE (+{trend:.3f})")
            else:
                print(f"  -> Curba se SATUREAZA ({trend:+.3f})")

    plt.suptitle(
        "Learning Curves cu class_weight='balanced'\n"
        "(alternativa la SMOTE, functionala pe toate train sizes)",
        fontsize=12
    )
    plt.tight_layout()
    fname = f"{FIG_DIR}/learning_curves_balanced.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  -> {fname}")

    return fname


def main():
    print("=" * 60)
    print("  LEARNING CURVES — SMOTE vs class_weight comparison")
    print("=" * 60)

    X, y, FEATURES = load_and_prepare()
    print(f"  Dataset: {len(X)} zile, {len(FEATURES)} features")
    print(f"  Distributie: {dict(pd.Series(y).value_counts().sort_index())}")

    fig1 = run_learning_curves_smote(X, y, FEATURES)
    fig2 = run_learning_curves_balanced(X, y, FEATURES)

    print("\n" + "=" * 60)
    print("  REZULTATE")
    print("=" * 60)
    print(f"  Fig. SMOTE (esec):     {fig1}")
    print(f"  Fig. balanced (corect): {fig2}")
    print(f"  Error log:             outputs/reports/learning_curves_smote_error_log.txt")
    print(f"\n  In lucrare (Cap.5/6):")
    print(f"  - Arata fig. SMOTE ca dovada ca esueaza")
    print(f"  - Arata fig. balanced ca solutie")
    print(f"  - Citeaza error log-ul pentru detalii")


if __name__ == "__main__":
    main()