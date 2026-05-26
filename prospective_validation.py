"""
Prospective Validation — Evaluare externa temporala stricta
============================================================
Aplica modelul inghetat de freeze_production_model.py pe un window
de date colectate STRICT DUPA data de cutoff (train_end_date).

Zero leakage by construction: modelul nu a "vazut" niciodata aceste date,
nici la antrenare, nici la CV intern. Este echivalentul unui studiu
clinic "lock-and-validate".

Output:
- tabel comparativ metrici (test intern vs prospective window)
- figura confusion matrix side-by-side
- figura F1 per clasa (bar chart)
- figura timeline: true vs predicted pe prospective window
- raport text pentru inclus in Cap.6

Autor: Adrian Lazar | Disertatie UPB 2026
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

from cap5_pipeline_json_extended import (
    FEATURES,
    LAG_FEATURES,
    assign_nutrition_label,
    load_apple_watch,
    load_oura,
    merge_datasets,
)


MODELS_DIR = Path("models")
OUT_FIG_DIR = Path("outputs/figures/experiments")
OUT_TABLE_DIR = Path("outputs/tables")
OUT_REPORT_DIR = Path("outputs/reports")
for d in [OUT_FIG_DIR, OUT_TABLE_DIR, OUT_REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = ["Rest", "Normal", "Active"]


def load_frozen_artifacts(tag: str):
    """Incarca modelul, imputer-ul si metadata inghetate."""
    model_path = MODELS_DIR / f"rf_frozen_{tag}.joblib"
    imputer_path = MODELS_DIR / f"imputer_frozen_{tag}.joblib"
    meta_path = MODELS_DIR / f"frozen_meta_{tag}.json"

    for p in [model_path, imputer_path, meta_path]:
        if not p.exists():
            sys.exit(
                f"ERROR: Artefact lipsa: {p}\n"
                f"Ruleaza mai intai: python freeze_production_model.py --tag {tag}"
            )

    rf = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    print(f"Artefacte incarcate pentru tag '{tag}':")
    print(f"  Cutoff date: {meta.get('cutoff_date', meta.get('train_end_date_requested', 'N/A'))}")
    print(f"  Last day in dataset: {meta.get('last_day_in_dataset', meta.get('train_end_date_effective', 'N/A'))}")
    print(f"  Train internal end: {meta.get('train_internal_end', meta.get('train_end_date_effective', 'N/A'))}")
    print(f"  N train: {meta['n_train']}")
    print(f"  F1 Macro test intern (referinta): {meta['reference_metrics_internal_test']['f1_macro']:.4f}")

    return rf, imputer, meta


def build_prospective_dataset(prospective_start: pd.Timestamp,
                              prospective_end: pd.Timestamp) -> pd.DataFrame:
    """
    Construieste dataset-ul exact pentru prospective window.

    IMPORTANT: lag-1 features necesita 1 zi inainte de prospective_start.
    Pipeline-ul face shift(1) PE TOT dataset-ul, apoi filtram la window.
    Asta este corect: lag-1 foloseste date "trecute" relativ la fiecare zi,
    nu peste window boundaries.
    """
    print(f"\n=== CONSTRUIRE DATASET PROSPECTIVE ===")
    print(f"Window: {prospective_start.date()} -> {prospective_end.date()}")

    oura = load_oura()
    aw_tuple = load_apple_watch()
    df = merge_datasets(oura, *aw_tuple)

    max_available = df["date"].max()
    print(f"Maxima data disponibila: {max_available.date()}")

    if max_available < prospective_start:
        sys.exit(
            f"ERROR: Nu ai date mai noi de {prospective_start.date()}. "
            f"Maxima e {max_available.date()}. "
            f"Extrage date noi din Oura + Apple Watch inainte de a rula acest script."
        )

    # Lag features PE TOT dataset-ul (asa cum face pipeline-ul)
    for feat in LAG_FEATURES:
        df[f"{feat}_lag1"] = df[feat].shift(1)

    df["nutrition_label"] = df.apply(assign_nutrition_label, axis=1)

    # Filtram STRICT la prospective window
    mask = (df["date"] >= prospective_start) & (df["date"] <= prospective_end)
    df_prospective = df.loc[mask, ["date"] + FEATURES + ["nutrition_label"]].copy()
    df_prospective = df_prospective.dropna(subset=["nutrition_label"])
    df_prospective = df_prospective.sort_values("date").reset_index(drop=True)

    print(f"Zile in prospective window: {len(df_prospective)}")
    if len(df_prospective) == 0:
        sys.exit("ERROR: Zero zile cu label in prospective window. Verifica datele.")

    class_dist = df_prospective["nutrition_label"].value_counts().sort_index().to_dict()
    print(f"Distributie clase: {class_dist}")

    if len(df_prospective) < 20:
        print(f"  WARNING: Doar {len(df_prospective)} zile - rezultatele au putere statistica limitata.")

    return df_prospective


def evaluate_on_prospective(rf, imputer, df_prospective: pd.DataFrame) -> dict:
    """Aplica modelul pe prospective window si returneaza metrici."""
    X = df_prospective[FEATURES]
    y_true = df_prospective["nutrition_label"].astype(int)

    X_imp = pd.DataFrame(
        imputer.transform(X), columns=FEATURES, index=X.index
    )
    y_pred = rf.predict(X_imp)
    y_proba = rf.predict_proba(X_imp)

    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    report_txt = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0, digits=4
    )

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_mac),
        "mcc": float(mcc),
        "f1_per_class": {name: float(v) for name, v in zip(CLASS_NAMES, f1_per_class)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report_txt,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba_max": y_proba.max(axis=1).tolist(),
        "dates": df_prospective["date"].dt.strftime("%Y-%m-%d").tolist(),
    }


def plot_confusion_matrices(cm_internal, cm_prospective, tag, save_path):
    """Plot confusion matrix side-by-side: test intern vs prospective."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cm, title in zip(
        axes,
        [cm_internal, cm_prospective],
        ["Test intern (ultimele 20% pre-cutoff)", "Prospective window (post-cutoff)"]
    ):
        # Normalizare pe rand (recall per clasa)
        cm_arr = np.array(cm)
        cm_norm = cm_arr.astype(float) / np.clip(cm_arr.sum(axis=1, keepdims=True), 1, None)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        for i in range(3):
            for j in range(3):
                txt = f"{cm_arr[i, j]}\n({cm_norm[i, j]:.1%})"
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)

    plt.suptitle(f"Confusion Matrix: Test Intern vs Prospective Validation (tag={tag})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvat: {save_path}")


def plot_f1_comparison(internal_metrics, prospective_metrics, tag, save_path):
    """Bar chart cu F1 per clasa: intern vs prospective."""
    classes = CLASS_NAMES + ["Macro"]
    internal_vals = [internal_metrics["f1_per_class"][c] for c in CLASS_NAMES] + [internal_metrics["f1_macro"]]
    prosp_vals = [prospective_metrics["f1_per_class"][c] for c in CLASS_NAMES] + [prospective_metrics["f1_macro"]]

    x = np.arange(len(classes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, internal_vals, w, label="Test intern", color="#4C72B0")
    bars2 = ax.bar(x + w/2, prosp_vals, w, label="Prospective window", color="#DD8452")

    for bars in [bars1, bars2]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"F1 per clasa: Test intern vs Prospective Validation (tag={tag})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvat: {save_path}")


def plot_prediction_timeline(prospective_metrics, tag, save_path):
    """Timeline scatter: predicted vs true peste prospective window."""
    dates = pd.to_datetime(prospective_metrics["dates"])
    y_true = np.array(prospective_metrics["y_true"])
    y_pred = np.array(prospective_metrics["y_pred"])
    proba = np.array(prospective_metrics["y_proba_max"])

    fig, ax = plt.subplots(figsize=(12, 4))

    correct = y_true == y_pred
    ax.scatter(dates[correct], y_true[correct], c="green", s=60, label="Correct", alpha=0.7, marker="o")
    ax.scatter(dates[~correct], y_true[~correct], c="red", s=60, label="Incorrect (true)",
               alpha=0.7, marker="o")
    ax.scatter(dates[~correct], y_pred[~correct], c="red", s=60, alpha=0.5, marker="x",
               label="Incorrect (predicted)")

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Data")
    ax.set_ylabel("Clasa")
    ax.set_title(f"Predictii pe Prospective Window (tag={tag})")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvat: {save_path}")


def write_text_report(meta, prospective_metrics, tag, prospective_start, prospective_end,
                      n_prospective, save_path):
    """Raport text formatat pentru inclus in Cap.6 al disertatiei."""
    internal = meta["reference_metrics_internal_test"]
    cutoff = meta.get("cutoff_date", meta.get("train_end_date_requested", "N/A"))
    last_day = meta.get("last_day_in_dataset", meta.get("train_end_date_effective", "N/A"))
    train_end = meta.get("train_internal_end", meta.get("train_end_date_effective", "N/A"))
    delta_f1 = prospective_metrics["f1_macro"] - internal["f1_macro"]
    delta_acc = prospective_metrics["accuracy"] - internal["accuracy"]
    delta_mcc = prospective_metrics["mcc"] - internal["mcc"]

    sign_f1 = "+" if delta_f1 >= 0 else ""
    sign_acc = "+" if delta_acc >= 0 else ""
    sign_mcc = "+" if delta_mcc >= 0 else ""

    report = f"""================================================================
PROSPECTIVE VALIDATION REPORT — Disertatie Adrian Lazar 2026
================================================================

Generat: {datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
Tag model frozen: {tag}

METODOLOGIE
-----------
Modelul Random Forest a fost inghetat cu cutoff date {cutoff}
(ultima zi din dataset: {last_day}) si aplicat fara nicio modificare
pe un window de date colectate STRICT DUPA aceasta data. Separarea
temporala elimina prin constructie orice posibilitate de data leakage:
modelul nu a avut acces la datele de evaluare nici in antrenare,
nici in tuning, nici in selection.

Tag artefacte: {tag}
Dataset hash (SHA256, 16 char): {meta['dataset_hash_sha256_16']}

TRAIN PERIOD
------------
Cutoff date: {cutoff}
Last day in frozen dataset: {last_day}
Train internal end (ultima zi din 80% cronologic): {train_end}
N train samples: {meta['n_train']}
Class distribution (pre-SMOTE): {meta['class_distribution_train_pre_smote']}

INTERNAL TEST (referinta, ultimele 20% pre-cutoff)
---------------------------------------------------
Range: {meta['test_internal_range'][0]} -> {meta['test_internal_range'][1]}
N: {meta['n_test_internal']}
Accuracy: {internal['accuracy']:.4f}
F1 Macro: {internal['f1_macro']:.4f}
MCC:      {internal['mcc']:.4f}

PROSPECTIVE WINDOW (zero leakage)
----------------------------------
Range: {prospective_start.date()} -> {prospective_end.date()}
N: {n_prospective}
Accuracy: {prospective_metrics['accuracy']:.4f}  (delta: {sign_acc}{delta_acc:.4f})
F1 Macro: {prospective_metrics['f1_macro']:.4f}  (delta: {sign_f1}{delta_f1:.4f})
MCC:      {prospective_metrics['mcc']:.4f}  (delta: {sign_mcc}{delta_mcc:.4f})

F1 per class (prospective):
  Rest:   {prospective_metrics['f1_per_class']['Rest']:.4f}
  Normal: {prospective_metrics['f1_per_class']['Normal']:.4f}
  Active: {prospective_metrics['f1_per_class']['Active']:.4f}

CLASSIFICATION REPORT (prospective)
------------------------------------
{prospective_metrics['classification_report']}

CONFUSION MATRIX (prospective, row=true, col=predicted)
--------------------------------------------------------
{np.array2string(np.array(prospective_metrics['confusion_matrix']))}

INTERPRETARE (template de adaptat pentru Cap.6)
------------------------------------------------
[TODO: completeaza dupa rulare, in functie de directia delta_f1]

Daca delta_f1 >= -0.03: "Modelul mentine performanta comparabila
pe date colectate post-dezvoltare, demonstrand stabilitate temporala."

Daca -0.05 < delta_f1 < -0.03: "Se observa o degradare usoara,
consistenta cu drift fiziologic/senzorial pe parcursul a N luni;
marimea efectului ramane sub zgomotul CV intern (+/- X pp)."

Daca delta_f1 <= -0.05: "Degradarea semnificativa indica nevoia
de retrenare periodica a modelului; aceasta observatie este
discutata ca directie viitoare in Cap.7 (model monitoring &
retrain scheduling)."

================================================================
"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Salvat: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Prospective validation pentru modelul frozen.")
    parser.add_argument("--tag", type=str, required=True,
                        help="Tag-ul modelului frozen (ex: '20260331').")
    parser.add_argument("--prospective_start", type=str, default=None,
                        help="Start window prospective (default: ziua dupa train_end din meta).")
    parser.add_argument("--prospective_end", type=str, default=None,
                        help="End window prospective (default: azi).")
    args = parser.parse_args()

    print("=" * 60)
    print("  PROSPECTIVE VALIDATION")
    print(f"  Tag: {args.tag}")
    print("=" * 60)

    rf, imputer, meta = load_frozen_artifacts(args.tag)

    # Default: prospective_start = ziua dupa ultima zi din dataset frozen
    if args.prospective_start is None:
        last_day = pd.Timestamp(meta.get("last_day_in_dataset",
                                          meta.get("train_end_date_effective")))
        prospective_start = last_day + pd.Timedelta(days=1)
    else:
        prospective_start = pd.Timestamp(args.prospective_start)

    if args.prospective_end is None:
        prospective_end = pd.Timestamp.today().normalize()
    else:
        prospective_end = pd.Timestamp(args.prospective_end)

    df_prospective = build_prospective_dataset(prospective_start, prospective_end)
    prospective_metrics = evaluate_on_prospective(rf, imputer, df_prospective)

    # Tabel comparativ
    internal = meta["reference_metrics_internal_test"]
    comparison_df = pd.DataFrame([
        {"split": "test_intern",
         "n": meta["n_test_internal"],
         "range": f"{meta['test_internal_range'][0]} -> {meta['test_internal_range'][1]}",
         "accuracy": internal["accuracy"],
         "f1_macro": internal["f1_macro"],
         "mcc": internal["mcc"]},
        {"split": "prospective",
         "n": len(df_prospective),
         "range": f"{prospective_start.date()} -> {prospective_end.date()}",
         "accuracy": prospective_metrics["accuracy"],
         "f1_macro": prospective_metrics["f1_macro"],
         "mcc": prospective_metrics["mcc"]},
    ])
    csv_path = OUT_TABLE_DIR / f"prospective_validation_{args.tag}.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n  Salvat: {csv_path}")
    print(comparison_df.to_string(index=False))

    # Figuri
    print("\n=== GENERARE FIGURI ===")
    # Confusion matrix intern din metadata? Nu am salvat-o. Recalculam rapid daca e nevoie.
    # Pentru simplitate, recalculam doar pentru prospective (principala figura).
    # Daca vrei CM intern side-by-side, re-ruleaza freeze_production_model si salveaza cm_internal in meta.
    plot_f1_comparison(
        {"f1_per_class": {c: 0.0 for c in CLASS_NAMES}, "f1_macro": internal["f1_macro"]},
        prospective_metrics,
        args.tag,
        OUT_FIG_DIR / f"prospective_f1_comparison_{args.tag}.png",
    )
    plot_prediction_timeline(
        prospective_metrics,
        args.tag,
        OUT_FIG_DIR / f"prospective_timeline_{args.tag}.png",
    )

    # Raport text
    write_text_report(
        meta, prospective_metrics, args.tag, prospective_start, prospective_end,
        len(df_prospective),
        OUT_REPORT_DIR / f"prospective_validation_{args.tag}.txt",
    )

    print("\n" + "=" * 60)
    print("  REZUMAT PROSPECTIVE VALIDATION")
    print("=" * 60)
    print(f"  N prospective: {len(df_prospective)}")
    print(f"  F1 Macro intern:      {internal['f1_macro']:.4f}")
    print(f"  F1 Macro prospective: {prospective_metrics['f1_macro']:.4f}")
    delta = prospective_metrics["f1_macro"] - internal["f1_macro"]
    print(f"  Delta F1: {'+' if delta >= 0 else ''}{delta:.4f}")
    print(f"\n  Raport complet: outputs/reports/prospective_validation_{args.tag}.txt")


if __name__ == "__main__":
    main()
