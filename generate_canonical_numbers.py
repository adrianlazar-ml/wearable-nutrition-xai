"""
Generate Canonical Numbers Reference
=====================================
Citeste TOATE fisierele CSV din outputs/tables/ + metadata din
models/frozen_meta_*.json si genereaza CIFRE_CANONICE.md,
un document structurat pe teme, usor de cautat cu Ctrl+F.

Utilitate: tinut deschis cand editezi disertatia. Pentru orice
cifra suspecta in text -> cauti in acest document -> verifici.

Autor: Adrian Lazar | Disertatie UPB 2026
"""

import json
from pathlib import Path

import pandas as pd


TABLES_DIR = Path("outputs/tables")
MODELS_DIR = Path("models")
OUTPUT_FILE = Path("CIFRE_CANONICE.md")


def fmt_pct(x, decimals=1):
    """Format float as percent with comma as decimal separator (RO style)."""
    return f"{x * 100:.{decimals}f}%".replace(".", ",")


def fmt_f(x, decimals=3):
    """Format float with comma (RO style)."""
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}f}".replace(".", ",")


def fmt_int(x):
    """Format int with thousand separator (RO style: punct)."""
    if pd.isna(x):
        return "N/A"
    return f"{int(x):,}".replace(",", ".")


def section(title, content):
    return f"\n## {title}\n\n{content}\n"


def load_meta():
    """Incarca metadata modelului frozen (cel mai recent)."""
    metas = sorted(MODELS_DIR.glob("frozen_meta_*.json"))
    if not metas:
        return None
    # Iei pe cel mai nou (dupa mtime)
    latest = max(metas, key=lambda p: p.stat().st_mtime)
    with open(latest, encoding="utf-8") as f:
        return json.load(f), latest.name


def build_dataset_section(meta):
    """Sectiune dataset sizes."""
    lines = []
    lines.append("### Zile calendaristice per sursa\n")
    # Aceste cifre nu sunt in metadata, le stiu din run-ul pipeline-ului
    lines.append("| Sursa | N zile |")
    lines.append("|---|---|")
    lines.append("| Oura Ring (combinat) | **1.009** |")
    lines.append("| Apple Watch JSON (Health Auto Export) | **1.083** |")
    lines.append("| Dataset combinat (left join pe date) | **1.009** |")
    lines.append("")
    lines.append("**Perioada acoperita:** 2023-06-09 -> 2026-03-30 (pipeline rulat 19.04.2026, cutoff 2026-03-31).")
    lines.append("")
    lines.append(f"### Dataset final pentru model\n")
    lines.append(f"- **Total zile utile** (dupa lag-1 + dropna label): **{meta['n_total_days']}**")
    lines.append(f"- **Train** (80% cronologic): **{meta['n_train']}** zile")
    lines.append(f"- **Test intern** (20% cronologic): **{meta['n_test_internal']}** zile")
    lines.append(f"  - Range: `{meta['test_internal_range'][0]}` -> `{meta['test_internal_range'][1]}`")
    return "\n".join(lines)


def build_class_distribution_section(meta):
    """Distributia claselor: full, train, test, post-SMOTE."""
    lines = []
    lines.append("Clasele sunt indexate 0=Rest, 1=Normal, 2=Active.\n")
    lines.append("### Distributie pe dataset complet (N=958)\n")
    total = sum(meta["class_distribution_full"].values())
    lines.append("| Clasa | N | % |")
    lines.append("|---|---|---|")
    for code, name in [("0", "Rest"), ("1", "Normal"), ("2", "Active")]:
        n = meta["class_distribution_full"][code]
        pct = n / total
        lines.append(f"| {name} | **{n}** | **{fmt_pct(pct)}** |")
    lines.append(f"| **Total** | **{total}** | **100%** |")
    lines.append("")

    lines.append("### Distributie train (pre-SMOTE, N=766)\n")
    lines.append("| Clasa | N |")
    lines.append("|---|---|")
    for code, name in [("0", "Rest"), ("1", "Normal"), ("2", "Active")]:
        n = meta["class_distribution_train_pre_smote"][code]
        lines.append(f"| {name} | **{n}** |")
    lines.append("")

    lines.append("### Distributie train (post-SMOTE k=3)\n")
    lines.append("Echilibrat la clasa majoritara:\n")
    for code, name in [("0", "Rest"), ("1", "Normal"), ("2", "Active")]:
        n = meta["class_distribution_train_post_smote"][code]
        lines.append(f"- {name}: **{n}**")
    lines.append("")

    lines.append("### Distributie test intern (N=192)\n")
    lines.append("| Clasa | N |")
    lines.append("|---|---|")
    for code, name in [("0", "Rest"), ("1", "Normal"), ("2", "Active")]:
        n = meta["class_distribution_test_internal"][code]
        lines.append(f"| {name} | **{n}** |")
    return "\n".join(lines)


def build_main_models_section():
    """Tabel principal RF vs SVM vs XGB."""
    df = pd.read_csv(TABLES_DIR / "valori_disertatie.csv")
    df = df[df["Model"] != "Baseline"].copy()

    lines = []
    lines.append("Rezultate pe 19 features lag-1, sursa JSON, test temporal 20%.\n")
    lines.append("| Model | F1 Macro | Accuracy | F1 Rest | F1 Normal | F1 Active | CV F1 (+/-) |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        cv_str = f"{fmt_f(r['cv_f1_mean'], 4)} (+/- {fmt_f(r['cv_f1_std'], 4)})"
        lines.append(
            f"| {r['Model']} | **{fmt_f(r['f1_macro'], 4)}** | "
            f"{fmt_pct(r['accuracy'])} | {fmt_f(r['f1_rest'], 2)} | "
            f"{fmt_f(r['f1_normal'], 2)} | {fmt_f(r['f1_active'], 2)} | {cv_str} |"
        )
    lines.append("")
    lines.append("**Baseline DummyClassifier (most_frequent):** F1 Macro = 0,281, Accuracy = 72,9%")
    lines.append("")
    lines.append("### Support pe test set\n")
    df_full = pd.read_csv(TABLES_DIR / "valori_disertatie.csv")
    r = df_full[df_full["Model"] == "Random Forest"].iloc[0]
    lines.append(f"- Rest: **{int(r['support_rest'])}**")
    lines.append(f"- Normal: **{int(r['support_normal'])}**")
    lines.append(f"- Active: **{int(r['support_active'])}**")
    lines.append(f"- Total: **{int(r['support_rest'] + r['support_normal'] + r['support_active'])}**")
    return "\n".join(lines)


def build_ablation_section():
    df = pd.read_csv(TABLES_DIR / "ablation_study_table.csv")
    lines = ["Configuratii RF, 19 features (toate cazurile folosesc lag-1, SMOTE k=3).\n"]
    lines.append("| Configuratie | N features | F1 Macro | Accuracy | F1 Rest | F1 Normal | F1 Active |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['config']} | {int(r['n_features'])} | **{fmt_f(r['f1_macro'], 3)}** | "
            f"{fmt_pct(r['accuracy'])} | {fmt_f(r['f1_rest'], 3)} | "
            f"{fmt_f(r['f1_normal'], 3)} | {fmt_f(r['f1_active'], 3)} |"
        )

    # Calcule delta pentru narativ
    f1_oura = df[df["config"].str.contains("Oura", na=False)].iloc[0]["f1_macro"]
    f1_aw = df[df["config"].str.contains("Apple", na=False)].iloc[0]["f1_macro"]
    f1_combined = df[df["config"].str.contains("Ambele", na=False)].iloc[0]["f1_macro"]
    lines.append("")
    lines.append("### Delta F1 Macro vs combined\n")
    lines.append(f"- Combined - Oura-only = +{(f1_combined - f1_oura) * 100:.1f} pp".replace(".", ","))
    lines.append(f"- Combined - AW-only = +{(f1_combined - f1_aw) * 100:.1f} pp".replace(".", ","))
    f1_active_aw = df[df["config"].str.contains("Apple", na=False)].iloc[0]["f1_active"]
    f1_active_combined = df[df["config"].str.contains("Ambele", na=False)].iloc[0]["f1_active"]
    ratio = f1_active_combined / max(f1_active_aw, 0.001)
    lines.append(f"- F1 Active: AW-only {fmt_f(f1_active_aw, 3)} -> Combined {fmt_f(f1_active_combined, 3)} (~{ratio:.1f}x)")
    return "\n".join(lines)


def build_threshold_section():
    df = pd.read_csv(TABLES_DIR / "sensitivity_threshold_table.csv")
    lines = ["Praguri pentru generarea labels (Rest/Normal/Active). RF, 19 features.\n"]
    lines.append("| Set | Rest readiness/sleep | Active readiness/sleep | Rest | Normal | Active | F1 Macro |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        rest_cond = f"<{int(r['r_rest'])}/<{int(r['s_rest'])}"
        active_cond = f">={int(r['r_active'])}/>={int(r['s_active'])}"
        best_marker = " ***" if "*" in str(r["threshold_set"]) else ""
        name = r["threshold_set"].replace(" *", "").replace("*", "")
        lines.append(
            f"| {name}{best_marker} | {rest_cond} | {active_cond} | "
            f"{int(r['n_rest'])} | {int(r['n_normal'])} | {int(r['n_active'])} | "
            f"**{fmt_f(r['f1_macro'], 3)}** |"
        )
    lines.append("")
    lines.append("**T3 selectat** ca optim. T4/T6 marcati cu `*` au F1 mai mare formal dar <5 eșantioane Active (inutile).")
    return "\n".join(lines)


def build_smote_section():
    df = pd.read_csv(TABLES_DIR / "smote_comparison_table.csv")
    lines = ["5 strategii de gestionare a dezechilibrului. RF, 19 features.\n"]
    lines.append("| Strategie | F1 Macro | Accuracy | F1 Rest | F1 Normal | F1 Active |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['strategy']} | **{fmt_f(r['f1_macro'], 3)}** | "
            f"{fmt_pct(r['accuracy'])} | {fmt_f(r['f1_rest'], 3)} | "
            f"{fmt_f(r['f1_normal'], 3)} | {fmt_f(r['f1_active'], 3)} |"
        )
    lines.append("")
    lines.append("**SMOTE k=3 ales** ca optim. k=5 produce F1 inferior (0,451).")
    return "\n".join(lines)


def build_confidence_section():
    df = pd.read_csv(TABLES_DIR / "confidence_threshold_analysis.csv")
    lines = ["Safety layer: daca max(predict_proba) < threshold, fallback la Normal.\n"]
    lines.append("| Threshold | F1 Macro | Accuracy | Coverage | N overridden | Mean cal error (kcal) | Max cal error |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        label = f" [{r['label']}]" if pd.notna(r["label"]) and r["label"] != "" else ""
        lines.append(
            f"| {r['threshold']}{label} | {fmt_f(r['f1_macro'], 3)} | "
            f"{fmt_pct(r['accuracy'])} | {fmt_pct(r['coverage_pct'] / 100)} | "
            f"{int(r['n_overridden'])} | {fmt_f(r['mean_cal_error'], 1)} | {int(r['max_cal_error'])} |"
        )
    lines.append("")
    lines.append("**Threshold 0.5 ales in cod** (demo_end_to_end_v2 + streamlit).")
    lines.append("Notat: flag-ul 'RECOMANDAT' in CSV la 0.6 este vestigial si ar trebui actualizat (vezi raport sanity check).")
    return "\n".join(lines)


def build_tuning_section():
    df = pd.read_csv(TABLES_DIR / "tuning_comparison.csv")
    df_bp = pd.read_csv(TABLES_DIR / "tuning_best_params.csv")

    lines = ["RandomizedSearchCV, 50 iteratii, 5-fold TimeSeriesSplit CV, metrica = F1 Macro.\n"]
    lines.append("### Rezultate tuning\n")
    lines.append("| Model | Default F1 CV | Tuned F1 CV | Delta (pp) |")
    lines.append("|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['Model']} | {fmt_f(r['Default_F1_Macro'], 4)} | "
            f"{fmt_f(r['Tuned_F1_Macro'], 4)} | **+{r['Delta_pp']:.2f}** |"
        )
    lines.append("")
    lines.append("**Interpretare:** Tuning post-fix CV leakage: RF +2,49pp, SVM +4,47pp, XGB +6,87pp. Dupa tuning, pe CV F1 Macro XGB (0,403) si SVM (0,401) depasesc RF (0,362), dar RF ramane best pe test F1 Macro (0,506) + QWK. Concluzia 'RF best overall' e nuantata: best pe metrica primara (test F1 Macro), competitiv pe CV.")
    lines.append("")
    lines.append("### Best params gasiti\n")
    for _, r in df_bp.iterrows():
        params = {k: r[k] for k in df_bp.columns if k != "Model" and pd.notna(r[k])}
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        lines.append(f"- **{r['Model']}:** {params_str}")
    return "\n".join(lines)


def build_new_models_section():
    df = pd.read_csv(TABLES_DIR / "new_models_comparison.csv")
    lines = ["Extensie: LR liniar si LightGBM comparati cu RF/XGB.\n"]
    lines.append("| Model | F1 Macro | Accuracy | CV F1 mean | CV F1 std | Train time (s) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['Model']} | **{fmt_f(r['f1_macro'], 3)}** | "
            f"{fmt_pct(r['accuracy'])} | {fmt_f(r['cv_f1_mean'], 4)} | "
            f"{fmt_f(r['cv_f1_std'], 4)} | {fmt_f(r['train_time'], 3)} |"
        )
    lines.append("")
    lines.append("**LR F1=0,342** confirma non-liniaritatea problemei. **LightGBM F1=0,471** similar cu XGBoost, RF ramane best.")
    return "\n".join(lines)


def build_mcc_qwk_section():
    df = pd.read_csv(TABLES_DIR / "mcc_qwk_results.csv")
    lines = ["Metrici robuste la dezechilibru si ordinalitate.\n"]
    lines.append("| Model | F1 Macro | MCC | QWK | Accuracy |")
    lines.append("|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['Model']} | **{fmt_f(r['F1 Macro'], 4)}** | "
            f"**{fmt_f(r['MCC'], 4)}** | **{fmt_f(r['QWK'], 4)}** | {fmt_f(r['Accuracy'], 4)} |"
        )
    lines.append("")
    lines.append("**MCC** (Matthews Correlation Coefficient): robust la dezechilibru. **QWK** (Quadratic Weighted Kappa): captureaza ordinalitatea Rest<Normal<Active.")
    return "\n".join(lines)


def build_calibration_section():
    df = pd.read_csv(TABLES_DIR / "calibration_brier_scores.csv")
    lines = ["Brier Score per clasa (mai mic = mai bine calibrat). RF, 19 features.\n"]
    lines.append("| Model | Rest | Normal | Active | Average |")
    lines.append("|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['Model']} | {fmt_f(r['Rest'], 3)} | "
            f"{fmt_f(r['Normal'], 3)} | {fmt_f(r['Active'], 3)} | "
            f"**{fmt_f(r['Average'], 3)}** |"
        )
    lines.append("")
    lines.append("Brier < 0,20 considerat bine calibrat. Valori confirma confidence threshold 0.50.")
    return "\n".join(lines)


def build_rolling_section():
    df = pd.read_csv(TABLES_DIR / "rolling_features_comparison.csv")
    lines = ["Baseline lag-1 vs extensie cu rolling means (3, 7, 14 zile).\n"]
    lines.append("| Config | N features | F1 Macro | CV F1 mean |")
    lines.append("|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['Config']} | {int(r['n_features'])} | "
            f"**{fmt_f(r['f1_macro_test'], 3)}** | {fmt_f(r['cv_f1_mean'], 4)} |"
        )
    delta = (df.iloc[0]["f1_macro_test"] - df.iloc[1]["f1_macro_test"]) * 100
    lines.append("")
    lines.append(f"**Rolling features DAUNEAZA:** -{delta:.1f} pp pe test.".replace(".", ","))
    lines.append("Justificare: pe N=958 (766 train) cele 36 features produc overfitting. Learning curves nu satureaza - inca cresc la ~131 zile (endpoint-ul range-ului TimeSeriesSplit pe train), deci semnalul e limitat de volumul de date, nu de numarul de features.")
    return "\n".join(lines)


def build_circularity_section():
    df = pd.read_csv(TABLES_DIR / "circularity_sanity_check.csv")
    lines = ["Sanity check: cat contribuie features 'potential circulare' (derivate din Oura scores) la clasificare?\n"]
    lines.append("| Variant | N features | F1 Macro | CV F1 mean |")
    lines.append("|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['variant']} | {int(r['n_features'])} | "
            f"**{fmt_f(r['f1_macro'], 4)}** | {fmt_f(r['f1_cv_mean'], 4)} |"
        )
    lines.append("")
    baseline = df.iloc[0]["f1_macro"]
    strict = df.iloc[2]["f1_macro"]
    delta = (baseline - strict) * 100
    lines.append(f"**Drop baseline -> strict (doar fiziologice brute):** -{delta:.1f} pp.".replace(".", ","))
    lines.append(f"**Semi-strict (fara readiness+sleep) creste marginal:** {(df.iloc[1]['f1_macro'] - baseline) * 100:+.1f} pp.".replace(".", ","))
    lines.append("")
    lines.append("**Interpretare:** circularitatea nu e o problema majora. Modelul extrage semnal din features fiziologice brute (HRV, RHR, HR, pasi, calorii), nu doar din derivate.")
    return "\n".join(lines)


def build_model_config_section(meta):
    lines = []
    lines.append("### Hiperparametri finali (RF, model de productie)\n")
    cfg = meta["model_config"]
    for k, v in cfg.items():
        lines.append(f"- `{k}` = **{v}**")
    lines.append("")
    lines.append("### Preprocessing\n")
    pre = meta["preprocessing"]
    for k, v in pre.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("### Scheme de etichetare T3\n")
    lines.append(f"- Rest: `readiness_score < 60 OR sleep_score < 60` -> label **0**")
    lines.append(f"- Active: `readiness_score >= 85 AND sleep_score >= 75` -> label **2**")
    lines.append(f"- Normal: restul zilelor -> label **1**")
    lines.append("")
    lines.append("### Features (19 lag-1)\n")
    lines.append("**Oura (11):** " + ", ".join(f"`{f}`" for f in meta["features"][:11]))
    lines.append("")
    lines.append("**Apple Watch (8):** " + ", ".join(f"`{f}`" for f in meta["features"][11:]))
    return "\n".join(lines)


def build_reproducibility_section(meta, meta_filename):
    lines = [
        "Pentru dovada reproducibilitatii (util daca comisia AI/ML intreaba):\n"
    ]
    lines.append(f"- **Frozen tag:** `{meta['tag']}`")
    lines.append(f"- **Frozen at (UTC):** `{meta['frozen_at_utc']}`")
    lines.append(f"- **Cutoff date** (data <= care s-au luat datele): `{meta.get('cutoff_date', meta.get('train_end_date_requested', 'N/A'))}`")
    lines.append(f"- **Last day in dataset** (efectiv): `{meta.get('last_day_in_dataset', meta.get('train_end_date_effective', 'N/A'))}`")
    lines.append(f"- **Train internal end** (ultima zi din 80% train): `{meta.get('train_internal_end', meta.get('train_end_date_effective', 'N/A'))}`")
    lines.append(f"- **Test internal range:** `{meta['test_internal_range'][0]}` -> `{meta['test_internal_range'][1]}`")
    lines.append(f"- **Dataset SHA256 (first 16 char):** `{meta['dataset_hash_sha256_16']}`")
    lines.append(f"- **Metadata file:** `models/{meta_filename}`")
    lines.append("")
    lines.append("### Versiuni librarii\n")
    env = meta["environment"]
    for k, v in env.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("### Artefacte salvate\n")
    for k, v in meta["artifacts"].items():
        lines.append(f"- {k}: `{v}`")
    return "\n".join(lines)


def main():
    result = load_meta()
    if result is None:
        raise FileNotFoundError(
            "Nu am gasit nicio metadata frozen. Ruleaza mai intai freeze_production_model.py."
        )
    meta, meta_filename = result

    sections = []
    sections.append("# Cifre canonice — Disertație Adrian Lazar 2026\n")
    sections.append(f"Generat automat din `outputs/tables/` si `models/{meta_filename}`.\n")
    sections.append(f"**Cutoff date:** {meta.get('cutoff_date', meta.get('train_end_date_requested', 'N/A'))}  |  "
                    f"**Last day in dataset:** {meta.get('last_day_in_dataset', meta.get('train_end_date_effective', 'N/A'))}  |  "
                    f"**Dataset hash:** `{meta['dataset_hash_sha256_16']}`\n")
    sections.append("---")
    sections.append("""
## Cuprins

1. [Dataset sizes](#1-dataset-sizes)
2. [Distribuții clase](#2-distribuții-clase)
3. [Rezultate modele principale (RF / SVM / XGB)](#3-rezultate-modele-principale)
4. [Ablation study (Oura / AW / Combined)](#4-ablation-study)
5. [Threshold sensitivity (T1–T7)](#5-threshold-sensitivity)
6. [SMOTE comparison](#6-smote-comparison)
7. [Confidence threshold](#7-confidence-threshold)
8. [Hyperparameter tuning](#8-hyperparameter-tuning)
9. [Modele extra (LR, LightGBM)](#9-modele-extra)
10. [MCC & QWK](#10-mcc--qwk)
11. [Calibration (Brier)](#11-calibration)
12. [Rolling features (respins)](#12-rolling-features-respins)
13. [Circularity sanity check](#13-circularity-sanity-check)
14. [Model config & features](#14-model-config--features)
15. [Reproducibilitate](#15-reproducibilitate)
""")
    sections.append("---")

    sections.append(section("1. Dataset sizes", build_dataset_section(meta)))
    sections.append(section("2. Distribuții clase", build_class_distribution_section(meta)))
    sections.append(section("3. Rezultate modele principale", build_main_models_section()))
    sections.append(section("4. Ablation study", build_ablation_section()))
    sections.append(section("5. Threshold sensitivity", build_threshold_section()))
    sections.append(section("6. SMOTE comparison", build_smote_section()))
    sections.append(section("7. Confidence threshold", build_confidence_section()))
    sections.append(section("8. Hyperparameter tuning", build_tuning_section()))
    sections.append(section("9. Modele extra", build_new_models_section()))
    sections.append(section("10. MCC & QWK", build_mcc_qwk_section()))
    sections.append(section("11. Calibration", build_calibration_section()))
    sections.append(section("12. Rolling features (respins)", build_rolling_section()))
    sections.append(section("13. Circularity sanity check", build_circularity_section()))
    sections.append(section("14. Model config & features", build_model_config_section(meta)))
    sections.append(section("15. Reproducibilitate", build_reproducibility_section(meta, meta_filename)))

    content = "\n".join(sections)
    OUTPUT_FILE.write_text(content, encoding="utf-8")

    # Stats
    n_lines = content.count("\n")
    n_tables = content.count("\n| ")
    print(f"Generat: {OUTPUT_FILE}")
    print(f"  Linii: {n_lines}")
    print(f"  Randuri tabel: ~{n_tables}")
    print(f"  Sectiuni: 15")


if __name__ == "__main__":
    main()
