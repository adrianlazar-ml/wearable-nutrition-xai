"""
Streamlit UI V2 - Sistem AI pentru Recomandări Nutriționale Personalizate
=========================================================================
Disertație: AI pentru nutriție personalizată folosind date din wearables
Autor: Adrian Lazar | Coordonator: Prof. Cătălin Buiu, UPB
Apărare: Iunie-Iulie 2026

V2 — Schimbări față de V1:
  - Sursă date primară: JSON Extended (19 features din Health Auto Export)
  - Explicabilitate: 3 taburi (Best Model / Comparație Algoritmi / Calitate Date)
  - Metrici actualizate post-fix rhr_avg (30.03.2026): RF best F1 Macro 0.506
  - Suport RF (cel mai bun model pe 19 features post-fix)
  - Grafice SHAP: organizate in outputs/figures/ (post-reorganizare 31.03.2026)

Pagini:
  1. Profil Utilizator — date biometrice, calcul BMR/TDEE
  2. Dashboard Zilnic — clasificare ML + recomandare nutrițională
  3. Explicabilitate — SHAP organized pe 3 taburi tematice

Rulare:
  pip install streamlit plotly --break-system-packages
  streamlit run streamlit_app_v2.py

Cerințe: Graficele SHAP trebuie generate anterior cu:
  - cap5_compare_algorithms.py (XML)
  - cap5_compare_algorithms_json.py (JSON 16 feat)
  - cap5_compare_algorithms_json_ext.py (JSON 19 feat)
  - generate_waterfall_svm.py + fix_svm_barplot.py (SVM SHAP)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

from nutrition_recommender import get_nutrition_recommendation

# ============================================================
# CONFIG & PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="AI Nutrition – Oura + Apple Watch",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data"
MODEL_DIR = "models"
AW_JSON_DIR = "apple_watch_api"

# Post-reorganizare (31.03.2026): figurile s-au mutat din data/ in outputs/figures/
FIG_EXT = "outputs/figures/json_extended_pipeline"
FIG_JSON = "outputs/figures/json_pipeline"
FIG_XML = "outputs/figures/xml_pipeline"
FIG_EXP = "outputs/figures/experiments"

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("AI Nutrition System")
st.sidebar.caption("Oura Ring + Apple Watch → ML → Recomandări")

page = st.sidebar.radio(
    "Navigare",
    ["📊 Profil Utilizator", "🎯 Dashboard Zilnic", "🔍 Explicabilitate"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Disertație Master UPB**  \n"
    "Adrian Lazar  \n"
    "Coord: Prof. Cătălin Buiu"
)


# ============================================================
# HELPERS
# ============================================================
# --- Features definitions ---
LAG_FEATURES_16 = [
    "readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
    "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
    "steps_oura", "respiratory_rate", "temp_deviation",
    "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2",
]
LAG_FEATURES_19 = LAG_FEATURES_16 + [
    "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate",
]
FEATURES_16 = [f"{f}_lag1" for f in LAG_FEATURES_16]
FEATURES_19 = [f"{f}_lag1" for f in LAG_FEATURES_19]


def calculate_bmr(weight_kg, height_cm, age, gender):
    """Mifflin-St Jeor equation."""
    if gender == "male":
        return round(10 * weight_kg + 6.25 * height_cm - 5 * age + 5)
    return round(10 * weight_kg + 6.25 * height_cm - 5 * age - 161)


def calculate_tdee(bmr, activity_level):
    multipliers = {
        "Sedentar (birou, fără sport)": 1.2,
        "Ușor activ (1-2 antrenamente/săpt)": 1.375,
        "Moderat activ (3-5 antrenamente/săpt)": 1.55,
        "Foarte activ (6-7 antrenamente/săpt)": 1.725,
    }
    return round(bmr * multipliers.get(activity_level, 1.2))


# get_nutrition_recommendation_local ELIMINAT (FND-203): logica de recomandare e
# consolidata in nutrition_recommender.get_nutrition_recommendation (import sus).
# UI consuma cheia 'key_foods' din modul (vezi rec["key_foods"]).


def try_load_dataset():
    """
    Încearcă să încarce dataset-ul, cu prioritate:
    1. JSON Extended (19 features) — cel mai bun
    2. JSON 16 features — fallback
    3. XML 16 features — legacy
    Returnează (df, source_name, features_list) sau (None, None, None).
    """
    candidates = [
        ("dataset_combinat_json_extended.csv", "JSON Extended (19 features)", FEATURES_19),
        ("dataset_combinat_json.csv", "JSON (16 features)", FEATURES_16),
        ("dataset_combinat.csv", "XML (16 features, legacy)", FEATURES_16),
    ]
    for filename, source_name, features in candidates:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df, source_name, features
    return None, None, None


def try_load_model():
    """Încearcă să încarce modelul salvat."""
    try:
        import joblib
        rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
        imp = joblib.load(os.path.join(MODEL_DIR, "imputer.joblib"))
        return rf, imp
    except Exception:
        return None, None


def show_image_if_exists(path, caption=None):
    """Afișează o imagine dacă fișierul există. Returnează True dacă există."""
    if os.path.exists(path):
        st.image(path, width="stretch")
        if caption:
            st.caption(caption)
        return True
    return False


# ============================================================
# STATE: Profil utilizator (persistent în sesiune)
# ============================================================
if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "Adrian",
        "weight_kg": 119.9,
        "height_cm": 187.0,
        "age": 41,
        "gender": "male",
        "body_fat_pct": 36.3,
        "goal": "fat_loss",
        "activity_level": "Ușor activ (1-2 antrenamente/săpt)",
    }


def get_lean_mass(p):
    return round(p["weight_kg"] * (1 - p["body_fat_pct"] / 100), 2)


# ============================================================
# PAGE 1: PROFIL UTILIZATOR
# ============================================================
if page == "📊 Profil Utilizator":
    st.title("Profil Utilizator")
    st.markdown(
        "Configurează datele biometrice. Acestea sunt folosite pentru "
        "calculul recomandărilor nutriționale personalizate."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Date Biometrice")
        p = st.session_state.profile

        p["name"] = st.text_input("Nume", value=p["name"])
        p["weight_kg"] = st.number_input(
            "Greutate (kg)", min_value=40.0, max_value=250.0,
            value=p["weight_kg"], step=0.1, format="%.1f",
        )
        p["height_cm"] = st.number_input(
            "Înălțime (cm)", min_value=140.0, max_value=220.0,
            value=p["height_cm"], step=0.5, format="%.1f",
        )
        p["age"] = st.number_input(
            "Vârstă", min_value=18, max_value=100, value=p["age"]
        )
        p["gender"] = st.selectbox(
            "Gen", ["male", "female"],
            index=0 if p["gender"] == "male" else 1,
        )
        p["body_fat_pct"] = st.number_input(
            "Body Fat (%)", min_value=5.0, max_value=60.0,
            value=p["body_fat_pct"], step=0.1, format="%.1f",
        )
        p["goal"] = st.selectbox(
            "Obiectiv", ["fat_loss", "maintenance", "muscle_gain"],
            index=["fat_loss", "maintenance", "muscle_gain"].index(p["goal"]),
        )
        p["activity_level"] = st.selectbox(
            "Nivel activitate",
            [
                "Sedentar (birou, fără sport)",
                "Ușor activ (1-2 antrenamente/săpt)",
                "Moderat activ (3-5 antrenamente/săpt)",
                "Foarte activ (6-7 antrenamente/săpt)",
            ],
            index=1,
        )

    with col2:
        st.subheader("Metrici Derivate")
        lean = get_lean_mass(p)
        bmr = calculate_bmr(p["weight_kg"], p["height_cm"], p["age"], p["gender"])
        tdee = calculate_tdee(bmr, p["activity_level"])
        bmi = round(p["weight_kg"] / (p["height_cm"] / 100) ** 2, 1)
        base_cal = round(25 * p["weight_kg"])

        col_a, col_b = st.columns(2)
        col_a.metric("BMR (Mifflin-St Jeor)", f"{bmr} kcal")
        col_b.metric("TDEE estimat", f"{tdee} kcal")

        col_c, col_d = st.columns(2)
        col_c.metric("BMI", f"{bmi}")
        col_d.metric("Lean Mass", f"{lean} kg")

        col_e, col_f = st.columns(2)
        col_e.metric("Fat Mass", f"{round(p['weight_kg'] - lean, 1)} kg")
        col_f.metric("Base Cal (AI4Food)", f"{base_cal} kcal")

        st.markdown("---")
        st.markdown("**Formula AI4Food** (Lacruz-Pleguezuelos et al., 2025)")
        st.markdown(
            f"25 kcal/kg × {p['weight_kg']} kg = **{base_cal} kcal/zi**  \n"
            f"Comparat cu TDEE ({tdee} kcal): formula AI4Food are limitare pe "
            f"obezitate (poate fi întreținere/surplus, nu deficit) — vezi Cap.7."
        )

        st.markdown("---")
        st.subheader("Comparație Recomandări per Tip de Zi")

        full_profile = {**p, "lean_mass_kg": lean, "bmr_kcal": bmr}

        comp_data = []
        for label_id, label_name in [(0, "Rest"), (1, "Normal"), (2, "Active")]:
            rec = get_nutrition_recommendation(label_id, full_profile)
            comp_data.append({
                "Tip Zi": label_name,
                "Calorii": rec["calories_target"],
                "Proteine (g)": rec["protein_g"],
                "Carbs (g)": rec["carbs_g"],
                "Grăsimi (g)": rec["fat_g"],
                "Hidratare (ml)": rec["hydration_ml"],
            })

        st.dataframe(
            pd.DataFrame(comp_data).set_index("Tip Zi"),
            width="stretch",
        )


# ============================================================
# PAGE 2: DASHBOARD ZILNIC
# ============================================================
elif page == "🎯 Dashboard Zilnic":
    st.title("Dashboard Zilnic")
    st.markdown(
        "Clasificare ML a stării fiziologice + recomandare nutrițională personalizată."
    )

    p = st.session_state.profile
    lean = get_lean_mass(p)
    bmr = calculate_bmr(p["weight_kg"], p["height_cm"], p["age"], p["gender"])
    full_profile = {**p, "lean_mass_kg": lean, "bmr_kcal": bmr}

    input_mode = st.radio(
        "Sursă clasificare",
        ["Manual (selectează tipul zilei)", "Automat (model ML pe date reale)"],
        horizontal=True,
    )

    if input_mode == "Manual (selectează tipul zilei)":
        st.markdown("---")
        col_m1, col_m2 = st.columns([1, 2])

        with col_m1:
            readiness = st.slider("Readiness Score (Oura)", 0, 100, 72)
            sleep = st.slider("Sleep Score (Oura)", 0, 100, 78)

            if readiness < 60 or sleep < 60:
                auto_label = 0
            elif readiness >= 85 and sleep >= 75:
                auto_label = 2
            else:
                auto_label = 1

            label_names = {0: "Rest", 1: "Normal", 2: "Active"}
            label_colors = {0: "🔴", 1: "🟡", 2: "🟢"}
            st.markdown(
                f"### Clasificare: {label_colors[auto_label]} {label_names[auto_label]} Day"
            )
            st.caption(
                "Praguri: Rest (<60 readiness sau <60 sleep) | "
                "Active (≥85 readiness ȘI ≥75 sleep) | Normal (restul)"
            )

        with col_m2:
            rec = get_nutrition_recommendation(auto_label, full_profile)

            st.subheader(f"Recomandare: {rec['day_type']}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Calorii", f"{rec['calories_target']} kcal")
            c2.metric("Proteine", f"{rec['protein_g']}g")
            c3.metric("Carbohidrați", f"{rec['carbs_g']}g")
            c4.metric("Grăsimi", f"{rec['fat_g']}g")

            try:
                import plotly.graph_objects as go

                fig = go.Figure(data=[go.Pie(
                    labels=["Proteine", "Carbohidrați", "Grăsimi"],
                    values=[
                        rec["protein_g"] * 4,
                        rec["carbs_g"] * 4,
                        rec["fat_g"] * 9,
                    ],
                    hole=0.4,
                    marker_colors=["#5DCAA5", "#F0997B", "#85B7EB"],
                    textinfo="label+percent",
                    textfont_size=13,
                )])
                fig.update_layout(
                    height=280,
                    margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")
            except ImportError:
                st.info("Instalează `plotly` pentru grafice: `pip install plotly`")

            st.markdown(f"**Hidratare:** {rec['hydration_ml']} ml")
            st.markdown(f"**Timing:** {rec['timing']}")
            st.markdown(f"**Raționale:** {rec['rationale']}")

            with st.expander("Alimente recomandate"):
                for food in rec["key_foods"]:
                    st.markdown(f"- {food}")

            with st.expander("De evitat"):
                for item in rec["avoid"]:
                    st.markdown(f"- {item}")

    else:
        # Mod automat: model ML pe date reale
        st.markdown("---")
        rf, imputer = try_load_model()

        if rf is None:
            st.warning(
                "Modelul ML nu a fost găsit. "
                "Rulează `python demo_end_to_end_v2.py` pentru a genera "
                "`models/rf_model.joblib` și `models/imputer.joblib`."
            )
            st.info(
                "Alternativ, folosește modul **Manual** pentru a testa "
                "recomandările cu scoruri introduse manual."
            )
        else:
            dataset, source_name, features_list = try_load_dataset()
            if dataset is None:
                st.warning(
                    "Dataset-ul nu a fost găsit. Rulează unul din pipeline-uri:\n"
                    "- `python cap5_pipeline_json_extended.py` (recomandat, 19 features)\n"
                    "- `python cap5_pipeline_json.py` (JSON 16 features)"
                )
            else:
                st.success(
                    f"Model încărcat. Dataset: **{len(dataset)} zile** "
                    f"— sursă: {source_name}"
                )

                # Detectăm features din dataset
                features_in_dataset = [c for c in dataset.columns if c.endswith("_lag1")]
                model_n_features = rf.n_features_in_

                # Verificăm compatibilitatea model-dataset
                if model_n_features != len(features_in_dataset):
                    st.warning(
                        f"Modelul a fost antrenat pe {model_n_features} features, "
                        f"dar dataset-ul are {len(features_in_dataset)} features. "
                        f"Re-antrenează modelul cu `demo_end_to_end.py` pe "
                        f"același dataset."
                    )
                else:
                    last_row = dataset.iloc[-1]
                    feat_values = pd.DataFrame(
                        [last_row[features_in_dataset].values],
                        columns=features_in_dataset,
                    )
                    feat_imp = pd.DataFrame(
                        imputer.transform(feat_values),
                        columns=features_in_dataset,
                    )

                    label = int(rf.predict(feat_imp)[0])
                    proba = rf.predict_proba(feat_imp)[0]

                    # Safety layer: confidence threshold
                    CONFIDENCE_THRESHOLD = 0.50
                    max_conf = float(np.max(proba))
                    original_label = label
                    if max_conf < CONFIDENCE_THRESHOLD:
                        label = 1  # Fallback Normal

                    label_names = {0: "Rest", 1: "Normal", 2: "Active"}
                    label_colors = {0: "🔴", 1: "🟡", 2: "🟢"}

                    # Detectăm data predicției din oura_combined.csv
                    pred_date_str = "N/A"
                    oura_path = os.path.join(DATA_DIR, "oura_combined.csv")
                    if os.path.exists(oura_path):
                        try:
                            oura_dates = pd.read_csv(oura_path, usecols=["date"])
                            oura_dates["date"] = pd.to_datetime(oura_dates["date"])
                            # Ultima zi din dataset = ultima zi cu date complete
                            pred_date = oura_dates["date"].iloc[-1]
                            pred_date_str = pred_date.strftime("%d.%m.%Y")
                        except Exception:
                            pass

                    today_str = datetime.now().strftime("%d.%m.%Y")
                    if pred_date_str != "N/A" and pred_date_str != today_str:
                        st.info(
                            f"Predicție pentru **{pred_date_str}** "
                            f"(ultima zi cu date complete în dataset). "
                            f"Pentru date actualizate, rulează "
                            f"`python parse_health_auto_export.py` și "
                            f"`python demo_end_to_end_v2.py`."
                        )

                    st.markdown(
                        f"### Predicție ML: {label_colors[label]} "
                        f"{label_names[label]} Day"
                    )

                    if label != original_label:
                        st.warning(
                            f"Confidence scăzută ({max_conf:.1%} < {CONFIDENCE_THRESHOLD:.0%}). "
                            f"Predicția originală ({label_names[original_label]}) a fost "
                            f"suprascrisă cu **Normal** (fallback conservator, ±200 kcal max)."
                        )
                    else:
                        st.caption(
                            f"Confidence: {max_conf:.1%} "
                            f"(threshold: {CONFIDENCE_THRESHOLD:.0%})"
                        )

                    pc1, pc2, pc3 = st.columns(3)
                    pc1.metric("P(Rest)", f"{proba[0]:.1%}")
                    pc2.metric("P(Normal)", f"{proba[1]:.1%}")
                    pc3.metric("P(Active)", f"{proba[2]:.1%}")

                    rec = get_nutrition_recommendation(label, full_profile)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Calorii (kcal)", f"{rec['calories_target']}")
                    c2.metric("Proteine (g)", f"{rec['protein_g']}")
                    c3.metric("Carbs / Grăsimi (g)",
                              f"{rec['carbs_g']} / {rec['fat_g']}")

                    st.markdown(f"**Raționale:** {rec['rationale']}")


# ============================================================
# PAGE 3: EXPLICABILITATE (3 TABURI)
# ============================================================
elif page == "🔍 Explicabilitate":
    st.title("Explicabilitate Model (SHAP)")
    st.markdown(
        "Transparență în predicțiile ML: ce features influențează "
        "clasificarea Rest/Normal/Active?"
    )

    tab1, tab2, tab3 = st.tabs([
        "🏆 Cel Mai Bun Model",
        "⚖️ Comparație Algoritmi",
        "🔬 Calitate Date (XML vs JSON)",
    ])

    # ==========================================================
    # TAB 1: BEST MODEL — RF Extended (19 features)
    # ==========================================================
    with tab1:
        st.subheader("Random Forest — 19 features — F1 Macro: 50.6%")
        st.markdown(
            "Cel mai performant model (post-fix rhr_avg, 30.03.2026), antrenat pe "
            "19 features (16 originale + sleep deep, sleep REM, respiratory rate "
            "din Apple Watch). SHAP calculat cu TreeExplainer."
        )

        st.markdown("---")
        st.markdown("**SHAP Feature Importance — Random Forest (TreeExplainer)**")
        found = show_image_if_exists(
            f"{FIG_EXT}/ext_shap_bar_random_forest.png",
            "Mean |SHAP value| per feature — Random Forest (modelul principal), 19 features. "
            "aw_hr_mean rămâne #1, confirmând complementaritatea Oura + Apple Watch."
        )

        st.markdown("---")
        st.markdown("**SHAP Waterfall — Random Forest (per clasă)**")
        found_wf = show_image_if_exists(
            f"{FIG_EXT}/ext_shap_waterfall_all_random_forest.png",
            "Contribuția fiecărui feature la decizia de clasificare "
            "pentru o zi reprezentativă din fiecare clasă (Rest/Normal/Active)."
        )

        # XGBoost — comparativ (figuri regenerate Sprint 2)
        with st.expander("XGBoost — detalii SHAP (summary, dependence) [comparativ]"):
            show_image_if_exists(
                f"{FIG_EXT}/ext_shap_summary_xgboost.png",
                "SHAP Summary (beeswarm) — XGBoost, 19 features."
            )
            show_image_if_exists(
                f"{FIG_EXT}/ext_shap_dependence_xgboost.png",
                "SHAP Dependence plots — top 5 features."
            )

        # Metrici model — tabel actualizat cu rezultate reale
        st.markdown("---")
        st.subheader("Metrici Comparative")

        # D14a: cifre dinamice din CSV (post-Sprint 2), nu hardcodate
        _TBL = "outputs/tables"
        try:
            _vdf = pd.read_csv(f"{_TBL}/valori_disertatie.csv").set_index("Model")
            _ndf = pd.read_csv(f"{_TBL}/new_models_comparison.csv").set_index("Model")

            def _cvf1(df, m):
                return f"{df.loc[m, 'cv_f1_mean'] * 100:.1f}%"

            def _f1a(df, m):
                return f"{df.loc[m, 'f1_active']:.2f}"

            metrics_data = {
                "Algoritm": [
                    "Random Forest — 19 feat",
                    "SVM (RBF) — 19 feat",
                    "XGBoost — 19 feat",
                    "LightGBM — 19 feat",
                    "Logistic Regression — 19 feat",
                    "Baseline (majority)",
                ],
                "CV F1 Macro": [
                    _cvf1(_vdf, "Random Forest"),
                    _cvf1(_vdf, "SVM (RBF)"),
                    _cvf1(_vdf, "XGBoost"),
                    _cvf1(_ndf, "LightGBM"),
                    _cvf1(_ndf, "Logistic Regression"),
                    "~28%",
                ],
                "F1 Active": [
                    _f1a(_vdf, "Random Forest"),
                    _f1a(_vdf, "SVM (RBF)"),
                    _f1a(_vdf, "XGBoost"),
                    _f1a(_ndf, "LightGBM"),
                    _f1a(_ndf, "Logistic Regression"),
                    "0.00",
                ],
                "Observații": [
                    "Best pe test F1 Macro (50,6%) + QWK; nuanțat pe CV",
                    "Best calibrat (Brier 0,156)",
                    "Beneficiază cel mai mult de tuning (+6,87pp)",
                    "Similar XGBoost, gradient boosting alternativ",
                    "Slab — confirmă non-liniaritatea problemei",
                    "Prezice mereu clasa dominantă (Normal)",
                ],
            }
        except (FileNotFoundError, KeyError) as _e:
            metrics_data = {
                "Algoritm": ["(CSV-uri pipeline lipsă)"],
                "CV F1 Macro": [f"rulează pipeline-ul ({type(_e).__name__})"],
                "F1 Active": ["—"],
                "Observații": ["generează outputs/tables/valori_disertatie.csv + new_models_comparison.csv"],
            }
        st.dataframe(
            pd.DataFrame(metrics_data).set_index("Algoritm"),
            width="stretch",
        )
        st.caption(
            "F1 Macro e metrica principală — corectă pentru class imbalance sever "
            "(Rest: ~202, Normal: ~710, Active: ~46). "
            "F1 Active arată capacitatea modelului de a detecta clasa minoritară."
        )

    # ==========================================================
    # TAB 2: COMPARAȚIE ALGORITMI (RF vs XGBoost vs SVM)
    # ==========================================================
    with tab2:
        st.subheader("Comparație: RF vs XGBoost vs SVM")
        st.markdown(
            "Evaluare pe split temporal 80/20 cu TimeSeriesSplit CV (5-fold), "
            "pipeline canonic Extended (19 features, JSON + sleep stages). "
            "Sub-taburile pre-canonice JSON 16 / XML 16 au fost eliminate (D13): "
            "erau dinainte de fix-ul rhr_avg + CV leakage, deci necomparabile apple-to-apple."
        )

        st.markdown("**19 features — JSON + sleep stages (post-fix Sprint 1-2)**")
        show_image_if_exists(
            f"{FIG_EXT}/ext_confusion_matrices.png",
            "Matrici de confuzie — RF / XGBoost / SVM pe 19 features."
        )
        show_image_if_exists(
            f"{FIG_EXT}/ext_feature_importance_comparison.png",
            "Feature importance: RF vs XGBoost (19 features)."
        )
        show_image_if_exists(
            f"{FIG_EXT}/ext_pdp_top3_xgboost.png",
            "PDP — Top 3 features, XGBoost (19 features)."
        )
        show_image_if_exists(
            f"{FIG_EXT}/ext_pdp_multiclass_xgboost.png",
            "PDP Multi-class — feature dominant (19 features)."
        )

    # ==========================================================
    # TAB 3: CALITATE DATE — XML vs JSON
    # ==========================================================
    with tab3:
        st.subheader("Calitate Date: XML vs JSON (Health Auto Export)")
        st.warning(
            "Snapshot martie 2026, scripturi arhivate orfane, reproducibilitate "
            "compromisă; comparație data-quality, nu evaluare model canonic "
            "(vezi Cap.7 limitare D14 (c))."
        )
        st.markdown(
            "Analiza comparativă demonstrează că sursa XML (export.xml) "
            "**dublu-contorizează pașii** (MAE = 9,106 pași, r = 0.98), "
            "în timp ce JSON-ul din Health Auto Export oferă date deduplicate. "
            "Finding-ul SHAP cheie (**aw_hr_mean ca feature #1**) rămâne "
            "stabil în ambele surse — demonstrând robustețea modelului."
        )

        st.markdown("---")
        st.markdown("#### Comparație metrici XML vs JSON (16 features)")

        comp_metrics = {
            "Metrică": [
                "RF Accuracy (temporal)",
                "RF F1 Macro (temporal)",
                "XGBoost CV F1 Macro",
                "SHAP Feature #1",
                "Pași zilnici (mediu)",
                "SpO2 format",
            ],
            "XML": [
                "66.5%", "~38%", "42.3%",
                "aw_hr_mean", "~18,000 (dublu)",
                "Fracție (0.93)",
            ],
            "JSON": [
                "61.8%", "~36%", "42.4%",
                "aw_hr_mean", "~9,000 (corect)",
                "Procent (93.6%)",
            ],
            "Interpretare": [
                "Accuracy mai mică = date corecte (nu e rău)",
                "Comparabil — modelul nu se bazează pe pași",
                "Practic identic — robustețe confirmat",
                "IDENTIC — finding cheie stabil",
                "XML dublu-contorizează",
                "Diferență de unitate, nu de valoare",
            ],
        }
        st.dataframe(
            pd.DataFrame(comp_metrics).set_index("Metrică"),
            width="stretch",
        )

        st.markdown("---")
        st.markdown("#### Corelații globale XML vs JSON")
        corr_data = {
            "Metrică": ["HR mean", "HRV mean", "Steps", "SpO2"],
            "Corelație (r)": ["~1.00", "0.99", "0.98", "N/A (unitate diferită)"],
            "MAE": ["~0.1 bpm", "~0.5 ms", "9,106 pași", "N/A"],
            "Verdict": [
                "Identic",
                "Identic",
                "XML dublu-contorizează",
                "Necesită normalizare",
            ],
        }
        st.dataframe(
            pd.DataFrame(corr_data).set_index("Metrică"),
            width="stretch",
        )

        st.caption(
            "Concluzie: JSON (Health Auto Export) e sursa corectă. "
            "XML-ul e util ca validare, dar nu ca sursă primară. "
            "Discuție detaliată în Cap.6 (Limitări și calitatea datelor)."
        )

        # Afișăm side-by-side SHAP bar plots dacă există
        st.markdown("---")
        st.markdown("#### SHAP Feature Importance: XML vs JSON (robustețe)")
        col_xml, col_json = st.columns(2)
        with col_xml:
            st.markdown("**XML source**")
            show_image_if_exists(f"{FIG_XML}/shap_bar_xgboost.png")
        with col_json:
            st.markdown("**JSON source**")
            show_image_if_exists(f"{FIG_JSON}/json_shap_bar_xgboost.png")

    # ==========================================================
    # SECȚIUNE COMUNĂ: Despre SHAP
    # ==========================================================
    st.markdown("---")
    st.subheader("Despre SHAP")
    st.markdown(
        "**SHAP (SHapley Additive exPlanations)** descompune fiecare predicție "
        "în contribuția individuală a fiecărui feature. Permite:\n\n"
        "- **Global**: care features contează cel mai mult în general?\n"
        "- **Local**: de ce a clasificat modelul *această zi* ca Rest/Normal/Active?\n\n"
        "Referințe: Susnjak & Griffin (2025) – SHAP în clinical XAI; "
        "Napa et al. (2025) – RF + SHAP + Streamlit pentru predicție CVD."
    )


# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Surse date:**  \n"
    "Oura Ring Gen3/Gen4  \n"
    "Apple Watch Ultra/Ultra3  \n\n"
    "**Pipeline:**  \n"
    "JSON Extended (19 features)  \n"
    "Health Auto Export → iCloud  \n\n"
    "**Referințe cheie:**  \n"
    "Lacruz-Pleguezuelos 2025  \n"
    "Helms 2014, Burke 2011  \n"
    "Susnjak & Griffin 2025"
)