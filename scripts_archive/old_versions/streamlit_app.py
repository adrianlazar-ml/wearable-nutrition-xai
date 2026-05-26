"""
Streamlit UI - Sistem AI pentru Recomandări Nutriționale Personalizate
=====================================================================
Disertație: AI pentru nutriție personalizată folosind date din wearables
Autor: Adrian Lazar | Coordonator: Prof. Cătălin Buiu, UPB
Apărare: Iunie-Iulie 2026

Pagini:
  1. Profil Utilizator - date biometrice, calcul BMR/TDEE
  2. Dashboard Zilnic - clasificare ML + recomandare nutrițională
  3. Explicabilitate - SHAP feature importance, vizualizare model

Rulare:
  pip install streamlit plotly --break-system-packages
  streamlit run streamlit_app.py

Cerințe: Scripturile existente (nutrition_recommender.py, etc.)
         trebuie să fie în același director.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

# ============================================================
# CONFIG & PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="AI Nutrition – Oura + Apple Watch",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths - ajustează dacă structura e diferită
DATA_DIR = "data"
MODEL_DIR = "models"

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


def get_nutrition_recommendation_local(label, profile):
    """
    Generează recomandare nutrițională.
    Reproducere din nutrition_recommender.py pentru independență Streamlit.
    
    Referințe:
    - AI4Food (Lacruz-Pleguezuelos et al., 2025): 25 kcal/kg/zi
    - Helms et al. (2014): 1.6-2.2 g/kg lean mass proteine
    - Burke et al. (2011): periodizare carbohidrați
    - Campos-Nonato et al. (2017): high-protein la obezitate
    """
    base_cal = round(25 * profile["weight_kg"])
    lean = profile["lean_mass_kg"]

    configs = {
        0: {  # REST
            "day_type": "Rest Day",
            "cal_adj": -200,
            "protein_mult": 2.0,
            "fat_pct": 0.30,
            "rationale": (
                "Readiness scăzut sau somn insuficient. "
                "Prioritate: recuperare și reducere inflamație. "
                "Proteine 2g/kg lean mass (Helms et al., 2014). Deficit ~700 kcal."
            ),
            "foods": [
                "Pui/curcan la grătar – proteină slabă",
                "Somon/sardine – omega-3 antiinflamator",
                "Legume verzi (broccoli, spanac) – minerale",
                "Fructe de pădure – antioxidanți",
            ],
            "avoid": [
                "Alimente procesate și zahăr adăugat",
                "Alcool – inhibă recuperarea",
            ],
            "hydration_ml": 3000,
            "timing": "3 mese principale + 1-2 gustări proteice",
        },
        1: {  # NORMAL
            "day_type": "Normal Day",
            "cal_adj": 0,
            "protein_mult": 1.8,
            "fat_pct": 0.28,
            "rationale": (
                "Stare fiziologică normală. Alimentație echilibrată "
                "cu deficit ~500 kcal. Protocol AI4Food: 25 kcal/kg/zi "
                "(Lacruz-Pleguezuelos et al., 2025)."
            ),
            "foods": [
                "Carne slabă (pui, vită, pește alb)",
                "Orez brun / cartofi dulci / quinoa",
                "Avocado / nuci / semințe",
                "Leguminoase (linte, năut) – fibre + proteine",
            ],
            "avoid": [
                "Fast food și alimente ultra-procesate",
                "Băuturi cu zahăr adăugat",
            ],
            "hydration_ml": 3500,
            "timing": "3 mese principale echilibrate",
        },
        2: {  # ACTIVE
            "day_type": "Active Day",
            "cal_adj": 200,
            "protein_mult": 1.8,
            "fat_pct": 0.25,
            "rationale": (
                "Readiness ridicat + somn bun. Capacitate maximă. "
                "Carbohidrați crescuți pentru performanță "
                "(Burke et al., 2011). Deficit redus ~300 kcal."
            ),
            "foods": [
                "Pre-workout: banană + fulgi ovăz",
                "Post-workout: shake proteic + orez",
                "Carne slabă sau pește",
                "Cartofi / paste / orez – reumplere glicogen",
            ],
            "avoid": [
                "Grăsimi în exces pre-workout",
                "Fibre excesive pre-workout",
            ],
            "hydration_ml": 4000,
            "timing": "Pre-workout 2-3h înainte, post-workout 30-60min",
        },
    }

    cfg = configs[label]
    calories = base_cal + cfg["cal_adj"]
    protein_g = round(lean * cfg["protein_mult"])
    fat_g = round(calories * cfg["fat_pct"] / 9)
    carb_g = round((calories - protein_g * 4 - fat_g * 9) / 4)
    carb_g = max(carb_g, 100)

    return {
        "day_type": cfg["day_type"],
        "calories_target": calories,
        "protein_g": protein_g,
        "carbs_g": carb_g,
        "fat_g": fat_g,
        "hydration_ml": cfg["hydration_ml"],
        "timing": cfg["timing"],
        "rationale": cfg["rationale"],
        "foods": cfg["foods"],
        "avoid": cfg["avoid"],
        "base_calories": base_cal,
    }


def try_load_dataset():
    """Încearcă să încarce dataset-ul combinat. Returnează None dacă nu există."""
    path = os.path.join(DATA_DIR, "dataset_combinat.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def try_load_model():
    """Încearcă să încarce modelul salvat. Returnează (rf, imputer) sau (None, None)."""
    try:
        import joblib
        rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
        imp = joblib.load(os.path.join(MODEL_DIR, "imputer.joblib"))
        return rf, imp
    except Exception:
        return None, None


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
        "body_fat_pct": 36.2,
        "goal": "fat_loss",
        "activity_level": "Ușor activ (1-2 antrenamente/săpt)",
    }

# Calculăm lean mass din profil
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
            f"Deficit standard: ~500 kcal față de TDEE ({tdee} kcal)"
        )

        st.markdown("---")
        st.subheader("Comparație Recomandări per Tip de Zi")

        full_profile = {**p, "lean_mass_kg": lean, "bmr_kcal": bmr}

        comp_data = []
        for label_id, label_name in [(0, "Rest"), (1, "Normal"), (2, "Active")]:
            rec = get_nutrition_recommendation_local(label_id, full_profile)
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

    # Profil curent
    p = st.session_state.profile
    lean = get_lean_mass(p)
    bmr = calculate_bmr(p["weight_kg"], p["height_cm"], p["age"], p["gender"])
    full_profile = {**p, "lean_mass_kg": lean, "bmr_kcal": bmr}

    # Modul input: manual sau din model
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

            # Clasificare pe baza pragurilor Oura
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
            rec = get_nutrition_recommendation_local(auto_label, full_profile)

            st.subheader(f"Recomandare: {rec['day_type']}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Calorii", f"{rec['calories_target']} kcal")
            c2.metric("Proteine", f"{rec['protein_g']}g")
            c3.metric("Carbohidrați", f"{rec['carbs_g']}g")
            c4.metric("Grăsimi", f"{rec['fat_g']}g")

            # Macro chart
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
                for food in rec["foods"]:
                    st.markdown(f"- {food}")

            with st.expander("De evitat"):
                for item in rec["avoid"]:
                    st.markdown(f"- {item}")

    else:
        # Mod automat: încearcă să încarce modelul
        st.markdown("---")
        rf, imputer = try_load_model()

        if rf is None:
            st.warning(
                "Modelul ML nu a fost găsit. "
                "Rulează `python demo_end_to_end.py` pentru a genera "
                "`models/rf_model.joblib` și `models/imputer.joblib`."
            )
            st.info(
                "Alternativ, folosește modul **Manual** pentru a testa "
                "recomandările cu scoruri introduse manual."
            )
        else:
            dataset = try_load_dataset()
            if dataset is None:
                st.warning(
                    "Dataset-ul combinat nu a fost găsit. "
                    "Rulează `python cap5_pipeline.py` pentru a genera "
                    "`data/dataset_combinat.csv`."
                )
            else:
                st.success(
                    f"Model încărcat. Dataset: {len(dataset)} zile."
                )

                features = [c for c in dataset.columns if c.endswith("_lag1")]
                last_row = dataset.iloc[-1]

                feat_values = pd.DataFrame(
                    [last_row[features].values], columns=features
                )
                feat_imp = pd.DataFrame(
                    imputer.transform(feat_values), columns=features
                )

                label = int(rf.predict(feat_imp)[0])
                proba = rf.predict_proba(feat_imp)[0]

                label_names = {0: "Rest", 1: "Normal", 2: "Active"}
                label_colors = {0: "🔴", 1: "🟡", 2: "🟢"}

                st.markdown(
                    f"### Predicție ML: {label_colors[label]} "
                    f"{label_names[label]} Day"
                )

                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("P(Rest)", f"{proba[0]:.1%}")
                pc2.metric("P(Normal)", f"{proba[1]:.1%}")
                pc3.metric("P(Active)", f"{proba[2]:.1%}")

                rec = get_nutrition_recommendation_local(label, full_profile)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Calorii", f"{rec['calories_target']} kcal")
                c2.metric("Proteine", f"{rec['protein_g']}g")
                c3.metric("Carbohidrați", f"{rec['carbs_g']}g")
                c4.metric("Grăsimi", f"{rec['fat_g']}g")

                st.markdown(f"**Raționale:** {rec['rationale']}")


# ============================================================
# PAGE 3: EXPLICABILITATE
# ============================================================
elif page == "🔍 Explicabilitate":
    st.title("Explicabilitate Model (SHAP)")
    st.markdown(
        "Transparență în predicțiile ML: ce features influențează "
        "clasificarea Rest/Normal/Active? "
        "(Susnjak & Griffin, 2025; Napa et al., 2025)"
    )

    # Încearcă să încarce imaginile SHAP generate de cap5_compare_algorithms.py
    shap_images = {
        "SHAP Summary (Global)": "data/shap_summary_xgboost.png",
        "SHAP Feature Importance (Bar)": "data/shap_bar_xgboost.png",
        "SHAP Dependence (Top 5)": "data/shap_dependence_xgboost.png",
        "SHAP Waterfall – Rest Day": "data/shap_waterfall_rest_xgboost.png",
        "SHAP Waterfall – Normal Day": "data/shap_waterfall_normal_xgboost.png",
        "SHAP Waterfall – Active Day": "data/shap_waterfall_active_xgboost.png",
        "SHAP Waterfall – Toate clasele": "data/shap_waterfall_all_xgboost.png",
        "PDP – Top 3 Features": "data/pdp_top3_xgboost.png",
        "PDP – Multi-class (feature dominant)": "data/pdp_multiclass_xgboost.png",
        "Feature Importance: RF vs XGBoost": "data/feature_importance_comparison.png",
        "Matrici de Confuzie": "data/confusion_matrices.png",
    }

    found_any = False
    for title, path in shap_images.items():
        if os.path.exists(path):
            found_any = True
            st.subheader(title)
            st.image(path, width="stretch")
            st.markdown("---")

    if not found_any:
        st.info(
            "Imaginile SHAP nu au fost găsite. "
            "Rulează `python cap5_compare_algorithms.py` pentru a genera "
            "graficele în directorul `data/`."
        )

        st.markdown("---")
        st.subheader("Demo: Feature Importance (date simulate)")
        st.markdown(
            "Mai jos e un exemplu demonstrativ cu feature importance "
            "generată din date simulate, pentru a arăta structura UI."
        )

        # Demo feature importance din date mock
        features_demo = [
            "aw_hr_mean_lag1", "sleep_score_lag1", "readiness_score_lag1",
            "hrv_mean_lag1", "aw_steps_lag1", "rhr_lowest_lag1",
            "activity_score_lag1", "aw_active_calories_lag1",
            "recovery_index_lag1", "hrv_balance_lag1",
            "steps_oura_lag1", "rhr_avg_lag1",
            "aw_hrv_mean_lag1", "respiratory_rate_lag1",
            "temp_deviation_lag1", "aw_spo2_lag1",
        ]
        importance_demo = np.array([
            0.142, 0.118, 0.105, 0.088, 0.082, 0.072,
            0.068, 0.062, 0.055, 0.048,
            0.042, 0.038, 0.030, 0.022, 0.018, 0.010,
        ])

        fi_df = pd.DataFrame({
            "Feature": features_demo,
            "Importance": importance_demo,
        }).sort_values("Importance", ascending=True)

        try:
            import plotly.express as px

            fig = px.bar(
                fi_df, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#85B7EB", "#5DCAA5", "#F0997B"],
            )
            fig.update_layout(
                height=500,
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False,
                yaxis_title="",
                xaxis_title="Feature Importance (Random Forest)",
            )
            st.plotly_chart(fig, width="stretch")
        except ImportError:
            st.bar_chart(fi_df.set_index("Feature")["Importance"])

        st.caption(
            "Notă: `aw_hr_mean` (Apple Watch Heart Rate) e dominant – "
            "demonstrează complementaritatea Oura + Apple Watch. "
            "Acesta e un finding cheie al disertației."
        )

    # Secțiune informativă
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

    st.subheader("Metrici Model")
    metrics_data = {
        "Algoritm": ["Random Forest", "XGBoost", "SVM (RBF)", "Baseline (majority)"],
        "CV F1 Macro": ["~38%", "~42%", "~37%", "~20%"],
        "Test F1 Macro": ["~52%", "~55%", "~58%", "~20%"],
        "Observații": [
            "Stabil, interpretabil, baseline proiect",
            "Best CV performance, gradient boosting",
            "Best test F1, dar variance mai mare",
            "Prezice mereu clasa dominantă (Rest)",
        ],
    }
    st.dataframe(
        pd.DataFrame(metrics_data).set_index("Algoritm"),
        width="stretch",
    )
    st.caption(
        "Metricile sunt orientative (din ultima rulare). "
        "F1 Macro e metrica principală datorită class imbalance sever "
        "(Rest: ~757, Normal: ~202, Active: ~45)."
    )


# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Surse date:**  \n"
    "Oura Ring Gen3/Gen4  \n"
    "Apple Watch Ultra/Ultra3  \n\n"
    "**Referințe cheie:**  \n"
    "Lacruz-Pleguezuelos 2025  \n"
    "Helms 2014, Burke 2011  \n"
    "Susnjak & Griffin 2025"
)
