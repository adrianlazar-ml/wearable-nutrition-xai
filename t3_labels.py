"""
t3_labels.py - Schema canonica de etichetare T3 (Rest / Normal / Active)
========================================================================
Sursa unica de adevar pentru pragurile T3 si maparea etichetelor.

Praguri (vezi Cap.5 + CIFRE_CANONICE.md sectiunea 14):
  Rest   (0): readiness_score < 60 OR sleep_score < 60
  Active (2): readiness_score >= 85 AND sleep_score >= 75
  Normal (1): restul zilelor

Inlocuieste logica T3 duplicata anterior inline in
cap5_pipeline_json_extended.py, experiments_new.py si streamlit_app_v2.py
(FND-210), plus constantele de afisare label_names / label_colors (FND-215).

Praguri NU se schimba (vezi Ground Truth in the tooling context file): refactor doar
elimina duplicarea, logica ramane bit-identica.

Autor: Adrian Lazar | Disertatie UPB 2026
"""

import numpy as np
import pandas as pd

# Praguri canonice T3
T3_REST_CUTOFF = 60        # readiness sau sleep sub acest prag -> Rest
T3_ACTIVE_READINESS = 85   # readiness >= acest prag SI ...
T3_ACTIVE_SLEEP = 75       # ... sleep >= acest prag -> Active

# Mapare canonica eticheta -> nume / culoare (FND-215)
LABEL_NAMES = {0: "Rest", 1: "Normal", 2: "Active"}
LABEL_COLORS = {0: "🔴", 1: "🟡", 2: "🟢"}


def classify_day(readiness, sleep):
    """Clasificare T3 pe valori scalare. Returneaza int 0/1/2.

    Nu trateaza NaN (folosit pe input direct, ex. slidere UI).
    """
    if readiness < T3_REST_CUTOFF or sleep < T3_REST_CUTOFF:
        return 0
    elif readiness >= T3_ACTIVE_READINESS and sleep >= T3_ACTIVE_SLEEP:
        return 2
    else:
        return 1


def assign_nutrition_label(row):
    """Clasificare T3 pe rand de DataFrame (pentru df.apply(..., axis=1)).

    Returneaza 0/1/2, sau np.nan daca readiness_score / sleep_score lipsesc.
    Comportament identic cu varianta inline anterioara.
    """
    if pd.isna(row["readiness_score"]) or pd.isna(row["sleep_score"]):
        return np.nan
    return classify_day(row["readiness_score"], row["sleep_score"])
