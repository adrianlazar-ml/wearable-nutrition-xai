"""
Modul recomandari nutritionale personalizate - v2.

Genereaza recomandari bazate pe clasificarea ML (Rest/Normal/Active)
si profilul biometric al utilizatorului.

Referinte:
- AI4Food (Lacruz-Pleguezuelos et al., 2025): 25 kcal/kg/zi
- Yumuk et al. (2015): European Guidelines for Obesity Management
- Helms et al. (2014): 1.6-2.2 g/kg lean mass proteine
- Campos-Nonato et al. (2017): high-protein la obezitate
- Burke et al. (2011): periodizare carbohidrati

Rulare: py nutrition_recommender.py
"""

USER_PROFILE = {
    "name": "Adrian",
    "weight_kg": 119.9,
    "height_cm": 187,
    "age": 41,
    "gender": "male",
    "bmr_kcal": 2162,
    "body_fat_pct": 36.3,
    "visceral_fat": 16,
    "protein_pct": 14.9,
    "goal": "fat_loss"
}

# AI4Food: "Daily caloric intake was calculated as 25 kcal per kg
# of body weight, equivalent to a daily reduction of 500 kcal"
# Sursa: Lacruz-Pleguezuelos et al. (2025), Clinical Nutrition
BASE_CALORIES = round(25 * USER_PROFILE["weight_kg"])  # ~2998 kcal


def get_lean_mass(profile):
    """Lean Body Mass = weight * (1 - body_fat%/100). LBM (nu SMM Xiaomi); vezi Cap.7."""
    return round(profile["weight_kg"] * (1 - profile["body_fat_pct"] / 100), 2)

# Simboluri text-safe (functioneaza pe orice terminal)
SYMBOLS = {"Rest": "[REST]", "Normal": "[NORM]", "Active": "[ACTV]"}


def validate_macros(calories, protein_g, fat_g, carb_g):
    """
    Validează și ajustează macronutrienții pentru a asigura:
    1. Carbs >= 100g (minim fiziologic pentru funcție cerebrală)
    2. Totalul caloric din macro nu depășește target-ul caloric
    3. Dacă protein + fat depășesc bugetul, redistribuie proporțional

    Returnează (protein_g, fat_g, carb_g) validate.
    """
    min_carbs_g = 100
    min_carbs_kcal = min_carbs_g * 4  # 400 kcal

    protein_kcal = protein_g * 4
    fat_kcal = fat_g * 9
    remaining_for_carbs = calories - protein_kcal - fat_kcal

    if remaining_for_carbs >= min_carbs_kcal:
        # Caz normal: suficient buget pentru carbs
        carb_g = round(remaining_for_carbs / 4)
        return protein_g, fat_g, carb_g

    # Caz problematic: protein + fat consumă prea mult din buget
    # Strategia: păstrează protein (prioritar), reduce fat, forțează carbs=100g
    budget_after_carbs = calories - min_carbs_kcal
    if protein_kcal <= budget_after_carbs:
        # Ajustează doar fat-ul
        fat_g = round((budget_after_carbs - protein_kcal) / 9)
    else:
        # Caz extrem: redistribuie proporțional
        total_pf_kcal = protein_kcal + fat_kcal
        scale = budget_after_carbs / total_pf_kcal
        protein_g = round(protein_g * scale)
        fat_g = round(fat_g * scale)

    return protein_g, fat_g, min_carbs_g


def get_nutrition_recommendation(label, profile):
    """
    Genereaza recomandare nutritionala personalizata.

    Label: 0=Rest, 1=Normal, 2=Active

    Strategie calorica (AI4Food + Burke et al.):
    - Rest Day:   base - 200 kcal
    - Normal Day: base             (AI4Food 25 kcal/kg; vezi Cap.7 limitare obezitate)
    - Active Day: base + 200 kcal
    """
    lean_mass = get_lean_mass(profile)
    base_cal = round(25 * profile["weight_kg"])  # AI4Food: 25 kcal/kg

    if label == 0:  # REST DAY
        calories = base_cal - 200
        protein_g = round(lean_mass * 2.0)       # Helms et al. (2014)
        fat_g = round(calories * 0.30 / 9)       # 30% calorii
        carb_g = round((calories - protein_g * 4 - fat_g * 9) / 4)
        protein_g, fat_g, carb_g = validate_macros(calories, protein_g, fat_g, carb_g)

        return {
            "day_type": "Rest Day",
            "calories_target": calories,
            "protein_g": protein_g,
            "carbs_g": carb_g,
            "fat_g": fat_g,
            "key_foods": [
                "Pui/curcan la gratar - proteina slaba",
                "Somon/sardine - omega-3 antiinflamator",
                "Legume verzi (broccoli, spanac) - minerale",
                "Fructe de padure - antioxidanti",
                "Oua - aminoacizi esentiali"
            ],
            "avoid": [
                "Alimente procesate si zahar adaugat",
                "Alcool - inhiba recuperarea",
                "Carbohidrati simpli in exces"
            ],
            "hydration_ml": 3000,
            "timing": "3 mese principale + 1-2 gustari proteice",
            "rationale": (
                "Readiness scazut sau somn insuficient. "
                "Prioritate: recuperare si reducere inflamatie. "
                "Proteine 2g/kg lean mass. Aport redus (base - 200 kcal); "
                "vezi Cap.7 limitare formula AI4Food pe obezitate."
            )
        }

    elif label == 1:  # NORMAL DAY
        calories = base_cal
        protein_g = round(lean_mass * 1.8)       # Campos-Nonato et al. (2017)
        fat_g = round(calories * 0.28 / 9)       # 28% calorii
        carb_g = round((calories - protein_g * 4 - fat_g * 9) / 4)
        protein_g, fat_g, carb_g = validate_macros(calories, protein_g, fat_g, carb_g)

        return {
            "day_type": "Normal Day",
            "calories_target": calories,
            "protein_g": protein_g,
            "carbs_g": carb_g,
            "fat_g": fat_g,
            "key_foods": [
                "Carne slaba (pui, vita, peste alb)",
                "Orez brun/cartofi dulci/quinoa",
                "Avocado/nuci/seminte",
                "Legume colorate - micronutrienti",
                "Leguminoase (linte, naut) - fibre + proteine"
            ],
            "avoid": [
                "Fast food si alimente ultra-procesate",
                "Bauturi cu zahar adaugat"
            ],
            "hydration_ml": 3500,
            "timing": "3 mese principale echilibrate",
            "rationale": (
                "Stare fiziologica normala. Alimentatie echilibrata "
                "la base caloric (AI4Food 25 kcal/kg/zi); vezi Cap.7 limitare "
                "formula pe obezitate (poate fi intretinere, nu deficit)."
            )
        }

    else:  # ACTIVE DAY
        calories = base_cal + 200
        protein_g = round(lean_mass * 1.8)
        fat_g = round(calories * 0.25 / 9)       # 25% calorii
        carb_g = round((calories - protein_g * 4 - fat_g * 9) / 4)
        protein_g, fat_g, carb_g = validate_macros(calories, protein_g, fat_g, carb_g)

        return {
            "day_type": "Active Day",
            "calories_target": calories,
            "protein_g": protein_g,
            "carbs_g": carb_g,
            "fat_g": fat_g,
            "key_foods": [
                "Pre-workout: banana + fulgi ovaz",
                "Post-workout: shake proteic + orez",
                "Carne slaba sau peste",
                "Cartofi/paste/orez - reumplere glicogen",
                "Fructe - vitamine + carbs naturali"
            ],
            "avoid": [
                "Grasimi in exces pre-workout",
                "Fibre excesive pre-workout"
            ],
            "hydration_ml": 4000,
            "timing": "Pre-workout 2-3h inainte, post-workout 30-60min",
            "pre_workout": "60-80g carbs + 20-30g proteine, 2-3h inainte",
            "post_workout": "40-60g carbs + 30-40g proteine, in 30-60min",
            "rationale": (
                "Readiness ridicat + somn bun. Capacitate maxima. "
                "Carbs crescuti pentru performanta (Burke et al., 2011). "
                "Aport crescut (base + 200 kcal); vezi Cap.7 limitare AI4Food pe obezitate."
            )
        }


def macro_split(rec):
    cal = rec["calories_target"]
    p = round(rec["protein_g"] * 4 / cal * 100)
    c = round(rec["carbs_g"] * 4 / cal * 100)
    f = round(rec["fat_g"] * 9 / cal * 100)
    return f"P:{p}% C:{c}% F:{f}%"


def print_recommendation(rec):
    cal_check = rec["protein_g"] * 4 + rec["carbs_g"] * 4 + rec["fat_g"] * 9

    print(f"\n{'='*55}")
    print(f"  {rec['day_type']}")
    print(f"{'='*55}")
    print(f"  Calorii:      {rec['calories_target']} kcal")
    print(f"  Proteine:     {rec['protein_g']}g")
    print(f"  Carbohidrati: {rec['carbs_g']}g")
    print(f"  Grasimi:      {rec['fat_g']}g")
    print(f"  Macro split:  {macro_split(rec)}")
    print(f"  Hidratare:    {rec['hydration_ml']}ml")
    print(f"  Timing:       {rec['timing']}")
    print(f"  [Check: {cal_check} kcal din macro vs {rec['calories_target']} target]")
    print(f"\n  Rationale: {rec['rationale']}")
    print(f"\n  Alimente recomandate:")
    for food in rec["key_foods"]:
        print(f"    + {food}")
    print(f"\n  Evita:")
    for item in rec["avoid"]:
        print(f"    - {item}")
    if "pre_workout" in rec:
        print(f"\n  Pre-workout:  {rec['pre_workout']}")
        print(f"  Post-workout: {rec['post_workout']}")


def print_comparative_table():
    """FIX: Tabel comparativ corect, fara bug de duplicare."""
    rest = get_nutrition_recommendation(0, USER_PROFILE)
    norm = get_nutrition_recommendation(1, USER_PROFILE)
    actv = get_nutrition_recommendation(2, USER_PROFILE)

    print(f"\n{'='*55}")
    print("  SUMAR COMPARATIV")
    print(f"{'='*55}")
    print(f"  {'Metric':<15} {'Rest':>10} {'Normal':>10} {'Active':>10}")
    print(f"  {'-'*45}")

    metrics = [
        ("Calorii", "calories_target"),
        ("Proteine (g)", "protein_g"),
        ("Carbs (g)", "carbs_g"),
        ("Grasimi (g)", "fat_g"),
        ("Hidratare (ml)", "hydration_ml"),
    ]
    for label, key in metrics:
        print(f"  {label:<15} {rest[key]:>10} {norm[key]:>10} {actv[key]:>10}")

    print(f"  {'Macro split':<15} {macro_split(rest):>10} {macro_split(norm):>10} {macro_split(actv):>10}")


if __name__ == "__main__":
    p = USER_PROFILE
    print("=" * 55)
    print("  MODUL RECOMANDARI NUTRITIONALE v2")
    print("  (Evidence-based, calibrat pe literatura proiectului)")
    print("=" * 55)
    print(f"\n  Profil: {p['name']}")
    print(f"  Greutate: {p['weight_kg']} kg | Lean mass: {get_lean_mass(p)} kg")
    print(f"  BMR: {p['bmr_kcal']} kcal | Body fat: {p['body_fat_pct']}%")
    print(f"  Obiectiv: {p['goal']}")
    print(f"\n  Formula calorica: 25 kcal/kg = {BASE_CALORIES} kcal/zi")
    print(f"  (Conform AI4Food, Lacruz-Pleguezuelos et al., 2025)")

    for label in [0, 1, 2]:
        rec = get_nutrition_recommendation(label, p)
        print_recommendation(rec)

    print_comparative_table()
