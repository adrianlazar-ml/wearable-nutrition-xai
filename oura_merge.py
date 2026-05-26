"""
Combina datele Oura din doua surse:
1. oura_historical.csv (export din aplicatie, iunie 2023 - iulie 2025)
2. Date API v2 (sleep.csv, readiness.csv, activity.csv, sleep_metrics.csv)

Output: data/oura_combined.csv - dataset unic fara duplicate.

Fix v1.1: rhr_avg < rhr_lowest → NaN (valori fiziologic imposibile
din bug agregare sleep_metrics.csv, ~60 zile afectate pe Gen4).

Rulare: py oura_merge.py
"""

import pandas as pd
import numpy as np
import ast


# Datele de upgrade device (specifice Adrian)
OURA_GEN4_DATE = "2025-05-30"
AW_ULTRA3_DATE = "2026-02-05"


def find_temp_column(columns):
    """
    FIX: Gaseste coloana de temperatura indiferent de encoding.
    CSV-ul din Oura poate avea Â°C, Ã‚Â°C, sau alte variante
    in functie de encoding-ul sistemului.
    """
    for col in columns:
        if "Temperature Deviation" in col:
            return col
    return None


def load_historical():
    print("=== LOAD HISTORICAL ===")
    hist = pd.read_csv("data/oura_historical.csv")
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.replace("None", np.nan)

    # FIX: Gaseste coloana de temperatura robust
    temp_col = find_temp_column(hist.columns)
    if temp_col is None:
        print("  WARN: Coloana Temperature Deviation nu a fost gasita!")
        hist["temp_deviation"] = np.nan
        select_cols = [
            "date", "Readiness Score", "Sleep Score", "Average HRV",
            "Lowest Resting Heart Rate", "Average Resting Heart Rate",
            "Recovery Index Score", "HRV Balance Score", "Activity Score",
            "Steps", "Respiratory Rate",
        ]
    else:
        select_cols = [
            "date", "Readiness Score", "Sleep Score", "Average HRV",
            "Lowest Resting Heart Rate", "Average Resting Heart Rate",
            "Recovery Index Score", "HRV Balance Score", "Activity Score",
            "Steps", "Respiratory Rate", temp_col,
        ]

    hist_clean = hist[select_cols].copy()
    hist_clean.columns = [
        "date", "readiness_score", "sleep_score", "hrv_mean",
        "rhr_lowest", "rhr_avg", "recovery_index", "hrv_balance",
        "activity_score", "steps_oura", "respiratory_rate",
    ] + (["temp_deviation"] if temp_col else [])

    # Adauga temp_deviation daca lipsea
    if "temp_deviation" not in hist_clean.columns:
        hist_clean["temp_deviation"] = np.nan

    for col in hist_clean.columns[1:]:
        hist_clean[col] = pd.to_numeric(hist_clean[col], errors="coerce")

    print(f"  Historical: {len(hist_clean)} zile "
          f"({hist_clean['date'].min().date()} -> {hist_clean['date'].max().date()})")
    return hist_clean


def load_api_data():
    print("\n=== LOAD API DATE ===")

    # Readiness + contributors
    readiness = pd.read_csv("data/readiness.csv")
    readiness["date"] = pd.to_datetime(readiness["day"])
    readiness["contributors"] = readiness["contributors"].apply(ast.literal_eval)
    contrib = pd.json_normalize(readiness["contributors"])
    readiness = pd.concat([readiness, contrib], axis=1)

    # Sleep daily
    sleep_api = pd.read_csv("data/sleep.csv")
    sleep_api["date"] = pd.to_datetime(sleep_api["day"])
    sleep_api = sleep_api[["date", "score"]].rename(columns={"score": "sleep_score"})

    # Sleep metrics (hrv, rhr, respiratory rate)
    sleep_metrics = pd.read_csv("data/sleep_metrics.csv")
    sleep_metrics["date"] = pd.to_datetime(sleep_metrics["date"])
    sleep_metrics = sleep_metrics.rename(columns={
        "average_hrv": "hrv_mean",
        "lowest_heart_rate": "rhr_lowest",
        "average_breath": "respiratory_rate",
        "average_heart_rate": "rhr_avg"
    })

    # Activity
    activity_api = pd.read_csv("data/activity.csv")
    activity_api["date"] = pd.to_datetime(activity_api["day"])
    activity_api = activity_api[["date", "steps", "score"]].rename(columns={
        "steps": "steps_oura",
        "score": "activity_score"
    })

    # Construim API dataframe
    api = readiness[["date", "score", "hrv_balance", "recovery_index",
                      "temperature_deviation"]].copy()
    api.columns = ["date", "readiness_score", "hrv_balance",
                    "recovery_index", "temp_deviation"]

    api = api.merge(sleep_api, on="date", how="left")
    api = api.merge(
        sleep_metrics[["date", "hrv_mean", "rhr_lowest", "respiratory_rate", "rhr_avg"]],
        on="date", how="left"
    )
    api = api.merge(
        activity_api[["date", "steps_oura", "activity_score"]],
        on="date", how="left"
    )

    print(f"  API: {len(api)} zile "
          f"({api['date'].min().date()} -> {api['date'].max().date()})")
    print(f"  Valori lipsa API:")
    print(api.isnull().sum().to_string(header=False))
    return api


def combine(hist_clean, api):
    print("\n=== COMBINARE ===")

    # Aliniem coloanele API cu historical
    target_cols = hist_clean.columns.tolist()
    for col in target_cols:
        if col not in api.columns:
            api[col] = np.nan
    api = api[target_cols]

    # Doar zilele care NU sunt deja in historical
    api_new = api[~api["date"].isin(hist_clean["date"])].copy()
    print(f"  Zile noi din API: {len(api_new)}")

    combined = pd.concat([hist_clean, api_new], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    # Device flags
    combined["oura_device"] = np.where(
        combined["date"] < OURA_GEN4_DATE, "Gen3", "Gen4"
    )
    combined["aw_device"] = np.where(
        combined["date"] < AW_ULTRA3_DATE, "Ultra", "Ultra3"
    )

    # ── DATA QUALITY FIX: rhr_avg ──────────────────────────────
    # Bug identificat în agregarea API (sleep_metrics.csv): pe ~60 zile
    # rhr_avg < rhr_lowest, ceea ce este fiziologic imposibil (media
    # ritmului cardiac nu poate fi mai mică decât minimul).
    # Cauza: sleep_metrics.csv agregă rhr_avg peste toate sesiunile
    # de somn (long_sleep + naps + rest), incluzând sesiuni parțiale
    # cu HR foarte scăzut sau incomplet.
    # Fix: setăm valorile imposibile la NaN → imputare cu mediană
    # în pipeline-urile downstream.
    # Afectează exclusiv perioada API v2 (iul 2025+), toate pe Gen4.
    invalid_rhr = (
        combined["rhr_avg"].notna()
        & combined["rhr_lowest"].notna()
        & (combined["rhr_avg"] < combined["rhr_lowest"])
    )
    n_invalid = invalid_rhr.sum()
    if n_invalid > 0:
        combined.loc[invalid_rhr, "rhr_avg"] = np.nan
        print(f"\n  rhr_avg fix: {n_invalid} zile cu rhr_avg < rhr_lowest → NaN")
    # ───────────────────────────────────────────────────────────

    print(f"\n  Dataset final: {len(combined)} zile")
    print(f"  Perioada: {combined['date'].min().date()} -> {combined['date'].max().date()}")
    print(f"  Gen3: {(combined['oura_device']=='Gen3').sum()} zile")
    print(f"  Gen4: {(combined['oura_device']=='Gen4').sum()} zile")
    print(f"\n  Valori lipsa per coloana:")
    print(combined.isnull().sum().to_string(header=False))

    return combined


def main():
    hist_clean = load_historical()
    api = load_api_data()
    combined = combine(hist_clean, api)

    combined.to_csv("data/oura_combined.csv", index=False)
    print(f"\n  Salvat: data/oura_combined.csv")
    return combined


if __name__ == "__main__":
    main()