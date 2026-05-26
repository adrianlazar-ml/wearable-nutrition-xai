"""
parse_health_auto_export.py — Parsează JSON-urile din Health Auto Export (iCloud)
==================================================================================
Citește toate fișierele HealthAutoExport-*.json din folderul iCloud,
agregă metrici pe zi, și generează un CSV cu un rând per zi.

Rulare: python parse_health_auto_export.py

Autor: Adrian Lazar | Disertație UPB 2026
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────
ICLOUD_FOLDER = Path(
    r"C:\Users\Adrian\iCloudDrive"
    r"\iCloud~com~ifunography~HealthExport\New Automation"
)
OUTPUT_CSV = Path(__file__).parent / "apple_watch_api" / "apple_health_daily.csv"

# Cum agregăm fiecare metrică: (coloană_output, funcție_agregare)
# "sum" = total zilnic (pași, energie), "mean" = medie zilnică (HR, HRV)
METRIC_CONFIG = {
    # ── Cardiac ──
    # heart_rate uses Avg/Min/Max → split into heart_rate_avg/min/max
    "heart_rate_avg": [
        ("aw_hr_mean", "mean"),
    ],
    "heart_rate_min": [
        ("aw_hr_min", "min"),
    ],
    "heart_rate_max": [
        ("aw_hr_max", "max"),
    ],
    "heart_rate_variability": [
        ("aw_hrv_mean", "mean"),
    ],
    "resting_heart_rate": [
        ("aw_rhr", "mean"),
    ],
    "walking_heart_rate_average": [
        ("aw_walking_hr", "mean"),
    ],
    "cardio_recovery": [
        ("aw_cardio_recovery", "mean"),
    ],
    "vo2_max": [
        ("aw_vo2_max", "mean"),
    ],
    # ── Respiratory / SpO2 ──
    "respiratory_rate": [
        ("aw_resp_rate", "mean"),
    ],
    "blood_oxygen_saturation": [
        ("aw_spo2", "mean"),
    ],
    "breathing_disturbances": [
        ("aw_breathing_dist", "sum"),
    ],
    # ── Activity ──
    "step_count": [
        ("aw_steps", "sum"),
    ],
    "active_energy": [
        ("aw_active_energy_kj", "sum"),
    ],
    "basal_energy_burned": [
        ("aw_basal_energy_kj", "sum"),
    ],
    "walking_running_distance": [
        ("aw_distance_km", "sum"),
    ],
    "flights_climbed": [
        ("aw_flights_climbed", "sum"),
    ],
    "apple_exercise_time": [
        ("aw_exercise_min", "sum"),
    ],
    "physical_effort": [
        ("aw_physical_effort", "mean"),
    ],
    # ── Sleep ──
    # ── Sleep (split from sleep_analysis) ──
    "sleep_total": [
        ("aw_sleep_hours", "sum"),
    ],
    "sleep_deep": [
        ("aw_sleep_deep_hours", "sum"),
    ],
    "sleep_rem": [
        ("aw_sleep_rem_hours", "sum"),
    ],
    "sleep_core": [
        ("aw_sleep_core_hours", "sum"),
    ],
    "sleep_awake": [
        ("aw_sleep_awake_hours", "sum"),
    ],
    "apple_sleeping_wrist_temperature": [
        ("aw_wrist_temp", "mean"),
    ],
    # ── Body composition ──
    "weight_body_mass": [
        ("aw_weight_kg", "mean"),
    ],
    "body_fat_percentage": [
        ("aw_body_fat_pct", "mean"),
    ],
    "body_mass_index": [
        ("aw_bmi", "mean"),
    ],
    "lean_body_mass": [
        ("aw_lean_mass_kg", "mean"),
    ],
    # ── Mobility ──
    "walking_speed": [
        ("aw_walking_speed_kmh", "mean"),
    ],
    "walking_step_length": [
        ("aw_step_length_cm", "mean"),
    ],
    "walking_asymmetry_percentage": [
        ("aw_walk_asymmetry_pct", "mean"),
    ],
    "walking_double_support_percentage": [
        ("aw_walk_dbl_support_pct", "mean"),
    ],
    "stair_speed_up": [
        ("aw_stair_up_ms", "mean"),
    ],
    "stair_speed_down": [
        ("aw_stair_down_ms", "mean"),
    ],
    # ── Standup ──
    "apple_stand_hour": [
        ("aw_stand_hours", "sum"),
    ],
    "apple_stand_time": [
        ("aw_stand_min", "sum"),
    ],
}

# Metrici care au format special (nu qty) dar trebuie procesate
SPECIAL_METRICS = {"heart_rate", "sleep_analysis"}

# Toate metricile pe care le recunoaștem
ALL_KNOWN_METRICS = set(METRIC_CONFIG.keys()) | SPECIAL_METRICS

# ── LOGGING ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("parse_health_export")


def parse_date(date_str: str) -> str:
    """Extrage doar data (YYYY-MM-DD) din formatul Health Auto Export."""
    # Format: "2024-11-03 01:18:00 +0200"
    return date_str[:10]


def load_json_file(filepath: Path) -> list[dict]:
    """Încarcă un fișier JSON și returnează lista de (date, metric, qty)."""
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Skip {filepath.name}: {e}")
        return rows

    metrics = data.get("data", {}).get("metrics", [])
    for metric in metrics:
        name = metric.get("name", "")
        if name not in ALL_KNOWN_METRICS:
            continue

        for entry in metric.get("data", []):
            date_str = entry.get("date", "")
            if not date_str:
                continue
            day = parse_date(date_str)

            # ── heart_rate: uses Avg/Min/Max instead of qty ──
            if name == "heart_rate":
                for key, suffix in [("Avg", "avg"), ("Min", "min"), ("Max", "max")]:
                    val = entry.get(key)
                    if val is not None:
                        rows.append({
                            "date": day,
                            "metric": f"heart_rate_{suffix}",
                            "qty": float(val),
                        })
                continue

            # ── sleep_analysis: uses totalSleep/deep/rem/core/awake ──
            if name == "sleep_analysis":
                for key, suffix in [
                    ("totalSleep", "total"),
                    ("deep", "deep"),
                    ("rem", "rem"),
                    ("core", "core"),
                    ("awake", "awake"),
                ]:
                    val = entry.get(key)
                    if val is not None:
                        rows.append({
                            "date": day,
                            "metric": f"sleep_{suffix}",
                            "qty": float(val),
                        })
                continue

            # ── Standard metrics: use qty ──
            qty = entry.get("qty")
            if qty is not None:
                rows.append({
                    "date": day,
                    "metric": name,
                    "qty": float(qty),
                })
    return rows


def aggregate_daily(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Agregă datele raw pe zi conform METRIC_CONFIG."""
    dates = sorted(df_raw["date"].unique())
    results = []

    for date in dates:
        day_data = df_raw[df_raw["date"] == date]
        row = {"date": date}

        for metric_name, columns in METRIC_CONFIG.items():
            metric_df = day_data[day_data["metric"] == metric_name]["qty"]

            for col_name, agg_func in columns:
                if metric_df.empty:
                    row[col_name] = np.nan
                elif agg_func == "mean":
                    row[col_name] = round(metric_df.mean(), 2)
                elif agg_func == "sum":
                    row[col_name] = round(metric_df.sum(), 2)
                elif agg_func == "min":
                    row[col_name] = round(metric_df.min(), 2)
                elif agg_func == "max":
                    row[col_name] = round(metric_df.max(), 2)

        results.append(row)

    return pd.DataFrame(results)


def main():
    # Detectează automat folderul (local test vs. iCloud)
    if ICLOUD_FOLDER.exists():
        scan_folder = ICLOUD_FOLDER
    else:
        # Fallback: folder curent (pentru testare)
        scan_folder = Path(__file__).parent
        logger.warning(f"iCloud folder not found, scanning: {scan_folder}")

    logger.info(f"Scanning: {scan_folder}")

    # Găsește toate JSON-urile
    json_files = sorted(scan_folder.glob("HealthAutoExport-*.json"))
    if not json_files:
        logger.error("Nu am găsit fișiere HealthAutoExport-*.json!")
        return

    logger.info(f"Found {len(json_files)} JSON files")
    logger.info(f"  First: {json_files[0].name}")
    logger.info(f"  Last:  {json_files[-1].name}")

    # Parsează toate fișierele
    all_rows = []
    parsed_count = 0
    for jf in json_files:
        rows = load_json_file(jf)
        all_rows.extend(rows)
        if rows:
            parsed_count += 1

    if not all_rows:
        logger.error("No data parsed from any file!")
        return

    df_raw = pd.DataFrame(all_rows)
    logger.info(f"\nParsed {parsed_count}/{len(json_files)} files")
    logger.info(f"Total raw records: {len(df_raw):,}")
    logger.info(f"Metrics found: {sorted(df_raw['metric'].unique().tolist())}")
    logger.info(f"Date range: {df_raw['date'].min()} → {df_raw['date'].max()}")

    # Agregare zilnică
    df_daily = aggregate_daily(df_raw)

    # Salvare
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    df_daily.to_csv(OUTPUT_CSV, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Apple Health Daily Aggregates — DONE")
    print(f"{'='*60}")
    print(f"  Days:      {len(df_daily)}")
    print(f"  Range:     {df_daily['date'].iloc[0]} → {df_daily['date'].iloc[-1]}")
    print(f"  Columns:   {len(df_daily.columns)}")
    print(f"  Output:    {OUTPUT_CSV}")
    print()

    # Completeness check
    print("  Metric completeness (non-null days):")
    for col in df_daily.columns:
        if col == "date":
            continue
        non_null = df_daily[col].notna().sum()
        pct = non_null / len(df_daily) * 100
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"    {col:30s} {non_null:>4d}/{len(df_daily)} {bar} {pct:.0f}%")

    print(f"\n  Preview (first 3 rows):")
    print(df_daily.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
