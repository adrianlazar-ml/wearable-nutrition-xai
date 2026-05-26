"""
compare_xml_vs_json.py — Compară datele Apple Health din XML vs JSON
=====================================================================
Rulare: python compare_xml_vs_json.py
"""
import pandas as pd
import numpy as np

print("=" * 65)
print("  XML (apple_health_parse.py) vs JSON (Health Auto Export)")
print("=" * 65)

# ── Sursa nouă: JSON daily aggregates ──
new = pd.read_csv("apple_watch_api/apple_health_daily.csv")
new["date"] = pd.to_datetime(new["date"]).dt.strftime("%Y-%m-%d")

# ── Sursa veche: XML raw CSVs → agregare zilnică ──
print("\nLoading XML data (may take a minute)...")

# Heart Rate
hr = pd.read_csv("data/apple_health/heart_rate.csv")
hr["date"] = hr["startDate"].str[:10]
hr_daily = hr.groupby("date")["value"].agg(
    xml_hr_mean="mean", xml_hr_min="min", xml_hr_max="max"
).round(2).reset_index()

# HRV
hrv = pd.read_csv("data/apple_health/hrv.csv")
hrv["date"] = hrv["startDate"].str[:10]
hrv_daily = hrv.groupby("date")["value"].mean().round(2).reset_index()
hrv_daily.columns = ["date", "xml_hrv_mean"]

# Steps
steps = pd.read_csv("data/apple_health/steps.csv")
steps["date"] = steps["startDate"].str[:10]
steps_daily = steps.groupby("date")["value"].sum().round(0).reset_index()
steps_daily.columns = ["date", "xml_steps"]

# SpO2
spo2 = pd.read_csv("data/apple_health/spo2.csv")
spo2["date"] = spo2["startDate"].str[:10]
spo2_daily = spo2.groupby("date")["value"].mean().round(2).reset_index()
spo2_daily.columns = ["date", "xml_spo2"]

# Active calories
ac = pd.read_csv("data/apple_health/active_calories.csv")
ac["date"] = ac["startDate"].str[:10]
ac_daily = ac.groupby("date")["value"].sum().round(1).reset_index()
ac_daily.columns = ["date", "xml_active_cal"]

# ── Merge XML ──
xml = hr_daily.copy()
for df in [hrv_daily, steps_daily, spo2_daily, ac_daily]:
    xml = xml.merge(df, on="date", how="outer")

# ── Merge XML + JSON ──
compare = xml.merge(
    new[["date", "aw_hr_mean", "aw_hr_min", "aw_hr_max",
         "aw_hrv_mean", "aw_steps", "aw_spo2", "aw_active_energy_kj"]],
    on="date", how="inner"
)

# Convert JSON active energy kJ → kcal for fair comparison
compare["json_active_cal"] = (compare["aw_active_energy_kj"] / 4.184).round(1)

print(f"\nOverlapping days: {len(compare)}")
print(f"Date range: {compare['date'].min()} → {compare['date'].max()}")

# ── Comparație pe 5 zile random ──
sample = compare.dropna(subset=["xml_hr_mean", "aw_hr_mean"]).sample(
    n=min(5, len(compare)), random_state=42
).sort_values("date")

print(f"\n{'─'*65}")
print(f"  Sample comparison (5 random days)")
print(f"{'─'*65}")

for _, row in sample.iterrows():
    print(f"\n  Date: {row['date']}")
    print(f"  {'Metric':<20s} {'XML':>10s} {'JSON':>10s} {'Diff%':>8s}")
    print(f"  {'─'*50}")

    pairs = [
        ("HR mean",   row["xml_hr_mean"],  row["aw_hr_mean"]),
        ("HR min",    row["xml_hr_min"],   row["aw_hr_min"]),
        ("HR max",    row["xml_hr_max"],   row["aw_hr_max"]),
        ("HRV mean",  row["xml_hrv_mean"], row["aw_hrv_mean"]),
        ("Steps",     row["xml_steps"],    row["aw_steps"]),
        ("SpO2",      row["xml_spo2"],     row["aw_spo2"]),
        ("Active cal", row["xml_active_cal"], row["json_active_cal"]),
    ]
    for name, xml_val, json_val in pairs:
        if pd.notna(xml_val) and pd.notna(json_val) and xml_val != 0:
            diff = abs(json_val - xml_val) / xml_val * 100
            flag = " ⚠" if diff > 10 else " ✓"
            print(f"  {name:<20s} {xml_val:>10.1f} {json_val:>10.1f} {diff:>7.1f}%{flag}")
        else:
            print(f"  {name:<20s} {str(xml_val):>10s} {str(json_val):>10s}     N/A")

# ── Statistici globale ──
print(f"\n{'─'*65}")
print(f"  Global correlation (all overlapping days)")
print(f"{'─'*65}")

metrics = [
    ("HR mean",  "xml_hr_mean",  "aw_hr_mean"),
    ("HRV mean", "xml_hrv_mean", "aw_hrv_mean"),
    ("Steps",    "xml_steps",    "aw_steps"),
    ("SpO2",     "xml_spo2",     "aw_spo2"),
]

for name, xml_col, json_col in metrics:
    valid = compare[[xml_col, json_col]].dropna()
    if len(valid) > 2:
        corr = valid[xml_col].corr(valid[json_col])
        mae = (valid[xml_col] - valid[json_col]).abs().mean()
        print(f"  {name:<15s}  r={corr:.4f}  MAE={mae:.2f}  (n={len(valid)})")

print(f"\n{'='*65}")
print(f"  DONE")
print(f"{'='*65}")
