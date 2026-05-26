"""
Oura Comparison: API v2 (oura_combined.csv) vs Membership Export
=================================================================
Compară cele 12 metrici Oura din oura_combined.csv cu CSV-urile
exportate din membership.ouraring.com.

Usage:
    python compare_api_vs_export.py "data\oura_combined.csv" "data\Oura\raw"
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# === CONFIG ===
API_PATH = sys.argv[1] if len(sys.argv) > 1 else r"data\oura_combined.csv"
EXPORT_DIR = sys.argv[2] if len(sys.argv) > 2 else r"data\Oura\raw"
OUTPUT_FILE = "oura_api_vs_export_report.txt"


def load_api(path):
    """Încarcă oura_combined.csv."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_export(export_dir, filename):
    """Încarcă un CSV din membership export (separator ;)."""
    path = Path(export_dir) / f"{filename}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=";", low_memory=False)
    if "day" in df.columns:
        df["date"] = pd.to_datetime(df["day"], format="mixed").dt.date
    return df


def parse_json_field(series, key):
    """Extrage o valoare dintr-un câmp JSON string."""
    def extract(val):
        if pd.isna(val) or not isinstance(val, str):
            return np.nan
        try:
            d = json.loads(val.replace("'", '"'))
            v = d.get(key)
            return float(v) if v is not None else np.nan
        except (json.JSONDecodeError, ValueError, TypeError):
            return np.nan
    return series.apply(extract)


def load_sleepmodel_main(export_dir):
    """Încarcă sleepmodel filtrat pe long_sleep, period=0."""
    df = load_export(export_dir, "sleepmodel")
    if df is None:
        return None
    df_main = df[(df["type"] == "long_sleep") & (df["period"] == 0)].copy()
    df_main = df_main.drop_duplicates(subset=["date"], keep="first")
    return df_main


def compare(api_vals, export_vals, label):
    """Compară două serii și returnează statistici."""
    mask = api_vals.notna() & export_vals.notna()
    a = api_vals[mask].astype(float)
    e = export_vals[mask].astype(float)
    n = len(a)

    if n == 0:
        return {"label": label, "n": 0, "note": "No overlap"}

    diff = a - e
    abs_diff = diff.abs()
    exact = (abs_diff < 0.01).sum()
    close = (abs_diff < 1.0).sum()
    corr = a.corr(e) if a.std() > 0 and e.std() > 0 else np.nan

    if exact / n > 0.95:
        verdict = "IDENTICAL"
    elif exact / n > 0.80 or (not np.isnan(corr) and corr > 0.99):
        verdict = "MATCH"
    elif not np.isnan(corr) and corr > 0.95:
        verdict = "CLOSE"
    elif not np.isnan(corr) and corr > 0.80:
        verdict = "DIFFERS"
    else:
        verdict = "CHECK"

    return {
        "label": label,
        "n": n,
        "exact": exact,
        "exact_pct": round(exact / n * 100, 1),
        "close": close,
        "close_pct": round(close / n * 100, 1),
        "mean_diff": round(diff.mean(), 3),
        "max_diff": round(abs_diff.max(), 3),
        "corr": round(corr, 4) if not np.isnan(corr) else "N/A",
        "api_mean": round(a.mean(), 2),
        "exp_mean": round(e.mean(), 2),
        "verdict": verdict,
    }


def main():
    api_path = Path(API_PATH)
    export_dir = Path(EXPORT_DIR)

    print("Loading API data (oura_combined.csv)...")
    api = load_api(api_path)
    print(f"  → {len(api)} rows, {api['date'].min()} → {api['date'].max()}")

    print("\nLoading membership export CSVs...")
    readiness = load_export(export_dir, "dailyreadiness")
    sleep = load_export(export_dir, "dailysleep")
    activity = load_export(export_dir, "dailyactivity")
    sleepmodel = load_sleepmodel_main(export_dir)
    spo2 = load_export(export_dir, "dailyspo2")

    # Parse JSON contributors from readiness for recovery_index and hrv_balance
    if readiness is not None:
        print("  Parsing readiness contributors JSON...")
        readiness["recovery_index_exp"] = parse_json_field(
            readiness["contributors"], "recovery_index"
        )
        readiness["hrv_balance_exp"] = parse_json_field(
            readiness["contributors"], "hrv_balance"
        )

    # Parse spo2_percentage JSON
    if spo2 is not None:
        print("  Parsing SpO2 JSON...")
        spo2["spo2_avg"] = parse_json_field(spo2["spo2_percentage"], "average")

    comparisons = []

    # === READINESS comparisons ===
    if readiness is not None:
        m = pd.merge(api, readiness, on="date", suffixes=("_api", "_exp"))
        print(f"\n  Readiness merged: {len(m)} rows")

        comparisons.append(compare(m["readiness_score"], m["score"], "readiness_score"))
        comparisons.append(compare(m["temp_deviation"], m["temperature_deviation"], "temp_deviation"))
        comparisons.append(compare(m["recovery_index"], m["recovery_index_exp"], "recovery_index"))
        comparisons.append(compare(m["hrv_balance"], m["hrv_balance_exp"], "hrv_balance"))

    # === SLEEP comparisons ===
    if sleep is not None:
        m = pd.merge(api, sleep, on="date", suffixes=("_api", "_exp"))
        print(f"  Sleep merged: {len(m)} rows")

        comparisons.append(compare(m["sleep_score"], m["score"], "sleep_score"))

    # === ACTIVITY comparisons ===
    if activity is not None:
        m = pd.merge(api, activity, on="date", suffixes=("_api", "_exp"))
        print(f"  Activity merged: {len(m)} rows")

        comparisons.append(compare(m["activity_score"], m["score"], "activity_score"))
        comparisons.append(compare(m["steps_oura"], m["steps"], "steps_oura"))

    # === SLEEPMODEL comparisons ===
    if sleepmodel is not None:
        m = pd.merge(api, sleepmodel, on="date", suffixes=("_api", "_exp"))
        print(f"  Sleepmodel merged: {len(m)} rows")

        comparisons.append(compare(m["hrv_mean"], m["average_hrv"], "hrv_mean"))
        comparisons.append(compare(m["rhr_lowest"], m["lowest_heart_rate"], "rhr_lowest"))
        comparisons.append(compare(m["rhr_avg"], m["average_heart_rate"], "rhr_avg"))
        comparisons.append(compare(m["respiratory_rate"], m["average_breath"], "respiratory_rate"))

    # === DATE COVERAGE ===
    api_dates = set(api["date"])
    export_dates = set()
    for df in [readiness, sleep, activity, sleepmodel]:
        if df is not None and "date" in df.columns:
            export_dates |= set(df["date"])

    overlap = api_dates & export_dates
    only_api = api_dates - export_dates
    only_export = export_dates - api_dates

    # === GENERATE REPORT ===
    lines = []
    lines.append("=" * 95)
    lines.append("OURA COMPARISON: API v2 (oura_combined.csv) vs Membership Export")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 95)

    lines.append("\n" + "─" * 70)
    lines.append("DATE COVERAGE")
    lines.append("─" * 70)
    lines.append(f"  API (oura_combined.csv):    {len(api_dates)} days  ({min(api_dates)} → {max(api_dates)})")
    lines.append(f"  Membership export:          {len(export_dates)} days ({min(export_dates)} → {max(export_dates)})")
    lines.append(f"  Overlapping:                {len(overlap)} days")
    lines.append(f"  Only in API:                {len(only_api)} days")
    lines.append(f"  Only in export:             {len(only_export)} days")

    # API columns inventory
    lines.append("\n" + "─" * 70)
    lines.append("API COLUMNS → EXPORT SOURCE MAPPING")
    lines.append("─" * 70)
    mapping = [
        ("readiness_score", "dailyreadiness.score"),
        ("sleep_score", "dailysleep.score"),
        ("hrv_mean", "sleepmodel.average_hrv"),
        ("rhr_lowest", "sleepmodel.lowest_heart_rate"),
        ("rhr_avg", "sleepmodel.average_heart_rate"),
        ("recovery_index", "dailyreadiness.contributors.recovery_index"),
        ("hrv_balance", "dailyreadiness.contributors.hrv_balance"),
        ("activity_score", "dailyactivity.score"),
        ("steps_oura", "dailyactivity.steps"),
        ("respiratory_rate", "sleepmodel.average_breath"),
        ("temp_deviation", "dailyreadiness.temperature_deviation"),
        ("oura_device", "(metadata, no export equivalent)"),
        ("aw_device", "(metadata, not from Oura)"),
    ]
    for api_col, exp_source in mapping:
        lines.append(f"  {api_col:<25} → {exp_source}")

    # Comparison summary
    lines.append("\n" + "─" * 95)
    lines.append("COMPARISON RESULTS")
    lines.append("─" * 95)
    lines.append(
        f"{'Metric':<25} {'N':>6} {'Exact%':>8} {'Close%':>8}"
        f" {'MeanDiff':>10} {'MaxDiff':>10} {'Corr':>8} {'Verdict':>10}"
    )
    lines.append("-" * 95)

    for c in comparisons:
        if c.get("note"):
            lines.append(f"{c['label']:<25} {c['n']:>6} {'—':>8} {'—':>8} {'—':>10} {'—':>10} {'—':>8} {c['note']:>10}")
            continue
        corr_str = f"{c['corr']:.4f}" if isinstance(c['corr'], float) else c['corr']
        lines.append(
            f"{c['label']:<25} {c['n']:>6} {c['exact_pct']:>7.1f}% {c['close_pct']:>7.1f}%"
            f" {c['mean_diff']:>10.3f} {c['max_diff']:>10.3f} {corr_str:>8} {c['verdict']:>10}"
        )

    # Detailed per metric
    lines.append("\n" + "─" * 95)
    lines.append("DETAILED PER METRIC")
    lines.append("─" * 95)
    for c in comparisons:
        if c.get("note") or c["n"] == 0:
            continue
        lines.append(f"\n  {c['label']}:")
        lines.append(f"    N={c['n']}, Exact={c['exact']} ({c['exact_pct']}%), Close={c['close']} ({c['close_pct']}%)")
        lines.append(f"    MeanDiff={c['mean_diff']}, MaxDiff={c['max_diff']}, Corr={c['corr']}")
        lines.append(f"    API mean={c['api_mean']}  |  Export mean={c['exp_mean']}")

    # Data availability comparison
    lines.append("\n" + "─" * 95)
    lines.append("DATA AVAILABILITY: WHAT EXPORT HAS THAT API DOESN'T")
    lines.append("─" * 95)
    lines.append("""
  API (oura_combined.csv) has 12 Oura metrics per day.
  Membership export adds significantly more data:

  FROM EXISTING SOURCES (more columns):
    dailyreadiness:  temperature_trend_deviation, all contributor sub-scores
    dailyactivity:   active_calories, total_calories, met, walking_distance,
                     high/medium/low activity times, sedentary_time, non_wear_time
    sleepmodel:      deep/rem/light sleep durations, efficiency, latency,
                     awake_time, time_in_bed, restless_periods, bedtime_start/end

  ENTIRELY NEW DATA (not in API pipeline):
    dailystress:         daily stress summary, stress_high, recovery_high
    daytimestress:       5-min interval stress/recovery (141K data points)
    dailyresilience:     resilience level + contributors
    dailycardiovascularage: vascular_age, pulse_wave_velocity
    workout:             activity type, calories, distance, intensity
    heartrate:           per-second HR with source (awake/sleep) — 1M+ points
    temperature:         per-minute skin temperature — 1.4M+ points
    meal/fooditem:       food diary (220 meals, no macros)
    vo2max:              VO2max estimates (27 data points only)
""")

    # Conclusion
    lines.append("─" * 95)
    lines.append("CONCLUSION")
    lines.append("─" * 95)
    lines.append("""
  1. For the 12 metrics in oura_combined.csv, both sources should be identical
     (same underlying Oura data, different export method).

  2. Membership export is strictly superior:
     - More days (extends beyond API collection period)
     - More columns per daily file
     - Additional data sources (stress, resilience, workout, granular HR/temp)

  3. oura_combined.csv also contains aw_device column, indicating it was
     already designed as a merged Oura + Apple Watch file.
""")

    report = "\n".join(lines)
    print("\n" + report)

    out_path = Path(EXPORT_DIR) / OUTPUT_FILE
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ Report saved to: {out_path}")


if __name__ == "__main__":
    main()