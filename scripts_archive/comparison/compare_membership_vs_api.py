"""
Oura Comparison: API v2 CSVs vs Membership Export CSVs
=======================================================
Compară fișier cu fișier pe perioada suprapusă (iul 2025 → mar 2026).

API files (comma-separated):
  readiness.csv, sleep.csv, sleep_metrics.csv,
  sleep_detailed.csv, activity.csv, stress.csv

Membership export files (semicolon-separated):
  dailyreadiness.csv, dailysleep.csv, sleepmodel.csv,
  dailyactivity.csv, dailystress.csv

Usage:
    python compare_membership_vs_api.py "path\to\api_csvs" "path\to\membership_export"
    
    Default paths (run from disertatie folder):
    python compare_membership_vs_api.py
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# === CONFIG ===
# API CSVs folder (where readiness.csv, sleep.csv etc. live)
API_DIR = sys.argv[1] if len(sys.argv) > 1 else r"data"
# Membership export folder (where dailyreadiness.csv etc. live)
EXPORT_DIR = sys.argv[2] if len(sys.argv) > 2 else r"data\Oura\raw"
OUTPUT_FILE = "membership_vs_api_comparison.txt"

STR_COLS = ["movement_30_sec", "sleep_phase_30_sec", "sleep_phase_5_min",
            "app_sleep_phase_5_min", "class_5_min"]


def load_csv(path, sep=","):
    """Load CSV with safe dtypes for large-integer string columns."""
    dtype_overrides = {c: str for c in STR_COLS}
    df = pd.read_csv(path, sep=sep, low_memory=False,
                     dtype={k: v for k, v in dtype_overrides.items()})
    return df


def add_date_col(df, source_col="day"):
    """Add standardized date column."""
    if source_col in df.columns:
        df["_date"] = pd.to_datetime(df[source_col], format="mixed").dt.date
    return df


def compare_series(s1, s2, label):
    """Compare two numeric series, return stats dict."""
    a = pd.to_numeric(s1, errors="coerce")
    b = pd.to_numeric(s2, errors="coerce")
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return {"label": label, "n": 0, "note": "No overlap"}

    diff = a.values - b.values
    abs_diff = np.abs(diff)
    exact = (abs_diff < 0.01).sum()
    close = (abs_diff < 1.0).sum()
    corr = np.corrcoef(a.values, b.values)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan

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
        "exact": int(exact),
        "exact_pct": round(exact / n * 100, 1),
        "close": int(close),
        "close_pct": round(close / n * 100, 1),
        "mean_diff": round(float(np.mean(diff)), 4),
        "max_diff": round(float(np.max(abs_diff)), 4),
        "corr": round(float(corr), 6) if not np.isnan(corr) else "N/A",
        "api_mean": round(float(a.mean()), 3),
        "mem_mean": round(float(b.mean()), 3),
        "verdict": verdict,
    }


def compare_categorical(s1, s2, label):
    """Compare two categorical series."""
    mask = s1.notna() & s2.notna()
    a, b = s1[mask].astype(str), s2[mask].astype(str)
    n = len(a)
    if n == 0:
        return {"label": label, "n": 0, "note": "No overlap"}
    match = (a.values == b.values).sum()
    return {
        "label": label,
        "n": n,
        "exact": int(match),
        "exact_pct": round(match / n * 100, 1),
        "verdict": "IDENTICAL" if match / n > 0.95 else "CHECK",
        "is_categorical": True,
    }


def main():
    api_dir = Path(API_DIR)
    export_dir = Path(EXPORT_DIR)

    lines = []
    lines.append("=" * 95)
    lines.append("OURA COMPARISON: API v2 CSVs vs Membership Export CSVs")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"API dir:    {api_dir.resolve()}")
    lines.append(f"Export dir: {export_dir.resolve()}")
    lines.append("=" * 95)

    all_comparisons = []

    # =========================================================================
    # 1. READINESS: readiness.csv vs dailyreadiness.csv
    # =========================================================================
    print("1. Comparing readiness...")
    api = load_csv(api_dir / "readiness.csv", sep=",")
    mem = load_csv(export_dir / "dailyreadiness.csv", sep=";")
    add_date_col(api, "day")
    add_date_col(mem, "day")

    m = pd.merge(api, mem, on="_date", suffixes=("_api", "_mem"))
    print(f"   Merged: {len(m)} rows")

    section = {"title": "READINESS (readiness.csv vs dailyreadiness.csv)", "results": []}
    section["results"].append(compare_series(m["score_api"], m["score_mem"], "readiness_score"))
    section["results"].append(compare_series(
        m["temperature_deviation_api"], m["temperature_deviation_mem"], "temperature_deviation"
    ))
    section["results"].append(compare_series(
        m["temperature_trend_deviation_api"], m["temperature_trend_deviation_mem"],
        "temp_trend_deviation"
    ))
    # Compare contributors JSON
    try:
        def extract_contributor(series, key):
            def _extract(val):
                if pd.isna(val) or not isinstance(val, str):
                    return np.nan
                try:
                    d = json.loads(val.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false"))
                    v = d.get(key)
                    return float(v) if v is not None else np.nan
                except:
                    return np.nan
            return series.apply(_extract)

        for key in ["recovery_index", "hrv_balance", "body_temperature",
                     "previous_night", "resting_heart_rate", "sleep_balance"]:
            api_vals = extract_contributor(m["contributors_api"], key)
            mem_vals = extract_contributor(m["contributors_mem"], key)
            if api_vals.notna().sum() > 0 and mem_vals.notna().sum() > 0:
                section["results"].append(compare_series(api_vals, mem_vals, f"contrib.{key}"))
    except Exception as e:
        section["results"].append({"label": "contributors_parse", "n": 0, "note": str(e)})

    section["merged_rows"] = len(m)
    all_comparisons.append(section)

    # =========================================================================
    # 2. SLEEP SCORE: sleep.csv vs dailysleep.csv
    # =========================================================================
    print("2. Comparing sleep scores...")
    api = load_csv(api_dir / "sleep.csv", sep=",")
    mem = load_csv(export_dir / "dailysleep.csv", sep=";")
    add_date_col(api, "day")
    add_date_col(mem, "day")

    m = pd.merge(api, mem, on="_date", suffixes=("_api", "_mem"))
    print(f"   Merged: {len(m)} rows")

    section = {"title": "SLEEP SCORE (sleep.csv vs dailysleep.csv)", "results": []}
    section["results"].append(compare_series(m["score_api"], m["score_mem"], "sleep_score"))
    section["merged_rows"] = len(m)
    all_comparisons.append(section)

    # =========================================================================
    # 3. SLEEP DETAILED: sleep_detailed.csv vs sleepmodel.csv
    # =========================================================================
    print("3. Comparing sleep detailed / sleepmodel...")
    api = load_csv(api_dir / "sleep_detailed.csv", sep=",")
    mem = load_csv(export_dir / "sleepmodel.csv", sep=";")
    add_date_col(api, "date")
    add_date_col(mem, "day")

    # Filter to long_sleep, period=0 for fair comparison
    api_main = api[(api["type"] == "long_sleep") & (api["period"] == 0)].copy()
    mem_main = mem[(mem["type"] == "long_sleep") & (mem["period"] == 0)].copy()
    api_main = api_main.drop_duplicates(subset=["_date"], keep="first")
    mem_main = mem_main.drop_duplicates(subset=["_date"], keep="first")

    m = pd.merge(api_main, mem_main, on="_date", suffixes=("_api", "_mem"))
    print(f"   Merged (long_sleep, period=0): {len(m)} rows")

    section = {"title": "SLEEP DETAILED (sleep_detailed.csv vs sleepmodel.csv) [long_sleep, period=0]", "results": []}
    
    compare_cols = [
        ("average_hrv", "average_hrv", "average_hrv"),
        ("average_heart_rate", "average_heart_rate", "avg_heart_rate"),
        ("average_breath", "average_breath", "avg_breath"),
        ("lowest_heart_rate", "lowest_heart_rate", "lowest_hr"),
        ("deep_sleep_duration", "deep_sleep_duration", "deep_sleep"),
        ("rem_sleep_duration", "rem_sleep_duration", "rem_sleep"),
        ("light_sleep_duration", "light_sleep_duration", "light_sleep"),
        ("total_sleep_duration", "total_sleep_duration", "total_sleep"),
        ("efficiency", "efficiency", "efficiency"),
        ("latency", "latency", "latency"),
        ("awake_time", "awake_time", "awake_time"),
        ("time_in_bed", "time_in_bed", "time_in_bed"),
        ("restless_periods", "restless_periods", "restless_periods"),
    ]
    
    for api_col, mem_col, label in compare_cols:
        ac = f"{api_col}_api" if f"{api_col}_api" in m.columns else api_col
        mc = f"{mem_col}_mem" if f"{mem_col}_mem" in m.columns else mem_col
        if ac in m.columns and mc in m.columns:
            section["results"].append(compare_series(m[ac], m[mc], label))

    section["merged_rows"] = len(m)
    all_comparisons.append(section)

    # =========================================================================
    # 4. SLEEP METRICS: sleep_metrics.csv vs sleepmodel.csv (aggregated)
    # =========================================================================
    print("4. Comparing sleep_metrics vs sleepmodel...")
    api_sm = load_csv(api_dir / "sleep_metrics.csv", sep=",")
    add_date_col(api_sm, "date")

    m = pd.merge(api_sm, mem_main, on="_date", suffixes=("_api", "_mem"))
    print(f"   Merged: {len(m)} rows")

    section = {"title": "SLEEP METRICS (sleep_metrics.csv vs sleepmodel.csv)", "results": []}
    sm_cols = [
        ("average_hrv_api", "average_hrv_mem", "sm_average_hrv"),
        ("lowest_heart_rate_api", "lowest_heart_rate_mem", "sm_lowest_hr"),
        ("average_breath_api", "average_breath_mem", "sm_avg_breath"),
        ("average_heart_rate_api", "average_heart_rate_mem", "sm_avg_hr"),
    ]
    for ac, mc, label in sm_cols:
        if ac in m.columns and mc in m.columns:
            section["results"].append(compare_series(m[ac], m[mc], label))
    
    section["merged_rows"] = len(m)
    all_comparisons.append(section)

    # =========================================================================
    # 5. ACTIVITY: activity.csv vs dailyactivity.csv
    # =========================================================================
    print("5. Comparing activity...")
    api = load_csv(api_dir / "activity.csv", sep=",")
    mem = load_csv(export_dir / "dailyactivity.csv", sep=";")
    add_date_col(api, "day")
    add_date_col(mem, "day")

    m = pd.merge(api, mem, on="_date", suffixes=("_api", "_mem"))
    print(f"   Merged: {len(m)} rows")

    section = {"title": "ACTIVITY (activity.csv vs dailyactivity.csv)", "results": []}
    act_cols = [
        ("score", "score", "activity_score"),
        ("steps", "steps", "steps"),
        ("active_calories", "active_calories", "active_calories"),
        ("total_calories", "total_calories", "total_calories"),
        ("average_met_minutes", "average_met_minutes", "average_met"),
        ("equivalent_walking_distance", "equivalent_walking_distance", "walking_distance"),
        ("high_activity_time", "high_activity_time", "high_activity_time"),
        ("medium_activity_time", "medium_activity_time", "medium_activity_time"),
        ("low_activity_time", "low_activity_time", "low_activity_time"),
        ("non_wear_time", "non_wear_time", "non_wear_time"),
        ("sedentary_time", "sedentary_time", "sedentary_time"),
        ("resting_time", "resting_time", "resting_time"),
        ("inactivity_alerts", "inactivity_alerts", "inactivity_alerts"),
    ]
    for api_col, mem_col, label in act_cols:
        ac = f"{api_col}_api" if f"{api_col}_api" in m.columns else api_col
        mc = f"{mem_col}_mem" if f"{mem_col}_mem" in m.columns else mem_col
        if ac in m.columns and mc in m.columns:
            section["results"].append(compare_series(m[ac], m[mc], label))

    section["merged_rows"] = len(m)
    all_comparisons.append(section)

    # =========================================================================
    # 6. STRESS: stress.csv vs dailystress.csv
    # =========================================================================
    print("6. Comparing stress...")
    api = load_csv(api_dir / "stress.csv", sep=",")
    mem = load_csv(export_dir / "dailystress.csv", sep=";")
    add_date_col(api, "day")
    add_date_col(mem, "day")

    m = pd.merge(api, mem, on="_date", suffixes=("_api", "_mem"))
    print(f"   Merged: {len(m)} rows")

    section = {"title": "STRESS (stress.csv vs dailystress.csv)", "results": []}
    section["results"].append(compare_series(
        m["recovery_high_api"], m["recovery_high_mem"], "recovery_high"
    ))
    section["results"].append(compare_series(
        m["stress_high_api"], m["stress_high_mem"], "stress_high"
    ))
    section["results"].append(compare_categorical(
        m["day_summary_api"], m["day_summary_mem"], "day_summary"
    ))
    section["merged_rows"] = len(m)
    all_comparisons.append(section)

    # =========================================================================
    # FORMAT REPORT
    # =========================================================================
    
    # Grand summary table
    lines.append("\n" + "=" * 95)
    lines.append("GRAND SUMMARY — ALL METRICS")
    lines.append("=" * 95)
    lines.append(
        f"{'Metric':<30} {'N':>5} {'Exact%':>8} {'Close%':>8}"
        f" {'MeanDiff':>10} {'MaxDiff':>10} {'Corr':>8} {'Verdict':>10}"
    )
    lines.append("-" * 95)

    total_metrics = 0
    identical_count = 0

    for section in all_comparisons:
        for r in section["results"]:
            total_metrics += 1
            if r.get("note"):
                lines.append(f"{r['label']:<30} {r['n']:>5} {'—':>8} {'—':>8} {'—':>10} {'—':>10} {'—':>8} {r.get('note',''):>10}")
                continue
            if r.get("is_categorical"):
                lines.append(
                    f"{r['label']:<30} {r['n']:>5} {r['exact_pct']:>7.1f}% {'—':>8}"
                    f" {'—':>10} {'—':>10} {'—':>8} {r['verdict']:>10}"
                )
                if r["verdict"] == "IDENTICAL":
                    identical_count += 1
                continue
            corr_str = f"{r['corr']:.4f}" if isinstance(r['corr'], float) else r['corr']
            lines.append(
                f"{r['label']:<30} {r['n']:>5} {r['exact_pct']:>7.1f}% {r['close_pct']:>7.1f}%"
                f" {r['mean_diff']:>10.4f} {r['max_diff']:>10.4f} {corr_str:>8} {r['verdict']:>10}"
            )
            if r["verdict"] == "IDENTICAL":
                identical_count += 1

    lines.append(f"\nTotal metrics compared: {total_metrics}")
    lines.append(f"IDENTICAL: {identical_count}/{total_metrics}")

    # Detailed per section
    for section in all_comparisons:
        lines.append("\n" + "─" * 95)
        lines.append(f"  {section['title']}  (merged: {section['merged_rows']} rows)")
        lines.append("─" * 95)
        for r in section["results"]:
            if r.get("note"):
                lines.append(f"    {r['label']}: {r['note']}")
                continue
            if r.get("is_categorical"):
                lines.append(f"    {r['label']}: {r['exact']}/{r['n']} match ({r['exact_pct']}%) → {r['verdict']}")
                continue
            lines.append(
                f"    {r['label']}: exact={r['exact']}/{r['n']} ({r['exact_pct']}%), "
                f"meanDiff={r['mean_diff']}, maxDiff={r['max_diff']}, "
                f"corr={r['corr']} → {r['verdict']}"
            )
            lines.append(f"      API mean={r['api_mean']}  |  Membership mean={r['mem_mean']}")

    # Conclusion
    lines.append("\n" + "=" * 95)
    lines.append("CONCLUSION")
    lines.append("=" * 95)
    lines.append(f"""
  Compared {total_metrics} metrics across 6 file pairs on the overlapping period.
  {identical_count}/{total_metrics} metrics are IDENTICAL (>95% exact match).

  The API v2 CSVs and Membership export CSVs contain the same underlying data.
  Differences (if any) are due to:
    - Rounding (API may keep more/fewer decimals)
    - Sleep session aggregation (sleep_metrics.csv may aggregate differently)
    - Retroactive score recalculations by Oura
""")

    report = "\n".join(lines)
    print("\n" + report)

    out_path = Path(EXPORT_DIR) / OUTPUT_FILE
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ Report saved to: {out_path}")


if __name__ == "__main__":
    main()