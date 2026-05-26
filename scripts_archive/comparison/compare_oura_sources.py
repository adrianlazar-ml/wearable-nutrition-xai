"""
Oura Data Comparison: Historical CSV vs Membership Export
==========================================================
Compară datele din oura_historical.csv (export vechi / API-generated)
cu CSV-urile din exportul membership.ouraring.com.

Usage:
    python compare_oura_sources.py
    python compare_oura_sources.py "C:\path\to\historical.csv" "C:\path\to\export_folder"
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# === CONFIG ===
HISTORICAL_PATH = sys.argv[1] if len(sys.argv) > 1 else r"data\oura_historical.csv"
EXPORT_DIR = sys.argv[2] if len(sys.argv) > 2 else r"data\Oura\raw"
OUTPUT_FILE = "oura_comparison_report.txt"


def load_historical(path):
    """Încarcă historical CSV (comma-separated)."""
    df = pd.read_csv(path, sep=",", low_memory=False, na_values=["None", "none", ""])
    df["date"] = pd.to_datetime(df["date"], format="mixed").dt.date
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    return df


def load_export_csv(export_dir, filename, sep=";"):
    """Încarcă un CSV din exportul membership."""
    path = Path(export_dir) / f"{filename}.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return None
    df = pd.read_csv(path, sep=sep, low_memory=False)
    if "day" in df.columns:
        df["date"] = pd.to_datetime(df["day"], format="mixed").dt.date
    return df


def load_sleepmodel(export_dir):
    """Încarcă sleepmodel și filtrează doar long_sleep (somnul principal)."""
    df = load_export_csv(export_dir, "sleepmodel")
    if df is None:
        return None
    # Păstrăm doar somnul principal (long_sleep, period=0)
    # period=0 = main sleep, period>0 = naps
    df_main = df[(df["type"] == "long_sleep") & (df["period"] == 0)].copy()
    # Dacă sunt duplicate pe aceeași zi, păstrăm prima intrare
    df_main = df_main.drop_duplicates(subset=["date"], keep="first")
    return df_main


def compare_columns(merged, hist_col, export_col, col_label, is_duration=False):
    """Compară două coloane și returnează statistici."""
    # Clean data
    h = pd.to_numeric(merged[hist_col], errors="coerce")
    e = pd.to_numeric(merged[export_col], errors="coerce")

    mask = h.notna() & e.notna()
    h_clean = h[mask]
    e_clean = e[mask]
    n = mask.sum()

    if n == 0:
        return {"label": col_label, "n": 0, "note": "No overlapping data"}

    diff = h_clean - e_clean
    abs_diff = diff.abs()

    # Exact matches
    exact = (abs_diff < 0.01).sum()
    close = (abs_diff < 1.0).sum()

    # Correlation
    if h_clean.std() > 0 and e_clean.std() > 0:
        corr = h_clean.corr(e_clean)
    else:
        corr = np.nan

    result = {
        "label": col_label,
        "n": n,
        "exact_match": exact,
        "exact_pct": round(exact / n * 100, 1),
        "close_match": close,
        "close_pct": round(close / n * 100, 1),
        "diff_mean": round(diff.mean(), 2),
        "diff_median": round(diff.median(), 2),
        "diff_std": round(diff.std(), 2),
        "diff_max": round(abs_diff.max(), 2),
        "corr": round(corr, 4) if not np.isnan(corr) else "N/A",
        "hist_mean": round(h_clean.mean(), 2),
        "export_mean": round(e_clean.mean(), 2),
    }

    # Flag major discrepancies
    major = (abs_diff > abs_diff.quantile(0.95)).sum() if n > 10 else 0
    result["outliers_5pct"] = major

    return result


def format_report(comparisons, date_stats, lines_extra):
    """Generează raportul text."""
    lines = []
    lines.append("=" * 95)
    lines.append("OURA DATA COMPARISON: Historical CSV vs Membership Export")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 95)

    # Date overlap
    lines.append("\n" + "─" * 70)
    lines.append("DATE COVERAGE")
    lines.append("─" * 70)
    for k, v in date_stats.items():
        lines.append(f"  {k:<35} {v}")

    # Comparison summary table
    lines.append("\n" + "─" * 95)
    lines.append("COMPARISON SUMMARY")
    lines.append("─" * 95)
    lines.append(
        f"{'Metric':<35} {'N':>5} {'Exact%':>7} {'Close%':>7}"
        f" {'MeanDiff':>9} {'MaxDiff':>9} {'Corr':>7} {'Verdict':>10}"
    )
    lines.append("-" * 95)

    for c in comparisons:
        if c.get("note"):
            lines.append(f"{c['label']:<35} {c['n']:>5} {'—':>7} {'—':>7} {'—':>9} {'—':>9} {'—':>7} {c['note']:>10}")
            continue

        # Verdict logic
        if c["exact_pct"] > 95:
            verdict = "IDENTICAL"
        elif c["exact_pct"] > 80 or (isinstance(c["corr"], float) and c["corr"] > 0.99):
            verdict = "MATCH"
        elif isinstance(c["corr"], float) and c["corr"] > 0.95:
            verdict = "CLOSE"
        elif isinstance(c["corr"], float) and c["corr"] > 0.80:
            verdict = "DIFFERS"
        else:
            verdict = "CHECK"

        corr_str = f"{c['corr']:.4f}" if isinstance(c["corr"], float) else c["corr"]
        lines.append(
            f"{c['label']:<35} {c['n']:>5} {c['exact_pct']:>6.1f}% {c['close_pct']:>6.1f}%"
            f" {c['diff_mean']:>9.2f} {c['diff_max']:>9.2f} {corr_str:>7} {verdict:>10}"
        )

    # Detailed per-metric
    lines.append("\n" + "─" * 95)
    lines.append("DETAILED ANALYSIS PER METRIC")
    lines.append("─" * 95)

    for c in comparisons:
        if c.get("note") or c["n"] == 0:
            continue
        lines.append(f"\n  {c['label']}:")
        lines.append(f"    Overlapping days: {c['n']}")
        lines.append(f"    Exact matches (diff<0.01): {c['exact_match']} ({c['exact_pct']}%)")
        lines.append(f"    Close matches (diff<1.0):  {c['close_match']} ({c['close_pct']}%)")
        lines.append(f"    Difference: mean={c['diff_mean']}, median={c['diff_median']}, std={c['diff_std']}, max={c['diff_max']}")
        lines.append(f"    Correlation: {c['corr']}")
        lines.append(f"    Historical mean: {c['hist_mean']}  |  Export mean: {c['export_mean']}")

    # Extra info
    if lines_extra:
        lines.append("\n" + "─" * 95)
        lines.append("ADDITIONAL NOTES")
        lines.append("─" * 95)
        for le in lines_extra:
            lines.append(f"  {le}")

    lines.append("\n" + "=" * 95)
    lines.append("INTERPRETATION GUIDE")
    lines.append("=" * 95)
    lines.append("""
  IDENTICAL = >95% exact matches (diff < 0.01)
  MATCH     = >80% exact OR correlation > 0.99
  CLOSE     = Correlation > 0.95, minor differences
  DIFFERS   = Correlation 0.80-0.95, systematic differences
  CHECK     = Correlation < 0.80, investigate further

  Common reasons for differences:
  - Historical CSV may aggregate sleep differently (main sleep only vs all periods)
  - Rounding differences (historical may round to integer, export keeps decimals)
  - Time zone handling (timestamp vs day assignment)
  - None/null handling (historical uses string "None", export uses NaN)
  - Score recalculations (Oura may retroactively update scores)
""")

    return "\n".join(lines)


def main():
    hist_path = Path(HISTORICAL_PATH)
    export_dir = Path(EXPORT_DIR)

    if not hist_path.exists():
        print(f"ERROR: Historical CSV not found: {hist_path}")
        sys.exit(1)
    if not export_dir.exists():
        print(f"ERROR: Export directory not found: {export_dir}")
        sys.exit(1)

    print("Loading historical CSV...")
    hist = load_historical(hist_path)
    print(f"  → {len(hist)} rows, date range: {hist['date'].min()} → {hist['date'].max()}")

    print("\nLoading export CSVs...")
    readiness = load_export_csv(export_dir, "dailyreadiness")
    sleep = load_export_csv(export_dir, "dailysleep")
    activity = load_export_csv(export_dir, "dailyactivity")
    spo2 = load_export_csv(export_dir, "dailyspo2")
    sleepmodel = load_sleepmodel(export_dir)
    stress = load_export_csv(export_dir, "dailystress")

    # Date coverage stats
    hist_dates = set(hist["date"])
    export_dates = {}
    for name, df in [("readiness", readiness), ("sleep", sleep),
                      ("activity", activity), ("sleepmodel", sleepmodel)]:
        if df is not None and "date" in df.columns:
            export_dates[name] = set(df["date"])

    all_export = set()
    for v in export_dates.values():
        all_export |= v

    overlap = hist_dates & all_export

    date_stats = {
        "Historical CSV days": len(hist_dates),
        "Historical range": f"{min(hist_dates)} → {max(hist_dates)}",
        "Export days (readiness)": len(export_dates.get("readiness", set())),
        "Export days (activity)": len(export_dates.get("activity", set())),
        "Export days (sleepmodel)": len(export_dates.get("sleepmodel", set())),
        "Overlapping days": len(overlap),
        "Only in historical": len(hist_dates - all_export),
        "Only in export": len(all_export - hist_dates),
    }

    # === MERGE AND COMPARE ===
    comparisons = []
    extra_notes = []

    # --- READINESS ---
    if readiness is not None:
        m = pd.merge(hist, readiness, on="date", suffixes=("_hist", "_exp"))
        print(f"  Readiness merged: {len(m)} rows")

        comparisons.append(compare_columns(m, "Readiness Score", "score", "Readiness Score"))
        # Find temperature deviation column (encoding varies: °C, Â°C, etc.)
        temp_col = [c for c in m.columns if "Temperature Deviation" in c and "Trend" not in c]
        if temp_col:
            comparisons.append(compare_columns(
                m, temp_col[0], "temperature_deviation",
                "Temperature Deviation"
            ))
        comparisons.append(compare_columns(
            m, "Temperature Trend Deviation", "temperature_trend_deviation",
            "Temp Trend Deviation"
        ))

    # --- SLEEP SCORE ---
    if sleep is not None:
        m = pd.merge(hist, sleep, on="date", suffixes=("_hist", "_exp"))
        print(f"  Sleep merged: {len(m)} rows")
        comparisons.append(compare_columns(m, "Sleep Score", "score", "Sleep Score"))

    # --- ACTIVITY ---
    if activity is not None:
        m = pd.merge(hist, activity, on="date", suffixes=("_hist", "_exp"))
        print(f"  Activity merged: {len(m)} rows")

        comparisons.append(compare_columns(m, "Activity Score", "score", "Activity Score"))
        comparisons.append(compare_columns(m, "Steps", "steps", "Steps"))
        comparisons.append(compare_columns(m, "Activity Burn", "active_calories", "Active Calories"))
        comparisons.append(compare_columns(m, "Total Burn", "total_calories", "Total Calories"))
        comparisons.append(compare_columns(m, "Average MET", "average_met_minutes", "Average MET"))
        comparisons.append(compare_columns(
            m, "Equivalent Walking Distance", "equivalent_walking_distance",
            "Walking Distance"
        ))
        comparisons.append(compare_columns(
            m, "Medium Activity Time", "medium_activity_time", "Medium Activity Time"
        ))
        comparisons.append(compare_columns(
            m, "High Activity Time", "high_activity_time", "High Activity Time"
        ))
        comparisons.append(compare_columns(
            m, "Low Activity Time", "low_activity_time", "Low Activity Time"
        ))
        comparisons.append(compare_columns(m, "Non-wear Time", "non_wear_time", "Non-wear Time"))

    # --- SLEEPMODEL (detailed sleep data) ---
    if sleepmodel is not None:
        m = pd.merge(hist, sleepmodel, on="date", suffixes=("_hist", "_exp"))
        print(f"  Sleepmodel merged: {len(m)} rows")

        comparisons.append(compare_columns(
            m, "Total Sleep Duration", "total_sleep_duration", "Total Sleep Duration"
        ))
        comparisons.append(compare_columns(
            m, "Deep Sleep Duration", "deep_sleep_duration", "Deep Sleep Duration"
        ))
        comparisons.append(compare_columns(
            m, "REM Sleep Duration", "rem_sleep_duration", "REM Sleep Duration"
        ))
        comparisons.append(compare_columns(
            m, "Light Sleep Duration", "light_sleep_duration", "Light Sleep Duration"
        ))
        comparisons.append(compare_columns(
            m, "Average HRV", "average_hrv", "Average HRV"
        ))
        comparisons.append(compare_columns(
            m, "Average Resting Heart Rate", "average_heart_rate", "Avg Resting HR"
        ))
        comparisons.append(compare_columns(
            m, "Lowest Resting Heart Rate", "lowest_heart_rate", "Lowest Resting HR"
        ))
        comparisons.append(compare_columns(
            m, "Respiratory Rate", "average_breath", "Respiratory Rate"
        ))
        comparisons.append(compare_columns(
            m, "Sleep Efficiency", "efficiency", "Sleep Efficiency"
        ))
        comparisons.append(compare_columns(
            m, "Sleep Latency", "latency", "Sleep Latency"
        ))
        comparisons.append(compare_columns(
            m, "Awake Time", "awake_time", "Awake Time"
        ))
        # Total Bedtime has trailing space in some exports
        bedtime_col = [c for c in m.columns if c.strip() == "Total Bedtime"]
        if bedtime_col:
            comparisons.append(compare_columns(
                m, bedtime_col[0], "time_in_bed", "Total Bedtime"
            ))
        comparisons.append(compare_columns(
            m, "Restless Sleep", "restless_periods", "Restless Periods"
        ))

        # Check sleepmodel type distribution
        all_sm = load_export_csv(export_dir, "sleepmodel")
        if all_sm is not None:
            type_counts = all_sm["type"].value_counts()
            extra_notes.append(f"Sleepmodel type distribution: {dict(type_counts)}")
            period_counts = all_sm["period"].value_counts().head()
            extra_notes.append(f"Sleepmodel period distribution: {dict(period_counts)}")

    # --- SPO2 ---
    if spo2 is not None:
        # spo2_percentage is JSON like {"average": 97.915}, need to extract
        try:
            import json
            spo2["spo2_avg"] = spo2["spo2_percentage"].apply(
                lambda x: json.loads(x.replace("'", '"'))["average"]
                if pd.notna(x) and isinstance(x, str) else np.nan
            )
            m = pd.merge(hist, spo2, on="date", suffixes=("_hist", "_exp"))
            print(f"  SpO2 merged: {len(m)} rows")
            # Historical doesn't have SpO2 directly, skip if not present
            extra_notes.append(f"SpO2 data: {spo2['spo2_avg'].notna().sum()} days with data, "
                             f"mean={spo2['spo2_avg'].mean():.1f}%")
        except Exception as e:
            extra_notes.append(f"SpO2 parsing note: {e}")

    # --- STRESS ---
    if stress is not None:
        m_stress = pd.merge(hist, stress, on="date", suffixes=("_hist", "_exp"))
        stress_summary = stress["day_summary"].value_counts()
        extra_notes.append(f"Daily stress summary distribution: {dict(stress_summary)}")
        extra_notes.append(f"Stress data covers {len(stress)} days")

    # === Generate report ===
    report = format_report(comparisons, date_stats, extra_notes)
    print("\n" + report)

    # Save
    out_path = Path(EXPORT_DIR) / OUTPUT_FILE
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ Report saved to: {out_path}")


if __name__ == "__main__":
    main()