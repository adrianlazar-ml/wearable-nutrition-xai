"""
Oura CSV Export Explorer v2
============================
Fix: Oura CSV export uses semicolon (;) as delimiter.
Scanează toate CSV-urile din exportul Oura și generează un raport detaliat.

Usage:
    python explore_oura_export_v2.py
    python explore_oura_export_v2.py /path/to/oura_export_folder
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# === CONFIG ===
EXPORT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
OUTPUT_FILE = "oura_export_report_v2.txt"

# Fișierele relevante pentru pipeline-ul de clasificare Rest/Normal/Active
RELEVANT_FILES = {
    "dailyreadiness", "dailysleep", "dailyactivity", "dailyspo2",
    "heartrate", "temperature", "sleepmodel", "dailystress",
    "daytimestress", "dailycardiovascularage", "dailyresilience",
    "workout", "vo2max", "sleeptime"
}


def detect_separator(filepath):
    """Detectează separatorul CSV (virgulă sau punct-virgulă)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    if header.count(";") > header.count(","):
        return ";"
    return ","


def find_date_columns(df):
    """Identifică coloanele care conțin date/timestamp-uri."""
    date_cols = []
    for col in df.columns:
        col_lower = col.lower()
        # Check by name first
        if any(kw in col_lower for kw in ["day", "date", "timestamp", "datetime", "start_d", "end_d"]):
            sample = df[col].dropna().head(20)
            if len(sample) == 0:
                continue
            try:
                pd.to_datetime(sample, format="mixed")
                date_cols.append(col)
            except (ValueError, TypeError):
                try:
                    pd.to_datetime(sample)
                    date_cols.append(col)
                except (ValueError, TypeError):
                    continue
    return date_cols


def get_date_range(df, date_cols):
    """Returnează date range din prima coloană de date găsită."""
    # Prefer 'day' column over 'timestamp'
    priority = sorted(date_cols, key=lambda c: (0 if c.lower() == "day" else 1))
    for col in priority:
        try:
            dates = pd.to_datetime(df[col], errors="coerce", format="mixed").dropna()
            if len(dates) > 0:
                return col, dates.min(), dates.max(), len(dates.dt.date.unique())
        except Exception:
            continue
    return None, None, None, None


def analyze_csv(filepath):
    """Analizează un singur CSV și returnează un dict cu info."""
    info = {
        "filename": Path(filepath).stem,
        "size_kb": os.path.getsize(filepath) / 1024,
        "error": None,
    }

    sep = detect_separator(filepath)
    info["separator"] = sep

    try:
        # Try reading with detected separator
        df = pd.read_csv(filepath, sep=sep, low_memory=False)
    except Exception as e:
        info["error"] = str(e)
        return info

    # If we still get 1 column and it has semicolons, force semicolon
    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        try:
            df = pd.read_csv(filepath, sep=";", low_memory=False)
            info["separator"] = ";"
        except Exception:
            pass

    info["rows"] = len(df)
    info["columns"] = len(df.columns)
    info["column_names"] = list(df.columns)

    if len(df) == 0 or len(df.columns) <= 1:
        return info

    # Data types
    info["dtypes"] = {}
    for col in df.columns:
        info["dtypes"][col] = str(df[col].dtype)

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    info["missing"] = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns
        if missing[col] > 0
    }
    info["total_missing_pct"] = round(
        df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 1
    )

    # Date range
    date_cols = find_date_columns(df)
    info["date_columns"] = date_cols
    date_col, date_min, date_max, n_unique_days = get_date_range(df, date_cols)
    info["date_col_used"] = date_col
    info["date_min"] = str(date_min)[:10] if date_min else None
    info["date_max"] = str(date_max)[:10] if date_max else None
    info["unique_days"] = n_unique_days

    # Numeric stats (skip ID and long text columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    info["numeric_columns"] = numeric_cols
    if numeric_cols:
        info["numeric_stats"] = {}
        for col in numeric_cols[:15]:  # Limit to first 15 numeric cols
            s = df[col].dropna()
            if len(s) > 0:
                info["numeric_stats"][col] = {
                    "min": round(float(s.min()), 2),
                    "max": round(float(s.max()), 2),
                    "mean": round(float(s.mean()), 2),
                    "median": round(float(s.median()), 2),
                    "std": round(float(s.std()), 2),
                    "non_null": int(len(s)),
                }

    # Sample (primele 3 rânduri, formatat compact)
    try:
        info["sample_head"] = df.head(3).to_string(max_colwidth=50)
    except Exception:
        info["sample_head"] = "(could not display sample)"

    return info


def format_report(results):
    """Generează raportul text formatat."""
    lines = []
    lines.append("=" * 90)
    lines.append("OURA CSV EXPORT - EXPLORATORY REPORT v2")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 90)

    # Sortează: relevante primele, apoi alfabetic
    results.sort(
        key=lambda r: (
            0 if r["filename"] in RELEVANT_FILES else 1,
            r["filename"],
        )
    )

    # Summary table
    lines.append("\n" + "=" * 90)
    lines.append("SUMMARY TABLE")
    lines.append("=" * 90)
    lines.append(
        f"{'File':<30} {'Rows':>10} {'Cols':>5} {'Size(KB)':>10}"
        f" {'Date Min':>12} {'Date Max':>12} {'Days':>6} {'Rel':>4}"
    )
    lines.append("-" * 90)

    empty_files = []
    relevant_files = []
    other_files = []

    for r in results:
        rows = r.get("rows", 0)
        cols = r.get("columns", 0)
        size = r.get("size_kb", 0)
        dmin = r.get("date_min", "") or ""
        dmax = r.get("date_max", "") or ""
        days = r.get("unique_days", "") or ""
        relevant = "***" if r["filename"] in RELEVANT_FILES else ""

        if rows == 0 or (cols <= 1 and size < 2):
            empty_files.append(r["filename"])
            status = " (EMPTY)"
        else:
            status = ""

        lines.append(
            f"{r['filename']:<30} {rows:>10,} {cols:>5} {size:>10,.0f}"
            f" {dmin:>12} {dmax:>12} {str(days):>6} {relevant:>4}{status}"
        )

        if r["filename"] in RELEVANT_FILES and rows > 0 and cols > 1:
            relevant_files.append(r)
        elif rows > 0 and cols > 1:
            other_files.append(r)

    lines.append(f"\nTotal files: {len(results)}")
    lines.append(f"Empty/unparseable: {len(empty_files)} → {', '.join(empty_files)}")
    lines.append(f"Relevant for pipeline: {len(relevant_files)}")
    lines.append(f"Other (non-empty): {len(other_files)}")

    # Detailed analysis per relevant file
    lines.append("\n" + "=" * 90)
    lines.append("DETAILED ANALYSIS - RELEVANT FILES")
    lines.append("=" * 90)

    for r in relevant_files:
        lines.append(f"\n{'─' * 70}")
        lines.append(f"  {r['filename'].upper()}")
        lines.append(f"{'─' * 70}")
        lines.append(f"  Rows: {r.get('rows', 0):,}  |  Columns: {r.get('columns', 0)}  |  Sep: '{r.get('separator', ',')}'")
        lines.append(f"  Size: {r.get('size_kb', 0):,.0f} KB")

        if r.get("date_min"):
            lines.append(
                f"  Date range: {r['date_min']} → {r['date_max']}"
                f"  ({r.get('unique_days', '?')} unique days)"
            )

        # Column list
        col_names = r.get("column_names", [])
        lines.append(f"\n  Columns ({len(col_names)}):")
        # Print columns with types
        dtypes = r.get("dtypes", {})
        for col in col_names:
            dtype = dtypes.get(col, "?")
            lines.append(f"    - {col:<40} [{dtype}]")

        # Missing data
        if r.get("missing"):
            lines.append(f"\n  Missing values (total {r['total_missing_pct']}% of all cells):")
            sorted_missing = sorted(
                r["missing"].items(), key=lambda x: x[1]["pct"], reverse=True
            )
            for col, m in sorted_missing[:15]:  # Top 15
                bar = "█" * int(m["pct"] / 5) if m["pct"] > 0 else ""
                lines.append(f"    {col:<35} {m['count']:>8} ({m['pct']:>5.1f}%) {bar}")

        # Numeric stats
        if r.get("numeric_stats"):
            lines.append(f"\n  Numeric stats:")
            lines.append(
                f"    {'Column':<30} {'Min':>10} {'Max':>10} {'Mean':>10}"
                f" {'Median':>10} {'Std':>10} {'Count':>8}"
            )
            lines.append(f"    {'-'*88}")
            for col, s in r["numeric_stats"].items():
                lines.append(
                    f"    {col:<30} {s['min']:>10.1f} {s['max']:>10.1f}"
                    f" {s['mean']:>10.1f} {s['median']:>10.1f}"
                    f" {s['std']:>10.1f} {s['non_null']:>8,}"
                )

        # Sample
        if r.get("sample_head"):
            lines.append(f"\n  Sample (first 3 rows):")
            for sl in r["sample_head"].split("\n"):
                lines.append(f"    {sl}")

    # Other non-empty files
    if other_files:
        lines.append("\n" + "=" * 90)
        lines.append("OTHER NON-EMPTY FILES")
        lines.append("=" * 90)

        for r in other_files:
            lines.append(f"\n{'─' * 50}")
            lines.append(f"  {r['filename']}: {r.get('rows', 0):,} rows, {r.get('columns', 0)} cols")
            lines.append(f"  Columns: {', '.join(r.get('column_names', []))}")
            if r.get("date_min"):
                lines.append(f"  Date range: {r['date_min']} → {r['date_max']} ({r.get('unique_days', '?')} days)")
            if r.get("numeric_stats"):
                for col, s in list(r["numeric_stats"].items())[:5]:
                    lines.append(
                        f"    {col}: min={s['min']}, max={s['max']}, mean={s['mean']:.1f}"
                    )

    # Pipeline relevance assessment
    lines.append("\n" + "=" * 90)
    lines.append("PIPELINE RELEVANCE ASSESSMENT")
    lines.append("=" * 90)
    lines.append("""
FILES ALREADY USED IN PIPELINE (via API / oura_combined.csv):
  - dailyreadiness → readiness_score, temperature_deviation
  - dailysleep → sleep_score (+ contributors for sleep staging)
  - dailyactivity → steps, active_calories, met (activity metrics)
  - dailyspo2 → spo2_percentage

NEW DATA AVAILABLE FROM CSV EXPORT (not in current pipeline):
  - dailystress → day_summary, stress_high, recovery_high
    → Potential new features for stress-aware classification
  - daytimestress → granular stress/recovery (5-min intervals)
    → Could compute daily stress variability, peak stress, etc.
  - dailyresilience → resilience level + contributors
    → sleep_recovery and stress contributors
  - dailycardiovascularage → vascular_age, pulse_wave_velocity
    → Long-term cardiovascular health indicator
  - workout → activity type, calories, distance, intensity
    → Direct signal for Active classification
  - vo2max → fitness level over time (only 27 data points)
  - sleeptime → optimal bedtime recommendation
  - heartrate → per-second HR data
    → Could compute resting HR, HR variability beyond Oura's built-in
  - temperature → per-minute skin temperature
    → Already captured in readiness as temperature_deviation

RECOMMENDATION:
  Priority 1 (quick wins): dailystress, workout — daily-level, direct signal
  Priority 2 (moderate effort): daytimestress aggregated, dailyresilience
  Priority 3 (high effort, diminishing returns): heartrate aggregation,
    dailycardiovascularage (limited coverage)
""")

    return "\n".join(lines)


def main():
    export_dir = Path(EXPORT_DIR)
    if not export_dir.exists():
        print(f"ERROR: Directory not found: {export_dir}")
        print(f"Usage: python {sys.argv[0]} /path/to/oura_csv_folder")
        sys.exit(1)

    csv_files = sorted(export_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {export_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files in {export_dir}")
    print("Analyzing (with semicolon separator detection)...\n")

    results = []
    for i, f in enumerate(csv_files, 1):
        print(f"  [{i}/{len(csv_files)}] {f.name} ({os.path.getsize(f)/1024:,.0f} KB)...")
        result = analyze_csv(f)
        results.append(result)

    report = format_report(results)

    # Print to console
    print("\n" + report)

    # Save to file
    output_path = export_dir / OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write(report)
    print(f"\n✅ Report saved to: {output_path}")


if __name__ == "__main__":
    main()