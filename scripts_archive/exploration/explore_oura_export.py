"""
Oura CSV Export Explorer
========================
Scanează toate CSV-urile din exportul Oura și generează un raport detaliat.
Rulează din folderul unde ai extras ZIP-ul, sau setează EXPORT_DIR mai jos.

Usage:
    python explore_oura_export.py
    python explore_oura_export.py /path/to/oura_export_folder
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# === CONFIG ===
# Schimbă calea dacă CSV-urile sunt în alt folder
EXPORT_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
OUTPUT_FILE = "oura_export_report.txt"

# Fișierele relevante pentru pipeline-ul de clasificare Rest/Normal/Active
RELEVANT_FILES = {
    "dailyreadiness", "dailysleep", "dailyactivity", "dailyspo2",
    "heartrate", "temperature", "sleepmodel", "dailystress",
    "daytimestress", "dailycardiovascularage", "dailyresilience",
    "workout", "vo2max"
}


def find_date_columns(df):
    """Identifică coloanele care conțin date/timestamp-uri."""
    date_cols = []
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(20)
            if len(sample) == 0:
                continue
            try:
                pd.to_datetime(sample)
                date_cols.append(col)
            except (ValueError, TypeError):
                continue
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
    return date_cols


def get_date_range(df, date_cols):
    """Returnează date range din prima coloană de date găsită."""
    for col in date_cols:
        try:
            dates = pd.to_datetime(df[col], errors="coerce").dropna()
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

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        info["error"] = str(e)
        return info

    info["rows"] = len(df)
    info["columns"] = len(df.columns)
    info["column_names"] = list(df.columns)
    info["dtypes"] = {col: str(df[col].dtype) for col in df.columns}

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
    ) if len(df) > 0 and len(df.columns) > 0 else 0

    # Date range
    date_cols = find_date_columns(df)
    info["date_columns"] = date_cols
    date_col, date_min, date_max, n_unique_days = get_date_range(df, date_cols)
    info["date_col_used"] = date_col
    info["date_min"] = str(date_min) if date_min else None
    info["date_max"] = str(date_max) if date_max else None
    info["unique_days"] = n_unique_days

    # Numeric stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    info["numeric_columns"] = numeric_cols
    if numeric_cols:
        info["numeric_stats"] = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) > 0:
                info["numeric_stats"][col] = {
                    "min": round(float(s.min()), 2),
                    "max": round(float(s.max()), 2),
                    "mean": round(float(s.mean()), 2),
                    "std": round(float(s.std()), 2),
                    "non_null": int(len(s)),
                }

    # Sample (primele 3 rânduri)
    info["sample_head"] = df.head(3).to_string(max_colwidth=40)

    return info


def format_report(results):
    """Generează raportul text formatat."""
    lines = []
    lines.append("=" * 80)
    lines.append("OURA CSV EXPORT - EXPLORATORY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 80)

    # Sortează: relevante primele, apoi alfabetic
    results.sort(
        key=lambda r: (
            0 if r["filename"] in RELEVANT_FILES else 1,
            r["filename"],
        )
    )

    # Summary table
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY TABLE")
    lines.append("=" * 80)
    lines.append(
        f"{'File':<35} {'Rows':>8} {'Cols':>5} {'Size(KB)':>10} {'Days':>6} {'Relevant':>8}"
    )
    lines.append("-" * 80)

    empty_files = []
    relevant_files = []
    other_files = []

    for r in results:
        rows = r.get("rows", 0)
        cols = r.get("columns", 0)
        size = r.get("size_kb", 0)
        days = r.get("unique_days", "-") or "-"
        relevant = "***" if r["filename"] in RELEVANT_FILES else ""

        if rows == 0 or size < 2:
            empty_files.append(r["filename"])
            status = "(EMPTY)"
        else:
            status = ""

        lines.append(
            f"{r['filename']:<35} {rows:>8,} {cols:>5} {size:>10,.0f} {str(days):>6} {relevant:>8} {status}"
        )

        if r["filename"] in RELEVANT_FILES and rows > 0:
            relevant_files.append(r)
        elif rows > 0:
            other_files.append(r)

    lines.append(f"\nTotal files: {len(results)}")
    lines.append(f"Empty/header-only: {len(empty_files)} → {', '.join(empty_files)}")
    lines.append(f"Relevant for pipeline: {len(relevant_files)}")
    lines.append(f"Other (non-empty): {len(other_files)}")

    # Detailed analysis per relevant file
    lines.append("\n" + "=" * 80)
    lines.append("DETAILED ANALYSIS - RELEVANT FILES")
    lines.append("=" * 80)

    for r in relevant_files:
        lines.append(f"\n{'─' * 60}")
        lines.append(f"📊 {r['filename'].upper()}")
        lines.append(f"{'─' * 60}")
        lines.append(f"  Rows: {r.get('rows', 0):,}  |  Columns: {r.get('columns', 0)}")
        lines.append(f"  Size: {r.get('size_kb', 0):,.0f} KB")

        if r.get("date_min"):
            lines.append(
                f"  Date range: {r['date_min'][:10]} → {r['date_max'][:10]}"
                f"  ({r.get('unique_days', '?')} unique days)"
            )

        lines.append(f"\n  Columns: {', '.join(r.get('column_names', []))}")

        # Missing data
        if r.get("missing"):
            lines.append(f"\n  Missing values (total {r['total_missing_pct']}%):")
            for col, m in sorted(
                r["missing"].items(), key=lambda x: x[1]["pct"], reverse=True
            ):
                if m["pct"] > 0:
                    lines.append(f"    {col:<30} {m['count']:>8} ({m['pct']}%)")

        # Numeric stats
        if r.get("numeric_stats"):
            lines.append(f"\n  Numeric columns stats:")
            lines.append(
                f"    {'Column':<30} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'N':>8}"
            )
            for col, s in r["numeric_stats"].items():
                lines.append(
                    f"    {col:<30} {s['min']:>10.1f} {s['max']:>10.1f}"
                    f" {s['mean']:>10.1f} {s['std']:>10.1f} {s['non_null']:>8,}"
                )

        # Sample
        if r.get("sample_head"):
            lines.append(f"\n  Sample (first 3 rows):")
            for sl in r["sample_head"].split("\n"):
                lines.append(f"    {sl}")

    # Detailed analysis for non-empty, non-relevant files
    if other_files:
        lines.append("\n" + "=" * 80)
        lines.append("OTHER NON-EMPTY FILES (may be useful)")
        lines.append("=" * 80)

        for r in other_files:
            lines.append(f"\n{'─' * 40}")
            lines.append(f"  {r['filename']}: {r.get('rows', 0):,} rows, {r.get('columns', 0)} cols")
            lines.append(f"  Columns: {', '.join(r.get('column_names', []))}")
            if r.get("date_min"):
                lines.append(f"  Date range: {r['date_min'][:10]} → {r['date_max'][:10]}")

    lines.append("\n" + "=" * 80)
    lines.append("NEXT STEPS / RECOMANDĂRI")
    lines.append("=" * 80)
    lines.append("""
1. Verifică date range-ul fișierelor relevante — ar trebui să se suprapună
   cu datele din oura_combined.csv (generat prin API).

2. Compară câteva zile între CSV export și API data pentru consistență
   (readiness score, sleep score, steps).

3. CSV-urile heartrate și temperature conțin date la nivel de minut/5-min —
   necesită agregare la nivel zilnic pentru a fi utile în pipeline.

4. dailystress și daytimestress sunt date noi față de ce ai prin API —
   evaluează dacă merită integrate ca features.

5. sleepmodel conține date detaliate sleep staging — util pentru
   sleep_deep și sleep_rem features din extended pipeline.
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
    print("Analyzing...\n")

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