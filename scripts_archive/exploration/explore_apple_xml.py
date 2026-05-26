"""
Apple Health XML Export Explorer
=================================
Explorează cele 9 CSV-uri parsate din export.xml.
Analizează surse, date ranges, statistici, contaminare multi-sursă.

Usage:
    python explore_apple_xml.py
    python explore_apple_xml.py "C:\path\to\apple_health_folder"
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

XML_DIR = sys.argv[1] if len(sys.argv) > 1 else r"data\apple_health"
OUTPUT_FILE = "apple_xml_exploration_report.txt"


def analyze_csv(filepath):
    """Analizează un CSV din XML export."""
    info = {"filename": filepath.stem, "size_kb": filepath.stat().st_size / 1024}

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        info["error"] = str(e)
        return info

    info["rows"] = len(df)
    info["columns"] = list(df.columns)

    if len(df) == 0:
        return info

    # Sources
    if "sourceName" in df.columns:
        src = df["sourceName"].value_counts()
        info["sources"] = {str(k): int(v) for k, v in src.items()}
        info["source_count"] = len(src)

        # Per-source stats
        info["per_source"] = {}
        for source_name in src.index:
            sdf = df[df["sourceName"] == source_name]
            src_info = {"count": len(sdf), "pct": round(len(sdf) / len(df) * 100, 1)}
            if "startDate" in sdf.columns:
                dates = pd.to_datetime(sdf["startDate"], format="mixed", errors="coerce")
                src_info["date_min"] = str(dates.min())[:10] if dates.notna().any() else None
                src_info["date_max"] = str(dates.max())[:10] if dates.notna().any() else None
            info["per_source"][str(source_name)] = src_info

    # Date range
    if "startDate" in df.columns:
        dates = pd.to_datetime(df["startDate"], format="mixed", errors="coerce")
        info["date_min"] = str(dates.min())[:10]
        info["date_max"] = str(dates.max())[:10]
        info["unique_days"] = len(dates.dt.date.unique())

    # Value stats
    if "value" in df.columns:
        numeric_vals = pd.to_numeric(df["value"], errors="coerce")
        nn = numeric_vals.notna().sum()
        if nn > 0:
            info["value_stats"] = {
                "count": int(nn),
                "min": round(float(numeric_vals.min()), 4),
                "max": round(float(numeric_vals.max()), 4),
                "mean": round(float(numeric_vals.mean()), 4),
                "std": round(float(numeric_vals.std()), 4),
                "median": round(float(numeric_vals.median()), 4),
            }
        else:
            # Categorical values (like sleep types)
            info["value_categories"] = df["value"].value_counts().to_dict()

    # Unit
    if "unit" in df.columns:
        info["unit"] = df["unit"].iloc[0] if df["unit"].notna().any() else None

    # Per-source value comparison (for contaminated metrics)
    if "sourceName" in df.columns and "value" in df.columns:
        numeric_vals = pd.to_numeric(df["value"], errors="coerce")
        if numeric_vals.notna().sum() > 0 and info.get("source_count", 0) > 1:
            info["source_value_comparison"] = {}
            for source_name in df["sourceName"].unique():
                svals = numeric_vals[df["sourceName"] == source_name].dropna()
                if len(svals) > 0:
                    info["source_value_comparison"][str(source_name)] = {
                        "count": int(len(svals)),
                        "mean": round(float(svals.mean()), 4),
                        "std": round(float(svals.std()), 4),
                    }

    # Daily aggregation check (how many records per day)
    if "startDate" in df.columns:
        dates = pd.to_datetime(df["startDate"], format="mixed", errors="coerce")
        daily_counts = dates.dt.date.value_counts()
        info["records_per_day"] = {
            "min": int(daily_counts.min()),
            "max": int(daily_counts.max()),
            "mean": round(float(daily_counts.mean()), 1),
            "median": int(daily_counts.median()),
        }

    # Apple Watch only stats (filtered)
    if "sourceName" in df.columns:
        aw_mask = df["sourceName"].str.contains("Watch", na=False)
        aw_count = aw_mask.sum()
        info["aw_only_count"] = int(aw_count)
        info["aw_only_pct"] = round(aw_count / len(df) * 100, 1) if len(df) > 0 else 0

        if aw_count > 0 and "value" in df.columns:
            aw_vals = pd.to_numeric(df.loc[aw_mask, "value"], errors="coerce").dropna()
            if len(aw_vals) > 0:
                info["aw_only_value_stats"] = {
                    "count": int(len(aw_vals)),
                    "mean": round(float(aw_vals.mean()), 4),
                    "std": round(float(aw_vals.std()), 4),
                }

    return info


def format_report(results):
    """Generează raportul."""
    lines = []
    lines.append("=" * 95)
    lines.append("APPLE HEALTH XML EXPORT - EXPLORATION REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 95)

    # Summary table
    lines.append("\n" + "─" * 95)
    lines.append("SUMMARY TABLE")
    lines.append("─" * 95)
    lines.append(
        f"{'File':<25} {'Rows':>10} {'SizeKB':>8} {'Days':>6}"
        f" {'Sources':>8} {'AW%':>6} {'DateRange':>25}"
    )
    lines.append("-" * 95)

    for r in sorted(results, key=lambda x: x.get("rows", 0), reverse=True):
        rows = r.get("rows", 0)
        size = r.get("size_kb", 0)
        days = r.get("unique_days", "")
        nsrc = r.get("source_count", 0)
        aw_pct = r.get("aw_only_pct", 0)
        dmin = r.get("date_min", "")
        dmax = r.get("date_max", "")
        drange = f"{dmin} → {dmax}" if dmin else ""
        lines.append(
            f"{r['filename']:<25} {rows:>10,} {size:>8,.0f} {str(days):>6}"
            f" {nsrc:>8} {aw_pct:>5.1f}% {drange:>25}"
        )

    # Contamination assessment
    lines.append("\n" + "─" * 95)
    lines.append("SOURCE CONTAMINATION ASSESSMENT")
    lines.append("─" * 95)

    for r in sorted(results, key=lambda x: x["filename"]):
        if r.get("error") or r.get("rows", 0) == 0:
            continue

        sources = r.get("sources", {})
        nsrc = len(sources)
        aw_pct = r.get("aw_only_pct", 0)

        if aw_pct >= 99:
            status = "CLEAN (AW only)"
        elif aw_pct >= 90:
            status = "QUASI-CLEAN"
        elif aw_pct > 0:
            status = "CONTAMINATED"
        else:
            status = "NO AW DATA"

        lines.append(f"\n  {r['filename'].upper()} — {status}")
        for src_name, src_info in r.get("per_source", {}).items():
            lines.append(
                f"    {src_name:<35} {src_info['count']:>8,} records ({src_info['pct']}%)"
                f"  [{src_info.get('date_min', '')} → {src_info.get('date_max', '')}]"
            )

        # Show value differences between sources if contaminated
        if nsrc > 1 and r.get("source_value_comparison"):
            lines.append(f"    Value comparison by source:")
            for src_name, stats in r["source_value_comparison"].items():
                lines.append(
                    f"      {src_name:<35} mean={stats['mean']:.2f}, std={stats['std']:.2f}, n={stats['count']:,}"
                )

    # Detailed per file
    lines.append("\n" + "─" * 95)
    lines.append("DETAILED STATISTICS PER FILE")
    lines.append("─" * 95)

    for r in sorted(results, key=lambda x: x["filename"]):
        if r.get("error") or r.get("rows", 0) == 0:
            continue

        lines.append(f"\n  {r['filename'].upper()}")
        lines.append(f"    Rows: {r['rows']:,}  |  Size: {r['size_kb']:,.0f} KB")
        lines.append(f"    Unit: {r.get('unit', '?')}")
        if r.get("date_min"):
            lines.append(f"    Date: {r['date_min']} → {r['date_max']} ({r.get('unique_days', '?')} days)")
        if r.get("records_per_day"):
            rpd = r["records_per_day"]
            lines.append(f"    Records/day: min={rpd['min']}, max={rpd['max']}, mean={rpd['mean']}, median={rpd['median']}")

        if r.get("value_stats"):
            vs = r["value_stats"]
            lines.append(f"    Values: min={vs['min']}, max={vs['max']}, mean={vs['mean']:.2f}, std={vs['std']:.2f}, median={vs['median']}")
        elif r.get("value_categories"):
            lines.append(f"    Categories:")
            for cat, cnt in r["value_categories"].items():
                lines.append(f"      {cat}: {cnt:,}")

        if r.get("aw_only_value_stats"):
            avs = r["aw_only_value_stats"]
            lines.append(f"    AW-only values: mean={avs['mean']:.2f}, std={avs['std']:.2f}, n={avs['count']:,}")

    # Recommendations
    lines.append("\n" + "=" * 95)
    lines.append("KEY FINDINGS & RECOMMENDATIONS")
    lines.append("=" * 95)
    lines.append("""
  XML EXPORT ISSUES:
  1. Multiple sources (Oura, iPhone, Apple Watch, Zepp Life) mixed in same files
  2. No deduplication — same measurement may appear from multiple sources
  3. Steps, HR, active_calories heavily contaminated (30-70% non-AW data)
  4. Sleep data: 67% Oura, 32% Apple Watch, 1% iPhone

  CLEAN XML FILES (usable as-is):
  - hrv.csv: 100% Apple Watch
  - spo2.csv: ~100% Apple Watch
  - basal_calories.csv: ~100% Apple Watch

  COMPARISON WITH JSON:
  - JSON Health Auto Export performs its own deduplication (source fields show merged sources)
  - JSON correctly deduplicates steps (finding: XML MAE ~9,106 steps/day vs JSON)
  - JSON sleep still mixes sources but doesn't double-count

  RECOMMENDATION:
  JSON (via parse_health_auto_export.py → apple_health_daily.csv) remains the
  better source for pipeline. XML is useful only as validation reference and
  for metrics JSON doesn't cover.
""")

    return "\n".join(lines)


def main():
    xml_dir = Path(XML_DIR)
    if not xml_dir.exists():
        print(f"ERROR: Directory not found: {xml_dir}")
        sys.exit(1)

    csv_files = sorted(xml_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files in {xml_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files in {xml_dir}")
    print("Analyzing (large files may take a moment)...\n")

    results = []
    for i, f in enumerate(csv_files, 1):
        print(f"  [{i}/{len(csv_files)}] {f.name} ({f.stat().st_size/1024:,.0f} KB)...")
        info = analyze_csv(f)
        results.append(info)

    report = format_report(results)
    print("\n" + report)

    out_path = xml_dir / OUTPUT_FILE
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write(report)
    print(f"\n✅ Report saved to: {out_path}")


if __name__ == "__main__":
    main()