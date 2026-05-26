"""
Apple Watch Comparison: XML Export vs JSON (apple_health_daily.csv)
===================================================================
Agregă XML CSVs la nivel zilnic (cu și fără filtrare pe sursă AW)
și compară cu apple_health_daily.csv generat din JSON Health Auto Export.

Usage:
    python compare_apple_xml_vs_json.py "data\apple_health" "apple_watch_api\apple_health_daily.csv"
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

XML_DIR = sys.argv[1] if len(sys.argv) > 1 else r"data\apple_health"
JSON_CSV = sys.argv[2] if len(sys.argv) > 2 else r"apple_watch_api\apple_health_daily.csv"
OUTPUT_FILE = "apple_xml_vs_json_comparison.txt"


def compare_series(s1, s2, label):
    """Compare two numeric series."""
    a = pd.to_numeric(s1, errors="coerce")
    b = pd.to_numeric(s2, errors="coerce")
    mask = a.notna() & b.notna()
    a, b = a[mask].values, b[mask].values
    n = len(a)
    if n == 0:
        return {"label": label, "n": 0, "note": "No overlap"}

    diff = a - b
    abs_diff = np.abs(diff)
    exact = (abs_diff < 0.01).sum()
    close_1 = (abs_diff < 1.0).sum()
    close_5pct = (abs_diff < np.maximum(np.abs(a), np.abs(b)) * 0.05 + 0.01).sum()
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan

    if not np.isnan(corr) and corr > 0.99:
        verdict = "MATCH"
    elif not np.isnan(corr) and corr > 0.95:
        verdict = "CLOSE"
    elif not np.isnan(corr) and corr > 0.80:
        verdict = "DIFFERS"
    else:
        verdict = "CHECK"

    # Override if exact match is very high
    if exact / n > 0.95:
        verdict = "IDENTICAL"

    return {
        "label": label,
        "n": n,
        "exact": int(exact),
        "exact_pct": round(exact / n * 100, 1),
        "close_5pct": int(close_5pct),
        "close_5pct_pct": round(close_5pct / n * 100, 1),
        "mean_diff": round(float(np.mean(diff)), 4),
        "median_diff": round(float(np.median(diff)), 4),
        "max_diff": round(float(np.max(abs_diff)), 4),
        "corr": round(corr, 6) if not np.isnan(corr) else "N/A",
        "xml_mean": round(float(np.mean(a)), 4),
        "json_mean": round(float(np.mean(b)), 4),
        "diff_pct": round(float(np.mean(diff) / (np.mean(b) + 1e-10) * 100), 2),
        "verdict": verdict,
    }


def load_xml_csv(xml_dir, filename):
    """Load an XML-parsed CSV."""
    path = Path(xml_dir) / f"{filename}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def aggregate_xml_daily(df, value_col="value", agg="sum", source_filter=None):
    """Aggregate XML granular data to daily level."""
    if df is None or len(df) == 0:
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["startDate"], format="mixed", errors="coerce").dt.date

    if source_filter:
        df = df[df["sourceName"].str.contains(source_filter, na=False)]

    df["_val"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["_val", "date"])

    if len(df) == 0:
        return None

    if agg == "sum":
        daily = df.groupby("date")["_val"].sum().reset_index()
    elif agg == "mean":
        daily = df.groupby("date")["_val"].mean().reset_index()
    elif agg == "min":
        daily = df.groupby("date")["_val"].min().reset_index()
    elif agg == "max":
        daily = df.groupby("date")["_val"].max().reset_index()

    daily.columns = ["date", "value"]
    return daily


def aggregate_sleep_xml(df, source_filter=None):
    """Aggregate sleep XML data to daily hours by category."""
    if df is None or len(df) == 0:
        return None

    df = df.copy()
    if source_filter:
        df = df[df["sourceName"].str.contains(source_filter, na=False)]

    df["start"] = pd.to_datetime(df["startDate"], format="mixed", errors="coerce")
    df["end"] = pd.to_datetime(df["endDate"], format="mixed", errors="coerce")
    df["duration_hours"] = (df["end"] - df["start"]).dt.total_seconds() / 3600
    df["date"] = df["start"].dt.date

    # Map sleep categories
    cat_map = {
        "HKCategoryValueSleepAnalysisAsleepDeep": "deep",
        "HKCategoryValueSleepAnalysisAsleepREM": "rem",
        "HKCategoryValueSleepAnalysisAsleepCore": "core",
        "HKCategoryValueSleepAnalysisAwake": "awake",
        "HKCategoryValueSleepAnalysisInBed": "inbed",
        "HKCategoryValueSleepAnalysisAsleepUnspecified": "unspecified",
    }
    df["category"] = df["value"].map(cat_map)

    results = []
    for date, grp in df.groupby("date"):
        row = {"date": date}
        for cat in ["deep", "rem", "core", "awake"]:
            cat_data = grp[grp["category"] == cat]["duration_hours"]
            row[f"sleep_{cat}"] = cat_data.sum() if len(cat_data) > 0 else 0
        row["sleep_total"] = row["sleep_deep"] + row["sleep_rem"] + row["sleep_core"]
        results.append(row)

    return pd.DataFrame(results)


def main():
    xml_dir = Path(XML_DIR)
    json_csv_path = Path(JSON_CSV)

    print("Loading JSON-based apple_health_daily.csv...")
    json_df = pd.read_csv(json_csv_path)
    json_df["date"] = pd.to_datetime(json_df["date"]).dt.date
    print(f"  → {len(json_df)} days, {json_df['date'].min()} → {json_df['date'].max()}")

    lines = []
    lines.append("=" * 100)
    lines.append("APPLE WATCH: XML Export vs JSON (apple_health_daily.csv)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 100)

    all_comparisons = []

    # =========================================================================
    # For each metric: compare XML (all sources) and XML (AW-only) vs JSON
    # =========================================================================

    metrics = [
        {
            "xml_file": "steps",
            "json_col": "aw_steps",
            "agg": "sum",
            "label": "Steps",
            "unit_factor": 1,
        },
        {
            "xml_file": "heart_rate",
            "json_col": "aw_hr_mean",
            "agg": "mean",
            "label": "HR Mean",
            "unit_factor": 1,
        },
        {
            "xml_file": "heart_rate",
            "json_col": "aw_hr_min",
            "agg": "min",
            "label": "HR Min",
            "unit_factor": 1,
        },
        {
            "xml_file": "heart_rate",
            "json_col": "aw_hr_max",
            "agg": "max",
            "label": "HR Max",
            "unit_factor": 1,
        },
        {
            "xml_file": "hrv",
            "json_col": "aw_hrv_mean",
            "agg": "mean",
            "label": "HRV Mean",
            "unit_factor": 1,
        },
        {
            "xml_file": "respiratory_rate",
            "json_col": "aw_resp_rate",
            "agg": "mean",
            "label": "Resp Rate",
            "unit_factor": 1,
        },
        {
            "xml_file": "spo2",
            "json_col": "aw_spo2",
            "agg": "mean",
            "label": "SpO2",
            "unit_factor": 100,  # XML is fraction, JSON is percentage
        },
        {
            "xml_file": "active_calories",
            "json_col": "aw_active_energy_kj",
            "agg": "sum",
            "label": "Active Cal",
            "unit_factor": 4.184,  # XML is kcal, JSON is kJ
        },
        {
            "xml_file": "basal_calories",
            "json_col": "aw_basal_energy_kj",
            "agg": "sum",
            "label": "Basal Cal",
            "unit_factor": 4.184,  # XML is kcal, JSON is kJ
        },
    ]

    for m in metrics:
        print(f"Processing {m['label']}...")
        df_xml = load_xml_csv(xml_dir, m["xml_file"])
        if df_xml is None:
            continue

        # XML ALL sources
        xml_all = aggregate_xml_daily(df_xml, agg=m["agg"])
        # XML AW-only
        xml_aw = aggregate_xml_daily(df_xml, agg=m["agg"], source_filter="Watch")

        if xml_all is not None:
            xml_all["value"] = xml_all["value"] * m["unit_factor"]
            merged_all = pd.merge(xml_all, json_df[["date", m["json_col"]]],
                                  on="date", how="inner")
            comp_all = compare_series(
                merged_all["value"], merged_all[m["json_col"]],
                f"{m['label']} (XML all vs JSON)"
            )
            all_comparisons.append(comp_all)

        if xml_aw is not None and len(xml_aw) > 0:
            xml_aw["value"] = xml_aw["value"] * m["unit_factor"]
            merged_aw = pd.merge(xml_aw, json_df[["date", m["json_col"]]],
                                 on="date", how="inner")
            comp_aw = compare_series(
                merged_aw["value"], merged_aw[m["json_col"]],
                f"{m['label']} (XML AW-only vs JSON)"
            )
            all_comparisons.append(comp_aw)

    # =========================================================================
    # SLEEP: Special handling
    # =========================================================================
    print("Processing Sleep...")
    df_sleep = load_xml_csv(xml_dir, "sleep")
    if df_sleep is not None:
        # XML all sources
        sleep_all = aggregate_sleep_xml(df_sleep)
        # XML AW-only
        sleep_aw = aggregate_sleep_xml(df_sleep, source_filter="Watch")
        # XML Oura-only (for comparison)
        sleep_oura = aggregate_sleep_xml(df_sleep, source_filter="Oura")

        for sleep_data, suffix in [(sleep_all, "all"), (sleep_aw, "AW-only"), (sleep_oura, "Oura-only")]:
            if sleep_data is None or len(sleep_data) == 0:
                continue
            merged = pd.merge(sleep_data, json_df, on="date", how="inner")
            if len(merged) == 0:
                continue

            for xml_col, json_col, label in [
                ("sleep_total", "aw_sleep_hours", f"Sleep Total ({suffix})"),
                ("sleep_deep", "aw_sleep_deep_hours", f"Sleep Deep ({suffix})"),
                ("sleep_rem", "aw_sleep_rem_hours", f"Sleep REM ({suffix})"),
                ("sleep_core", "aw_sleep_core_hours", f"Sleep Core ({suffix})"),
                ("sleep_awake", "aw_sleep_awake_hours", f"Sleep Awake ({suffix})"),
            ]:
                if xml_col in merged.columns and json_col in merged.columns:
                    comp = compare_series(merged[xml_col], merged[json_col], label)
                    all_comparisons.append(comp)

    # =========================================================================
    # FORMAT REPORT
    # =========================================================================
    lines.append("\n" + "=" * 100)
    lines.append("COMPARISON SUMMARY")
    lines.append("=" * 100)
    lines.append(
        f"{'Metric':<40} {'N':>5} {'Exact%':>7} {'~5%%':>7}"
        f" {'MeanDiff':>10} {'Diff%':>8} {'Corr':>8} {'Verdict':>10}"
    )
    lines.append("-" * 100)

    for c in all_comparisons:
        if c.get("note"):
            lines.append(f"{c['label']:<40} {c['n']:>5} {'—':>7} {'—':>7} {'—':>10} {'—':>8} {'—':>8} {c.get('note',''):>10}")
            continue
        corr_str = f"{c['corr']:.4f}" if isinstance(c['corr'], float) else c['corr']
        lines.append(
            f"{c['label']:<40} {c['n']:>5} {c['exact_pct']:>6.1f}% {c['close_5pct_pct']:>6.1f}%"
            f" {c['mean_diff']:>10.2f} {c['diff_pct']:>7.1f}% {corr_str:>8} {c['verdict']:>10}"
        )

    # Detailed
    lines.append("\n" + "─" * 100)
    lines.append("DETAILED RESULTS")
    lines.append("─" * 100)
    for c in all_comparisons:
        if c.get("note") or c["n"] == 0:
            continue
        lines.append(f"\n  {c['label']}:")
        lines.append(f"    N={c['n']}, Exact={c['exact']} ({c['exact_pct']}%), Within 5%={c['close_5pct']} ({c['close_5pct_pct']}%)")
        lines.append(f"    MeanDiff={c['mean_diff']}, MedianDiff={c['median_diff']}, MaxDiff={c['max_diff']}")
        lines.append(f"    Corr={c['corr']}, DiffPct={c['diff_pct']}%")
        lines.append(f"    XML mean={c['xml_mean']}  |  JSON mean={c['json_mean']}")

    # Conclusions
    lines.append("\n" + "=" * 100)
    lines.append("CONCLUSIONS")
    lines.append("=" * 100)
    lines.append("""
  This comparison reveals the impact of source mixing and deduplication:

  XML ALL SOURCES vs JSON:
  - If they match closely → JSON doesn't deduplicate, just passes through
  - If XML > JSON → XML double-counts, JSON deduplicates correctly

  XML AW-ONLY vs JSON:
  - If they match → JSON effectively uses only AW data for this metric
  - If they differ → JSON includes non-AW sources (Oura, iPhone)

  Key question: Does JSON (Health Auto Export) deduplicate correctly?
  The steps comparison is the litmus test (known ~9,106 MAE from XML double-counting).
""")

    report = "\n".join(lines)
    print("\n" + report)

    out_path = Path(XML_DIR) / OUTPUT_FILE
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ Report saved to: {out_path}")


if __name__ == "__main__":
    main()