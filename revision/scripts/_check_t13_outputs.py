"""
_check_t13_outputs.py - T1.3 Task 3 pass criteria (determinism, LF, md == csv, figure).

1. t13_calibration_table.md carries exactly the values of t13_calibration.csv:
   every markdown cell is re-derived from the csv row (split, fit_folds,
   variant, metric) with the same 4-decimal / integer formatting; cell count
   equals the number of (table, row, metric) slots.
2. Every text output (csv, md, json) and CIFRE_REVIZIE.md contain no CRLF.
3. Figure files exist, the png opens (matplotlib.image.imread), run_meta says
   2 panels / 4 axes (2 reliability + 2 histogram).
4. Determinism: with --previous DIR (a copy of results/ from an earlier run),
   sha256 identical for csv / md / png / pdf / npy / joblib; json identical
   without run_date and elapsed_seconds.

Exits 1 on any FAIL. Evidence script, kept.
Run from repo root:
    python revision/scripts/_check_t13_outputs.py [--previous DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_rf import INT_METRICS, METRIC_TITLE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
TEXT_OUTPUTS = ["t13_oof_probabilities_rf.csv", "t13_oof_folds_rf.csv", "t13_calibration.csv",
                "t13_calibration_table.md", "t13_reliability_bins.csv", "t13_run_meta.json",
                "t13_safety_layer_variants.csv", "t13_safety_layer_variants_table.md"]
BINARY_OUTPUTS = ["figures/t13_reliability_rf.png", "figures/t13_reliability_rf.pdf",
                  "figures/t13_safety_layer_coverage.png", "figures/t13_safety_layer_coverage.pdf",
                  "models/calibrator_platt_rf.joblib", "models/calibrator_isotonic_rf.joblib"]
SAFETY_FLOAT_COLS = {"threshold", "coverage_pct", "mean_cal_error", "f1_macro", "accuracy", "f1_active"}
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_cell(r: pd.Series, metric: str) -> str:
    if np.isnan(r.point):
        return "n/a"
    if metric in INT_METRICS:
        return f"{int(round(r.point))}"
    if np.isnan(r.ci_low):
        return f"{r.point:.4f}"
    return f"{r.point:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]"


def compare_md(csv_path: Path, md_path: Path) -> None:
    df = pd.read_csv(csv_path)
    title_to_metric = {v: k for k, v in METRIC_TITLE.items()}
    lines = md_path.read_text(encoding="utf-8").splitlines()
    split = None
    header: list[str] | None = None
    n_cells = n_bad = 0
    for ln in lines:
        if ln.startswith("### "):
            split = "oof" if ln.startswith("### OOF") else "test"
            header = None
            continue
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if header is None:
            header = cells
            check(header[:2] == ["fit_folds", "variant"], f"table header starts with fit_folds, variant: {header[:2]}")
            check(all(h in title_to_metric for h in header[2:]), f"all metric titles known: {header[2:]}")
            continue
        if set(ln.replace("|", "").strip()) <= {"-"}:
            continue
        ff, variant = cells[0], cells[1]
        for title, cell in zip(header[2:], cells[2:]):
            m = title_to_metric[title]
            sel = df[(df.split == split) & (df.fit_folds == ff) & (df.variant == variant) & (df.metric == m)]
            exp = "n/a" if len(sel) == 0 else expected_cell(sel.iloc[0], m)
            n_cells += 1
            if exp != cell:
                n_bad += 1
                print(f"   mismatch {split}/{ff}/{variant}/{m}: md '{cell}' vs csv '{exp}'")
    check(n_bad == 0, f"{md_path.name}: all {n_cells} cells == csv values ({n_bad} mismatches)")
    check(n_cells >= 100, f"{md_path.name}: {n_cells} cells compared (expected >= 100)")


def compare_safety_md(csv_path: Path, md_path: Path) -> None:
    """Both tables of the stage C md (0.50 rows, full grid) must equal the csv rows keyed by (variant, threshold)."""
    df = pd.read_csv(csv_path)
    lines = [ln for ln in md_path.read_text(encoding="utf-8").splitlines() if ln.startswith("|")]
    header: list[str] | None = None
    n_cells = n_bad = n_rows = 0
    for ln in lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if set(ln.replace("|", "").strip()) <= {"-"}:
            continue
        if cells[0] == "variant":
            header = cells
            check(header == list(df.columns), f"{md_path.name}: header == csv columns")
            continue
        assert header is not None
        row = dict(zip(header, cells))
        sel = df[(df.variant == row["variant"]) & (np.isclose(df.threshold, float(row["threshold"])))]
        check(len(sel) == 1, f"{md_path.name}: unique csv row for {row['variant']} @ {row['threshold']}")
        n_rows += 1
        for c in header:
            v = sel.iloc[0][c]
            exp = v if c == "variant" else (f"{float(v):.4f}" if c in SAFETY_FLOAT_COLS else f"{int(v)}")
            n_cells += 1
            if exp != row[c]:
                n_bad += 1
                print(f"   mismatch {row['variant']}/{row['threshold']}/{c}: md '{row[c]}' vs csv '{exp}'")
    check(n_bad == 0, f"{md_path.name}: all {n_cells} cells in {n_rows} rows == csv values ({n_bad} mismatches)")
    check(n_rows == len(df) + 3, f"{md_path.name}: {n_rows} md rows == {len(df)} grid rows + 3 canonical rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--previous", type=Path, default=None)
    args = ap.parse_args()

    print("=== 1. md == csv ===")
    compare_md(RES / "t13_calibration.csv", RES / "t13_calibration_table.md")
    compare_safety_md(RES / "t13_safety_layer_variants.csv", RES / "t13_safety_layer_variants_table.md")

    print("\n=== 2. LF only ===")
    for name in TEXT_OUTPUTS:
        check(b"\r\n" not in (RES / name).read_bytes(), f"no CRLF in results/{name}")
    check(b"\r\n" not in (ROOT / "revision" / "CIFRE_REVIZIE.md").read_bytes(), "no CRLF in CIFRE_REVIZIE.md")

    print("\n=== 3. figure ===")
    import matplotlib.image as mpimg
    png = RES / "figures" / "t13_reliability_rf.png"
    pdf = RES / "figures" / "t13_reliability_rf.pdf"
    img = mpimg.imread(png)
    check(img.ndim == 3 and img.shape[0] > 500 and img.shape[1] > 1000, f"png opens, shape {img.shape}")
    check(pdf.exists() and pdf.read_bytes()[:5] == b"%PDF-", f"pdf exists and starts with %PDF- ({pdf.stat().st_size} bytes)")
    img2 = mpimg.imread(RES / "figures" / "t13_safety_layer_coverage.png")
    pdf2 = RES / "figures" / "t13_safety_layer_coverage.pdf"
    check(img2.ndim == 3 and img2.shape[0] > 300, f"safety-layer png opens, shape {img2.shape}")
    check(pdf2.exists() and pdf2.read_bytes()[:5] == b"%PDF-", f"safety-layer pdf exists ({pdf2.stat().st_size} bytes)")
    meta = json.loads((RES / "t13_run_meta.json").read_text(encoding="utf-8"))
    check(meta.get("figure_panels") == 2 and meta.get("figure_axes") == 4,
          f"run_meta: figure_panels {meta.get('figure_panels')} == 2, figure_axes {meta.get('figure_axes')} == 4")
    check(meta.get("sanity_checks_failed") == [], f"run_meta: sanity_checks_failed == [] ({meta.get('sanity_checks_failed')})")

    if args.previous is not None:
        print(f"\n=== 4. determinism vs {args.previous} ===")
        for name in TEXT_OUTPUTS + BINARY_OUTPUTS:
            cur, prev = RES / name, args.previous / name
            if name == "t13_run_meta.json":
                a = json.loads(cur.read_text(encoding="utf-8"))
                b = json.loads(prev.read_text(encoding="utf-8"))
                for k in ("run_date", "elapsed_seconds"):
                    a.pop(k, None), b.pop(k, None)
                check(a == b, f"{name}: identical without run_date / elapsed_seconds")
            else:
                check(sha256(cur) == sha256(prev), f"{name}: sha256 identical ({sha256(cur)[:16]})")

    print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}"))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
