"""audit-apply (2026-10-26, AUDIT_DECISIONS #17, process part): every numeric cell of the manuscript and Supplementary
tables that is derived from a csv in revision/results/ must equal the csv value rounded ONCE to the printed number of
decimals (round(csv, d) == printed; rounding on the decimal value of the csv, half away from zero or, on an exact tie such
as 682.5, half to even as Python's format() does in the Supplementary producers; both conventions are accepted on ties). Finding F6.1 of the audit found two cells of Table 11 rounded twice (through the
4-decimal markdown table); this script makes the check permanent on all csv-derived tables.

Mapping table -> csv -> columns is explicit below (TABLES). Base tables of the submitted manuscript (Tables 1, 2, 3, 5, 7,
12, 13, 14) and text columns (search spaces, selected configurations, SHAP feature names) are not derived from the
revision csv files and are not checked. Only csv files are read; no model is trained.

Usage (repo root): python revision/scripts/_check_table_rounding.py [--out revision/audit/table_rounding_check.md]
Exit code 1 if any cell mismatches.
"""
from __future__ import annotations

import argparse
import re
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path

import docx
import pandas as pd
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "revision" / "manuscript"
RES = ROOT / "revision" / "results"
NUM = re.compile(r"[-+]?\d[\d,]*\.\d+|[-+]?\d[\d,]*")


def rd(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / name, keep_default_na=False, na_values=[""])


def body_tables(path: Path) -> dict[str, Table]:
    """caption text (paragraph starting with 'Table') -> the Word table that follows it."""
    d = docx.Document(str(path))
    out, last_cap = {}, None
    for child in d.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            t = Paragraph(child, d).text.strip()
            if re.match(r"Table (S?\d+|A\.\d+)\.", t):
                last_cap = t
        elif tag == "tbl" and last_cap is not None:
            out[last_cap] = Table(child, d)
            last_cap = None
    return out


def grid(tbl: Table) -> list[list[str]]:
    return [[c.text.strip() for c in r.cells] for r in tbl.rows]


def col_index(header: list[str], prefix: str) -> int:
    hits = [i for i, h in enumerate(header) if h.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"STOP: column '{prefix}' matched {len(hits)} headers in {header}")
    return hits[0]


def rhu(x: float, d: int) -> float:
    """Round half away from zero on the decimal representation of the csv value (not on the binary float)."""
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def rhe(x: float, d: int) -> float:
    """Round half to even on the decimal representation (the convention of Python's format(), used by the Supplementary
    producers); differs from rhu() only on exact ties such as 682.5."""
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_EVEN))


def row_index(rows: list[list[str]], prefix: str) -> int:
    exact = [i for i, r in enumerate(rows) if r[0] == prefix]
    if len(exact) == 1:
        return exact[0]
    hits = [i for i, r in enumerate(rows) if r[0].startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"STOP: row '{prefix}' matched {len(hits)} rows")
    return hits[0]


def compare_cell(cell: str, expected: list[float]) -> str | None:
    """None if every number in the cell equals round(expected, printed decimals); else a message."""
    toks = NUM.findall(cell)
    if len(toks) != len(expected):
        return f"{len(toks)} numbers printed, {len(expected)} expected ({cell!r})"
    for tok, exp in zip(toks, expected):
        clean = tok.replace(",", "").lstrip("+")
        d = len(clean.split(".")[1]) if "." in clean else 0
        printed = float(clean)
        if pd.isna(exp):
            return f"expected value missing for {tok!r}"
        if abs(rhu(exp, d) - printed) > 1e-9 and abs(rhe(exp, d) - printed) > 1e-9:
            return f"{tok} printed, csv {float(exp):.6f} -> {rhu(exp, d):.{d}f}"
    return None


# ----------------------------------------------------------------------------------------------
# expected values per table: list of (row prefix, column prefix, [expected numbers in print order])
# ----------------------------------------------------------------------------------------------
def spec_table4():
    t11, t21 = rd("t11_extended_metrics.csv"), rd("t21_extra_models.csv")
    sysmap = {"Random Forest": "rf", "XGBoost": "xgb", "SVM (RBF)": "svm", "Persistence": "persistence", "Majority": "majority"}

    def v(df, s, m):
        r = df[(df.system == s) & (df.metric == m)]
        assert len(r) == 1, (s, m)
        return r.iloc[0]

    out = []
    for col, s in sysmap.items():
        out.append(("Macro F1", col, [v(t11, s, "f1_macro").value]))
        for row, m in (("F1 Rest", "f1_rest"), ("F1 Normal", "f1_normal"), ("F1 Active", "f1_active")):
            out.append((row, col, [v(t21, s, m).value]))
        out.append(("Accuracy", col, [v(t11, s, "accuracy").value * 100]))
        for row, m in (("Balanced accuracy", "balanced_accuracy"), ("Cohen's kappa", "cohen_kappa"), ("G-mean", "g_mean"),
                       ("Macro ROC-AUC", "roc_auc_ovr_macro"), ("Macro PR-AUC", "pr_auc_macro")):
            if s in ("persistence", "majority") and m in ("roc_auc_ovr_macro", "pr_auc_macro"):
                continue  # n/a
            r = v(t11, s, m)
            ci = [] if (pd.isna(r.ci_low) or s == "majority" or (m == "g_mean" and s == "persistence")) else [r.ci_low, r.ci_high]
            out.append((row, col, [r.value] + ci))
        out.append(("MCC", col, [v(t11, s, "mcc").value]))
        out.append(("QWK", col, [v(t11, s, "qwk").value]))
    return out


def spec_table6():
    t21, cmp21, pb = rd("t21_extra_models.csv"), rd("t21_comparisons.csv"), rd("t12_paired_bootstrap.csv")
    rows = {"Random Forest (reported)": "rf", "CatBoost": "catboost", "XGBoost": "xgb", "SVM (RBF)": "svm",
            "FT-Transformer": "ftt", "LightGBM": "lgbm", "Persistence baseline": "persistence", "TabNet": "tabnet",
            "Majority baseline": "majority"}
    out = []
    for label, s in rows.items():
        f1 = t21[(t21.system == s) & (t21.metric == "f1_macro")].iloc[0]
        out.append((label, "Macro F1", [f1.value, f1.ci_low, f1.ci_high]))
        out.append((label, "F1 Active", [t21[(t21.system == s) & (t21.metric == "f1_active")].iloc[0].value]))
        out.append((label, "MCC", [t21[(t21.system == s) & (t21.metric == "mcc")].iloc[0].value]))
        if s in ("lgbm", "catboost", "tabnet", "ftt"):
            r = cmp21[cmp21.model == s].iloc[0]
            out.append((label, "RF minus model", [r.delta_rf_minus_model_f1_macro, r.delta_rf_minus_model_f1_macro_ci_low,
                                                   r.delta_rf_minus_model_f1_macro_ci_high]))
        elif s in ("xgb", "svm", "persistence"):
            pair = {"xgb": "RF-XGB", "svm": "RF-SVM", "persistence": "RF-persistence"}[s]
            r = pb[(pb.pair == pair) & (pb.metric == "f1_macro")].iloc[0]
            out.append((label, "RF minus model", [r.diff_point, r.ci_low, r.ci_high]))
    return out


def spec_table8():
    logo = rd("t15_logo.csv")
    rows = {"None (full model)": "none", "Composite scores": "G1_composite_scores", "Heart rate": "G2_heart_rate",
            "HRV": "G3_hrv", "Sleep architecture": "G4_sleep_architecture", "Activity and energy": "G5_activity_energy",
            "Respiration, temperature, SpO2": "G6_resp_thermal_spo2"}
    out = []
    for label, g in rows.items():
        r = logo[logo.group == g].iloc[0]
        if g != "none":
            out.append((label, "Features", [19 - int(r.n_features)]))  # the table prints the size of the removed group
        else:
            out.append((label, "Features", [r.n_features]))
        out.append((label, "Macro F1", [r.f1_macro]))
        out.append((label, "F1 Active", [r.f1_active]))
        if g != "none":
            out.append((label, "Difference in macro F1", [r.delta_f1_macro, r.delta_f1_macro_ci_low, r.delta_f1_macro_ci_high]))
    return out


def spec_table9():
    g = rd("t31_threshold_grid.csv")
    out = []
    for label, cell in (("T2", "R55_A85"), ("T3 (reported)", "R60_A85"), ("T4 *", "R60_A90"), ("T5", "R65_A85"), ("T6 *", "R65_A90")):
        r = g[g.cell == cell].iloc[0]
        out.append((label, "Macro F1", [r.f1_macro, r.f1_macro_ci_low, r.f1_macro_ci_high]))
        out.append((label, "Persistence", [r.pers_f1_macro, r.pers_f1_macro_ci_low, r.pers_f1_macro_ci_high]))
        out.append((label, "RF minus persistence", [r.delta_f1_macro, r.delta_f1_macro_ci_low, r.delta_f1_macro_ci_high]))
        out.append((label, "R/N/A", [r.n_rest, r.n_normal, r.n_active]))
    return out


def spec_agreement(rows_map: dict[str, str], cols=("Rest", "Normal", "Active")):
    tests, desc = rd("t32_tests.csv"), rd("t32_descriptives.csv")
    out = []
    for label, var in rows_map.items():
        for cls in cols:
            r = desc[(desc.variable == var) & (desc.cls == cls)].iloc[0]
            out.append((label, cls, [r["median"], r.p25, r.p75]))
        t = tests[tests.variable == var].iloc[0]
        out.append((label, "eps2", [t.epsilon2]))
        out.append((label, "Cliff's delta", [t.cliff_active_rest, t.cliff_active_rest_ci_low, t.cliff_active_rest_ci_high]))
    return out


def spec_table10():
    return spec_agreement({"Mean 24-h heart rate": "aw_hr_mean", "Respiratory rate": "aw_resp_rate", "HRV, SDNN": "aw_hrv_mean",
                           "SpO2": "aw_spo2", "Active calories": "aw_active_calories", "Steps": "aw_steps"})


def spec_table_s3():
    return spec_agreement({"Nocturnal HRV, ring": "hrv_mean", "Lowest nocturnal heart rate, ring": "rhr_lowest",
                           "Skin-temperature deviation, ring": "temp_deviation", "Steps, ring": "steps_oura",
                           "Active calories, watch, day t-1": "aw_active_calories_lag1", "Steps, watch, day t-1": "aw_steps_lag1",
                           "Steps, ring, day t-1": "steps_oura_lag1"})


def spec_table11():
    ew = rd("t33_expanding_window.csv")
    ew["fold"] = ew["fold"].astype(str)
    rows = {"Dec 2023 to Jul 2024": "4", "Jul 2024 to Feb 2025": "3", "Mar 2025 to Sep 2025": "2",
            "Sep 2025 to Mar 2026 (reported)": "1", "Pooled (768 days)": "pooled"}
    out = []
    for label, f in rows.items():
        r = ew[ew.fold == f].iloc[0]
        out.append((label, "Macro F1", [r.f1_macro, r.f1_macro_ci_low, r.f1_macro_ci_high]))
        out.append((label, "F1 Active", [r.f1_active, r.f1_active_ci_low, r.f1_active_ci_high]))
        out.append((label, "Test R/N/A", [r.test_rest, r.test_normal, r.test_active]))
        if f != "pooled":
            out.append((label, "Train days", [r.n_train, r.train_active]))
            out.append((label, "Persistence", [r.pers_f1_macro, r.pers_f1_macro_ci_low, r.pers_f1_macro_ci_high]))
            out.append((label, "RF minus persistence", [r.delta_f1_macro, r.delta_f1_macro_ci_low, r.delta_f1_macro_ci_high]))
    return out


def spec_table_a1():
    tr, cmp22, seeds, cv = rd("t22_test_results.csv"), rd("t22_comparisons.csv"), rd("t22_seeds.csv"), rd("t22_cv_results.csv")
    rows = {"Random Forest": "rf", "XGBoost": "xgb", "SVM (RBF)": "svm", "LightGBM": "lgbm", "CatBoost": "catboost",
            "TabNet": "tabnet", "FT-Transformer": "ftt"}
    out = []
    for label, m in rows.items():
        r = tr[tr.model == m].iloc[0]
        out.append((label, "CV macro F1", [r.cv_f1_mean, r.cv_f1_sd]))
        out.append((label, "Test macro F1", [r.f1_macro, r.f1_macro_ci_low, r.f1_macro_ci_high]))
        out.append((label, "Test F1 Active", [r.f1_active]))
        c = cmp22[(cmp22.model == m) & (cmp22.comparison == "a")].iloc[0]
        out.append((label, "Tuned minus reported RF", [c.diff_point, c.ci_low, c.ci_high]))
        s = seeds[seeds.model == m].f1_macro.astype(float)
        out.append((label, "Seed range", [s.min()] if s.nunique() == 1 else [s.min(), s.max()]))
        mc = cv[cv.model == m].copy()
        mc["f1_mean"] = mc["f1_mean"].astype(float)
        ref = mc[mc.config_id.astype(str) == "ref"]
        assert len(ref) == 1, (m, len(ref))
        rank = int((mc.f1_mean > float(ref.iloc[0].f1_mean)).sum()) + 1
        out.append((label, "Reference rank", [rank, len(mc)]))
    return out


def spec_table_s1a():
    mc = rd("t12_mcnemar.csv")
    out = []
    for _, r in mc.iterrows():
        exp = [r.n01, r.n10, r.chi2_cc, r.p_cc, r.p_exact]
        if r.p_cc_holm not in ("", "n/a") and not pd.isna(r.p_cc_holm):
            exp.append(float(r.p_cc_holm))
        out.append((r.pair, "n01", [r.n01]))
        out.append((r.pair, "n10", [r.n10]))
        out.append((r.pair, "Chi-square", [r.chi2_cc]))
        out.append((r.pair, "p (cc)", [r.p_cc]))
        out.append((r.pair, "p (exact)", [r.p_exact]))
        if len(exp) == 6:
            out.append((r.pair, "Holm", [exp[5]]))
    return out


def check_table_s1b(tbl_rows: list[list[str]]) -> tuple[int, list[str]]:
    pb = rd("t12_paired_bootstrap.csv")
    hdr = tbl_rows[0]
    ci_i, d_i, p_i = col_index(hdr, "95% CI"), col_index(hdr, "Difference"), col_index(hdr, "Proportion")
    n, bad = 0, []
    for row in tbl_rows[1:]:
        metric = "f1_macro" if row[1] == "macro F1" else "balanced_accuracy"
        r = pb[(pb.pair == row[0]) & (pb.metric == metric)].iloc[0]
        for ci, exp in ((d_i, [r.diff_point]), (ci_i, [r.ci_low, r.ci_high]), (p_i, [r.prop_diff_gt_0])):
            msg = compare_cell(row[ci], exp)
            n += 1
            if msg:
                bad.append(f"row {row[0]} / {row[1]}, col {hdr[ci]}: {msg}")
    return n, bad


def check_table_s2(tbl_rows: list[list[str]], part: str) -> tuple[int, list[str]]:
    cal = rd("t13_calibration.csv")
    cal = cal[cal.split == "test"]
    hdr = tbl_rows[0]
    n, bad = 0, []

    def pick(fit, var, metric):
        d = cal[(cal.fit_folds.astype(str) == fit) & (cal.variant == var) & (cal.metric == metric)]
        assert len(d) == 1, (fit, var, metric)
        return d.iloc[0]

    cols_a = [("ECE top-label", "ece_top", True), ("MCE top-label", "mce_top", False), ("Brier multiclass", "brier_multiclass", True),
              ("Log loss", "log_loss", True)]
    cols_b = [("ECE Active", "ece_active", False), ("Brier Active", "brier_active", True), ("ECE Rest", "ece_rest", False),
              ("Brier Rest", "brier_rest", False), ("ECE Normal", "ece_normal", False), ("Brier Normal", "brier_normal", False)]
    for row in tbl_rows[1:]:
        fit, var = row[0], row[1]
        for cp, metric, ci in (cols_a if part == "a" else cols_b):
            r = pick(fit, var, metric)
            exp = [r.point] + ([r.ci_low, r.ci_high] if ci else [])
            msg = compare_cell(row[col_index(hdr, cp)], exp)
            n += 1
            if msg:
                bad.append(f"row {fit} / {var}, col {cp}: {msg}")
    return n, bad


def check_table_s2c(tbl_rows: list[list[str]]) -> tuple[int, list[str]]:
    sl = rd("t13_safety_layer_variants.csv")
    hdr = tbl_rows[0]
    n, bad = 0, []
    cols = [("Coverage", lambda r: [r.coverage_pct]), ("Days overridden", lambda r: [r.n_overridden]),
            ("Mean caloric error", lambda r: [r.mean_cal_error]), ("Maximum error", lambda r: [r.max_cal_error]),
            ("Macro F1", lambda r: [r.f1_macro]), ("Accuracy", lambda r: [r.accuracy]), ("F1 Active", lambda r: [r.f1_active]),
            ("States issued", lambda r: [r.n_pred_rest, r.n_pred_normal, r.n_pred_active])]
    for row in tbl_rows[1:]:
        d = sl[(sl.variant == row[0]) & (sl.threshold.astype(float) == 0.5)]
        assert len(d) == 1
        r = d.iloc[0]
        for cp, f in cols:
            msg = compare_cell(row[col_index(hdr, cp)], [float(x) for x in f(r)])
            n += 1
            if msg:
                bad.append(f"row {row[0]}, col {cp}: {msg}")
    return n, bad


def check_spec(tbl_rows: list[list[str]], spec: list[tuple[str, str, list]]) -> tuple[int, list[str]]:
    hdr = tbl_rows[0]
    n, bad = 0, []
    for row_p, col_p, exp in spec:
        cell = tbl_rows[row_index(tbl_rows, row_p)][col_index(hdr, col_p)]
        msg = compare_cell(cell, [float(x) for x in exp])
        n += 1
        if msg:
            bad.append(f"row '{row_p}', col '{col_p}': {msg}")
    return n, bad


TABLES = [
    # (docx, caption prefix, checker, csv sources)
    ("manuscript", "Table 4.", lambda rows: check_spec(rows, spec_table4()), "t11_extended_metrics.csv, t21_extra_models.csv (per-class F1)"),
    ("manuscript", "Table 6.", lambda rows: check_spec(rows, spec_table6()), "t21_extra_models.csv, t21_comparisons.csv, t12_paired_bootstrap.csv"),
    ("manuscript", "Table 8.", lambda rows: check_spec(rows, spec_table8()), "t15_logo.csv"),
    ("manuscript", "Table 9.", lambda rows: check_spec(rows, spec_table9()), "t31_threshold_grid.csv (rows T2-T6; T1 and T7 are base rows)"),
    ("manuscript", "Table 10.", lambda rows: check_spec(rows, spec_table10()), "t32_descriptives.csv, t32_tests.csv"),
    ("manuscript", "Table 11.", lambda rows: check_spec(rows, spec_table11()), "t33_expanding_window.csv"),
    ("manuscript", "Table A.1.", lambda rows: check_spec(rows, spec_table_a1()), "t22_test_results.csv, t22_comparisons.csv, t22_seeds.csv, t22_cv_results.csv (reference rank)"),
    ("supplementary", "Table S1. (a)", lambda rows: check_spec(rows, spec_table_s1a()), "t12_mcnemar.csv"),
    ("supplementary", "Table S1. (b)", check_table_s1b, "t12_paired_bootstrap.csv"),
    ("supplementary", "Table S2. (a)", lambda rows: check_table_s2(rows, "a"), "t13_calibration.csv"),
    ("supplementary", "Table S2. (b)", lambda rows: check_table_s2(rows, "b"), "t13_calibration.csv"),
    ("supplementary", "Table S2. (c)", check_table_s2c, "t13_safety_layer_variants.csv"),
    ("supplementary", "Table S3.", lambda rows: check_spec(rows, spec_table_s3()), "t32_descriptives.csv, t32_tests.csv"),
]
NOT_CHECKED = "Tables 1, 2, 3, 5, 7, 12, 13, 14 (submitted manuscript, not derived from revision/results/); text columns of Table A.1 and the SHAP column of Table 11"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "revision" / "audit" / "table_rounding_check.md"))
    args = ap.parse_args()
    docs = {"manuscript": body_tables(MAN / "KBS_Manuscript_R1.docx"), "supplementary": body_tables(MAN / "KBS_Supplementary_R1.docx")}
    lines = ["# Table rounding check (round(csv, d) == printed cell)", "",
             "audit-apply, AUDIT_DECISIONS #17 (process). Script: revision/scripts/_check_table_rounding.py. Every numeric cell of the",
             "tables below is compared with the full-precision value of its source csv in revision/results/, rounded once to the printed",
             "number of decimals. Not checked: " + NOT_CHECKED + ".", "",
             "| Document | Table | Source csv | Cells checked | Mismatches |", "|---|---|---|---|---|"]
    total, total_bad, details = 0, 0, []
    for doc, cap, checker, src in TABLES:
        caps = [c for c in docs[doc] if c.startswith(cap)]
        if len(caps) != 1:
            raise SystemExit(f"STOP: caption '{cap}' matched {len(caps)} tables in {doc}")
        n, bad = checker(grid(docs[doc][caps[0]]))
        total += n
        total_bad += len(bad)
        lines.append(f"| {doc} | {cap.rstrip('.')} | {src} | {n} | {len(bad)} |")
        details += [f"- {cap} {b}" for b in bad]
        print(f"[{'PASS' if not bad else 'FAIL'}] {doc} {cap} {n} cells, {len(bad)} mismatches")
        for b in bad:
            print("    ", b)
    lines += ["", f"Total: {total} cells checked, {total_bad} mismatches."]
    if details:
        lines += ["", "## Mismatches", ""] + details
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{total} cells, {total_bad} mismatches -> {args.out}")
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
