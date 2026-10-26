"""
watch_only_sensitivity.py - T3.2b: sensitivity of the aw_hr_mean label agreement (T3.2) to the
source mix of the Apple Health heart-rate records (audit finding F4.1).

Background. The daily feature aw_hr_mean is the parser mean over ALL HealthKit heart-rate records
of the day, whichever device wrote them (Apple Watch, Oura app, iPhone). The audit P0 pass
(revision/audit/_check_f4_empirical.py) recomputed, per labelled day, the same mean over all
records ("parser_all") and over the records whose source does not contain "Oura" ("watch_only"),
and wrote both to revision/audit/p0_parser_vs_watch_only.csv. watch_only is undefined on the days
with no non-Oura heart-rate record (632 of the 952 labelled days with a heart-rate record).

What this script does (register only; no figure, no manuscript change):
  1. keeps the labelled days (label of day d from models/dataset_snapshot_20260331.csv, column
     nutrition_label, as in T3.2: label of day t against the value of day t) on which watch_only is
     defined;
  2. reports n and the R / N / A distribution of these days, their distribution per calendar year
     (is the subset a particular period?), the per-class median [p25, p75] of watch_only AND of
     parser_all on the SAME days, Cliff's delta Active-Rest (and the other two pairs) with a 95 %
     percentile CI (1,000 resamples within each class, numpy default_rng(42), a fresh generator per
     series so both series are resampled on the same day indices) for both series, using
     cliffs_delta / cliffs_delta_ci imported from label_agreement.py, and the Spearman correlation
     between the two series on these days;
  3. PASS checks: identity of parser_all with the day-t aw_hr_mean of the T3.2 frame
     (common.load_raw) on the common days, |diff| <= 0.011 (parser_all is rounded to 2 decimals in
     the audit csv); identity of the snapshot label with the T3.2 label on the common days; the
     Spearman reproduces 0.6768 (printed by _check_f4_empirical.py on 320 days) when the subset is
     the same 320 days, otherwise the reason for the difference is reported.

The subset is defined by the presence of at least one non-Oura heart-rate record on the day; it is
not a random sample of the 958 labelled days. The result is a sensitivity analysis of the T3.2
aw_hr_mean row, not a validation.

Outputs (revision/results/): t32b_watch_only_hr.csv, t32b_watch_only_hr_table.md,
t32b_run_meta.json. Seed 42. 0 warnings under `python -W error`. LF line endings.
Run from repo root:  python -W error revision/scripts/watch_only_sensitivity.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SEED, build_dataset, load_raw  # noqa: E402
from label_agreement import CLASS_NAMES, N_BOOT, PAIR_NAMES, PAIRS, cliffs_delta, cliffs_delta_ci  # noqa: E402
from statistical_tests import LABELS, md_table, write_text  # noqa: E402

RES = ROOT / "revision" / "results"
AUDIT_CSV = ROOT / "revision" / "audit" / "p0_parser_vs_watch_only.csv"
SNAPSHOT = ROOT / "models" / "dataset_snapshot_20260331.csv"
METRIC = "heart_rate"
FEATURE = "aw_hr_mean"
SERIES = ["parser_all", "watch_only"]
IDENTITY_TOL = 0.011                 # parser_all is rounded to 2 decimals in the audit csv
# Documented exception (audit F4, block [P0.3]): apple_health_daily.csv was built before the export of 2026-03-29 was
# complete (aw_hr_mean 85.66 in the csv vs 86.48 from the complete file). The audit csv value (complete file) is kept;
# the identity check passes only if every mismatching day is in this set, and the mismatches are reported.
KNOWN_IDENTITY_EXCEPTIONS = {"2026-03-29"}
F4_SPEARMAN = 0.6768                 # _check_f4_empirical_output.txt, line "Spearman(parser_all, watch_only) heart_rate"
F4_N = 320
T32_CLIFF_ACTIVE_REST = -0.840       # CIFRE_REZIVIE row T3.2 aw_hr_mean, all 958 days (reference only)
CANON_N = 958
YEARS = [2023, 2024, 2025, 2026]


def load_subset() -> tuple[pd.DataFrame, dict]:
    """Labelled days (snapshot label of day d) with a defined watch_only heart-rate mean on day d."""
    pv = pd.read_csv(AUDIT_CSV)
    hr = pv[pv.metric == METRIC].copy()
    hr["date"] = pd.to_datetime(hr["date"])
    snap = pd.read_csv(SNAPSHOT, usecols=["date", "nutrition_label"])
    snap["date"] = pd.to_datetime(snap["date"])
    snap["label"] = snap["nutrition_label"].astype(int)
    merged = hr.merge(snap[["date", "label"]], on="date", how="inner")
    sub = merged.dropna(subset=["watch_only", "parser_all"]).sort_values("date").reset_index(drop=True)
    info = dict(n_hr_rows=int(len(hr)), n_hr_labelled_flag=int(hr.labelled.sum()), n_snapshot=int(len(snap)),
                n_labelled_with_hr=int(len(merged)), n_watch_only_undefined=int(merged.watch_only.isna().sum()),
                n_subset=int(len(sub)))
    return sub, info


def describe(sub: pd.DataFrame, series: str) -> dict:
    row: dict = dict(series=series, n=int(len(sub)))
    for c in LABELS:
        s = sub.loc[sub.label == c, series]
        nm = CLASS_NAMES[c].lower()
        row[f"n_{nm}"] = int(len(s))
        row[f"median_{nm}"] = float(s.median())
        row[f"p25_{nm}"] = float(s.quantile(0.25))
        row[f"p75_{nm}"] = float(s.quantile(0.75))
    for pair in PAIRS:
        a, b = (sub.loc[sub.label == pair[0], series].to_numpy(), sub.loc[sub.label == pair[1], series].to_numpy())
        rng = np.random.default_rng(SEED)              # fresh generator per series and pair: same indices for both series
        d = cliffs_delta(a, b)
        lo, hi = cliffs_delta_ci(a, b, rng, N_BOOT)
        row[f"cliff_{PAIR_NAMES[pair]}"] = d
        row[f"cliff_{PAIR_NAMES[pair]}_ci_low"] = lo
        row[f"cliff_{PAIR_NAMES[pair]}_ci_high"] = hi
    return row


def main(res: Path = RES) -> None:
    t0 = time.time()
    fails: list[str] = []

    def check(cond: bool, msg: str, abort: bool = False) -> None:
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)
            if abort:
                sys.exit(f"ABORT: {msg}")

    print("=== T3.2b subset ===")
    sub, info = load_subset()
    n = len(sub)
    dist = [int((sub.label == c).sum()) for c in LABELS]
    print(f"  heart_rate rows {info['n_hr_rows']}, labelled (csv flag) {info['n_hr_labelled_flag']}, snapshot rows "
          f"{info['n_snapshot']}, labelled days with a heart-rate record {info['n_labelled_with_hr']}, watch_only "
          f"undefined {info['n_watch_only_undefined']}, subset n {n}")
    print(f"  R / N / A on the subset: {dist[0]} / {dist[1]} / {dist[2]}")
    check(n > 0 and min(dist) >= 2, f"subset non-empty, every class has >= 2 days ({dist})", abort=True)
    check(info["n_hr_labelled_flag"] == info["n_labelled_with_hr"] and info["n_snapshot"] == CANON_N,
          f"csv 'labelled' flag == snapshot dates ({info['n_hr_labelled_flag']} days with a heart-rate record of "
          f"{CANON_N} labelled)")

    # ---- per-year distribution (is the subset a particular period?)
    snap = pd.read_csv(SNAPSHOT, usecols=["date", "nutrition_label"])
    snap["date"] = pd.to_datetime(snap["date"])
    snap["label"] = snap["nutrition_label"].astype(int)
    per_year: dict[str, dict] = {}
    for yr in YEARS:
        s_all = snap[snap.date.dt.year == yr]
        s_sub = sub[sub.date.dt.year == yr]
        per_year[str(yr)] = dict(n_subset=int(len(s_sub)), n_labelled=int(len(s_all)),
                                 share=float(len(s_sub) / len(s_all)) if len(s_all) else None,
                                 subset_rest_normal_active=[int((s_sub.label == c).sum()) for c in LABELS],
                                 labelled_rest_normal_active=[int((s_all.label == c).sum()) for c in LABELS],
                                 first=s_sub.date.min().strftime("%Y-%m-%d") if len(s_sub) else None,
                                 last=s_sub.date.max().strftime("%Y-%m-%d") if len(s_sub) else None)
        print(f"  {yr}: subset {len(s_sub)} of {len(s_all)} labelled days ({per_year[str(yr)]['share'] or 0:.3f}); "
              f"R/N/A subset {per_year[str(yr)]['subset_rest_normal_active']} vs labelled "
              f"{per_year[str(yr)]['labelled_rest_normal_active']}; span {per_year[str(yr)]['first']} .. "
              f"{per_year[str(yr)]['last']}")
    check(sum(v["n_subset"] for v in per_year.values()) == n, "per-year counts sum to n")

    # ---- identity with the T3.2 frame (day t value of aw_hr_mean, day t label)
    print("\n=== identity with the T3.2 frame (common.load_raw / build_dataset) ===")
    df = load_raw()
    X, y, dates = build_dataset(df)
    frame = pd.DataFrame({"date": dates.to_numpy(), "label_t32": y.to_numpy(dtype=int)})
    frame[FEATURE] = df.set_index("date").loc[dates, FEATURE].to_numpy(dtype=float)
    check(len(frame) == CANON_N and list(frame.label_t32.value_counts().sort_index()) == [202, 710, 46],
          "T3.2 frame: 958 labelled days, 202 / 710 / 46", abort=True)
    m = sub.merge(frame, on="date", how="left")
    check(bool(m[FEATURE].notna().all()), f"every subset day has a day-t {FEATURE} in the T3.2 frame")
    diff = (m.parser_all - m[FEATURE]).abs()
    max_diff = float(diff.max())
    mism = m.loc[diff > IDENTITY_TOL, ["date", "parser_all", FEATURE]]
    mism_days = [d.strftime("%Y-%m-%d") for d in mism.date]
    for r in mism.itertuples():
        print(f"  mismatch {r.date:%Y-%m-%d}: parser_all {r.parser_all:.2f} vs {FEATURE} {getattr(r, FEATURE):.2f} "
              f"({'documented exception' if f'{r.date:%Y-%m-%d}' in KNOWN_IDENTITY_EXCEPTIONS else 'UNDOCUMENTED'})")
    check(set(mism_days) <= KNOWN_IDENTITY_EXCEPTIONS,
          f"identity parser_all == {FEATURE}(t) (|diff| <= {IDENTITY_TOL}) on {len(m) - len(mism)} / {len(m)} subset days; "
          f"mismatches {mism_days} all in the documented exception set {sorted(KNOWN_IDENTITY_EXCEPTIONS)} "
          f"(max |diff| {max_diff:.4f})")
    check(bool((m.label == m.label_t32).all()), "snapshot label == T3.2 label (build_dataset) on every subset day")
    # the same identity on ALL labelled days with a heart-rate record, for completeness
    hr_all = pd.read_csv(AUDIT_CSV)
    hr_all = hr_all[(hr_all.metric == METRIC)].copy()
    hr_all["date"] = pd.to_datetime(hr_all["date"])
    m_all = frame.merge(hr_all[["date", "parser_all"]], on="date", how="inner")
    diff_all = (m_all.parser_all - m_all[FEATURE]).abs()
    mism_all = [d.strftime("%Y-%m-%d") for d in m_all.loc[diff_all > IDENTITY_TOL, "date"]]
    check(len(m_all) == info["n_labelled_with_hr"] and set(mism_all) <= KNOWN_IDENTITY_EXCEPTIONS,
          f"identity parser_all == {FEATURE}(t) on {len(m_all) - len(mism_all)} / {len(m_all)} labelled days with a "
          f"record; mismatches {mism_all} (max |diff| {float(diff_all.max()):.4f})")

    # ---- descriptives and Cliff's delta on the same days, both series
    print("\n=== descriptives, Cliff's delta (Active-Rest, Active-Normal, Normal-Rest), Spearman ===")
    rows = [describe(sub, s) for s in SERIES]
    rho, p_rho = stats.spearmanr(sub.parser_all, sub.watch_only)
    rho, p_rho = float(rho), float(p_rho)
    for r in rows:
        r["spearman_parser_vs_watch"] = rho
        r["n_days_differing"] = int(((sub.parser_all - sub.watch_only).abs() > 1e-9).sum())
        print(f"  {r['series']:11s} median R / N / A {r['median_rest']:.2f} / {r['median_normal']:.2f} / "
              f"{r['median_active']:.2f}; Cliff A-R {r['cliff_active_rest']:+.3f} [{r['cliff_active_rest_ci_low']:+.3f}, "
              f"{r['cliff_active_rest_ci_high']:+.3f}]; A-N {r['cliff_active_normal']:+.3f} "
              f"[{r['cliff_active_normal_ci_low']:+.3f}, {r['cliff_active_normal_ci_high']:+.3f}]; N-R "
              f"{r['cliff_normal_rest']:+.3f} [{r['cliff_normal_rest_ci_low']:+.3f}, {r['cliff_normal_rest_ci_high']:+.3f}]")
    d_pw = sub.parser_all - sub.watch_only
    print(f"  parser_all minus watch_only on the {n} days: median {d_pw.median():+.2f}, p5 {d_pw.quantile(.05):+.2f}, "
          f"p95 {d_pw.quantile(.95):+.2f}, max |diff| {d_pw.abs().max():.2f}, days differing {rows[0]['n_days_differing']}")
    # how many records feed each mean on these days (the Watch-only mean often rests on a handful of samples)
    rec = dict(n_all_median=float(sub.n_all.median()), n_all_p25=float(sub.n_all.quantile(.25)),
               n_all_p75=float(sub.n_all.quantile(.75)), n_noo_median=float(sub.n_noo.median()),
               n_noo_p25=float(sub.n_noo.quantile(.25)), n_noo_p75=float(sub.n_noo.quantile(.75)),
               n_noo_min=int(sub.n_noo.min()), n_noo_max=int(sub.n_noo.max()),
               days_noo_le_5=int((sub.n_noo <= 5).sum()), share_noo_median=float((sub.n_noo / sub.n_all).median()))
    print(f"  records per day: all sources median {rec['n_all_median']:.0f} [{rec['n_all_p25']:.0f}, {rec['n_all_p75']:.0f}]; "
          f"non-Oura median {rec['n_noo_median']:.0f} [{rec['n_noo_p25']:.0f}, {rec['n_noo_p75']:.0f}], min {rec['n_noo_min']}, "
          f"max {rec['n_noo_max']}, days with <= 5 non-Oura records {rec['days_noo_le_5']}, median non-Oura share "
          f"{rec['share_noo_median']:.3f}")
    print(f"  Spearman(parser_all, watch_only) = {rho:.4f} (p {p_rho:.2e}) on {n} days")
    if n == F4_N:
        check(round(rho, 4) == F4_SPEARMAN, f"Spearman reproduces {F4_SPEARMAN} from _check_f4_empirical on the same "
                                             f"{F4_N} days ({rho:.4f})")
    else:
        print(f"  NOTE Spearman not compared with {F4_SPEARMAN}: subset has {n} days, _check_f4 used {F4_N} "
              f"(labelled flag of the csv vs snapshot merge)")
    for r in rows:
        for nm in PAIR_NAMES.values():
            check(-1 <= r[f"cliff_{nm}"] <= 1 and r[f"cliff_{nm}_ci_low"] <= r[f"cliff_{nm}"] <= r[f"cliff_{nm}_ci_high"],
                  f"{r['series']}: Cliff {nm} in [-1, 1], CI contains the point")
    check(-1 <= rho <= 1, "Spearman in [-1, 1]")
    check(all(r["n_rest"] + r["n_normal"] + r["n_active"] == n for r in rows), "n per class sums to n")

    # ---- outputs
    out = pd.DataFrame(rows)
    out.to_csv(res / "t32b_watch_only_hr.csv", index=False, lineterminator="\n")
    fmt = {c: "%.4f" for c in out.columns if out[c].dtype.kind == "f"}
    md = ["# T3.2b - watch-only heart-rate sensitivity of the T3.2 aw_hr_mean row (register only)", "",
          f"Subset: {n} labelled days (R / N / A {dist[0]} / {dist[1]} / {dist[2]}) with at least one non-Oura "
          f"heart-rate record on day t; label of day t from models/dataset_snapshot_20260331.csv; values from "
          f"revision/audit/p0_parser_vs_watch_only.csv (metric heart_rate). parser_all = mean over all HealthKit "
          f"records of the day (== feature aw_hr_mean of day t within 0.011 on {len(m) - len(mism)} / {len(m)} days; "
          f"the exception {mism_days} is the documented incomplete-export day, csv 85.66 vs 86.48); watch_only = "
          f"mean over the records whose source does not contain \"Oura\" (non-Oura records per day: median "
          f"{rec['n_noo_median']:.0f} [{rec['n_noo_p25']:.0f}, {rec['n_noo_p75']:.0f}] of {rec['n_all_median']:.0f} "
          f"[{rec['n_all_p25']:.0f}, {rec['n_all_p75']:.0f}] in total; {rec['days_noo_le_5']} days with <= 5). "
          f"Cliff's delta = first class minus second, 95 % "
          f"percentile CI from {N_BOOT} resamples within each class (default_rng({SEED}), fresh per series and pair). "
          f"Spearman(parser_all, watch_only) = {rho:.4f} on the same {n} days. Seed {SEED}.", ""]
    md += md_table(out, fmt)
    md += ["", "Per-year distribution of the subset (n subset / n labelled days of the year, share; R / N / A on the subset):",
           ""]
    for yr, v in per_year.items():
        md.append(f"- {yr}: {v['n_subset']} / {v['n_labelled']} ({(v['share'] or 0):.3f}); R / N / A "
                  f"{' / '.join(str(k) for k in v['subset_rest_normal_active'])}; span {v['first']} .. {v['last']}")
    md += ["", f"Reference (T3.2, all {CANON_N} days, parser_all): Cliff's delta Active-Rest {T32_CLIFF_ACTIVE_REST:+.3f}.",
           "The subset is defined by the presence of a non-Oura heart-rate record, not at random; this is a "
           "sensitivity analysis, not a validation."]
    write_text(res / "t32b_watch_only_hr_table.md", "\n".join(md) + "\n")

    elapsed = round(time.time() - t0, 2)
    meta = dict(
        task="T3.2b watch-only heart-rate sensitivity (register only)",
        run_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), elapsed_seconds=elapsed, seed=SEED, n_boot=N_BOOT,
        rng="numpy default_rng(42), fresh generator per series and per pair (same resample indices for both series)",
        inputs=dict(audit_csv=str(AUDIT_CSV.relative_to(ROOT)), snapshot=str(SNAPSHOT.relative_to(ROOT)),
                    metric=METRIC, feature=FEATURE, label_column="nutrition_label (label of day t)"),
        subset_definition="labelled days with >= 1 non-Oura heart-rate record (watch_only defined); not random",
        interpretation="sensitivity of the T3.2 aw_hr_mean row to the record source mix, not a validation",
        counts=info, n=n, label_distribution=dist, per_year=per_year,
        identity=dict(tolerance=IDENTITY_TOL, max_abs_diff_subset=max_diff, max_abs_diff_all_labelled=float(diff_all.max()),
                      n_all_labelled=int(len(m_all)), mismatch_days_subset=mism_days, mismatch_days_all_labelled=mism_all,
                      documented_exceptions=sorted(KNOWN_IDENTITY_EXCEPTIONS),
                      labels_identical=bool((m.label == m.label_t32).all())),
        spearman=dict(rho=rho, p=p_rho, n=n, f4_reference=F4_SPEARMAN, f4_n=F4_N, reproduced=(n == F4_N and round(rho, 4) == F4_SPEARMAN)),
        parser_minus_watch=dict(median=float(d_pw.median()), p5=float(d_pw.quantile(.05)), p95=float(d_pw.quantile(.95)),
                                max_abs=float(d_pw.abs().max()), n_days_differing=rows[0]["n_days_differing"]),
        records_per_day=rec,
        results={r["series"]: {k: v for k, v in r.items() if k != "series"} for r in rows},
        t32_reference_cliff_active_rest_all_days=T32_CLIFF_ACTIVE_REST,
        outputs=["t32b_watch_only_hr.csv", "t32b_watch_only_hr_table.md", "t32b_run_meta.json"],
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
        sanity_checks_failed=fails,
    )
    write_text(res / "t32b_run_meta.json", json.dumps(meta, indent=2) + "\n")
    print(f"\nelapsed {elapsed} s")
    print("ALL CHECKS PASS" if not fails else "FAIL: " + "; ".join(fails))
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
