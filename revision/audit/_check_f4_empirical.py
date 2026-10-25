"""
_check_f4_empirical.py  (audit slice F4, 2026-10-25)
=====================================================
Evidence script for AUDIT_F4_provenance_shap_ablation_agreement.md.
Read-only with respect to the repository: writes only under revision/audit/.

Sub-commands (run from the repo root, `python -I revision/audit/_check_f4_empirical.py <cmd>`):

  p0   Provenance pass over the raw Health Auto Export JSON files in
       data/apple_health/New Automation/ (1,079 files, ~787 MB; declared
       exception to the 5-minute cap, under 10 minutes). Counts, per day and
       per metric, the records whose `source` contains "Oura" and those that
       do not; writes revision/audit/p0_source_mix.csv and
       revision/audit/p0_parser_vs_watch_only.csv; prints the per-year
       fractions, the first Oura date, the per-file date coverage, the
       sleep_analysis multiplicity and the parser-vs-Watch-only differences.
       Re-implements the parser's aggregation independently (mean of `Avg`
       for heart_rate, sum of `qty` for step_count / active_energy, sum of
       `deep` / `rem` for sleep_analysis) and, for three days, also calls the
       repository parser functions themselves (P0.4).

  t2   Evidence for the targeted questions of T2 (SHAP, ablation, agreement);
       imports the revision scripts in memory and never calls their main().

The two outputs are concatenated into _check_f4_empirical_output.txt.
"""
import sys
import json
import collections
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "revision" / "audit"
RAW = ROOT / "data" / "apple_health" / "New Automation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision" / "scripts"))

P0_METRICS = [
    "heart_rate", "active_energy", "step_count", "sleep_analysis",
    "respiratory_rate", "heart_rate_variability", "resting_heart_rate",
    "blood_oxygen_saturation",
]


def sha16(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# P0
# ----------------------------------------------------------------------------
def p0():
    import time
    t0 = time.time()
    files = sorted(RAW.glob("HealthAutoExport-*.json"))
    print(f"[P0.1] folder: {RAW}")
    print(f"[P0.1] files: {len(files)}; all .json: {all(f.suffix == '.json' for f in files)}")
    fdates = [f.stem.replace("HealthAutoExport-", "") for f in files]
    print(f"[P0.1] filename date range: {fdates[0]} .. {fdates[-1]}")
    full = pd.date_range("2023-06-09", "2026-03-30", freq="D").strftime("%Y-%m-%d")
    missing = sorted(set(full) - set(fdates))
    print(f"[P0.1] study window 2023-06-09..2026-03-30: {len(full)} days; files present: "
          f"{len(set(full) & set(fdates))}; missing filename dates: {len(missing)} -> {missing}")
    extra = sorted(set(fdates) - set(full))
    print(f"[P0.1] files outside the study window: {len(extra)} ({extra[0]}..{extra[-1]})" if extra else "[P0.1] none outside")

    # per (date, metric) counters
    cnt = collections.defaultdict(lambda: {"n_total": 0, "n_oura": 0, "n_watch_only": 0,
                                           "n_oura_pure": 0, "n_iphone": 0})
    # per (date, metric): parser-equivalent aggregates, all sources vs. excluding Oura
    agg = collections.defaultdict(lambda: {"all": [], "noo": [], "src": collections.Counter()})
    sleep_rows = []          # one row per sleep_analysis entry
    file_cov = []            # per file: n entries (8 metrics), n with entry date != file date
    bad_json = []
    first_oura = {}
    entry_dates_all = set()
    srcset = collections.Counter()

    for f, fd in zip(files, fdates):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # noqa
            bad_json.append((f.name, repr(e)[:80]))
            continue
        n_ent = 0
        n_off = 0
        for m in data.get("data", {}).get("metrics", []):
            name = m.get("name", "")
            if name not in P0_METRICS:
                continue
            for e in m.get("data", []):
                ds = e.get("date", "")
                if not ds:
                    continue
                day = ds[:10]
                n_ent += 1
                if day != fd:
                    n_off += 1
                entry_dates_all.add(day)
                src = e.get("source", "")
                srcset[(name, src)] += 1
                is_oura = "Oura" in src
                c = cnt[(day, name)]
                c["n_total"] += 1
                if is_oura:
                    c["n_oura"] += 1
                    if src == "Oura":
                        c["n_oura_pure"] += 1
                    if name not in first_oura or day < first_oura[name]:
                        first_oura[name] = day
                else:
                    c["n_watch_only"] += 1
                if "iPhone" in src:
                    c["n_iphone"] += 1
                a = agg[(day, name)]
                a["src"][src] += 1
                if name == "heart_rate":
                    v = e.get("Avg")
                elif name == "sleep_analysis":
                    v = (e.get("deep"), e.get("rem"), e.get("totalSleep"))
                    sleep_rows.append({"date": day, "file_date": fd, "source": src,
                                       "sleepStart": e.get("sleepStart", "")[:10],
                                       "sleepEnd": e.get("sleepEnd", "")[:10],
                                       "deep": e.get("deep"), "rem": e.get("rem"),
                                       "totalSleep": e.get("totalSleep")})
                else:
                    v = e.get("qty")
                if v is not None:
                    a["all"].append(v)
                    if not is_oura:
                        a["noo"].append(v)
        file_cov.append({"file_date": fd, "n_entries": n_ent, "n_off_date": n_off})

    print(f"[P0.2] parse time {time.time() - t0:.1f} s; unreadable files: {bad_json}")
    fc = pd.DataFrame(file_cov)
    print(f"[P0.2] per-file coverage: entries (8 metrics) per file min/median/max = "
          f"{fc.n_entries.min()} / {fc.n_entries.median():.0f} / {fc.n_entries.max()}; "
          f"files with 0 entries: {(fc.n_entries == 0).sum()} -> "
          f"{fc.loc[fc.n_entries == 0, 'file_date'].tolist()[:20]}")
    print(f"[P0.2] entries whose date differs from the file name date: {fc.n_off_date.sum()} "
          f"(files affected: {(fc.n_off_date > 0).sum()}) -> each file holds only its own day: "
          f"{'YES' if fc.n_off_date.sum() == 0 else 'NO'}")
    print(f"[P0.2] distinct entry dates: {len(entry_dates_all)}; "
          f"== distinct file dates with entries: {len(entry_dates_all) == (fc.n_entries > 0).sum()}")
    print("[P0.2] distinct source strings per metric (count of records):")
    for name in P0_METRICS:
        items = sorted(((s, n) for (m, s), n in srcset.items() if m == name), key=lambda x: -x[1])
        print(f"    {name}: " + "; ".join(f"{s!r}: {n}" for s, n in items))

    # p0_source_mix.csv
    rows = []
    for (day, name), c in sorted(cnt.items()):
        rows.append({"date": day, "metric": name, **c})
    mix = pd.DataFrame(rows)
    out = AUDIT / "p0_source_mix.csv"
    mix[["date", "metric", "n_total", "n_oura", "n_watch_only", "n_oura_pure", "n_iphone"]].to_csv(
        out, index=False, lineterminator="\n")
    print(f"[P0.2] written {out.relative_to(ROOT)}: {len(mix)} rows (date x metric), sha256[:16] {sha16(out)}")

    # labelled dates (958) and raw frame dates (1,009)
    snap = pd.read_csv(ROOT / "models" / "dataset_snapshot_20260331.csv")
    dcol = [c for c in snap.columns if "date" in c.lower()][0]
    lab_dates = set(pd.to_datetime(snap[dcol]).dt.strftime("%Y-%m-%d"))
    print(f"[P0.2] labelled dates from snapshot ({dcol}): {len(lab_dates)}; "
          f"range {min(lab_dates)}..{max(lab_dates)}; with a raw export file: {len(lab_dates & set(fdates))}; "
          f"without: {sorted(lab_dates - set(fdates))}")
    mix["year"] = mix["date"].str[:4]
    mix["labelled"] = mix["date"].isin(lab_dates)
    mix["any_oura"] = mix["n_oura"] > 0
    print("[P0.2] first calendar date with a record whose source contains 'Oura', per metric:")
    for name in P0_METRICS:
        print(f"    {name}: {first_oura.get(name, 'never')}")
    print("[P0.2] fraction of days with >= 1 Oura record, per metric and year "
          "(denominator = days having >= 1 record of that metric; 'lab' = restricted to the 958 labelled dates):")
    for name in P0_METRICS:
        sub = mix[mix.metric == name]
        parts = []
        for y, g in sub.groupby("year"):
            gl = g[g.labelled]
            parts.append(f"{y}: {g.any_oura.mean():.3f} ({g.any_oura.sum()}/{len(g)}), lab {gl.any_oura.mean() if len(gl) else float('nan'):.3f} ({gl.any_oura.sum()}/{len(gl)})")
        subl = sub[sub.labelled]
        parts.append(f"ALL lab: {subl.any_oura.mean():.3f} ({subl.any_oura.sum()}/{len(subl)})")
        print(f"    {name}: " + " | ".join(parts))
    print("[P0.2] share of RECORDS whose source contains 'Oura', per metric, labelled dates, per year:")
    for name in P0_METRICS:
        sub = mix[(mix.metric == name) & mix.labelled]
        parts = [f"{y}: {g.n_oura.sum() / max(g.n_total.sum(), 1):.3f}" for y, g in sub.groupby("year")]
        parts.append(f"ALL: {sub.n_oura.sum() / max(sub.n_total.sum(), 1):.3f}; pure-Oura records {sub.n_oura_pure.sum()}")
        print(f"    {name}: " + " | ".join(parts))
    print("[P0.2] days (labelled) where ALL records of the metric contain 'Oura' (no Watch-only record):")
    for name in P0_METRICS:
        sub = mix[(mix.metric == name) & mix.labelled]
        print(f"    {name}: {(sub.n_watch_only == 0).sum()} / {len(sub)}")

    # sleep_analysis multiplicity and date convention
    sl = pd.DataFrame(sleep_rows)
    sl["is_oura"] = sl.source.str.contains("Oura")
    per_day = sl.groupby("date").agg(n=("source", "size"), n_oura=("is_oura", "sum"),
                                     srcs=("source", lambda s: "|".join(sorted(set(s)))))
    print(f"[P0.2] sleep_analysis: entries {len(sl)}; days {len(per_day)}; days with >1 entry: "
          f"{(per_day.n > 1).sum()}; days with >1 entry AND mixed sources: "
          f"{((per_day.n > 1) & (per_day.n_oura > 0) & (per_day.n_oura < per_day.n)).sum()}; "
          f"days with an Oura entry: {(per_day.n_oura > 0).sum()}; days Oura-only: {(per_day.n_oura == per_day.n).sum()}")
    print(f"    source strings: {sl.source.value_counts().to_dict()}")
    print(f"    multiplicity distribution: {per_day.n.value_counts().sort_index().to_dict()}")
    pl = per_day[per_day.index.isin(lab_dates)]
    print(f"    on labelled dates: days {len(pl)}, >1 entry {(pl.n > 1).sum()}, with Oura {(pl.n_oura > 0).sum()}, Oura-only {(pl.n_oura == pl.n).sum()}")
    ok_end = (sl.sleepEnd == sl.date)
    ok_start = (sl.sleepStart == sl.date)
    print(f"    date convention: entry date == sleepEnd date on {ok_end.sum()}/{len(sl)} entries; "
          f"== sleepStart date on {ok_start.sum()}/{len(sl)} (previous-evening start: {(sl.sleepStart < sl.date).sum()})")
    print(f"    Watch entries: date==sleepEnd {ok_end[~sl.is_oura].sum()}/{(~sl.is_oura).sum()}; "
          f"Oura entries: date==sleepEnd {ok_end[sl.is_oura].sum()}/{sl.is_oura.sum()}")
    multi = per_day[per_day.n > 1].head(8)
    print(f"    first days with >1 entry:\n{multi.to_string()}")

    # parser-equivalent aggregates vs Watch-only, labelled dates
    rows = []
    for (day, name), a in agg.items():
        if name == "sleep_analysis":
            d_all = sum(x[0] for x in a["all"] if x[0] is not None)
            d_noo = sum(x[0] for x in a["noo"] if x[0] is not None)
            r_all = sum(x[1] for x in a["all"] if x[1] is not None)
            r_noo = sum(x[1] for x in a["noo"] if x[1] is not None)
            rows.append({"date": day, "metric": "sleep_deep_sum", "parser_all": round(d_all, 2),
                         "watch_only": round(d_noo, 2) if a["noo"] else np.nan, "n_all": len(a["all"]), "n_noo": len(a["noo"])})
            rows.append({"date": day, "metric": "sleep_rem_sum", "parser_all": round(r_all, 2),
                         "watch_only": round(r_noo, 2) if a["noo"] else np.nan, "n_all": len(a["all"]), "n_noo": len(a["noo"])})
            continue
        if name == "heart_rate":
            v_all = round(float(np.mean(a["all"])), 2) if a["all"] else np.nan
            v_noo = round(float(np.mean(a["noo"])), 2) if a["noo"] else np.nan
        elif name in ("step_count", "active_energy"):
            v_all = round(float(np.sum(a["all"])), 2) if a["all"] else np.nan
            v_noo = round(float(np.sum(a["noo"])), 2) if a["noo"] else np.nan
        else:
            v_all = round(float(np.mean(a["all"])), 2) if a["all"] else np.nan
            v_noo = round(float(np.mean(a["noo"])), 2) if a["noo"] else np.nan
        rows.append({"date": day, "metric": name, "parser_all": v_all, "watch_only": v_noo,
                     "n_all": len(a["all"]), "n_noo": len(a["noo"])})
    pv = pd.DataFrame(rows).sort_values(["metric", "date"])
    pv["labelled"] = pv.date.isin(lab_dates)
    pv["diff"] = pv.parser_all - pv.watch_only
    out2 = AUDIT / "p0_parser_vs_watch_only.csv"
    pv.to_csv(out2, index=False, lineterminator="\n")
    print(f"[P0.4-ext] written {out2.relative_to(ROOT)}: {len(pv)} rows, sha256[:16] {sha16(out2)}")
    print("[P0.4-ext] parser aggregate (all sources) minus Watch-only aggregate, labelled dates:")
    for name, g in pv[pv.labelled].groupby("metric"):
        d = g["diff"].dropna()
        nz = (d.abs() > 1e-9).sum()
        rel = (g["diff"] / g["parser_all"].replace(0, np.nan)).dropna()
        print(f"    {name}: days {len(g)}, watch_only undefined (no non-Oura record) {g.watch_only.isna().sum()}, "
              f"diff != 0 on {nz}; diff mean {d.mean():.3f}, median {d.median():.3f}, "
              f"p5 {d.quantile(.05):.3f}, p95 {d.quantile(.95):.3f}, max|diff| {d.abs().max():.3f}; "
              f"median |rel| {rel.abs().median():.4f}, p95 |rel| {rel.abs().quantile(.95):.4f}")
    # Spearman between the two variants on labelled days
    from scipy.stats import spearmanr
    for name in ["heart_rate", "step_count", "active_energy"]:
        g = pv[(pv.metric == name) & pv.labelled].dropna(subset=["parser_all", "watch_only"])
        print(f"    Spearman(parser_all, watch_only) {name}: {spearmanr(g.parser_all, g.watch_only)[0]:.4f} on {len(g)} days")

    # P0.3 cross-check against the aggregated csv actually used by the pipeline
    daily = pd.read_csv(ROOT / "apple_watch_api" / "apple_health_daily.csv")
    print(f"[P0.3] apple_health_daily.csv: {len(daily)} rows, {daily.date.min()}..{daily.date.max()}, sha256[:16] {sha16(ROOT / 'apple_watch_api' / 'apple_health_daily.csv')}")
    for name, col in [("heart_rate", "aw_hr_mean"), ("step_count", "aw_steps"), ("active_energy", "aw_active_energy_kj"),
                      ("sleep_deep_sum", "aw_sleep_deep_hours"), ("sleep_rem_sum", "aw_sleep_rem_hours"),
                      ("respiratory_rate", "aw_resp_rate"), ("heart_rate_variability", "aw_hrv_mean"),
                      ("blood_oxygen_saturation", "aw_spo2"), ("resting_heart_rate", "aw_rhr")]:
        g = pv[pv.metric == name].merge(daily[["date", col]], on="date", how="inner")
        both = g.dropna(subset=["parser_all", col])
        eq = np.isclose(both.parser_all, both[col], atol=0.011)
        print(f"    {col}: common dates {len(g)}, both non-null {len(both)}, re-implemented parser == csv (|d|<=0.011) on {eq.sum()} "
              f"(max |d| {np.abs(both.parser_all - both[col]).max():.3f}); first mismatches: "
              f"{both.loc[~eq, ['date', 'parser_all', col]].head(3).values.tolist()}")

    # P0.4: three days, repository parser functions, with and without Oura
    import parse_health_auto_export as phe
    days = ["2023-06-09", "2024-09-15", "2026-03-15"]
    print("[P0.4] three days through the repository parser (load_json_file + aggregate_daily), "
          "all records vs. records whose source does not contain 'Oura':")
    for d in days:
        f = RAW / f"HealthAutoExport-{d}.json"
        rows_all = phe.load_json_file(f)
        # replicate load_json_file but keep only non-Oura records: patch the json in memory
        data = json.loads(f.read_text(encoding="utf-8"))
        for m in data["data"]["metrics"]:
            m["data"] = [e for e in m.get("data", []) if "Oura" not in e.get("source", "")]
        tmpf = AUDIT / "_p0_tmp_noo.json"
        tmpf.write_text(json.dumps(data), encoding="utf-8")
        rows_noo = phe.load_json_file(tmpf)
        tmpf.unlink()
        a_all = phe.aggregate_daily(pd.DataFrame(rows_all)).set_index("date").loc[d]
        a_noo = phe.aggregate_daily(pd.DataFrame(rows_noo)).set_index("date").loc[d]
        csvrow = daily.set_index("date").loc[d]
        for col in ["aw_hr_mean", "aw_steps", "aw_active_energy_kj", "aw_sleep_deep_hours", "aw_resp_rate"]:
            va, vn, vc = a_all.get(col, np.nan), a_noo.get(col, np.nan), csvrow.get(col, np.nan)
            print(f"    {d} {col}: parser(all)={va} csv={vc} match={np.isclose(va, vc, atol=0.011, equal_nan=True)} | "
                  f"excl-Oura={vn} diff={va - vn if pd.notna(vn) else 'n/a'} "
                  f"({(va - vn) / va * 100 if pd.notna(vn) and va else float('nan'):.1f}%)")
    print(f"[P0] total time {time.time() - t0:.1f} s")


# ----------------------------------------------------------------------------
# T2 (filled in after the P0 pass)
# ----------------------------------------------------------------------------
def t2():
    import inspect
    import time
    import joblib
    import shap
    from sklearn.metrics import f1_score
    import common
    import shap_stability as ss
    import group_ablation as ga
    import label_agreement as la
    import statistical_tests as st
    t0 = time.time()
    RES = ROOT / "revision" / "results"
    F = common.FEATURES
    print(f"[T2] versions: shap {shap.__version__}; scripts imported in memory, no main() called")
    X, y, dates = common.build_dataset(common.load_raw())
    canon = common.canonical_split(X, y, dates)
    rf = joblib.load(RES / "models" / "rf_revision.joblib")
    imp = joblib.load(RES / "models" / "imputer_revision.joblib")
    ref = pd.read_csv(RES / "t14_shap_reference.csv").set_index("feature").loc[F]
    ref_mean = ref.mean_abs_shap.to_numpy()

    # ---- Q1 data used for the reference SHAP
    print("\n[Q1] reference SHAP: which matrix reproduces t14_shap_reference.csv")
    print(f"  raw X_test (rows 766..957) NaN count = {int(X.iloc[766:].isna().sum().sum())} -> imputation needed")
    print(f"  canon.X_test == imputer_revision.joblib.transform(raw X_test): max diff "
          f"{np.abs(imp.transform(X.iloc[766:]) - canon.X_test.to_numpy()).max():.1e}")
    ex = shap.TreeExplainer(rf)
    sv = ex.shap_values(canon.X_test)
    print(f"  shap_values(X_test) type {type(sv).__name__}, shape {np.asarray(sv).shape}; expected_value {np.round(ex.expected_value, 6).tolist()}")
    for name, M in [("imputed test, unscaled (192)", canon.X_test), ("imputed test, SCALED (192)", canon.X_test_scaled),
                    ("imputed train (766)", canon.X_train)]:
        m = ss.mean_abs_shap(rf, M)
        print(f"  mean|SHAP| on {name:30s}: max |diff vs reference| = {np.abs(m - ref_mean).max():.2e}; "
              f"top-5 {[F[i].replace('_lag1', '') for i in np.argsort(-m, kind='stable')[:5]]}")
    print(f"  rf_revision.joblib used as loaded (no refit): rf.random_state={rf.random_state}, n_estimators={rf.n_estimators}")

    # ---- Q2 aggregation
    print("\n[Q2] aggregation on the (192, 19, 3) array")
    arr = np.asarray(sv)
    mean_cls = np.abs(arr).mean(axis=(0, 2))
    sum_cls = np.abs(arr).mean(axis=0).sum(axis=1)
    act_only = np.abs(arr[:, :, 2]).mean(axis=0)
    for name, m in [("mean over samples and classes", mean_cls), ("SUM over classes", sum_cls), ("Active class only", act_only)]:
        print(f"  {name:32s}: aw_hr_mean {m[F.index('aw_hr_mean_lag1')]:.4f}; max|diff vs ref| {np.abs(m - ref_mean).max():.2e}; "
              f"top-5 {[F[i].replace('_lag1', '') for i in np.argsort(-m, kind='stable')[:5]]}")
    as_list = [arr[:, :, c] for c in range(3)]
    print(f"  aggregate_multiclass(array) vs aggregate_multiclass(list of 3 (192,19)): max |diff| "
          f"{np.abs(ss.aggregate_multiclass(arr, 192) - ss.aggregate_multiclass(as_list, 192)).max():.1e} "
          f"(np.asarray(list) shape {np.asarray(as_list).shape} -> branch (classes, samples, features))")
    print(f"  ambiguity guard: n_samples 192 != n_classes 3, so the shape[0] dispatch cannot misfire here")

    # ---- Q3 additivity
    print("\n[Q3] additivity")
    proba = rf.predict_proba(canon.X_test)
    add = np.abs(arr.sum(axis=1) + np.asarray(ex.expected_value)[None, :] - proba).max()
    print(f"  max |sum_f SHAP + expected_value - predict_proba| = {add:.3e}  (< 1e-12: {add < 1e-12}; < 1e-9: {add < 1e-9})")
    print(f"  max |predict_proba - test_predictions_rf.csv| = "
          f"{np.abs(proba - pd.read_csv(RES / 'test_predictions_rf.csv')[['p_rest', 'p_normal', 'p_active']].to_numpy()).max():.2e}")

    # ---- Q4 bootstrap
    print("\n[Q4] bootstrap mode")
    npy = np.load(RES / "bootstrap_train_indices_100x766_seed42.npy")
    regen = np.random.default_rng(42).integers(0, 766, size=(100, 766))
    print(f"  npy shape {npy.shape} dtype {npy.dtype}; == default_rng(42).integers(0, 766, (100, 766)): {np.array_equal(npy, regen)}; "
          f"min {npy.min()} max {npy.max()}; mean unique rows {np.mean([len(np.unique(r)) for r in npy]):.1f}")
    print(f"  raw train (rows 0..765) NaN count = {int(X.iloc[:766].isna().sum().sum())} -> rows are bootstrapped BEFORE imputation")
    print("  source bootstrap_split:\n    " + inspect.getsource(ss.bootstrap_split).replace("\n", "\n    "))
    bcsv = pd.read_csv(RES / "t14_shap_values_bootstrap.csv")
    for r in (0, 1):
        sp = ss.bootstrap_split(X, y, dates, npy[r])
        same_test = (np.array_equal(sp.y_test.to_numpy(), canon.y_test.to_numpy())
                     and list(sp.dates_test) == list(canon.dates_test) and len(sp.y_test) == 192)
        out = ss.run_one(sp, 42)
        got = bcsv[bcsv.run == r].set_index("feature").loc[F, "mean_abs_shap"].to_numpy()
        print(f"  run {r}: train rows {len(sp.y_train)} (unique {len(np.unique(npy[r]))}), SMOTE rows {len(sp.y_train_sm)}, "
              f"test = the 192 original days: {same_test}; re-run mean|SHAP| vs csv max diff {np.abs(out['mean_abs'] - got).max():.2e}; "
              f"ranks equal: {np.array_equal(ss.rank_desc(out['mean_abs']), bcsv[bcsv.run == r].set_index('feature').loc[F, 'rank'].to_numpy())}")
    print(f"  imputer of run 0 differs from the canonical imputer (medians): max |diff| "
          f"{np.abs(ss.bootstrap_split(X, y, dates, npy[0]).imputer.statistics_ - canon.imputer.statistics_).max():.3f}")

    # ---- Q5 seeds
    print("\n[Q5] seeds mode")
    print("  source fit_rf:\n    " + inspect.getsource(ss.fit_rf).replace("\n", "\n    "))
    scsv = pd.read_csv(RES / "t14_shap_values_seeds.csv")
    got42 = scsv[scsv.run == 42].set_index("feature").loc[F, "mean_abs_shap"].to_numpy()
    print(f"  csv run 42 vs reference csv: max diff {np.abs(got42 - ref_mean).max():.2e}; ranks equal "
          f"{np.array_equal(scsv[scsv.run == 42].set_index('feature').loc[F, 'rank'].to_numpy(), ref['rank'].to_numpy())}")
    out42 = ss.run_one(canon, 42)
    print(f"  re-run run 42 (canonical split, RF seed 42) vs reference (computed): max diff "
          f"{np.abs(out42['mean_abs'] - ss.mean_abs_shap(rf, canon.X_test)).max():.2e}; F1 macro {out42['f1_macro']:.4f}")
    out7 = ss.run_one(canon, 7)
    got7 = scsv[scsv.run == 7].set_index("feature").loc[F, "mean_abs_shap"].to_numpy()
    print(f"  re-run run 7 vs csv: max diff {np.abs(out7['mean_abs'] - got7).max():.2e}")
    print(f"  SMOTE train identical across seed runs (canon reused): rows {len(canon.y_train_sm)}; only rf.random_state varies")

    # ---- Q6 ranks, Jaccard, Kendall
    print("\n[Q6] ranks / Jaccard / Kendall")
    print("  rank_desc source: " + inspect.getsource(ss.rank_desc).strip().splitlines()[-1].strip())
    print("  kendall source:   " + inspect.getsource(ss.kendall).strip().splitlines()[-1].strip())
    ref_top = set(ref[ref["rank"] <= 5].index)
    for mode, csv, claim_j, claim_t in [("bootstrap", bcsv, 0.4904, 0.3909), ("seeds", scsv, 0.5276, 0.7506)]:
        jac, tau = [], []
        for r, g in csv.groupby("run"):
            g = g.set_index("feature").loc[F]
            top = set(g[g["rank"] <= 5].index)
            top_v = set(F[i] for i in np.argsort(-g.mean_abs_shap.to_numpy(), kind="stable")[:5])
            assert top == top_v
            jac.append(len(top & ref_top) / len(top | ref_top))
            tau.append(__import__("scipy").stats.kendalltau(g["rank"].to_numpy(), ref["rank"].to_numpy()).statistic)
        print(f"  {mode}: mean Jaccard top-5 {np.mean(jac):.4f} (claim {claim_j}), mean Kendall tau-b {np.mean(tau):.4f} (claim {claim_t}); "
              f"runs {len(jac)}; ties in ranks: {int((csv.groupby('run')['rank'].nunique() < 19).sum())} runs")

    # ---- Q7 partition
    print("\n[Q7] partition of the 19 features")
    groups = ga.build_groups("G1")
    ga.check_partition(groups)
    flat = [f for g in groups.values() for f in g]
    brief = {"G1": ["readiness_score", "sleep_score"], "G2": ["aw_hr_mean", "rhr_lowest", "rhr_avg", "recovery_index"],
             "G3": ["hrv_mean", "hrv_balance", "aw_hrv_mean"], "G4": ["aw_sleep_deep_hours", "aw_sleep_rem_hours"],
             "G5": ["activity_score", "steps_oura", "aw_steps", "aw_active_calories"],
             "G6": ["respiratory_rate", "aw_resp_rate", "temp_deviation", "aw_spo2"]}
    print(f"  n features {len(flat)}, unique {len(set(flat))}, set == FEATURES {set(flat) == set(F)}, sizes {[len(v) for v in groups.values()]}")
    print(f"  groups as sets == brief/CIFRE_REVIZIE listing: "
          f"{all(set(x.replace('_lag1', '') for x in groups[g]) == set(brief[g]) for g in brief)}")
    meta15 = json.loads((RES / "t15_run_meta.json").read_text())
    print(f"  t15_run_meta.json groups == build_groups('G1'): {meta15['groups'] == groups}; SLEEP_SCORE_GROUP {meta15['SLEEP_SCORE_GROUP']}")

    # ---- Q8 fit_eval
    print("\n[Q8] fit_eval")
    src = inspect.getsource(ga.fit_eval)
    print("  fit_eval keeps canonical order via subset_in_canonical_order: " + str("subset_in_canonical_order(features)" in src)
          + "; imputer refit: " + str("SimpleImputer" in src) + "; SMOTE refit: " + str("SMOTE(random_state=seed" in src)
          + "; RF from build_models: " + str('build_models()["rf"]' in src))
    yt, pf, prf = ga.fit_eval(X, y, F)
    tp = pd.read_csv(RES / "test_predictions_rf.csv")
    print(f"  full 19: y_pred == test_predictions_rf.csv {np.array_equal(pf, tp.y_pred.to_numpy())}; "
          f"max |proba diff| {np.abs(prf - tp[['p_rest', 'p_normal', 'p_active']].to_numpy()).max():.1e}; "
          f"F1 macro {ga.f1_macro(yt, pf):.6f}")
    for name, feats in [("oura_only", [f for f in F if not f.startswith("aw_")]), ("watch_only", [f for f in F if f.startswith("aw_")])]:
        _, p, _ = ga.fit_eval(X, y, feats)
        print(f"  {name} ({len(feats)}): F1 macro {ga.f1_macro(yt, p):.6f}, F1 Active {ga.f1_active(yt, p):.6f}")
    kept = [f for f in F if f not in set(groups["G5"])]
    _, p5, pr5 = ga.fit_eval(X, y, kept)
    g5 = pd.read_csv(RES / "t15_predictions_logo_G5_activity_energy.csv")
    print(f"  LOGO G5 ({len(kept)} kept): y_pred == t15_predictions_logo_G5 csv {np.array_equal(p5, g5.y_pred.to_numpy())}; "
          f"max |proba diff| {np.abs(pr5 - g5[['p_rest', 'p_normal', 'p_active']].to_numpy()).max():.1e}; F1 macro {ga.f1_macro(yt, p5):.6f}")
    print(f"  imputer medians in reduced space == full-space medians on the same columns (column-wise median): "
          f"{np.allclose(canon.imputer.statistics_[[F.index(f) for f in kept]], __import__('sklearn.impute', fromlist=['SimpleImputer']).SimpleImputer(strategy='median').fit(X.iloc[:766][kept]).statistics_)}")

    # ---- Q9 paired bootstrap
    print("\n[Q9] paired bootstrap of the delta")
    idx = np.load(RES / st.INDEX_FILE)
    print("  paired_bootstrap applies one row to y_true, pred_a, pred_b: "
          + str("yt = y_true[row]" in inspect.getsource(st.paired_bootstrap) and "pred_a[row]" in inspect.getsource(st.paired_bootstrap)
                and "pred_b[row]" in inspect.getsource(st.paired_bootstrap)))
    b = st.paired_bootstrap(yt, pf, p5, ga.f1_macro, idx)
    print(f"  G5: diff_point {b['diff_point']:+.6f} = F1(full) - F1(LOGO) = {ga.f1_macro(yt, pf) - ga.f1_macro(yt, p5):+.6f}; "
          f"CI [{b['ci_low']:.4f}, {b['ci_high']:.4f}]; n_used {b['n_used']}, n_skipped {b['n_skipped']}; "
          f"call in group_ablation.py:350 is paired_bootstrap(y_true, p_full, p, ...) -> A = full, B = LOGO -> positive = removal hurts")
    logo = pd.read_csv(RES / "t15_logo.csv").set_index("group")
    print(f"  t15_logo.csv G5: delta {logo.loc['G5_activity_energy', 'delta_f1_macro']:+.6f} "
          f"[{logo.loc['G5_activity_energy', 'delta_f1_macro_ci_low']:.4f}, {logo.loc['G5_activity_energy', 'delta_f1_macro_ci_high']:.4f}]")

    # ---- Q11 day-t frame
    print("\n[Q11] T3.2 frame")
    df = common.load_raw()
    frame, ident = la.build_frame(df)
    inv = pd.read_csv(RES / "t32_inventory.csv").set_index("variable")
    print(f"  raw daily frame rows {len(df)}; frame rows {len(frame)}; labelled dates == snapshot dates: "
          f"{list(pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')) == list(pd.read_csv(ROOT / 'models' / 'dataset_snapshot_20260331.csv').date)}")
    nn = {v: int(frame[v].isna().sum()) for v in la.VARIABLES}
    print(f"  NaN per variable in the frame (no imputation): {nn}; == inventory n_missing: "
          f"{all(nn[v] == int(inv.loc[v, 'n_missing']) for v in la.VARIABLES)}")
    daily = df.set_index("date")
    ok = 0
    for v in la.VARIABLES:
        a = np.nan_to_num(daily[v].shift(1).loc[frame.date].to_numpy(dtype=float), nan=-1e30)
        bb = np.nan_to_num(X[f"{v}_lag1"].to_numpy(dtype=float), nan=-1e30)
        ok += int(np.array_equal(a, bb))
    print(f"  independent identity: daily[v].shift(1) on the {len(df)}-row frame, taken at the 958 dates, == canonical lag-1 feature: {ok}/10")
    print(f"  script's ident flags all True: {bool(ident.all())}; NOTE lag_identity(df[v], df[v].shift(1)) (label_agreement.py:216-219) "
          f"compares a series with its own shift -> tautological; the effective test is the second clause (vs build_dataset X)")
    tests = pd.read_csv(RES / "t32_tests.csv").set_index("variable")
    print(f"  n per class in t32_tests == frame notna per class: "
          f"{all(int((frame.label == c)[frame[v].notna()].sum()) == int(tests.loc[v, k]) for v in la.VARIABLES for c, k in [(0, 'n_rest'), (1, 'n_normal'), (2, 'n_active')])}")

    # ---- Q12 directions
    print("\n[Q12] expected directions")
    print(f"  VARIABLES constant (label_agreement.py:87-108), expected per variable: "
          f"{ {v: m['expected'] for v, m in la.ALL_VARIABLES.items()} }")
    print(f"  t32_run_meta.json expected == code: "
          f"{ {v: d['expected'] for v, d in json.loads((RES / 't32_run_meta.json').read_text())['variables'].items()} == {v: m['expected'] for v, m in la.ALL_VARIABLES.items()} }")
    print("  unit test test_fixed_design asserts the same dict (test_label_agreement.py:113-120); git: single commit 8b09bd4 adds script, test and results together")

    # ---- Q13 / Q14 sources
    print("\n[Q13/Q14] sources")
    for fn in (la.cliffs_delta, la.cliffs_delta_ci, la.epsilon_squared):
        print("  " + inspect.getsource(fn).replace("\n", "\n  "))
    print("  kruskal call: " + [l.strip() for l in inspect.getsource(la.main).splitlines() if "stats.kruskal" in l][0])
    print("  Dunn tie term: " + [l.strip() for l in inspect.getsource(la.dunn_test).splitlines() if "tie =" in l][0])
    print("  rng: one default_rng(42) created at label_agreement.py:306, consumed sequentially over variables x pairs")

    # ---- Q15 provenance of level A (from P0 csv)
    print("\n[Q15] level-A variables by source (p0_source_mix.csv, 952 labelled dates with a raw export file)")
    mix = pd.read_csv(AUDIT / "p0_source_mix.csv")
    snap = pd.read_csv(ROOT / "models" / "dataset_snapshot_20260331.csv")
    mix = mix[mix.date.isin(set(snap.date))]
    for v, m in [("aw_hr_mean", "heart_rate"), ("aw_hrv_mean", "heart_rate_variability"), ("aw_active_calories", "active_energy"),
                 ("aw_steps", "step_count"), ("aw_resp_rate", "respiratory_rate"), ("aw_spo2", "blood_oxygen_saturation")]:
        s = mix[mix.metric == m]
        print(f"  {v:20s} <- {m:24s}: days {len(s)}, days with >=1 Oura record {int((s.n_oura > 0).sum())} ({(s.n_oura > 0).mean():.3f}), "
              f"days with NO non-Oura record {int((s.n_watch_only == 0).sum())}, Oura share of records {s.n_oura.sum() / s.n_total.sum():.3f}")
    print(f"[T2] elapsed {time.time() - t0:.1f} s")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "p0"
    {"p0": p0, "t2": t2}[cmd]()
