# AUDIT P1: what was checked during the dissertation about the Oura / Watch mix in Apple Health

Scope: report only. Sources: repository files (mtimes, since git history starts at the 2026-05-26 snapshot f7cc4ca), the two reports of 2026-03-30 in `outputs/reports/`, the dissertation docx files in `_referinte_upb/`, the KBS manuscripts in `revision/manuscript/`. No notebooks exist in the repository (`find -name "*.ipynb"`: none). Nothing outside `revision/audit/` was modified. Written 2026-10-25.

## 1. Chronology (what was known, when, from which evidence)

| When | Evidence | What it shows |
|---|---|---|
| 2023-06-09 | F4 P0.2 (`p0_source_mix.csv`) | Oura records are in Apple Health from the first study day; the mix is not a later regime. |
| 2026-03-25 21:33-21:36 | mtime `scripts_archive/old_versions/apple_health_parse.py`, `data/apple_health/*.csv` | XML parser; keeps `sourceName` per record (`:52`) and writes one csv per metric. |
| 2026-03-28 11:13 | mtime `cap5_pipeline.py` | XML 16-feature pipeline; `load_aw_metric()` groups by date with no source filter (`:45-49`). The "XML" column of Table 5.3 / Table 13 therefore also pools Oura, Watch and iPhone records. |
| 2026-03-28 16:22-16:43 | `apple_watch_api/api_sync.log`, `apple_health_auto.csv` | Trial of the Health Auto Export REST API (40 records, `source` column = "Health Auto Export"); abandoned, server script not in the repo. |
| 2026-03-28 17:22 | mtime `parse_health_auto_export.py` | JSON parser written; the string "source" does not occur in it (F4 P0.1). Only version that ever existed: one git commit (f7cc4ca), no copy in `backups/`, `git log -S source` hits only the snapshot. |
| 2026-03-28 20:00 | mtime `scripts_archive/comparison/compare_xml_vs_json.py` | First XML-vs-JSON comparison: XML aggregated over all sources (`:16-48`, no `sourceName` filter), 5 sample days plus r and MAE for HR mean, HRV, steps, SpO2 (`:97-115`). Output not saved; the "~9,106 steps MAE" figure quoted later most likely comes from this run (inference, no saved output). |
| 2026-03-28 20:12 | docstring `cap5_pipeline_json.py:7-12`, `:49` | Conclusion already written: "Steps si active_calories sunt deduplicate (JSON corect vs XML dublu-contor)". Framed as double counting, not as device mixing. |
| 2026-03-29 20:12 | mtime `apple_watch_api/apple_health_daily.csv` | Parser output used by the frozen pipeline (F4 P0.1 reproduces it). |
| 2026-03-30 21:07-21:08 | `scripts_archive/exploration/explore_apple_xml.py`, `outputs/reports/apple_xml_exploration_report.txt` | Per-source audit of the XML export. Oura share of records: heart_rate 61.3% (624,918 of 1,019,878), steps 71.1%, active_calories 36.8%, sleep 66.6%, respiratory_rate 1.9%; HRV, SpO2, basal 100% Watch. Per-source means printed (HR: Watch 93.94, Oura 90.33 bpm). Hardcoded conclusion (`:231-249`): "JSON Health Auto Export performs its own deduplication (source fields show merged sources)", "JSON sleep still mixes sources but doesn't double-count", recommendation JSON. |
| 2026-03-30 21:13-21:14 | `scripts_archive/comparison/compare_apple_xml_vs_json.py`, `outputs/reports/apple_xml_vs_json_comparison.txt` | Dual aggregation: XML all sources and XML Watch-only (`source_filter="Watch"`, `:86-87`, `:241`) against the JSON csv, 9 metrics; sleep also Oura-only (`:274`). Results on the Watch-only rows: HR mean XML-Watch 92.4 vs JSON 87.6 bpm, mean diff +4.74, r 0.79, verdict CHECK; steps XML-Watch 6,027 vs JSON 6,921 (-12.9%); active energy -2.9%; HRV and SpO2 identical. The script's own reading rule (`:340-342`): "XML AW-ONLY vs JSON: if they differ, JSON includes non-AW sources (Oura, iPhone)". The verdicts were not followed up. |
| 2026-03-31 | freeze | Canonical pipeline frozen on the pooled csv. |
| 2026-05-25 | `Disertatie_Lazar_Adrian_2026_Mai_CB.docx` [713], [1056]-[1058] | Dissertation text fixed: double counting attributed to "iPhone si Apple Watch raporteaza simultan", MAE 9.106, "ritm cardiac, HRV ... r >= 0,99", JSON "implementeaza deduplicare la nivel de sursa". Oura is never named as an Apple Health source. |
| 2026-05-26..29 | `AUDIT_NEW_2026-05-26.md:1043,1312,1338`, `audit/AUDIT_INDEPENDENT_2026-05-29.md:57,140,171` | The May audits mention HealthKit only for the Xiaomi scale path (Zepp Life -> HealthKit -> Health Auto Export) and call the XML csv files "raw HealthKit". No finding about Oura inside the Watch features. |
| 2026-06-02 | `scripts/canonical/xml_vs_json_accuracy.py` (a91f95f), `outputs/tables/steps_xml_vs_json_mae.csv` | Reproducible MAE: XML all sources 15,634 vs JSON 6,902 steps/day, MAE 8,734 on 1,069 days, r 0.96; still no source filter (`:80-96`). Replaces 9,106 in the text. |
| 2026-06-17 | `Lazar_Adrian_Lucrare_Disertatie_2026.pdf.docx` | Final dissertation (section 2 below). |
| 2026-06-21 | `PIPELINE_MAP.md:111,316` | "Sursa JSON deduplicata pentru Apple Watch"; explore_apple_xml "a confirmat contaminarea XML si superioritatea JSON". |
| 2026-10-25 | `AUDIT_F4_...md` P0, F4.1, F4.2, F4.4; `AUDIT_DECISIONS.md` #7 | First quantification of the mix inside the JSON records (Oura in 99.7% / 95.8% / 96.1% of HR / energy / step records on labelled days). |

## 2. What the dissertation states (final docx, paragraph index in brackets, chapter in angle brackets)

- [514] <3, Apple Watch>: "stocate local pe iPhone prin ... HealthKit"; export "XML ... sau automat prin ... Health Auto Export ... JSON".
- [749] <4, Arhitectura>: "S-a optat pentru sursa JSON ... problema de dublu-contorizare a pasilor in exportul XML".
- [851] <4, Integrarea>: "Apple Watch completeaza cu ... ritm cardiac continuu 24/7, pasi, calorii active".
- [854] <4, Sursa de date>: "pasii sunt dublu-contorizati din surse multiple (iPhone si Apple Watch raporteaza simultan)"; "ritm cardiac minim r = 0,99, HRV r = 0,98, ritm cardiac mediu r = 0,88" (these are the "XML all vs JSON" rows of the 2026-03-30 report, not the Watch-only rows).
- [921] <4, Colectarea>: the 8 `aw_` columns listed as Apple Watch features.
- [1006] <5, 6.4>: "exportul XML insumeaza pasi din surse multiple (iPhone si Apple Watch), in timp ce sursa JSON deduplica la nivel de sursa".
- [1237]-[1239] <7, Calitatea datelor>: "toate sursele din HealthKit (iPhone, Apple Watch, aplicatii terte), fara deduplicare"; [1238] "ritm cardiac, HRV ... r >= 0,99" (inconsistent with [854], where HR mean is 0.88).
- [1269] <8, Contributii>: "dublu-contorizare a pasilor in exportul XML Apple Health, cu MAE de 8,734 pasi/zi".
- No paragraph claims that the Watch features are free of Oura data; no paragraph names Oura as an Apple Health source. The independence claim appears only in the KBS manuscript: `KBS_Manuscript_R1.docx` [132] "Indicators from the watch ... are independent of the ring scores ... the only genuine validation"; [465] "watch indicators that do not enter the labeling rule and are independent of the ring scores"; Figure 1 caption [161] "JSON export with step deduplication"; 3.3 [111] "all sources (phone, watch, third-party apps)". Earlier drafts (Mai_CB, Sprint4_WIP, Sprint4_AIedited) carry the same claims with 9.106 and 66,5% / 61,8%.

## 3. What was actually checked (columns, interval, method)

- Columns: all nine Watch metrics, including heart rate, steps and active energy; not only the unaffected ones (HRV, SpO2). The T5 working hypothesis (check limited to HRV / SpO2 / Oura scores / sleep) is contradicted by both reports.
- Interval: 2023-04-11..2026-03-25 (XML export of 2026-03-28), 1,050-1,074 overlapping days per metric; the whole study window.
- Method, XML side: per-record `sourceName` counted and valued per source (explore), then aggregated with and without a `str.contains("Watch")` filter (compare). Method, JSON side: the daily csv only; the per-record `source` field of the JSON files was looked at ("source fields show merged sources") but never counted, and no JSON aggregate was ever built without the Oura-tagged buckets before F4 P0.4.
- Question asked: "does JSON double-count?" (steps as litmus test). Answer obtained: no. The second question that the same report could answer, "does JSON contain Oura samples?", had its indicator rows computed (HR mean r 0.79 CHECK, steps -12.9%) and the reading rule printed, but the conclusion written the same evening and kept in every later text is the first answer only. The merged source strings were taken as evidence of deduplication rather than of pooling.
- The XML "16f" pipeline used for Table 5.3 / Table 13 pools all sources too (`cap5_pipeline.py:45-49`), so the XML-vs-JSON accuracy comparison is pooled-vs-pooled on heart rate and energy, and differs mainly in steps.

## 4. Conclusion

Variant chosen: (a), with X made precise. The pooling was known at data-collection time and was quantified, but in the XML export and under the heading of double counting; the Watch-only-versus-JSON differences for heart rate, steps and energy were computed and left uninterpreted; the Oura content of the JSON records that feed the pipeline was not counted until the revision. Variant (b) is wrong because a quantification exists; the plain wording of (a) is wrong because the check did cover heart rate, steps and energy.

Paragraph for Discussion / Limitations (English):

"The presence of ring samples in the Apple Health record was known at data-collection time: a per-source audit of the native XML export (March 2026) showed that 61% of heart-rate, 71% of step and 37% of active-energy records carried the ring as source, and the comparison of that export with the JSON daily aggregates was run both over all sources and over watch-only records. That check was framed as a double-counting question and answered it (steps are not double-counted in the JSON export); it did not quantify how much of the JSON heart-rate, step and energy aggregates originates from the ring, and the watch-only comparison, which differed from the JSON values for heart rate and steps, was not followed up. The revision quantified the mix on the records actually used (ring samples in 96% to 100% of the daily records for these three features, on all labelled days) and reclassifies these features as Apple Health aggregates of both devices rather than watch-only measurements."

Open items: the saved output of the 2026-03-28 comparison (origin of 9,106) does not exist; the KBS R1 independence sentences ([132], [465]) and Figure 1 caption are the texts that need the repositioning already listed under `AUDIT_DECISIONS.md` #7 (open, question 9 to the co-author).
