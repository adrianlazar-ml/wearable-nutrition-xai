# PIPELINE_MAP.md - Harta pipeline-ului ML disertatie UPB

Nota: document generat prin analiza statica read-only pe 2026-06-21. NU contine cifre de rezultat (F1, MCC, QWK, accuracy, CV, etc.). Toate cifrele de performanta traiesc in CIFRE_CANONICE.md, sursa unica de adevar. Aici se descriu doar tipurile de artefacte (tabel, figura, model frozen), deciziile metodologice si fluxul de date.

---

## 1. FLUXUL PRINCIPAL (data flow)

Ordinea reala de executie de la ingestie (Oura + Apple Watch) pana la rezultatul final (UI + recomandare). Se folosesc DOAR versiunile canonice (production-path). Iteratiile intermediare XML 16f si JSON 16f sunt excluse din acest flux (vezi sectiunea 2 pentru lineage).

```
  [Oura Ring API v2]                         [Apple Health / iCloud Auto Export]
         |                                              |
   oura_auth.py  (OAuth2 -> .env token)                 |
         |                                              |
   oura_collect.py                            parse_health_auto_export.py
   data/sleep.csv, readiness.csv,             apple_watch_api/
   activity.csv, stress.csv,                  apple_health_daily.csv
   sleep_detailed.csv, sleep_metrics.csv               |
         |                                              |
   oura_merge.py                                        |
   (+ data/oura_historical.csv)                         |
         |                                              |
   data/oura_combined.csv ------------+-----------------+
                                      |
                                      v
                    cap5_pipeline_json_extended.py  (CANONIC, 19 features)
                    importa t3_labels.assign_nutrition_label
                    -> data/dataset_combinat_json_extended.csv
                                      |
              +-----------------------+------------------------+
              |                                                |
              v                                                v
   freeze_production_model.py                   cap5_compare_algorithms_json_ext.py
   (importa FEATURES/LAG_FEATURES/             (CANONIC RQ2, 19 features)
    assign_nutrition_label din                 -> figuri SHAP/PDP/CM
    cap5_pipeline_json_extended)               -> outputs/tables/mcc_qwk_results.csv
   -> models/rf_frozen_20260331.joblib
   -> models/imputer_frozen_20260331.joblib    generate_waterfall_combined_ext.py
   -> models/frozen_meta_20260331.json         (CANONIC waterfall XGB 19f)
   -> models/dataset_snapshot_20260331.csv     -> figura waterfall per clasa
              |
   patch_frozen_meta.py  (adauga f1/precision/recall per clasa + CM in frozen_meta)
              |
   sanity_check_frozen.py  (verifica reproducibilitate, read-only)
   bootstrap_ci_frozen.py  (CI 95% bootstrap pe test intern)
              |
              v
   generate_canonical_numbers.py
   citeste outputs/tables/*.csv + frozen_meta -> CIFRE_CANONICE.md
              |
              v
   [layer aplicatie]
   demo_end_to_end_v2.py  -> models/rf_model.joblib + models/imputer.joblib
   nutrition_recommender.py  (modul recomandare, importat)
   streamlit_app_v2.py  (UI: clasificare ML + recomandare + figuri SHAP)
   importa t3_labels.classify_day, nutrition_recommender.get_nutrition_recommendation
```

Ordine de executie (producer -> file -> consumer):

1. `oura_auth.py` -> `.env` (OURA_ACCESS_TOKEN) -> consumat de `oura_collect.py`.
2. `oura_collect.py` -> `data/sleep.csv`, `data/readiness.csv`, `data/activity.csv`, `data/stress.csv`, `data/sleep_detailed.csv`, `data/sleep_metrics.csv` -> consumate de `oura_merge.py`.
3. `oura_merge.py` (+ `data/oura_historical.csv`) -> `data/oura_combined.csv` -> consumat de aproape toate scripturile de clasificare, explicabilitate si experimente.
4. `parse_health_auto_export.py` -> `apple_watch_api/apple_health_daily.csv` -> consumat de toate scripturile JSON (16f si 19f).
5. `cap5_pipeline_json_extended.py` (CANONIC, importa `t3_labels`) -> `data/dataset_combinat_json_extended.csv` -> consumat de `streamlit_app_v2.py` (prioritar) si de `scripts/canonical/xml_vs_json_accuracy.py`.
6. `freeze_production_model.py` (importeaza simboluri din `cap5_pipeline_json_extended.py`) -> `models/rf_frozen_20260331.joblib`, `models/imputer_frozen_20260331.joblib`, `models/frozen_meta_20260331.json`, `models/dataset_snapshot_20260331.csv`.
7. `patch_frozen_meta.py` -> imbogateste `models/frozen_meta_20260331.json` (aditiv, idempotent).
8. `sanity_check_frozen.py` si `bootstrap_ci_frozen.py` -> verifica/cuantifica modelul frozen (CI in `outputs/bootstrap_ci/`).
9. `cap5_compare_algorithms_json_ext.py` (CANONIC RQ2) -> figuri SHAP/PDP/CM in `outputs/figures/json_extended_pipeline/` + `outputs/tables/mcc_qwk_results.csv`.
10. `generate_waterfall_combined_ext.py` (CANONIC waterfall) -> `outputs/figures/json_extended_pipeline/ext_shap_waterfall_combined_clean.png`.
11. `generate_canonical_numbers.py` -> citeste toate `outputs/tables/*.csv` + frozen_meta -> `CIFRE_CANONICE.md`.
12. `demo_end_to_end_v2.py` -> `models/rf_model.joblib`, `models/imputer.joblib` -> consumate de `streamlit_app_v2.py`.
13. `nutrition_recommender.py` (modul) -> importat de `demo_end_to_end_v2.py` si `streamlit_app_v2.py`.
14. `streamlit_app_v2.py` -> interfata web finala (consuma oura_combined.csv, apple_health_daily.csv, modele, dataset_combinat_json_extended.csv, tabele si figuri).

Dangling edges (de semnalat la aparare):

1. `data/oura_historical.csv` este input pentru `oura_merge.py` dar nu este produs de niciun script din pipeline (export manual istoric din aplicatia Oura, iunie 2023 - iulie 2025).
2. `data/apple_health/*.csv` (heart_rate, hrv, steps, active_calories, spo2) sunt inputuri pentru versiunile XML (`cap5_pipeline.py`, `cap5_compare_algorithms.py`, `confidence_threshold.py`) dar in pipeline-ul canonic ar fi produse de `scripts_archive/old_versions/apple_health_parse.py`, care este ARHIVAT. In fluxul canonic JSON aceste fisiere nu mai sunt folosite.
3. `outputs/tables/valori_disertatie.csv` este consumat de `streamlit_app_v2.py` si `generate_canonical_numbers.py`, produs de `extract_all_results.py` (utilitar, nu in flux principal).
4. `models/rf_model.joblib` si `models/imputer.joblib` (produse de `demo_end_to_end_v2.py`) sunt distincte de modelul frozen (`rf_frozen_20260331.joblib`); UI-ul foloseste `rf_model.joblib`, NU frozen. Frozen este folosit doar pentru sanity, bootstrap CI si validare prospectiva.

---

## 2. PER SCRIPT

Blocuri compacte per script ne-arhivat, grupate pe stadiu. Fiecare marcheaza `pipeline_role`.

### Stadiu: Ingestie / Integrare

**`oura_auth.py`** (canonical-production)
Implementeaza OAuth2 pentru Oura Ring API v2: deschide browserul, prinde callback pe localhost, salveaza access token in `.env`. Script one-time de autorizare, ruleaza inainte de colectare.
- Input: `.env` (CLIENT_ID, CLIENT_SECRET, REDIRECT_URI).
- Output: `.env` (ACCESS_TOKEN).
- Depends-on: nimic (intrare pipeline). Depended-on-by: `oura_collect.py`.

**`oura_collect.py`** (canonical-production)
Colecteaza date brute din Oura API v2 pe interval de date (implicit 2025-07-01 pana azi) din patru daily endpoints (sleep, readiness, activity, stress) plus metrici detaliate de somn agregate pe zi.
- Input: `.env` (ACCESS_TOKEN).
- Output: `data/sleep.csv`, `data/readiness.csv`, `data/activity.csv`, `data/stress.csv`, `data/sleep_detailed.csv`, `data/sleep_metrics.csv`.
- Depends-on: `oura_auth.py`. Depended-on-by: `oura_merge.py`.

**`oura_merge.py`** (canonical-production)
Combina exportul istoric Oura cu datele noi din API v2 fara duplicate, adauga device flags (Gen3/Gen4, Ultra/Ultra3) si aplica data quality fix pentru rhr_avg fiziologic imposibil. Pas critic de curatare.
- Input: `data/oura_historical.csv`, `data/readiness.csv`, `data/sleep.csv`, `data/activity.csv`, `data/sleep_metrics.csv`.
- Output: `data/oura_combined.csv`.
- Depends-on: `oura_collect.py` (+ export manual istoric). Depended-on-by: aproape toate scripturile downstream.

**`parse_health_auto_export.py`** (canonical-production)
Parseaza exporturile JSON din Apple Health (iCloud Auto Export), extrage metrici zilnice si le agrega pe zi (sum pentru activitate/energie/pasi, mean pentru cardiac/respirator). Sursa JSON deduplicata pentru Apple Watch.
- Input: `HealthAutoExport-*.json` (iCloud).
- Output: `apple_watch_api/apple_health_daily.csv`.
- Depends-on: export manual iCloud. Depended-on-by: toate scripturile JSON.

### Stadiu: Clasificare (pipeline ML)

**`t3_labels.py`** (canonical-production, LIBRARY MODULE)
Modul canonic pentru regula T3 (Rest/Normal/Active): centralizeaza pragurile si maparea nume/culori etichete. Sursa unica de adevar, elimina duplicarea logicii inline (FND-210). Nu este script executabil.
- Input: niciunul. Output: niciun artefact (modul importat).
- Depended-on-by: `cap5_pipeline_json_extended.py`, `experiments_new.py`, `streamlit_app_v2.py`, `prospective_validation.py`.

**`cap5_pipeline.py`** (intermediate-iteration)
Pipeline ML de clasificare nutritie cu ingestie XML a Apple Watch, 16 features lag-1, split temporal 80/20, SMOTE, Random Forest. Versiunea initiala a familiei pipeline. Logica T3 inline.
- Input: `data/oura_combined.csv`, `data/apple_health/*.csv` (XML).
- Output: `data/dataset_combinat.csv`.
- Lineage: XML 16f, prima veriga; inlocuit de `cap5_pipeline_json.py` (sursa JSON, logica identica).

**`cap5_pipeline_json.py`** (intermediate-iteration)
Identic cu `cap5_pipeline.py` dar cu ingestie JSON (Health Auto Export) in loc de XML, pentru a cuantifica impactul dublu-contorizarii pasilor/caloriilor din XML. 16 features lag-1.
- Input: `data/oura_combined.csv`, `apple_watch_api/apple_health_daily.csv`.
- Output: `data/dataset_combinat_json.csv`.
- Lineage: evoluat din `cap5_pipeline.py` (XML 16f); inlocuit de `cap5_pipeline_json_extended.py` (19f).

**`cap5_pipeline_json_extended.py`** (canonical-production)
Pipeline ML CANONIC al tezei: clasificare Rest/Normal/Active din 19 features lag-1 (16 originale + 3 noi sleep stages si resp rate), split temporal 80/20, TimeSeriesSplit CV, imputer + SMOTE pe train, Random Forest. Importa `t3_labels`. Alimenteaza modelul frozen.
- Input: `data/oura_combined.csv`, `apple_watch_api/apple_health_daily.csv`.
- Output: `data/dataset_combinat_json_extended.csv`.
- Lineage: extensie a JSON 16f cu 3 features noi (aw_sleep_deep_hours, aw_sleep_rem_hours, aw_resp_rate). VERSIUNEA CANONICA FINALA a familiei cap5_pipeline.
- Depended-on-by: `freeze_production_model.py`, `prospective_validation.py`, `scripts/canonical/compare_16f_vs_19f.py` (prin import de simboluri), `streamlit_app_v2.py` (consuma dataset).

### Stadiu: Comparatie algoritmi / RQ2

**`cap5_compare_algorithms.py`** (intermediate-iteration)
Compara RF/XGBoost/SVM pe date XML 16f cu split temporal si TimeSeriesSplit, genereaza SHAP (summary/bar/dependence/waterfall) si PDP. Versiunea XML initiala, fara MCC/QWK.
- Input: `data/oura_combined.csv`, `data/apple_health/*.csv`.
- Output: figuri in `outputs/figures/xml_pipeline/`.
- Lineage: XML 16f; inlocuit de `cap5_compare_algorithms_json.py` apoi `cap5_compare_algorithms_json_ext.py`.

**`cap5_compare_algorithms_json.py`** (intermediate-iteration)
Identic cu versiunea XML dar cu Apple Watch din JSON, pe 16 features. SHAP + PDP, fara MCC/QWK. Studiu de viabilitate JSON inainte de extensie.
- Input: `data/oura_combined.csv`, `apple_watch_api/apple_health_daily.csv`.
- Output: figuri in `outputs/figures/json_pipeline/` (prefix `json_`).
- Lineage: JSON 16f; inlocuit de `cap5_compare_algorithms_json_ext.py` (19f).

**`cap5_compare_algorithms_json_ext.py`** (canonical-production)
Versiunea CANONICA a RQ2: compara RF/XGBoost/SVM pe 19 features, split temporal 80/20 + TimeSeriesSplit CV, SHAP (summary/bar/dependence/waterfall), PDP si metrici MCC + QWK pentru Cap.6. Versiunea care defineste pipeline-ul frozen.
- Input: `data/oura_combined.csv`, `apple_watch_api/apple_health_daily.csv`.
- Output: figuri in `outputs/figures/json_extended_pipeline/` (prefix `ext_`), `outputs/tables/mcc_qwk_results.csv`.
- Lineage: JSON 19f, VERSIUNEA CANONICA FINALA a familiei cap5_compare_algorithms.
- Depended-on-by: `ablation_study.py` (referinta pipeline), `generate_canonical_numbers.py` (citeste mcc_qwk_results.csv).

### Stadiu: Explicabilitate (waterfall)

**`generate_waterfall_combined.py`** (intermediate-iteration)
Genereaza waterfall plots SHAP pe XGBoost cu 16 features (XML), TreeExplainer, 3 subplots verticale (Rest/Normal/Active) cu matplotlib pur.
- Input: `data/oura_combined.csv`, `data/apple_health/*.csv`.
- Output: `outputs/figures/xml_pipeline/shap_waterfall_all_xgboost.png`.
- Lineage: XML 16f; inlocuit de `generate_waterfall_combined_json.py` apoi `_ext`.

**`generate_waterfall_combined_json.py`** (intermediate-iteration)
Waterfall SHAP pe XGBoost cu 16 features dar sursa JSON. Script de tranzitie intre XML 16f si JSON 19f.
- Input: `data/oura_combined.csv`, `apple_watch_api/apple_health_daily.csv`.
- Output: `outputs/figures/json_pipeline/json_shap_waterfall_combined_clean.png`.
- Lineage: JSON 16f; inlocuit de `generate_waterfall_combined_ext.py`.

**`generate_waterfall_combined_ext.py`** (canonical-production)
Waterfall SHAP pe XGBoost cu 19 features (JSON + sleep stages), TreeExplainer, 3 subplots. Versiunea extinsa finala conectata la pipeline-ul frozen.
- Input: `data/oura_combined.csv`, `apple_watch_api/apple_health_daily.csv`.
- Output: `outputs/figures/json_extended_pipeline/ext_shap_waterfall_combined_clean.png`.
- Lineage: JSON 19f, versiunea CANONICA de waterfall.

**`generate_waterfall_svm.py`** (experiment)
Waterfall plots + SHAP bar pe SVM (RBF) cu 19 features folosind KernelExplainer (model-agnostic, lent). Analiza comparativa standalone, nu in pipeline-ul canonic.
- Input: `data/oura_combined.csv`, `apple_watch_api/apple_health_daily.csv`.
- Output: `outputs/figures/json_extended_pipeline/ext_shap_waterfall_svm_clean.png`, `ext_shap_bar_svm.png`.

### Stadiu: Utilitare (helper, nu pas in flux)

**`fix_svm_barplot.py`** (utility)
Regenereaza rapid doar SHAP bar plot pentru SVM (varianta condensata a `generate_waterfall_svm.py`). Helper single-purpose.
- Input: `data/oura_combined.csv`, `apple_watch_api/apple_health_daily.csv`.
- Output: `outputs/figures/json_extended_pipeline/ext_shap_bar_svm.png`.

**`patch_frozen_meta.py`** (utility)
Patch aditiv idempotent (D10): calculeaza F1/precision/recall per clasa + confusion matrix pe test intern si le injecteaza in `frozen_meta_20260331.json` fara a atinge modelul .joblib.
- Input: artefacte frozen + `dataset_snapshot_20260331.csv`.
- Output: `models/frozen_meta_20260331.json` (aditiv).

**`extract_all_results.py`** (utility)
Reconstruieste pipeline-ul principal in-situ si compileaza toate valorile numerice + tabelele de experiment intr-un raport unic de referinta.
- Input: `data/oura_combined.csv`, `apple_watch_api/apple_health_daily.csv`, `outputs/tables/*.csv`.
- Output: `outputs/reports/valori_disertatie.txt`, `outputs/tables/valori_disertatie.csv`.

**`audit_csv.py`** (utility)
Audit rapid read-only al CSV-urilor de date: shape, coloane, dtype, NA, date range, SHA256 hash.
- Input: `data/oura_combined.csv`, `apple_watch_api/apple_health_daily.csv`, `models/dataset_snapshot_20260331.csv`. Output: stdout.

**`sanity_check_frozen.py`** (utility)
Verifica reproducibilitatea modelului frozen fara re-antrenare: reincarca artefacte, reconstruieste test set 80/20, recalculeaza metrici si le compara cu canonicul din frozen_meta. Read-only.
- Input: artefacte frozen + `dataset_snapshot_20260331.csv`. Output: raport consola.

### Stadiu: Freeze / Reproducibilitate

**`freeze_production_model.py`** (canonical-production)
Antreneaza RandomForest pe date pana la cutoff (2026-03-31), salveaza modelul frozen + imputer + metadate SHA256 + snapshot dataset. Ruleaza o singura data; modelul frozen NU se mai modifica.
- Input: importa simboluri din `cap5_pipeline_json_extended.py` (FEATURES, LAG_FEATURES, assign_nutrition_label, load_apple_watch, load_oura, merge_datasets).
- Output: `models/rf_frozen_20260331.joblib`, `models/imputer_frozen_20260331.joblib`, `models/frozen_meta_20260331.json`, `models/dataset_snapshot_20260331.csv`.
- Depended-on-by: `sanity_check_frozen.py`, `patch_frozen_meta.py`, `bootstrap_ci_frozen.py`, `prospective_validation.py`.

**`bootstrap_ci_frozen.py`** (canonical-production)
Calculeaza intervale de incredere bootstrap 95% pentru metricile modelului frozen pe test intern, prin resampling pe perechile (y_true, y_pred) fara re-antrenare. Read-only.
- Input: artefacte frozen + `dataset_snapshot_20260331.csv`.
- Output: `outputs/bootstrap_ci/bootstrap_ci_frozen_test.csv`.

### Stadiu: Generare cifre canonice

**`generate_canonical_numbers.py`** (canonical-production)
Citeste toate CSV-urile din `outputs/tables/` + frozen_meta si construieste `CIFRE_CANONICE.md` (16 sectiuni: dataset, clase, modele, ablation, threshold, SMOTE, confidence, tuning, modele noi, MCC/QWK, calibrare, rolling, circularity, config, reproducibilitate, deltas 16f vs 19f).
- Input: toate `outputs/tables/*.csv` + `models/frozen_meta_20260331.json`.
- Output: `CIFRE_CANONICE.md` (sursa de adevar pentru cifre).

### Stadiu: Recomandare / Interfata

**`nutrition_recommender.py`** (canonical-production)
Modul de recomandari nutritionale personalizate bazat pe clasificarea ML si profilul biometric. Calculeaza distributie calorica si macronutrienti conform ghidurilor evidence-based. Importabil.
- Input: niciun fisier (profil hardcoded). Output: dictionar recomandare (in memorie).
- Depended-on-by: `streamlit_app_v2.py`, `demo_end_to_end_v2.py`.

**`demo_end_to_end_v2.py`** (canonical-production)
Demonstratie end-to-end V2: incarca date, pregateste 19 features lag-1, antreneaza RF cu split temporal 80/20 + SMOTE, prezice ziua cea mai recenta nevazuta, aplica confidence threshold si genereaza recomandare. Consistent cu pipeline-ul canonic.
- Input: `data/oura_combined.csv`, `apple_watch_api/apple_health_daily.csv`.
- Output: `models/rf_model.joblib`, `models/imputer.joblib`.
- Depends-on: `nutrition_recommender.py`. Depended-on-by: `streamlit_app_v2.py` (consuma modelele).

**`streamlit_app_v2.py`** (canonical-production)
Interfata Streamlit V2 cu trei pagini: profil + BMR/TDEE, dashboard zilnic cu clasificare ML + recomandare, explicabilitate cu figuri SHAP. Prioritizeaza dataset JSON Extended 19f.
- Input: `data/oura_combined.csv`, `apple_watch_api/apple_health_daily.csv`, `models/rf_model.joblib`, `models/imputer.joblib`, `data/dataset_combinat_json_extended.csv`, tabele si figuri din `outputs/`.
- Output: interfata web (session state).
- Depends-on: `nutrition_recommender.py`, `t3_labels.py`.

### In afara pipeline-ului inghetat

Aceste doua scripturi NU fac parte din pipeline-ul frozen al tezei. Sunt separate intentionat.

**`prospective_validation.py`** (not-thesis-pipeline)
Evaluare externa temporala a modelului frozen pe o fereastra de date colectate strict dupa cutoff-ul antrenarii. Motiv: validare prospectiva care ruleaza DUPA aparare (nota explicita la linia 2), nu face parte din pipeline-ul frozen al tezei. Construieste tabel comparativ si figuri pe date prospective cu zero leakage.
- Input: artefacte frozen + `data/oura_combined.csv` + `apple_watch_api/apple_health_daily.csv`; importa simboluri din `cap5_pipeline_json_extended.py`.
- Output: `outputs/tables/prospective_validation_*.csv`, figuri si raport text.

**`eswa_architecture.py`** (not-thesis-pipeline)
Generator bilingv (EN/RO) de figuri arhitecturale pentru manuscrisul ESWA (jurnal Elsevier), nu pentru teza. Motiv: produce diagrama de flux pentru submission de jurnal (nota explicita liniile 1-18, inclusiv disclosure GenAI per politica Elsevier). Output vectorial PDF + raster PNG.
- Input: niciunul. Output: 4 figuri (PDF + PNG, EN + RO) in `outputs/figures/eswa_architecture/`.

---

## 3. PIPELINE PRINCIPAL vs EXPERIMENT/ANALIZA

Scripturi de productie (production-pipeline): `oura_auth.py`, `oura_collect.py`, `oura_merge.py`, `parse_health_auto_export.py`, `t3_labels.py`, `cap5_pipeline_json_extended.py`, `cap5_compare_algorithms_json_ext.py`, `generate_waterfall_combined_ext.py`, `freeze_production_model.py`, `bootstrap_ci_frozen.py`, `generate_canonical_numbers.py`, `nutrition_recommender.py`, `demo_end_to_end_v2.py`, `streamlit_app_v2.py`. Plus utilitarele `patch_frozen_meta.py`, `sanity_check_frozen.py`, `extract_all_results.py`, `audit_csv.py`, `fix_svm_barplot.py`.

Scripturi de experiment / analiza: tabel mai jos, cu intrebarea metodologica si tipul de rezultat (fara cifre).

| Script | Intrebare metodologica | Tip rezultat/figura |
|---|---|---|
| `experiments_new.py` | Sunt hyperparametrii default OK? Sunt modele alternative (LR, LightGBM) superioare? Satureaza invatarea cu mai multe date? Sunt predictiile calibrate? Ajuta rolling features? | Tabel tuning default vs optimized, tabel modele noi, figura learning curves, diagrame calibrare/reliability, tabel Brier, tabel rolling 19f vs 34f (cifre in CIFRE_CANONICE.md) |
| `ablation_study.py` | Cat contribuie fiecare dispozitiv (Oura singur vs Apple Watch singur vs ambele) la performanta? | Tabel ablation pe 3 configuratii (11/8/19 features), bar charts F1, confusion matrices side-by-side (cifre in CIFRE_CANONICE.md) |
| `smote_comparison.py` | Care strategie de imbalance e cea mai buna (no resampling, class_weight balanced, RandomUnderSampler, SMOTE k=3, SMOTE k=5)? | Tabel comparativ 5 strategii + bar charts dual-panel (cifre in CIFRE_CANONICE.md) |
| `threshold_sensitivity.py` | Cat de robuste sunt rezultatele la variatia pragurilor T3 de clasificare (T1-T6 manuale + T7 quantile)? | Tabel sensitivity per set praguri, heatmap, distributii clase, line plot robustete (cifre in CIFRE_CANONICE.md) |
| `confidence_threshold.py` | Ce prag de confidenta minimizeaza eroarea calorica in worst-case (fallback Normal sub prag)? Praguri 0.50-0.70. NOTA: foloseste set redus 15 features. | Tabel analiza confidence (F1/coverage/eroare calorica), figura trade-off F1 vs coverage (cifre in CIFRE_CANONICE.md) |
| `circularity_sanity_check.py` | Cat din performanta vine din variabile circulare (lag-1 ai scorurilor de label) vs informatie fiziologica reala? Trei variante 19f/17f/14f. | Tabel CSV pe 3 variante feature (cifre in CIFRE_CANONICE.md) |
| `circularity_plot.py` | Vizualizare a rezultatului analizei de circularitate (citeste CSV-ul generat). | Doua figuri: F1 pe 3 variante feature, autocorelatie lag-1 a variabilelor de label |
| `learning_curves_comparison.py` | De ce esueaza SMOTE k=3 pe learning curves cu subseturi mici (sub k+1 instante Active) si de ce class_weight balanced functioneaza? | Doua figuri learning curves (SMOTE esuat vs balanced) + log de erori ca dovada |
| `bootstrap_ci_frozen.py` | Cat de incerta e fiecare metrica a modelului frozen (clasa Active are putine esantioane in test)? | CSV cu CI bootstrap 95% pe test intern + raport consola (cifre in CIFRE_CANONICE.md) |
| `scripts/canonical/compare_16f_vs_19f.py` | Cat aduc cele 3 features noi (sleep stages + resp rate) la RF/XGBoost/SVM (16f vs 19f)? | CSV cu deltas per model + bloc metadata; sanity-assert vs CIFRE_CANONICE.md sec.3 |
| `scripts/canonical/xml_vs_json_accuracy.py` | Cat difera acuratetea RF intre sursa XML si JSON la 16 features? Cat de mare e eroarea de pasi XML vs JSON? | CSV acuratete per sursa + CSV steps MAE XML vs JSON (cifre in CIFRE_CANONICE.md) |
| `generate_waterfall_svm.py` | Cum arata explicabilitatea SVM (RBF) cu SHAP model-agnostic, comparativ cu XGBoost? | Figura waterfall per clasa + SHAP bar plot pentru SVM (19f) |

---

## 4. SCRIPTURI ARHIVATE

Toate fisierele din `scripts_archive/`, marcate ARHIVAT, cu motivul scurt si ce le-a inlocuit.

### old_versions

1. `scripts_archive/old_versions/apple_health_parse.py` ARHIVAT. Parser XML brut Apple Health (9 metrici), pre-pipeline. Motiv vechi: ingestie XML cu dublu-contorizare pasi/calorii. Inlocuit de: pipeline-ul JSON (`parse_health_auto_export.py` + variantele cap5 JSON care folosesc apple_health_daily.csv).
2. `scripts_archive/old_versions/streamlit_app.py` ARHIVAT. UI Streamlit V1 cu 16 features XML si figuri din `data/`. Motiv vechi: bazat pe sursa XML legacy, figuri mutate ulterior in `outputs/`. Inlocuit de: `streamlit_app_v2.py` (19f JSON Extended).
3. `scripts_archive/old_versions/demo_end_to_end.py` ARHIVAT (marcat intermediate-iteration in metadata). Demo end-to-end V2/V3 cu 16 features XML. Motiv vechi: 16 features XML, feature engineering mai limitat. Inlocuit de: `cap5_pipeline.py` si variantele JSON/extended, respectiv `demo_end_to_end_v2.py` pentru demo.
4. `scripts_archive/old_versions/experiments_new_balanced.py` ARHIVAT. Variantа de development a celor 5 experimente (tuning, modele noi, learning curves, calibrare, rolling), majoritatea comentate in main, doar learning curves activ. Motiv vechi: snapshot incomplet de dezvoltare. Inlocuit de: `experiments_new.py` (versiunea consolidata in radacina).
5. `scripts_archive/old_versions/compute_mcc_qwk.py` ARHIVAT. Utilitar care replica pipeline-ul ext pentru a calcula MCC + QWK pentru Cap.6. Motiv vechi: functionalitate absorbita in pipeline-ul canonic. Inlocuit de: `cap5_compare_algorithms_json_ext.py` (calculeaza MCC/QWK nativ si scrie `mcc_qwk_results.csv`); mutare in arhiva ratificata (FND-401, D3).

### comparison

6. `scripts_archive/comparison/compare_xml_vs_json.py` ARHIVAT. Compara Apple Watch XML vs JSON pe 7 metrici (concordanta, MAE, % diferenta), print-only. Motiv vechi: analiza pre-decizie pentru alegerea sursei. Inlocuit de: nimic (decizie luata, nu mai ruleaza); auditul reproductibil traieste in `scripts/canonical/xml_vs_json_accuracy.py`.
7. `scripts_archive/comparison/compare_oura_sources.py` ARHIVAT. Compara Oura legacy CSV vs membership export pe ~30 metrici, salveaza raport. Motiv vechi: evaluare de sursa, nu versiune de pipeline. Inlocuit de: nimic (decizie luata).
8. `scripts_archive/comparison/compare_api_vs_export.py` ARHIVAT. Compara Oura API v2 (oura_combined) vs membership export pe 12 metrici. Motiv vechi: evaluare de sursa pre-decizie. Inlocuit de: nimic (decizie luata).
9. `scripts_archive/comparison/compare_membership_vs_api.py` ARHIVAT. Cel mai complet comparison Oura (40+ metrici, 6 perechi de fisiere). Motiv vechi: evaluare finala de sursa Oura. Inlocuit de: nimic (decizie luata).
10. `scripts_archive/comparison/compare_apple_xml_vs_json.py` ARHIVAT. Compara Apple Watch XML vs JSON pe 9 metrici cu dual aggregation (all vs Watch-only) pentru double-counting. Motiv vechi: investigatie de sursa Apple Watch. Inlocuit de: nimic (decizie luata, JSON ales).

### exploration

11. `scripts_archive/exploration/extract_anexa2.py` ARHIVAT. Exemplificare manuala a regulii T3 cu 2 randuri per clasa, pentru verificare etichete. Motiv vechi: script de documentare/verificare manuala. Inlocuit de: nimic (logica T3 centralizata in `t3_labels.py`).
12. `scripts_archive/exploration/extract_stats.py` ARHIVAT. Statistici descriptive pe 19 features merged, exploratoriu. Motiv vechi: verificare calitate date pre-pipeline. Inlocuit de: nimic (utilitarul `audit_csv.py` acopera audit-ul de date).
13. `scripts_archive/exploration/explore_oura_export.py` ARHIVAT. Scanare export Oura .zip (date ranges, missing, recomandari integrare). Motiv vechi: delimitator fix virgula, nu detecta punct-virgula. Inlocuit de: `scripts_archive/exploration/explore_oura_export_v2.py`.
14. `scripts_archive/exploration/explore_oura_export_v2.py` ARHIVAT. Versiunea imbunatatita cu auto-detectie delimitator. Motiv vechi: exploratoriu pre-pipeline, decizie de integrare deja luata. Inlocuit de: nimic (rol exploratoriu incheiat).
15. `scripts_archive/exploration/explore_apple_xml.py` ARHIVAT. Audit calitate Apple Health XML (detectie contaminare Oura/iPhone/Zepp, comparatie cu JSON). Motiv vechi: a confirmat contaminarea XML si superioritatea JSON. Inlocuit de: nimic (a fundamentat decizia de a folosi JSON).

---

## 5. PUNCTE DE ATENTIE PENTRU APARARE

Deciziile metodologice gasite in cod si locatia exacta. Agregat peste toate scripturile. Fara cifre de rezultat.

### Split temporal cronologic 80/20 (anti-leakage temporal)

Niciodata random shuffle. Train pe primele 80% zile cronologice, test pe ultimele 20%.
- `cap5_pipeline.py:188-200` (train_and_evaluate), `cap5_pipeline_json.py:188-200`, `cap5_pipeline_json_extended.py:211-224`.
- `cap5_compare_algorithms.py:137-141` (X.iloc[:split_idx], fara shuffle), `cap5_compare_algorithms_json.py:142-146`, `cap5_compare_algorithms_json_ext.py:146-150`.
- `freeze_production_model.py:112-116` (split_idx = int(len(df_model) * 0.8)).
- `demo_end_to_end_v2.py:51` (TRAIN_RATIO=0.8 definit), `:110-113` (split aplicat).
- `ablation_study.py:136`, `smote_comparison.py:106`, `threshold_sensitivity.py:160`, `confidence_threshold.py:111`, `circularity_sanity_check.py:139`.
- `extract_all_results.py:102-104`, `scripts/canonical/compare_16f_vs_19f.py:196-198` (split-ul real; 106-126 e build_dataset), `scripts/canonical/xml_vs_json_accuracy.py:44-45`.
- `sanity_check_frozen.py:62-64`, `patch_frozen_meta.py:30`, `bootstrap_ci_frozen.py:115` (reproduc exact split-ul frozen).
- Generatoarele waterfall: `generate_waterfall_combined.py:86-88`, `_json.py:78-80`, `_ext.py:82-84`, `generate_waterfall_svm.py:76-78`, `fix_svm_barplot.py:50`.

### Imputation (median) fit DOAR pe train

SimpleImputer strategy='median' invatat exclusiv pe train, apoi transform pe test (fara leakage de statistici din test).
- `cap5_pipeline.py:137-143` (_train_eval_core), `cap5_pipeline_json.py:141-147`, `cap5_pipeline_json_extended.py:165-171`.
- `cap5_compare_algorithms.py:151-153` (fit_transform pe train, transform pe test), `_json.py:156-158`, `_json_ext.py:160-162`.
- `freeze_production_model.py:123-126`, `demo_end_to_end_v2.py:121-124`.
- `ablation_study.py:141-147`, `smote_comparison.py:111-117`, `threshold_sensitivity.py:165-171`, `confidence_threshold.py:121-124`, `circularity_sanity_check.py:147-148`.
- `sanity_check_frozen.py:74` si `patch_frozen_meta.py:34` (imputer.transform, NU fit, pe test).
- `extract_all_results.py:106-108`, `scripts/canonical/xml_vs_json_accuracy.py:48-50`.
- Waterfall: `generate_waterfall_combined.py:90-92`, `_json.py:82-84`, `_ext.py:86-88`, `generate_waterfall_svm.py:80-82`, `fix_svm_barplot.py:52-54`.

### SMOTE aplicat DOAR pe training fold (post-imputation)

SMOTE(random_state=42, k_neighbors=3) pe train dupa imputare, niciodata pe test.
- `cap5_pipeline.py:148-149`, `cap5_pipeline_json.py:151-152`, `cap5_pipeline_json_extended.py:175-176`.
- `cap5_compare_algorithms.py:156-157` (k_neighbors=3), `_json.py:161`, `_json_ext.py:165-166`.
- `freeze_production_model.py:128-129`, `demo_end_to_end_v2.py:127-129`.
- `ablation_study.py:150-157`, `smote_comparison.py:204` (SMOTE k=3), `:214` (SMOTE k=5; doua dintre cele 5 strategii testate), `threshold_sensitivity.py:177-189` (k adaptiv), `confidence_threshold.py:127`, `circularity_sanity_check.py:150`.
- `extract_all_results.py:111-112`, `scripts/canonical/xml_vs_json_accuracy.py:52`.
- Waterfall: `generate_waterfall_combined.py:94-95`, `_json.py:86-87`, `_ext.py:90-91`, `generate_waterfall_svm.py:84-85`, `fix_svm_barplot.py:55-56`.

### Ordine scaler vs SMOTE pentru SVM (FND-105/FND-147)

Fix-ul cere StandardScaler fit pe X_train_imp PRE-SMOTE, apoi SMOTE in spatiu izotrop (evita scalare contaminata de esantioane sintetice). ATENTIE: nu toate scripturile aplica ordinea corectata.

Ordine CORECTATA (scaler fit pre-SMOTE):
- `cap5_compare_algorithms_json_ext.py:170-174` (scaler fit pe X_train_imp pre-SMOTE, varianta canonica RQ2).
- `scripts/canonical/compare_16f_vs_19f.py:210-215` (cale SVM: imputer -> scaler -> SMOTE, FND-105; CV pipe corect la 159-165).
- `extract_all_results.py:116-120` (imputer -> StandardScaler fit pe train -> SMOTE).

Ordine VECHE (scaler fit pe date deja SMOTE, NU implementeaza fix-ul FND-105):
- `cap5_compare_algorithms.py:160-162` (versiunea XML, scaler fit pe X_train_sm).
- `generate_waterfall_svm.py:88-89` si `fix_svm_barplot.py:57-58` (utilitare SVM SHAP: SMOTE la 84-85 / 55-56, apoi StandardScaler pe datele SMOTE). Figurile SHAP SVM provin din aceasta ordine, diferita de SVM-ul canonic. De mentionat la aparare daca figurile sunt in lucrare.

### Lag-1 feature engineering (datele de ieri prezic recomandarea de azi)

shift(1) pe toate features de baza. Acesta este si motivul anti-leakage temporal in feature space.
- `cap5_pipeline.py:84-91,113-114`, `cap5_pipeline_json.py:100-107,122-123`, `cap5_pipeline_json_extended.py:133-147`.
- `cap5_compare_algorithms.py:90-99` (16 coloane), `freeze_production_model.py:69-70`, `demo_end_to_end_v2.py:83-84`.
- `ablation_study.py:110-117`, `smote_comparison.py` (19 features lag-1), `threshold_sensitivity.py:144`, `confidence_threshold.py:44-50` (lista 15 features), `:92` (shift(1) aplicat), `circularity_sanity_check.py:115`, `learning_curves_comparison.py:61-70`.
- Waterfall: `generate_waterfall_combined.py:49-58`, `_json.py:59-60`, `_ext.py:63-64`, `generate_waterfall_svm.py:57-58`.

### Random seed fix 42 (reproducibilitate)

random_state=42 explicit pe toate componentele stochastice (SMOTE, RF, XGBoost, SVM, scaler, RandomizedSearchCV, split).
- `cap5_pipeline.py:148,155,210`, `cap5_pipeline_json.py:151,158`, `cap5_pipeline_json_extended.py:175,182`.
- `cap5_compare_algorithms_json_ext.py` (RF/XGB/SVM la liniile 192-203, SMOTE la 166).
- `freeze_production_model.py:128-136`, `demo_end_to_end_v2.py:127-136`.
- `ablation_study.py:63`, `smote_comparison.py:59`, `threshold_sensitivity.py:58`, `confidence_threshold.py:53`, `circularity_sanity_check.py:150,154`, `bootstrap_ci_frozen.py:48-49` (seed=42, N_BOOTSTRAP=2000).
- `learning_curves_comparison.py:104-123,246-260`, `scripts/canonical/compare_16f_vs_19f.py:76,163-172`, `scripts/canonical/xml_vs_json_accuracy.py:52-56`.

### Regula T3 de etichetare (Rest / Normal / Active)

Praguri canonice: Rest daca readiness < 60 OR sleep < 60; Active daca readiness >= 85 AND sleep >= 75; Normal in rest. Aceste praguri NU se schimba (Ground Truth in the tooling context file).
- Sursa unica de adevar: `t3_labels.py:24-27,39-44` (T3_REST_CUTOFF=60, T3_ACTIVE_READINESS=85, T3_ACTIVE_SLEEP=75).
- Importata de `cap5_pipeline_json_extended.py:30,149`, `experiments_new.py:132`, `streamlit_app_v2.py` (classify_day).
- Logica inline (pre-centralizare) inca prezenta in: `cap5_pipeline.py:94-108`, `cap5_pipeline_json.py:110-118`, `cap5_compare_algorithms*.py`, `freeze_production_model.py:195` (metadate), `demo_end_to_end_v2.py:87-94`, `ablation_study.py:120-127`, `smote_comparison.py:91-98`, `confidence_threshold.py:95-102`, `circularity_sanity_check.py:90-98`, generatoarele waterfall.

### TimeSeriesSplit CV (cross-validare temporala, shuffle=False)

TimeSeriesSplit n_splits=5 pentru CV, calculat pe TRAIN set (nu pe X,y complet) in versiunea canonica, pentru a evita leakage din test la evaluare.
- `cap5_pipeline.py:216`, `cap5_pipeline_json.py:215`, `cap5_pipeline_json_extended.py:239-243` (pe X_train_t, y_train_t).
- `cap5_compare_algorithms.py:199`, `_json.py:204`, `_json_ext.py:211` (cross_val_score pe TRAIN, liniile 267-268).
- `circularity_sanity_check.py:173`, `learning_curves_comparison.py:94,234` (init TimeSeriesSplit; shuffle=False in learning_curve la 154,276), `experiments_new.py:570` (shuffle=False).
- `extract_all_results.py:122,167-187` (ImbPipeline de-leaked, imputer refacut pe fiecare fold), `scripts/canonical/compare_16f_vs_19f.py:217,235-236` (ImbPipeline de-leaked pe X_train brut, fix FND-101).

### Confidence threshold (safety layer)

Daca max(predict_proba) < prag, predictia se suprascrie cu Normal (label 1), fallback conservator. Motivatie: limitarea erorii calorice in worst-case (Active->Rest = dubla eroarea fata de Active->Normal).
- `streamlit_app_v2.py:424-428` (prag 0.50), `demo_end_to_end_v2.py:242-249` (prag 0.50).
- `confidence_threshold.py:61` (testeaza 0.00/0.45/0.50/0.55/0.60/0.65/0.70), `:159` (fallback la label 1), `:232-233` (prag 0.60 recomandat), `:58` (CALORIC_MAP Rest=-200/Normal=0/Active=200).
- `experiments_new.py` (experiment 4 calibrare valideaza pragurile de confidenta).
- `prospective_validation.py`: NU aplica prag de confidenta; predictia la :148 e directa (`rf.predict`), iar :170 doar inregistreaza `y_proba_max` pentru raport. Pragul de siguranta exista in UI/demo, nu in validarea prospectiva.

### Class weighting (alternativa la SMOTE pentru learning curves)

class_weight='balanced' folosit unde SMOTE esueaza pe subseturi mici (clasa Active sub k+1=4 instante), demonstrat empiric.
- `learning_curves_comparison.py:135` (nota: SMOTE necesita minim k+1=4 instante per clasa), `:244-247,259-261` (varianta 2 cu class_weight balanced), `:155` (error_score=np.nan pentru a capta esecul SMOTE ca dovada).
- `experiments_new.py:531` (experiment 3 foloseste class_weight balanced in loc de SMOTE), `:553` (train_sizes linspace 0.2-1.0).
- `smote_comparison.py` (strategia 2 cost-sensitive class_weight balanced printre cele 5 testate).

### 16 features vs 19 features (feature expansion)

16f = 11 Oura + 5 Apple Watch de baza. 19f adauga 3 features noi din JSON: aw_sleep_deep_hours, aw_sleep_rem_hours, aw_resp_rate.
- 16f: `cap5_pipeline.py:84-89`, `cap5_pipeline_json.py:100-107`, `cap5_compare_algorithms.py:90-99`, `_json.py:95-104`, generatoarele waterfall XML/JSON.
- 19f canonic: `cap5_pipeline_json_extended.py:133-140`, `cap5_compare_algorithms_json_ext.py:98-104`, `generate_waterfall_combined_ext.py:53-59`, `generate_waterfall_svm.py:47-53`, `freeze_production_model.py` (prin import).
- Delta 16f vs 19f reprodusa cu sanity-assert in `scripts/canonical/compare_16f_vs_19f.py:82-89` (16f = 19f minus cele 3 features), `:91-97,286-292` (sanity check vs CIFRE_CANONICE.md sec.3).
- ATENTIE: `confidence_threshold.py:44-50` foloseste un set REDUS de 15 features (fara sleep_deep, sleep_rem, resp_rate), diferit de canonicul 19f, de mentionat la aparare.

### Data quality fix rhr_avg < rhr_lowest (bug API v2)

Valori fiziologic imposibile (rhr_avg < rhr_lowest, ~60 zile Gen4) setate la NaN pentru imputare downstream.
- Originea bug-ului si fix-ul: `oura_merge.py:162-180` (documentat in-code), `:171-179` (set NaN).
- Reaplicat in pipeline: `cap5_pipeline_json_extended.py:119-123`, `circularity_sanity_check.py:52-53`, `scripts_archive/old_versions/compute_mcc_qwk.py:85-88`.

### Reproducibilitate frozen (hash si verificare)

Hash dataset normalizat (cols sortate, lineterminator='\n', UTF-8, SHA256[:16]) pentru invarianta cross-platform; sanity check abort daca point estimate nu match canonicul.
- `freeze_production_model.py` (forteaza lineterminator='\n', salveaza SHA256 in frozen_meta).
- `sanity_check_frozen.py:46-48` (hash normalizat), `:62-64` (split replica), TOL=1e-6.
- `bootstrap_ci_frozen.py:128-161` (sanity check point estimate vs canonic, abort daca mismatch).
- `scripts/canonical/compare_16f_vs_19f.py:261-266` (hash verificat 63bc3a2915bae864), `:270-276` (split verificat), `:350-355` (abort daca 19f diverge de canonic).

### Conversii unitati (consistenta surse)

- Active energy kJ -> kcal (/4.184): `cap5_pipeline_json.py:61`, `cap5_compare_algorithms_json.py:77`, `demo_end_to_end_v2.py:71`.
- SpO2 ca % (JSON) vs fractie (XML): `cap5_pipeline_json.py:63-65`.
