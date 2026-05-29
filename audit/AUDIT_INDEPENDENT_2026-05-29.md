# AUDIT INDEPENDENT PROTOTIP PYTHON - DISERTATIE UPB

**Data:** 2026-05-29
**Mod:** READ-ONLY, de la zero, fara consultarea audit-urilor anterioare (independenta stricta)
**Subiect:** Pipeline ML clasificare stare fiziologica (Rest/Normal/Active) din date wearable Oura Ring + Apple Watch, Random Forest + SHAP, ~958 zile utile, split temporal 80/20.
**Metodologie:** workflow multi-agent (8 agenti mapare + 4 specialisti + 47 verificatori adversariali = 63 agenti, 50 scripturi, 7 axe), plus verificare independenta proprie pe calea canonica (sanity check frozen, recalcul hash read-only, spot-check pe 4 claims cu impact mare, verificare istoric git pentru secrete).

> Acest audit nu a modificat niciun fisier al proiectului. Singurul script rulat (`sanity_check_frozen.py`) nu scrie pe disc; restul au fost comenzi git/read. Patch-urile propuse sunt unified diff de aplicat manual.

---

## 1. Sumar executiv

| Severitate | Confirmate | Note |
|---|---|---|
| BLOCKER | 4 (3 teme distincte) | 2 sunt aceeasi cauza (CV leaky in scripturi legacy) |
| MAJOR | 37 + 1 (securitate, gasit la verificarea proprie) | grupate pe 7 axe |
| NICE-TO-HAVE | 76 | igiena, DRY, cod mort |
| Respinse la verificare | 6 | false positives eliminate adversarial |

**Reproducibilitate numerica: EXCELENTA.** Toate cele 26 de verificari numerice MATCH bit-perfect fata de `CIFRE_CANONICE.md`: model frozen (F1 Macro 0.506324, Accuracy 0.640625, MCC 0.141560), hash dataset `63bc3a2915bae864` (recalculat read-only, identic), si TOATE tabelele (modele principale, ablation, threshold T1-T7, SMOTE, confidence, tuning, calibration, rolling, circularity, deltas 16f vs 19f, versiuni librarii). Artefactele committed reproduc exact cifrele din lucrare.

**Verdict: DEFENSABIL CA PROOF-OF-CONCEPT ML LA NIVEL DE MASTER, CU REZERVE.**

Argumente PRO: calea canonica (`cap5_pipeline_json_extended` + `freeze_production_model` + experimentele 19f) este libera de data leakage (imputer si SMOTE fit-uite strict pe train, CV de-leaked in `TimeSeriesSplit` pe `X_train`, lag-1 aliniat corect). Model inghetat, hash, seed-uri (42), versiuni librarii pinned. Circularitatea autoregresiva este recunoscuta SI cuantificata (`circularity_sanity_check`).

Rezerve care TREBUIE tratate inainte de aparare (detalii la sectiunile 3-4):

1. Selectia design-ului (prag T3, SMOTE k=3, confidence threshold) se face pe setul de test, fara validation split dedicat (selection leakage).
2. Concluziile despre clasa Active se sprijina pe doar 8 esantioane in test (1-2 zile decid F1 Active), fara interval de incredere.
3. Studiu N=1 (un singur subiect) fara limitare explicita de generalizare in cod/metadata.
4. Exista scripturi legacy (non-`_ext`) cu CV leaky; nu contamineaza cifrele canonice actuale, dar sunt o capcana daca se ruleaza varianta gresita.
5. Cateva figuri poarta etichete numerice stale care contrazic cifrele canonice (ex. "SVM cel mai bun model, F1 52.1%").
6. Secret (token Oura) ramane in istoricul git dupa untracking (necesita rotatie).

---

## 2. Harta pipeline-ului (data flow)

**A. COLECTARE** (nu ruleaza in audit: lipsa token Oura + JSON iCloud, motiv raportat, fara estimari)

- `oura_auth.py` --(OAuth2)--> scrie `OURA_ACCESS_TOKEN` in `.env`
- `oura_collect.py` --(API Oura v2)--> `data/{sleep,readiness,activity,stress,sleep_detailed,sleep_metrics}.csv`
- `oura_merge.py` : `data/oura_historical.csv` (export manual) + CSV API --> `data/oura_combined.csv` **(aici se aplica fix-ul rhr_avg < rhr_lowest -> NaN)**
- `parse_health_auto_export.py` : JSON Health Auto Export --> `apple_watch_api/apple_health_daily.csv`

**B. CALEA CANONICA ML**

- `t3_labels.py` (praguri T3: Rest<60, Active>=85 & >=75; `assign_nutrition_label`) importat de pipeline/freeze/prospective
- `cap5_pipeline_json_extended.py` (19 features): `oura_combined.csv` + `apple_health_daily.csv` -> `merge_datasets` (sort dupa data + rhr fix) -> `engineer_features` (lag-1 `shift(1)`, label, `dropna(label)`) -> `train_and_evaluate` (split 80/20 cronologic, imputer median **train-only**, SMOTE k=3 **train-only**, RF; CV `ImbPipeline` in `TimeSeriesSplit` **pe X_train**) -> `data/dataset_combinat_json_extended.csv`
- `freeze_production_model.py` (importa din _extended) -> `models/rf_frozen_20260331.joblib`, `imputer_frozen_...`, `frozen_meta_...json`, `dataset_snapshot_...csv` (hash `63bc...`)
- `sanity_check_frozen.py` -> reproduce metrici (PASS), `prospective_validation.py` -> aplica frozen pe date noi

**C. EXPERIMENTE** (19f, lag/label canonic, citesc `apple_health_daily.csv`)

- `ablation_study.py` (Oura/AW/Combined), `smote_comparison.py`, `threshold_sensitivity.py` (T1-T7), `circularity_sanity_check.py` (A/B/C), `learning_curves_comparison.py` -> `outputs/tables/*.csv`
- `confidence_threshold.py` -> **DIVERGENT**: citeste `data/apple_health/*.csv` (raw HealthKit), **16 features**, nu modelul canonic 19f

**D. COMPARARE MODELE**

- `cap5_compare_algorithms_json_ext.py` **(CANONIC, 19f, de-leaked)** -> `new_models_comparison.csv`, `mcc_qwk_results.csv`, calibration, tuning
- `cap5_compare_algorithms.py` (XML 16f) si `cap5_compare_algorithms_json.py` (JSON 16f) = **LEGACY, CV leaky**
- `experiments_new.py` (tuning RandomizedSearchCV, LR/LightGBM, rolling, calibration), `compare_16f_vs_19f.py` (deltas Tabel 5.4)

**E. AGREGARE CIFRE**

- `extract_all_results.py` -> `outputs/tables/valori_disertatie.csv` (reimplementeaza load/label, DRY)
- `generate_canonical_numbers.py` : citeste `outputs/tables/*.csv` + `frozen_meta` -> `CIFRE_CANONICE.md`

**F. PLOTS**: `generate_waterfall_combined{,_ext,_json}.py`, `generate_waterfall_svm.py`, `fix_svm_barplot.py`, `circularity_plot.py` (autocorelatie hardcodata)

**G. DEMO/UI**: `demo_end_to_end_v2.py` (re-antreneaza `rf_model.joblib`), `streamlit_app_v2.py` (incarca `rf_model.joblib`, **NU** frozen), `nutrition_recommender.py` (`USER_PROFILE` hardcodat N=1)

**H. LEGACY/DEAD**: `cap5_pipeline.py` (16f XML), `cap5_pipeline_json.py` (16f JSON), `scripts_archive/**` (15 scripturi, neutilizate de calea activa)

---

## 3. Findings BLOCKER

### B1. CV leaky pe X,y INTREG (train+test) in scripturile legacy non-`_ext`

**Fisiere/linii:** `cap5_pipeline.py:217-218,236-237`, `cap5_pipeline_json.py:216-217`, `cap5_compare_algorithms.py:251-252`, `cap5_compare_algorithms_json.py:256-257`

**Cauza:** `cross_val_score(cv_pipeline, X, y, cv=tscv, ...)` primeste dataset-ul complet (958 = train 766 + test 192) in loc de `X_train`. In `TimeSeriesSplit`, fold-urile expanding includ inevitabil zilele din holdout-ul de 20%, deci "CV F1 Macro" raporteaza partial performanta pe test. Verificat independent la `cap5_pipeline_json.py:216` (confirmat bit-perfect).

**Impact asupra cifrelor canonice:** ZERO direct. Cifrele canonice (CV F1 RF 0.3369) provin din `cap5_compare_algorithms_json_ext.py:267`, care e corect (`X_train`). Acesta e un BLOCKER de RISC: daca cineva ruleaza/citeaza varianta legacy, obtine un CV optimist umflat care contrazice lucrarea. Recomand arhivarea sau corectarea celor 4 scripturi.

```diff
--- a/cap5_pipeline_json.py
+++ b/cap5_pipeline_json.py
@@ -213,8 +213,8 @@
     tscv = TimeSeriesSplit(n_splits=5)
-    cv_scores_t = cross_val_score(cv_pipeline, X, y, cv=tscv, scoring="accuracy")
-    f1_cv_t = cross_val_score(cv_pipeline, X, y, cv=tscv, scoring="f1_macro")
+    cv_scores_t = cross_val_score(cv_pipeline, X_train_t, y_train_t, cv=tscv, scoring="accuracy")
+    f1_cv_t = cross_val_score(cv_pipeline, X_train_t, y_train_t, cv=tscv, scoring="f1_macro")
```

(identic in `cap5_pipeline.py:217-218`, `cap5_compare_algorithms.py:251-252`, `cap5_compare_algorithms_json.py:256-257`; vezi si M16/M17 pentru ramura StratifiedKFold random)

### B2. Concluziile despre clasa Active se sprijina pe 8 esantioane in test

**Fisiere/linii:** `models/frozen_meta_20260331.json` (`class_distribution_test_internal`), `CIFRE_CANONICE.md:116-123,327-333`

**Cauza:** Distributia test = {Rest:44, Normal:140, Active:8}. F1 Active 0.4444 vine din confusion matrix randul Active `[0,4,4]` (recall 4/8, precision 4/10). Diferentele centrale ale Cap.5.4 (F1 Active "AW-only 0.100 -> Combined 0.444 ~4.4x"; delta 19f vs 16f "+9.15pp Active") se reduc, in termeni de esantioane, la a clasifica corect 1-2 zile Active din 8. Un singur sample mutat schimba F1 cu ~0.1.

**Impact:** Nu invalideaza cifrele (sunt reproductibile), dar invalideaza interpretarea lor CANTITATIVA fara interval de incredere. Submineaza robustetea afirmatiilor "RF best", "fuziunea Oura+AW ajuta Active", "cele 3 features noi adauga +9.15pp". Pentru o comisie AI/ML aceasta e prima intrebare.

**Fix (in lucrare, nu cod):** (1) raporteaza explicit support-ul per clasa (Active n=8) langa fiecare F1 Active; (2) adauga bootstrap CI pe test sau macar caveat "F1 Active estimat pe 8 zile, instabil"; (3) muta accentul concluziilor pe F1 Macro / MCC / QWK globale si pe CV F1.

### B3. Selectia design-ului (T3, SMOTE k, confidence threshold) se face pe SETUL DE TEST

**Fisiere/linii:** `threshold_sensitivity.py:159-228,376-378`, `smote_comparison.py:106-119,292-302`, `confidence_threshold.py:110-115,216-256`

**Cauza:** Cele 3 scripturi de "selectie" impart datele 80/20 o singura data si evalueaza pe ACELASI test intern (ultimele 192 zile), apoi declara "optim" configuratia cu cel mai mare F1 pe test: T3 (`threshold_sensitivity`), SMOTE k=3 (`smote_comparison`, `best = max(results, key=lambda r: r['f1_macro'])`), confidence 0.5/0.6 (`confidence_threshold`). Nu exista validation set separat. Selectie pe test = selection leakage clasic; metrica de test inceteaza sa fie estimator nepartinitor al generalizarii.

**Impact:** Cifrele raman corecte numeric, dar sectiunile 3, 5, 6, 7 din `CIFRE_CANONICE` nu pot fi prezentate ca "optim ales obiectiv". Pentru comisie, "am ales T3/k=3/0.5 fiindca dau cel mai bun F1 pe test" anuleaza statutul de holdout al testului.

**Fix (metodologic):** introdu un al treilea split temporal (train/val/test, ex. 70/10/20); selectia se face pe VAL, test atins o singura data la final. Alternativ, daca testul ramane fix, declara explicit in lucrare ca aceste alegeri sunt analize de sensibilitate post-hoc, NU selectie de model nepartinitoare.

---

## 4. Findings MAJOR (37 confirmate + 1 securitate)

### Securitate (adaugat la verificarea proprie, verificat in istoricul git)

**S1. Token Oura ramane in istoricul git dupa untracking.** `.env` (cu `OURA_CLIENT_SECRET` 43 char, `OURA_ACCESS_TOKEN` 45 char) a fost committat in `f7cc4ca`, scos din tracking in `91d8ee6`, dar blob-ul `b5381e3a43d81c345a81a1f5425e8d783b6583df` ramane recuperabil din istorie (`git cat-file -p b5381e3a` reda valorile). `.gitignore` + `git rm --cached` NU sterg secretul din istorie. **Actiune:** roteste tokenul/clientul Oura (regenereaza in Oura dev portal) si, daca repo-ul ajunge pe GitHub, curata istoria (`git filter-repo` / BFG). Nota: pe scala de severitate a acestui audit (axata pe stiinta) nu e BLOCKER, dar operational e prioritate inalta.

### DATA_LEAKAGE (in scripturi non-canonice; cifrele canonice neafectate)

- **M13/M14/M15** SVM in `cap5_compare_algorithms.py`/`_json.py`: `StandardScaler` fit pe date POST-SMOTE (puncte sintetice intra in mean/std), atat in holdout cat si in CV-fold. Varianta `_ext` a corectat ordinea (`imputer -> scaler -> smote`). Fix: muta scaler-ul inainte de SMOTE.
- **M16/M17** `cap5_pipeline.py`/`cap5_pipeline_json.py`: CV TimeSeriesSplit si StratifiedKFold ruleaza pe `X,y` full (acelasi root cause ca B1, ramura random).
- **M12** `confidence_threshold.py:216-256`: pragul de confidenta ales pe test (vezi si B3).
- **M18** `experiments_new.py:159-164,415-419,665-669`: scaler fit pe date post-SMOTE pentru LR/SVM. Impact marginal pe LR (0.342) si SVM Brier; tree-based neafectate.

### METODOLOGIE_ML

- **M26** Nicaieri bootstrap / interval de incredere / McNemar / p-value (verificat prin grep, zero rezultate). Diferentele RF 0.506 vs XGB 0.476 vs SVM 0.435 si delta +7.8pp combined vs Oura sunt prezentate ca reale, posibil in zgomot pe n=192.
- **M27** Singurul baseline e `DummyClassifier(most_frequent)`; lipseste `stratified` si un baseline pe clasa minoritara, slabind cadrul de evaluare pe dezechilibru.
- **M25/M28** `confidence_threshold.py` foloseste pipeline DIVERGENT (16f, raw HealthKit), deci F1 baseline 0.520 (sect.7) descrie alt model decat RF canonic 19f (0.5063). Pragul "recomandat" nu corespunde modelului de productie.
- **M29** Studiu N=1: `USER_PROFILE` hardcodat (greutate 119.9, body_fat 36.3); toate validarile sunt intra-subiect temporale. Lipseste limitarea explicita de generalizare in cod/metadata.

### CORECTITUDINE

- **M10** `prospective_validation.py:428`: `f1_per_class` trimis ca `{c: 0.0}` desi meta are valorile reale (0.3218/0.7527/0.4444). Figura `prospective_f1_comparison` ar arata fals barele interne la zero. Verificat independent. Fix:

```diff
--- a/prospective_validation.py
+++ b/prospective_validation.py
@@ -427,5 +427,5 @@
     plot_f1_comparison(
-        {"f1_per_class": {c: 0.0 for c in CLASS_NAMES}, "f1_macro": internal["f1_macro"]},
+        {"f1_per_class": internal["f1_per_class"], "f1_macro": internal["f1_macro"]},
         prospective_metrics,
```

- **M6** `shift(1)` presupune zile calendaristice consecutive, dar exista 42 de gap-uri (pana la 10 zile): lag-1 devine "ultima zi disponibila", nu "ieri". NU e leakage spre viitor (ramane in trecut), dar semantica "ieri" e incorecta pe gap-uri. Apare in toate scripturile cu `shift(1)`.
- **M4** `circularity_sanity_check.py:74-79`: autocorelatia lag-1 se calculeaza dupa `dropna()` INAINTE de `shift(1)`, traversand goluri (compacteaza zile neconsecutive). Submineaza dovada "autocorelatie slaba 0.234/0.054" care sustine argumentul anti-circularitate.
- **M7** `experiments_new.py:167-168`: cheie `X_test_imp` duplicata in dict (benign, valoarea finala corecta). Verificat independent.
- **M5** `generate_waterfall_combined_ext.py:204-217`: indexare fragila `expected_value[cls_id]` pentru SHAP XGBoost multiclass (risc IndexError pe versiuni shap noi).
- **M8** `oura_collect.py:57-62`: `sleep_metrics.average_heart_rate` agrega nap-uri/sesiuni partiale (valori 0.0), genereaza ~60 zile cu rhr_avg invalid (fixate downstream).
- **M9/M11** `oura_merge` prioritizeaza historical peste API pe zile comune (pierde metrici API); streamlit deriva data predictiei din `oura_combined` (1009 randuri) dar prezice pe `dataset_combinat_json_extended` (958), risc off-by-N.

### CIRCULARITATE

- **M1** Variant C "fiziologic brut" in `circularity_sanity_check.py:217-233` pastreaza `respiratory_rate_lag1` si `temp_deviation_lag1`, care sunt iesiri ale algoritmilor Oura (contribuitori la readiness), nu masuratori brute independente. Concluzia "circularitatea nu e o problema majora" e partial supravanduta.
- **M2/M3** Label T3 derivat din `readiness_score`/`sleep_score`(t); features = `*_lag1` ale acelorasi scoruri. Pe serii cu autocorelatie zilnica, scor(t-1) e proxy puternic pentru scor(t). Recunoscut partial (Variant B fara aceste scoruri da 0.5148, chiar mai mare), deci modelul NU depinde critic de ele, dar trebuie discutat onest in lucrare.

### REPRODUCTIBILITATE

- **M30** `generate_waterfall_combined.py` foloseste sursa AW diferita (raw HealthKit) si omite conversia kJ->kcal: dataset diferit de canonic.
- **M31** `freeze_production_model` reimplementeaza `engineer_features` inline (risc de drift; momentan bit-echivalent, hash corect).
- **M32** `oura_collect.py:28-29` `end_date=datetime.today()` face colectarea nedeterminista.
- **M33** UI (`streamlit_app_v2`) serveste `rf_model.joblib` (regenerat de demo) in loc de modelul frozen versionat.

### REPRO_NUMERICA (discrepante in figuri/text, nu in tabele)

- **M35** `generate_waterfall_svm.py:4,7,214`: eticheta "SVM cel mai bun model, F1 Macro 52.1%" e FALSA (canonic SVM 0.4347, cel mai slab; RF 0.5063 e best). Verificat independent. Stale, contrazice `CIFRE_CANONICE` sect.3/10.
- **M34/M19** Autocorelatia din figura circularitatii (`circularity_plot.py:100-107`) e hardcodata, nu produsa de `circularity_sanity_check` (care doar o printeaza). Figura nu e trasabila la date.
- **M36** Baseline `0.281`/`72.9%` hardcodate in `generate_canonical_numbers.py:143` (momentan match CSV, risc drift).
- **M37** Nota "RECOMANDAT la 0.6 vestigial" in `CIFRE_CANONICE` contrazice CSV-ul (RECOMANDAT e la 0.5).

### IGIENA_COD (MAJOR, cauze-radacina pentru leakage-ul legacy)

- **M20/M21/M22/M23** Duplicare masiva: 3 scripturi compare ~95% identice; `assign_nutrition_label` reimplementat inline in `cap5_pipeline.py`/`_json.py` (in loc de `t3_labels`); `fix_svm_barplot.py` dubleaza integral pipeline-ul din `generate_waterfall_svm.py`; `extract_all_results` reimplementeaza load/label. Aceasta e cauza pentru care fix-urile (de-leaking) au ajuns doar in `_ext`.
- **M24** `scripts_archive/old_versions/compute_mcc_qwk.py` scrie ACELASI `outputs/tables/mcc_qwk_results.csv` ca pipeline-ul activ, dar cu metodologie divergenta (capcana daca se ruleaza din greseala).

---

## 5. Findings NICE-TO-HAVE (76, lista concisa pe axa)

**CORECTITUDINE (21):** timezone in `parse_health_auto_export.parse_date` (taie 10 char); agregare `sum` pe sleep stages (risc dublare nap+somn); `aw_spo2 = aw_spo2` no-op (`cap5_pipeline_json.py:65`, `_extended.py:69`); primul rand (2023-06-10) supravietuieste cu toate features lag1 = NaN -> imputat; idiom `f1_score(labels=[lbl], average='macro')` pentru F1 per-clasa; `sanity_check_frozen`/`patch_frozen_meta` aliniere pozitionala accidentala dupa `imputer.transform`; cod mort in `draw_waterfall`; streamlit trimite la `demo_end_to_end.py` inexistent.

**IGIENA_COD (19):** cale absoluta hardcodata iCloud (`parse_health_auto_export.py:20-23`); cod LOAD/lag/label duplicat in 6 scripturi de experimente; mesaje `print('data/...')` cand fisierele se scriu in `outputs/...`; importuri nefolosite (`RandomForestClassifier`, `FancyArrowPatch`, `f1_score`); `CONFIDENCE_THRESHOLD=0.50` hardcodat in 2 locuri; `audit_csv.py` fara guard `__main__`; `experiments_new_balanced.py` near-duplicat arhivat.

**REPRODUCTIBILITATE (11):** `to_csv` fara `lineterminator`/`encoding` in `oura_merge.py:197` (risc CRLF); `n_jobs=-1` la `RandomizedSearchCV`/`learning_curve` (non-determinism potential); `freeze_production_model` suprascrie frozen fara protectie; `generate_canonical_numbers.load_meta` alege meta dupa `st_mtime` (nu dupa tag); KernelExplainer SVM fara seed; tab "Calitate Date" cu numere hardcodate.

**METODOLOGIE_ML (10):** fix rhr_avg doar in `_extended`, nu in `cap5_pipeline_json.py` (confound 16f vs 19f); header "Identic cu cap5_pipeline.py" inselator; SVM CV fara `probability=True` dar holdout cu; praguri quantile T7 calculate pe tot dataset-ul (pre-split, leaga eticheta de distributia globala); interpretarea learning curves bazata pe endpoint-ul range-ului TSCV.

**CIRCULARITATE (6):** include o verificare POZITIVA (tratarea lag-1 e onesta, Variant B >= A); Variant B pastreaza scoruri compozite corelate (hrv_balance).

**REPRO_NUMERICA (3):** F1 Rest/Normal/Active in CSV difera de tabel prin rotunjire; dataset sizes hardcodate in `generate_canonical_numbers`.

**DATA_LEAKAGE (6):** TOATE sunt verificari POZITIVE care confirma ca pipeline-ul canonic (imputer/SMOTE train-only, lag/label aliniat, rolling pre-split dar respins) NU are leakage.

---

## 6. Lista scripturi de rerulat dupa fixuri

Daca aplici fix-urile, ruleaza in aceasta ordine (pe Windows: `PYTHONUTF8=1 python ...`):

1. **Daca corectezi B1/M16/M17 (CV legacy):** doar `cap5_pipeline.py`, `cap5_pipeline_json.py`, `cap5_compare_algorithms.py`, `cap5_compare_algorithms_json.py`. NU afecteaza cifrele canonice (vin din `_ext`). Recomand mai degraba arhivarea lor.
2. **Daca introduci train/val/test pentru B3:** re-ruleaza `threshold_sensitivity.py`, `smote_comparison.py`, `confidence_threshold.py` si actualizeaza `CIFRE_CANONICE` sect.5/6/7. Cifrele se vor schimba (de asteptat).
3. **Fix M10 (prospective 0.0):** `prospective_validation.py` (doar regenereaza figura, nu atinge tabele).
4. **Fix M35 (eticheta SVM 52.1%):** `generate_waterfall_svm.py`, `fix_svm_barplot.py` (doar figuri).
5. **NU rula** `freeze_production_model.py` (ar suprascrie modelul inghetat; cifrele canonice depind de el).
6. **Dupa orice modificare a caii canonice:** `python sanity_check_frozen.py` trebuie sa ramana PASS pe toate liniile (acum: PASS bit-perfect).

**NU necesita rerulare** (cifre canonice deja corecte si reproductibile bit-perfect): intregul lant `cap5_pipeline_json_extended` -> `freeze` -> `extract_all_results` -> `generate_canonical_numbers`.

---

## Anexa A: 6 findings RESPINSE la verificarea adversariala (transparenta)

Eliminate ca false positives:

1. Device flags pe comparatie string (corect, datele-s sortate string-ISO `YYYY-MM-DD`).
2. "ImbPipeline SMOTE in CV ramane leaky" (fals: SMOTE in `cross_val_score` se fit-uieste per-fold).
3-5. "fix rhr_avg lipseste in circularity/extract/demo" (fals: fix-ul e centralizat in `oura_merge.py` si salvat in `oura_combined.csv`, deci toate scripturile downstream il mostenesc).
6. "figurile SHAP nu folosesc frozen" (corect factual, dar SHAP e explicabilitate, nu metrica raportata).

## Anexa B: Verificari numerice proprii (26/26 MATCH bit-perfect)

`sanity_check_frozen.py`: Accuracy 0.640625, F1 Macro 0.506324, MCC 0.141560, hash dataset `63bc3a2915bae864`, distributie test {0:44, 1:140, 2:8}, N test 192 - toate MATCH. Hash recalculat read-only (sort coloane + `to_csv(lineterminator='\n')` + utf-8 + sha256[:16]) = `63bc3a2915bae864` MATCH. Tabele complete `CIFRE_CANONICE` (modele principale, ablation, threshold T1-T7, SMOTE, confidence, tuning, best params, modele extra LR/LightGBM, calibration Brier, rolling, circularity A/B/C, deltas 16f vs 19f, versiuni librarii) - toate MATCH cu `outputs/tables/*.csv` si `frozen_meta`.
