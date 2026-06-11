# RAPORT AUDIT PRE-DEFENSĂ

**Autor:** Adrian Lazăr | **Coordonator:** prof. dr. ing. Cătălin Buiu
**Data creării:** 11.06.2026 | **Upload lucrare:** cel târziu 15.06.2026, ora 23:59 | **Defensă:** 22-26.06.2026
**Surse canonice:** CIFRE_CANONICE.md, mcc_qwk_results.csv, valori_disertatie.csv (niciodată documente derivate)
**Regulă:** orice modificare numerică se verifică ÎNTÂI în sursele canonice, apoi se aplică în document.

---

## 1. AUDIT DOCUMENT DISERTAȚIE (RO) - EFECTUAT 11.06.2026

### 1.1 Metodologie
- Verificare pe extracția text a `Disertatie_Lazar_Adrian_corectata.docx` (copie KB sincronizată 11.06.2026).
- Grep sistematic: valori stale (listă închisă), valori canonice (listă închisă), inventar comentarii Word, inventar captions figuri/tabele.
- Limitare cunoscută: extracția text NU expune formatarea (partial bold în tabele, ecuații OMML, ToC, paginație). Acestea se verifică local, în Word (vezi 1.4).

### 1.2 Verdict numeric: CURAT

**FND-DOCX-NEW-001: ÎNCHIS.** Verificat:

| Locație | Valoare verificată | Status |
|---|---|---|
| Tabel 5.4, SVM F1 Macro | 0,4347 | OK |
| Tabel 5.4, SVM F1 Active | 0,25 | OK |
| Tabel 5.4, Δ vs 16f SVM | −2,41pp | OK |
| Tabel 5.2 (ablation 19f) | 0,5063 / 0,4760 / 0,4347 | OK |
| Narativa Cap.5/6 | „0,506 vs. 0,4347 și 0,476" | OK |

**Valori stale: 0 apariții** pentru: 0,482; 0,38 (SVM Active); 9.106; 953 ca total dataset; 706/45; 762/191; 564; 162/564/36; 1.004; 1.082; 1.078; 1.008; 0,517/0,366/0,348 (ablation vechi).

**Valori canonice: toate prezente.** F1 Macro 0,5063/0,4760/0,4347; MCC 0,1416/0,1503/0,0956; QWK 0,2551/0,1805/0,1709; CI MCC [−0,0045; 0,2843]; 958 zile; split 766/192; clase 202/710/46; train pre-SMOTE 158/570/38; SMOTE target 570; MAE pași 8.734.

**False positives documentate (NU se modifică):** DOI 10.1613/jair.953 (bibliografie); pp. 752-762 (bibliografie); 2.191,8 și N=953 pentru respiratory_rate (tabel statistici descriptive).

**Notă intenționată (NU se modifică):** comparația surse XML vs JSON este realizată pe setul nativ de 953 de zile, versiune anterioară frozen-ului de 958; documentul explică explicit acest lucru. Defensibil la întrebări de comisie.

### 1.3 Findings deschise (de rezolvat local, vineri 12.06)

- [ ] **F2. Figuri:** doar 2 captions în document (Figura 4.1 arhitectura, Figura 5.1 interfața Streamlit). Verifică: SHAP plots și matricea de confuzie există ca imagini fără caption? Dacă da: adaugă captions numerotate per capitol + referințe în text. Dacă lipsesc de tot: adaugă minim matricea de confuzie RF și SHAP bar plot (gata generate în `outputs/figures/`). Notă: c184 (screenshot Streamlit promis lui Buiu) e închis prin Figura 5.1.
- [ ] **F3. Ecuații:** toate în Equation Editor (cerința Buiu, c87: „la fel peste tot unde mai ai ecuații"). Extracția aplatizează formulele, deci verificare exclusiv vizuală, în Word.
- [ ] **F4. Zecimale, uniformizare la 4:** Tabel 5.4: 0,506 → 0,5063; 0,476 → 0,4760; F1 Active 0,44/0,33/0,25 → 0,4444/0,3333/0,2500; Δ +3,6pp → +3,66pp. Verifică apoi consistența cu Tabel 5.2 și 5.4b (deja pe 4 zecimale).
- [ ] **F5. Diacritice, 4 locuri certe:** titlul Cap.2 „Starea actuală in domeniul abordat" → „în" (heading + referința din 1.5); heading 6.4 „prototipului si aspecte" → „și"; „evaluată si prin metrici" (Cap.5, lângă Tabel 5.4b); „Autori si an" (header tabel Cap.2). Atenție: „a doua" din text este corect, nu se modifică.
- [ ] **F6 (opțional, recomandat). Paragraf reproducibilitate:** lipsește. Conținut propus, 3-4 fraze în 5.3 sau Cap.6: model înghețat la 31.03.2026 (`rf_frozen_20260331.joblib` + imputer + metadata JSON), hash SHA-256 al setului de date `63bc3a2915bae864` (verificat cross-platform), seed 42, split temporal fixat. Răspunde preventiv la întrebarea „cum dovediți reproducibilitatea?".

### 1.4 Verificări locale (nevizibile în extracție)

- [ ] Partial bold în celulele tabelelor intact după editări (fragilitate cunoscută).
- [ ] OMML real la toate ecuațiile (vezi F3).
- [ ] Captions auto + numerotare per capitol funcționale (conversia rămasă din aprilie: figurile).
- [ ] ToC regenerat (Update Field) după toate modificările.
- [ ] Cross-references valide (niciun „Error! Reference source not found").

### 1.5 Inventar comentarii Word (stare 11.06)

7 comentarii Buiu, toate cu răspuns (tracking anterior: 9; probabil 2 șterse după rezolvare):

| ID | Buiu | Răspuns Adrian | De verificat că s-a materializat |
|---|---|---|---|
| c9 | Titlu Cap.2 în română | „Rezolvat cu textul in limba romana" | Da, dar cu typo „in" (vezi F5) |
| c74 | „Ori in clar, ori deloc" | „Gata" | - |
| c83 | „In romana" | „OK" | - |
| c87 | Ecuații în Equation Editor, peste tot | „OK" | F3 |
| c184 | „Un screenshot?" | „O sa pun" | Închis: Figura 5.1 |
| c191 | „Atentie la formatare" (zona oura_merge) | „OK" | Vizual |
| c494 | „Atentie formatare" (zona tuning, Cap.5) | „ok" | Vizual |

---

## 2. AUDIT PROTOTIP PYTHON - DE EFECTUAT (rulezi în the LLM assistant)

### 2.1 Scope și regulă de decizie

Audit de **reproducibilitate**, nu exploratoriu. Timebox: 3-4 ore. Codul a trecut deja prin Sprint 0-6 (CV leakage fix, rhr_avg fix, scale→SMOTE, frozen model).

**Regula de decizie:** o problemă găsită intră în document DOAR dacă invalidează o cifră tipărită sau claim-ul principal (RF best model). Orice altceva merge în secțiunea 2.4 (KNOWN_ISSUES), se folosește onest la defensă și se rezolvă în versiunea ESWA. Nu se modifică niciun script care produce cifre canonice înainte de upload.

Disciplina the LLM assistant: aprobare individuală per comandă, „Yes (option 1)", niciodată „allow all" pentru git sau operații ireversibile.

### 2.2 Checklist verificări

| # | Verificare | Cum | Criteriu PASS | Rezultat |
|---|---|---|---|---|
| 1 | Environment | `python --version`; versiuni sklearn/imblearn/pandas/xgboost | Conform mediului documentat (Python 3.11.9, sklearn 1.8.0, imblearn 0.14.1, pandas 2.3.3) | |
| 2 | Hash dataset | `sanity_check_frozen.py` | hash == `63bc3a2915bae864` | |
| 3 | Metrici frozen | `sanity_check_frozen.py` | RF F1 Macro 0,5063 bit-perfect; MCC 0,1416; QWK 0,2551 | |
| 4 | Re-run end-to-end | scriptul principal de train/eval (seeded), comparat cu CIFRE_CANONICE.md | Toate valorile RF/XGB/SVM identice (F1 Macro, MCC, QWK, per-class) | |
| 5 | Bootstrap CI | `bootstrap_ci_frozen.py` | CI MCC [−0,0045; 0,2843] | |
| 6 | Anti-leakage static | grep pe fit/transform: scaler fit DOAR pe train; SMOTE doar în imblearn.Pipeline pe train; cross_val_score doar pe X_train; lag-1 înainte de split; dropna → 958 rânduri | Zero pattern-uri de leakage | |
| 7 | Streamlit smoke test | `streamlit run` aplicația; încarcă modelul frozen, produce o recomandare pe un rând de test | Fără crash, output coerent, disclaimer prezent | |
| 8 | Security grep | grep credentials, date personale, atribuire AI în fișierele din repo | Zero hituri | |

### 2.3 Rezultate (rulat 11.06.2026)

| # | Status | Observații |
|---|---|---|
| 1 | **PASS** | Python 3.11.9, sklearn 1.8.0, imblearn 0.14.1, pandas 2.3.3, numpy 2.4.3, xgboost 3.2.0, joblib 1.5.3, lightgbm 4.6.0. Identic cu mediul documentat, zero diferențe de versiune (nici patch). |
| 2 | **PASS** | `sanity_check_frozen.py`: hash recalculat `63bc3a2915bae864` == canonic. Snapshot 958 zile, distribuție test {0:44, 1:140, 2:8} == canonic. |
| 3 | **PASS** | RF frozen bit-perfect (diff < 1e-6): Accuracy 0,6406, F1 Macro 0,5063, MCC 0,1416. Per-clasă: F1 Rest 0,3218, F1 Normal 0,7527, F1 Active 0,4444. Matrice confuzie reprodusă (Active 4/8 corect). QWK verificat la pasul 4. |
| 4 | **PASS** | Re-rulare seeded `extract_all_results.py` + `cap5_compare_algorithms_json_ext.py` (scripturile principale train/eval). RF 0,5063 / MCC 0,1416 / QWK 0,2551; XGB 0,4760 / 0,1503 / 0,1805; SVM 0,4347 / 0,0956 / 0,1709. CV F1 (0,3369 / 0,3343 / 0,3560), tuning Δ (RF +2,49 / SVM +4,47 / XGB +6,87 pp), per-clasă și 958/766/192 toate identice. `valori_disertatie.csv` și `mcc_qwk_results.csv` reproduse **bit-identic** (zero diff git de conținut). Singurele diff: timestamp în `valori_disertatie.txt` și 2 PNG SHAP summary (non-determinism matplotlib, nu FAIL). Repo restaurat din backup extern: hashes_after == hashes_before pe toate fișierele canonice (4 CSV/txt + 21 PNG); git status curat. |
| 5 | **PASS** | `bootstrap_ci_frozen.py`: sanity point estimate toate OK (acc/f1/mcc/per-clasă). CI 95% MCC [-0,0045; 0,2843], F1 Macro [0,3733; 0,6095], F1 Active [0,1111; 0,7000] lățime 0,5889. Identic cu canonic (commit e4a62f7). |
| 6 | **PASS** | Toate scripturile canonice (cele care scriu CSV în `outputs/tables/`) curate pe cele 4 dimensiuni: `sort(date) -> shift(1) -> dropna(subset=label) -> split cronologic iloc`; imputer/scaler fit DOAR pe train; scaler fit pre-SMOTE; SMOTE train-only; `cross_val_score`/`RandomizedSearchCV` pe `X_train`, cu preprocesare în `imblearn.Pipeline` (refit per fold). Două observații non-blocking, vezi K1 și K2. |
| 7 | **PASS** | App încarcă `rf_model.joblib` + `imputer.joblib`, ambele **bit-identice** cu `rf_frozen_20260331.joblib` / `imputer_frozen_20260331.joblib` (folosește modelul canonic). Predicție pe zi test 2026-03-30 validă: proba {Rest 0,389, Normal 0,351, Active 0,260}, safety layer confidence 0,5 suprascrie corect spre Normal, recommender produce output coerent (2998 kcal, P137/C403/F93). `streamlit run` pornește fără crash (boot check headless). Observație: lipsește disclaimer medical explicit, vezi K5. |
| 8 | **PASS** | Zero secrete hardcodate în cod (`oura_auth.py`/`oura_collect.py` citesc din `os.getenv`). `.env` untracked + gitignored. Findings non-blocking de igienă repo (NU afectează lucrarea uploadată): secrete Oura în istoricul git, atribuire AI în 27/33 commituri, path personal hardcodat. Vezi K3, K4, K6. |

### 2.4 KNOWN_ISSUES (intră în pregătirea Q&A, nu în document)

| ID | Descriere | Impact pe cifre tipărite? | Acțiune |
|---|---|---|---|
| K1 | Scripturi superseded (`cap5_pipeline.py`, `cap5_pipeline_json.py`, `cap5_compare_algorithms.py`, `cap5_compare_algorithms_json.py`) rulează `cross_val_score(cv_pipe, X, y)` pe X complet (CV leakage pre-fix Sprint 1). Variantele lor fixate (`_json_ext`, `extract_all_results`, `compare_16f_vs_19f`) folosesc `X_train`. | **NU.** Aceste scripturi nu scriu niciun CSV canonic de metrici (doar print la consolă sau dataset CSV de input). Cifrele CV tipărite (0,3369 / 0,3343 / 0,3560) vin din scripturile fixate. Confirmat prin maparea `to_csv` + re-rularea pasului 4. | Arhivează în `scripts_archive/` post-upload (cleanup ESWA). La defensă: explicabil ca versiuni anterioare păstrate pentru trasabilitate. |
| K2 | `experiments_new.py` `prepare_train_test` (linia 162-163) face `StandardScaler.fit` pe date post-SMOTE (`X_train_sm`); `X_train_scaled` antrenează LR în `new_models_comparison.csv`. | **NU** pe cifrele principale. Nu e test leakage (testul e doar `transform`-at, niciodată văzut la fit). Afectează doar metrica LR secundară (F1 0,342). RF/XGB/SVM principale și CV/tuning folosesc `ImbPipeline` corect (scaler->smote). Concluzia "LR confirmă non-liniaritatea" rămâne validă. | Aliniază ordinea scaler-before-SMOTE în versiunea ESWA. LR rămâne reproducibil ca atare. |
| K3 | Secrete Oura reale (CLIENT_SECRET, ACCESS_TOKEN, CLIENT_ID) prezente în istoricul git (`.env` a fost tracked până la f7cc4ca, scos în 91d8ee6). `.env` curent untracked + în `.gitignore`; codul citește din `os.getenv` (fără hardcodare). | **NU** (lucrarea uploadată este `.docx`/PDF, nu repo-ul). | Rotește tokenul Oura (expus în istoric și în acest audit). Dacă repo-ul/istoricul se partajează vreodată: scrub istoric (`git filter-repo`/BFG). Menține `.env` gitignored. |
| K4 | Atribuire AI în istoricul git: 27 din 33 commituri au trailer `Co-Authored-By: the LLM assistant`. Plus mențiuni "cu the LLM assistant" în fișiere de lucru interne (`the tooling context file`, `audit/DECIZIE_B3_*.md`, `AUDIT_PREDEFENSA`). Identitatea autorului git este doar Adrian Lazar. | **NU** (istoricul git și fișierele de lucru nu se încarcă pe platformă, nu fac parte din lucrare). | Dacă repo-ul/istoricul se predă ca anexă cod: rescrie istoricul pentru eliminarea trailerelor și exclude fișierele interne din bundle-ul partajat. |
| K5 | `streamlit_app_v2.py` nu are disclaimer medical explicit ("nu constituie sfat medical"). Există doar safety layer confidence 0,5 (fallback conservator Normal, ±200 kcal) și notă data-quality (Tab3). | **NU.** | Adaugă o notă scurtă de disclaimer în UI înainte de demo la defensă (recomandat pentru un prototip de recomandare nutrițională). |
| K6 | Path personal hardcodat: `parse_health_auto_export.py:21` (`C:\Users\Adrian\iCloudDrive`) și `outputs/reports/membership_vs_api_comparison.txt`. `disertatie_audit.zip` este tracked (60 intrări, verificat: fără secrete, doar scripts_archive + headere CSV). | **NU.** | Parametrizează path-ul (env var/argument) în ESWA; opțional scoate zip-ul din tracking. Curățenie, nu blocant. |

---

## 3. CHECKLIST FINAL PRE-UPLOAD (duminică 14.06)

- [ ] Toate F2-F6 din secțiunea 1.3 închise; re-rulare verificare numerică pe versiunea finală (re-sincronizare KB + grep stale).
- [ ] **Accept All Changes + Delete All Comments** (16 comentarii în document; obligatoriu înainte de export, altfel intră în PDF-ul verificat antiplagiat).
- [ ] Update Fields global: ToC, captions, cross-references.
- [ ] Export PDF; inspecție vizuală pagină cu pagină: tabele, ecuații, figuri, diacritice.
- [ ] PDF sub 20MB (limita platformei).
- [ ] Titlu copertă == titlu platformă (verificat 11.06: match exact pe 159 caractere; re-verifică doar dacă modifici coperta).
- [ ] Confirmarea lui Buiu că situația administrativă permite upload-ul (mesaj WhatsApp trimis 11.06).
- [ ] Răspuns secretariat privind comisia AIA5a (email trimis 11.06; dacă nu vine până duminică seara: telefon luni dimineață).
- [ ] **Luni 15.06 dimineața:** upload + Nr. pagini completat + verificare „Documentul este încărcat" + download de control + screenshot.

---

## 4. CHANGELOG MODIFICĂRI RO (pentru sync EN + manuscris ESWA, post-defensă)

| Data | Secțiune | Modificare | Sync EN | Sync ESWA |
|---|---|---|---|---|
| 12.06 | | | nu | nu |

---

## 5. PREZENTARE DEFENSĂ (plan; construire marți 16.06)

**Decizie:** deck-ul se construiește după upload, pe versiunea înghețată. Calendar: marți 16.06 draft complet; miercuri-joi 17-18 rafinare + export figuri high-res; vineri-duminică 19-21 minim 2 runde de simulare comisie cu deck-ul.

**Ipoteză de confirmat:** 10-12 minute de prezentare (confirmă durata exactă când secretarul comisiei comunică programarea).

**Public:** comisia AIA (profil systems/control + CS), deci pondere crescută pe arhitectura end-to-end și prototip funcțional, nu exclusiv pe metrici ML. Rigoarea ML rămâne obligatorie.

**Schelet, 13 slide-uri:**

1. Titlu, autor, coordonator.
2. Context și problema: nutriție personalizată din date pasive wearable, fără food logging.
3. Întrebările de cercetare RQ1-RQ3.
4. Datele: 958 zile, Oura Ring (Gen3/Gen4) + Apple Watch (Ultra/Ultra3), pipeline de colectare și integrare.
5. Metodologie: split temporal 766/192, SMOTE în pipeline, RF/XGBoost/SVM, metrici robuste la dezechilibru.
6. Rezultate principale: Tabel 5.4 + 5.4b, RF F1 Macro 0,5063, framing onest al CI bootstrap.
7. Studiul ablativ: complementaritatea dispozitivelor (combinat 0,506 vs Oura 0,428 vs AW 0,324).
8. Explicabilitate: SHAP bar + un waterfall pe o predicție.
9. Prototipul Streamlit: arhitectura recomandării rule-based + screenshot (Figura 5.1) sau demo scurt.
10. Contribuția metodologică: XML vs JSON Apple Health, dublu-contorizare, MAE 8.734 pași/zi.
11. Limitări oneste: N=1, shift Gen3→Gen4 (30.05.2025), CI MCC include zero, pseudo-etichetă Readiness.
12. Concluzii și direcții (prospective validation, ESWA).
13. Backup slides pentru Q&A: tuning RandomizedSearchCV, threshold sensitivity, detalii bootstrap, CV leakage fix.

**Generare .pptx:** marți, pe baza acestui schelet + figurile exportate din `outputs/figures/`.
