# Plan de revizie KBS (Knowledge-Based Systems)

Versiune: 14 septembrie 2026, actualizat 19 oct 2026 (§3, §8). Autor: Adrian Lazar, cu the LLM assistant ca partener de planificare.
Scop: input pentru proiectul de lucru cu codul disertației. Se continuă discuția de acolo, fază cu fază.

## 1. Context

- Manuscris: "AI System for Personalized Nutrition and Diet Recommendations Based on Wearable and mHealth Data" (Lazar & Buiu), trimis de prof. Buiu la Knowledge-Based Systems (Elsevier) după desk rejection la ESWA.
- Decizie editor (13 sept 2026): **Revision** (KBS nu distinge minor/major). Termen afișat: **1 decembrie 2026**. Scrisoarea menționează și "21 days for first revision"; data din Editorial Manager e cea care contează, de confirmat de Buiu.
- Poziția lui Buiu pe Reviewer 2: validarea nutrițională nu se poate face cu datele existente, se declară obiectiv viitor. Mergem înainte cu revizia.
- Buget de timp: ~8 ore/săptămână, 11 săptămâni, ~88 ore total. Planul e dimensionat pentru ~80 ore, cu ~8 ore rezervă.
- Pipeline: set de scripturi, freeze din 31 martie 2026, reproduce rezultatele din disertație. Pragurile de etichetare sunt (probabil) parametri, de verificat.

## 2. Reguli de lucru (nenegociabile)

1. **Cifrele canonice ale modelului raportat nu se schimbă.** Sursa de adevăr rămâne `CIFRE_CANONICE.md`. Toate analizele noi sunt aditive.
2. **Fiecare cifră nouă intră în `revision/CIFRE_REVIZIE.md`** cu: valoare, script care a produs-o, seed, data rulării. Nicio cifră nu intră în manuscris fără să fie în acest fișier.
3. **Test set-ul (192 zile) se atinge o singură dată per experiment.** Orice tuning sau selecție se face cu temporal CV exclusiv pe train (766 zile).
4. **Rezultatele neplăcute se raportează.** Calibrare slabă, diferențe nesemnificative, F1 instabil la praguri: toate intră în lucrare. Un reviewer convins de onestitate e mai valoros decât o cifră cosmetizată.
5. **Reproductibilitate:** seed 42 peste tot, `requirements.txt` înghețat, fiecare script rulabil din linia de comandă fără intervenție manuală.
6. Manuscris: engleză, APA 7, punct zecimal, virgulă la mii, fără em dash.

## 3. Cifre canonice de referință (din CIFRE_CANONICE.md)

- Dataset: 1,009 zile colectate; 958 utile; split temporal 766/192 (8 Active în test); seed 42; hash prefix 63bc3a2915bae864.
- Etichete T3: Rest (readiness<60 SAU sleep<60); Active (readiness>=85 ȘI sleep>=75); Normal restul. Distribuție 202/710/46.
- F1 Macro (19 feat): RF 0.5063, XGB 0.4760, SVM 0.4347. Accuracy: 64.1% / 68.8% / 63.0%. MCC: 0.1416 / 0.1503 / 0.0956. QWK: 0.2551 / 0.1805 / 0.1709.
- Bootstrap 95% CI (RF): Macro F1 [0.373, 0.610]; MCC [-0.0045, 0.2843].
- Ablație RF: Oura-only 0.428 (Active 0.286); Watch-only 0.324 (Active 0.100); Both 0.506 (Active 0.444).
- Baseline majority (DummyClassifier): F1 0.281, accuracy 72.9%.
- Modele: RF (100 arbori, depth 10, min leaf 5); XGB (100 est, depth 6, lr 0.1); SVM (RBF, C=1, gamma=scale). SMOTE k=3.
- Safety layer: prag confidence 0.50; eroare calorică max 200 kcal; coverage 80% în manuscris, 65.6% pe modelul canonic (vezi §8 punctul 5).

## 4. Maparea cerințelor reviewerilor pe task-uri

| ID | Cerință | Task | Fază |
|---|---|---|---|
| R1.1 | Justificare N=1, generalizare | Text în Discussion | F4 |
| R1.2 | LightGBM, CatBoost, TabNet, FT-Transformer | T2.1 | F2 |
| R1.3 | Teste statistice între modele | T1.2 | F1 |
| R1.4 | Hyperparameter search sistematic | T2.2 | F2 |
| R1.5 | Validarea pseudo-etichetelor | T3.1, T3.2 | F3 |
| R1.6 | Stabilitate SHAP | T1.4 | F1 |
| R1.7 | Ablație pe grupuri de feature-uri | T1.5 | F1 |
| R1.8 | Metrici pentru imbalance | T1.1 | F1 |
| R1.9 | Calibrare | T1.3 | F1 |
| R1.10 | Robustețe temporală | T3.3 | F3 |
| R2.a | Validare nutrițională / EE de referință | Concesie explicită (Buiu) | F4 |
| R2.b | Persistence baseline | T0.4 | F0 |
| R2.c | Venue fit | Paragraf cu ancore din aims and scope KBS | F4 |

## 5. Faze

### Faza 0: poarta de reproductibilitate (14-20 sept, ~8h)

**T0.1 Rerun complet.** Rulezi pipeline-ul înghețat cap-coadă.
Pass: F1 0.5063, accuracy 64.1%, MCC 0.1416 pentru RF, la a 4-a zecimală. Fail: te oprești și repari înainte de orice altceva.

**T0.2 Structură de revizie.** Creezi `revision/` cu `CIFRE_REVIZIE.md`, `scripts/`, `results/`. Salvezi pe disc predicțiile și probabilitățile pe test pentru RF, XGB, SVM (`results/test_predictions_{model}.csv`: date, y_true, y_pred, p_rest, p_normal, p_active). Faza 1 lucrează exclusiv pe aceste fișiere.

**T0.3 Praguri ca parametri.** Verifici că pragurile de etichetare (60/85/75) sunt argumente ale funcției de labeling. Dacă sunt hardcodate, le scoți în parametri și verifici că T0.1 dă aceleași cifre.

**T0.4 Persistence baseline (R2.b).** Predicția pentru ziua t = eticheta reală la t-1, pe cele 192 zile de test (prima zi folosește ultima zi din train). Metrici identice cu RF: macro F1, F1/clasă, accuracy, MCC, bootstrap 95% CI (1000 resamples).
Interpretare: persistence mult sub RF = argument nou; apropiat = discuție onestă; peste RF = reformulare a contribuției. Rezultatul decide tonul Fazei 4.
Ipoteză: F1 persistence sub RF, pentru că Active (46 zile în 3 ani) apare rar în zile consecutive.

### Faza 1: analize pe predicțiile existente (21 sept - 4 oct, ~16h)

Fără reantrenare (excepție T1.4, T1.5). Livrabil intermediar: pe 4 oct răspunzi la 5 din 10 puncte, asigurare pentru scenariul "21 de zile".

**T1.1 Metrici suplimentare (R1.8).** Pentru RF, XGB, SVM, persistence, majority: balanced accuracy, macro ROC-AUC (one-vs-rest), macro PR-AUC, G-mean, Cohen's kappa (kappa simplu, pe lângă QWK existent). Bootstrap CI pentru fiecare. Un tabel.

**T1.2 Teste statistice (R1.3).** McNemar pe perechi (RF-XGB, RF-SVM, XGB-SVM) pe corect/incorect per zi. Bootstrap paired difference pe macro F1 (aceleași resamples pentru ambele modele, CI al diferenței). Raportezi p-values și CI ale diferențelor.
Ipoteză: nicio diferență semnificativă la 0.05. Se raportează ca atare; consistent cu CI-urile largi.

**T1.3 Calibrare (R1.9).** Pentru RF (modelul din safety layer): reliability diagram (10 bins), ECE, Brier score multiclass. Evaluezi separat calibrarea pe clasa Active. Testezi isotonic regression / Platt antrenat pe un fold din train (nu pe test). Decizie: dacă ECE scade clar, safety layer-ul folosește probabilități calibrate și raportezi impactul pe coverage și eroare calorică; dacă nu, raportezi calibrarea brută și discuți limita.

**T1.4 Stabilitate SHAP (R1.6).** 100 de modele RF antrenate pe bootstrap samples din train (aceiași hiperparametri). Pentru fiecare: ranking SHAP (mean |SHAP|). Raportezi: frecvența cu care aw_hr_mean e pe locul 1; top-5 overlap (Jaccard) față de ranking-ul canonic; Kendall tau mediu între ranking-uri, cu CI. O figură (boxplot al rangului per feature).

**T1.5 Ablație pe grupuri (R1.7).** Leave-one-group-out si keep-one-group-in cu RF, 6 grupuri definite pe mecanism fiziologic (composite scores, heart rate, HRV, sleep architecture, activity/energy, respiration/thermal/SpO2), aceeasi procedura ca ablatia pe dispozitive, paired bootstrap fata de modelul complet pe matricea seed 42. Permutation importance: sacrificat (§8.11); se adauga doar daca ramane timp dupa F3.

### Faza 2: modele și tuning (5-18 oct, ~16h)

**T2.1 Modele suplimentare (R1.2).** LightGBM și CatBoost în același pipeline (split, SMOTE k=3, seed 42), parametri default apoi tunați (T2.2). TabNet: o singură rulare cu parametri rezonabili, raportată cu caveat de scară (766 rânduri). FT-Transformer: nu se rulează; se argumentează în răspuns că arhitecturile transformer pe tabular nu sunt adecvate sub ~1,000 de rânduri, cu 1-2 citări (de verificat: Grinsztajn et al. 2022, "Why do tree-based models still outperform deep learning on tabular data", NeurIPS).
Buget: dacă TabNet consumă peste 3 ore, se renunță și se argumentează ca FT-Transformer.

**T2.2 Hyperparameter search (R1.4).** Random search (30-50 configurații per model) pentru RF, XGB, SVM, LightGBM, CatBoost, cu TimeSeriesSplit (3-4 folduri) pe train, metrica macro F1. Test set atins o singură dată cu configurația câștigătoare. Tabel în anexă: spațiu de căutare, cea mai bună configurație, F1 CV, F1 test.
Regulă: dacă ierarhia pe test se schimbă față de default, raportezi ambele și discuți; nu alegi versiunea convenabilă.
Notă: un tuning există deja ca robustness check (nu a schimbat ierarhia); T2.2 îl înlocuiește cu o procedură raportabilă.

### Faza 3: punctele grele (19 oct - 1 nov, ~16h)

**T3.1 Sensibilitate la praguri (R1.5, R2 parțial).** Grilă: prag Rest în {55, 60, 65}, prag Active readiness în {80, 85, 90}, sleep fix 75 (opțional și sleep {70, 75, 80} dacă timpul permite). Pentru fiecare combinație: regenerezi etichetele, aceeași RF, același split. Raportezi: distribuția claselor, macro F1, F1 Active, F1 persistence, top-3 SHAP. Întrebări: RF rămâne peste persistence și majority în toate celulele? aw_hr_mean rămâne dominant? Cât variază F1?

**T3.2 Agreement cu indicatori independenți (R1.5).** Pentru fiecare clasă, statistici descriptive (medie, mediană, IQR) pe variabile care NU intră în definiția etichetei: HRV medie nocturnă, HR de repaus, active energy (Watch), temperatură deviație (Oura), pași. Test Kruskal-Wallis + post-hoc între clase. Dacă zilele Active au HRV mai mare și HR de repaus mai mic decât Rest, etichetele au sens fiziologic. Un tabel, un paragraf.

**T3.3 Robustețe temporală (R1.10).** Expanding-window: 3-4 folduri temporale pe cele 958 zile (de ex. train până la sfârșitul lui 2024, test S1 2025; train până la S1 2025, test S2 2025; etc.). RF cu parametrii canonici. Per fold: macro F1, F1 Active, numărul de Active în test, bootstrap CI. CI-urile vor fi largi; se raportează explicit.
Dacă timpul presează: 2 folduri și o frază care spune de ce nu mai multe.

### Faza 4: scriere (2-15 nov, ~16h)

**T4.1 Manuscris.** Modificări: tabel metrici extins (T1.1), paragraf teste statistice (T1.2), subsecțiune calibrare + figură (T1.3), figură stabilitate SHAP (T1.4), tabel ablație pe grupuri (T1.5), tabel modele + tuning în anexă (T2), subsecțiune validare etichete (T3.1, T3.2), tabel expanding-window (T3.3), persistence baseline în tabelul principal (T0.4). Discussion: paragraf generalizare N=1 (R1.1); limitare explicită validare nutrițională și EE de referință, formulată ca obiectiv viitor (R2.a, poziția lui Buiu). Introduction: paragraf de poziționare cu ancore din aims and scope KBS: "recommender systems and E-service personalization", "intelligent decision support systems, prediction systems and warning systems", "applications in healthcare" (R2.c).
Abstract: actualizat dacă intră persistence baseline sau o cifră nouă relevantă; limită de cuvinte verificată în KBS Guide for Authors.

**T4.2 Response letter.** Format Elsevier: pentru fiecare comentariu, textul comentariului, răspunsul, modificarea și localizarea (secțiune, pagină, tabel/figură). Ton: factual, fără apologie, concesii clare unde e cazul. Reviewer 2 primește mulțumiri pentru observația despre persistence baseline (e o omisiune reală).

**T4.3 CIFRE_CANONICE.md** actualizat cu secțiune "Revision 1", din `CIFRE_REVIZIE.md`.

### Faza 5: verificare și submisie (16 nov - 1 dec, ~8h + rezervă)

1. Verificare programatică: toate cifrele din manuscris vs CIFRE_CANONICE.md (script existent sau nou).
2. Review Buiu: manuscris + response letter, cu minimum o săptămână pentru feedback.
3. Referințe regenerate din Zotero (APA 7), inclusiv citările noi (T2.1).
4. Figuri hi-res separate pentru figurile noi.
5. Submisie cu cel puțin 5 zile înainte de 1 decembrie.

## 6. Calendar sintetic

| Săptămâna | Interval | Fază | Livrabil |
|---|---|---|---|
| 1 | 14-20 sept | F0 | Reproducere confirmată, praguri parametrizate, persistence baseline |
| 2-3 | 21 sept - 4 oct | F1 | Metrici, teste statistice, calibrare, SHAP stabilitate, ablație grupuri |
| 4-5 | 5-18 oct | F2 | LightGBM, CatBoost, (TabNet), tuning raportat |
| 6-7 | 19 oct - 1 nov | F3 | Sensibilitate praguri, agreement indicatori, expanding-window |
| 8-9 | 2-15 nov | F4 | Manuscris revizuit, response letter, CIFRE actualizate |
| 10-11 | 16 nov - 1 dec | F5 | Verificare, review Buiu, submisie |

Puncte de control cu Buiu: după F0 (rezultat persistence, confirmare termen), după F3 (cifre complete, înainte de scriere), după F4 (draft complet).

## 7. Riscuri

- **T0.1 nu reproduce.** Cel mai grav; se rezolvă înainte de orice.
- **Persistence baseline apropiat de RF.** Contribuția se reformulează spre sistem + explicabilitate, nu spre performanță predictivă. Se decide cu Buiu la primul punct de control.
- **Tuning-ul schimbă ierarhia.** Se raportează ambele; nu se ascunde.
- **F3 alunecă.** T3.3 se reduce la 2 folduri; T3.1 se reduce la grila 3x3 fără sleep.
- **Termenul real e 21 de zile.** Pe 4 oct există pachetul F0+F1 (6 puncte acoperite) și se negociază extensie pentru rest. De aceea F1 e înaintea F2.
- **Buget de ore depășit.** Ordinea de sacrificiu se aplica din start (§8.11): TabNet, permutation importance (T1.5), CatBoost, T2.2 redus la 20 configuratii si 3 folduri, grila sleep din T3.1, T3.3 la 2 folduri. Protejate: T1.4, T3.1 3x3, T3.2, T3.3 cu minimum 2 folduri, LightGBM default. F3 are prioritate peste F2.

## 8. Întrebări deschise (actualizat 19 oct 2026)

Rezolvate:
- (fost 2) Calibrarea nu intră în safety layer: decizie T1.3, 19 oct (sub probabilități calibrate sistemul emite doar Normal; vezi CIFRE_REVIZIE, note T1.3). Se raportează ca rezultat și ca limită.
- (fost 11) Decis 19 oct: ordinea de sacrificiu se aplica din start; lista in §7. De comunicat lui Buiu in update (§8.3).

Deschise:
1. Confirmarea datei de 1 decembrie în Editorial Manager (Buiu).
2. Fișierul exact submis la KBS (ipoteză: ESWA_Manuscript_Anonymized_v2.docx); de cerut lui Buiu.
3. Update Buiu netrimis; conținut: F0 + T1.1-T1.3 închid pe cifre R1.3, R1.8, R1.9, R2.b; corecția la 4.9 (punctul 5).
4. Limita de cuvinte a abstractului la KBS (Guide for Authors KBS, nu ESWA).
5. Errata CIFRE_CANONICE §7 la T4.3: tabelul provine din pipeline-ul XML 16 features (confidence_threshold.main(), commit f7cc4ca), nu din RF-ul canonic și nici din JSON 16f (care dă 0.4697). Cifre corecte pe canonic la 0.50: coverage 65.6% (66 overridden), eroare medie 58.3 kcal (71.9 la prag 0), max 200 kcal (se menține). Evidență: revision/scripts/_check_t13_safety_layer.py. Auditul de proveniență (19 oct): celelalte 15 secțiuni sunt pe modelul canonic.
6. Manuscris 4.9 corectat la T4.1 (coverage, eroare medie; „76 la 69" devine „71.9 la 58.3"); frază de dezvăluire în response letter („during the revision the authors identified and corrected...").
7. Eroarea calorică medie e minimizată de „mereu Normal" (54.2 kcal); nu se raportează singură, ci cu stările emise și F1 Active. Legătură directă cu R2.a (EE de referință).
8. demo_end_to_end_v2.py șterge și reantrenează rf_model.joblib la fiecare rulare, fără hash check pe dataset; azi coincide cu modelul înghețat (sha256 b8113662). De rezolvat înainte de repo-ul public cu DOI (F5).
9. Repo-ul public cu DOI (Zenodo): anonimizarea profilului din nutrition_recommender.py rămâne de făcut înainte.
10. (corectat 19 oct, după T3.3) Durata dataset-ului: 10 iunie 2023 - 30 martie 2026, aproximativ 2 ani și 9 luni (1,009 zile colectate / 958 utile; min / max verificat pe coloana date a dataset-ului etichetat); 18 septembrie 2025 este sfârșitul train-ului canonic, nu al dataset-ului (eroare din versiunea anterioară a acestui punct). „Three years" din manuscris se înlocuiește cu „approximately 33 months" sau „nearly three years" la F4.
12. Working tree-ul repo-ului are fișiere nestaged care nu țin de revizie: _referinte_upb/, eswa_architecture.py, outputs/figures/eswa_architecture/ și ștergerea disertatie_audit.zip. Înainte de repo-ul public cu DOI (F5): de decis ce se comite, ce intră în .gitignore, ce se șterge definitiv. Nu se atinge în F1-F4.
13. (adăugat 19 oct, după T3.2) Semantica claselor: T3.2 arată că Active (readiness >= 85, sleep >= 75) sunt zile de recuperare cu activitate REDUSĂ (Cliff's delta Active vs Rest -0.42 pe calorii active Watch), iar Rest sunt zile de strain; numele claselor sunt prescriptive. De decis cu Buiu la checkpoint-ul post-F3: păstrăm numele Rest / Normal / Active cu definiția prescriptivă explicită în Methods (recomandat; nimic nu se schimbă în cifre, figuri, cod) sau redenumim (ar atinge tabele, figuri, demo, Streamlit). Offsetul +200 / -200 kcal rămâne prescriptiv și nevalidat; intră la limite cu cifră (R2.a). Detalii în revision/TEXT_DECISIONS.md, secțiunea T3.2.