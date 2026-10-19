# Cifre canonice: Disertație Adrian Lazar 2026

Generat automat din `outputs/tables/` si `models/frozen_meta_20260331.json`.

**Cutoff date:** 2026-03-31  |  **Last day in dataset:** 2026-03-30  |  **Dataset hash:** `63bc3a2915bae864`

---

## Cuprins

1. [Dataset sizes](#1-dataset-sizes)
2. [Distribuții clase](#2-distribuții-clase)
3. [Rezultate modele principale (RF / SVM / XGB)](#3-rezultate-modele-principale)
4. [Ablation study (Oura / AW / Combined)](#4-ablation-study)
5. [Threshold sensitivity (T1–T7)](#5-threshold-sensitivity)
6. [SMOTE comparison](#6-smote-comparison)
7. [Confidence threshold](#7-confidence-threshold)
8. [Hyperparameter tuning](#8-hyperparameter-tuning)
9. [Modele extra (LR, LightGBM)](#9-modele-extra)
10. [MCC & QWK](#10-mcc--qwk)
11. [Calibration (Brier)](#11-calibration)
12. [Rolling features (respins)](#12-rolling-features-respins)
13. [Circularity sanity check](#13-circularity-sanity-check)
14. [Model config & features](#14-model-config--features)
15. [Reproducibilitate](#15-reproducibilitate)
16. [Deltas 16f vs 19f (Tabel 5.4)](#16-deltas-16f-vs-19f-tabel-54)

---

## 1. Dataset sizes

### Zile calendaristice per sursa

| Sursa | N zile |
|---|---|
| Oura Ring (combinat) | **1.009** |
| Apple Watch JSON (Health Auto Export) | **1.083** |
| Dataset combinat (left join pe date) | **1.009** |

**Perioada acoperita:** 2023-06-09 -> 2026-03-30 (pipeline rulat 19.04.2026, cutoff 2026-03-31).

### Dataset final pentru model

- **Total zile utile** (dupa lag-1 + dropna label): **958**
- **Train** (80% cronologic): **766** zile
- **Test intern** (20% cronologic): **192** zile
  - Range: `2025-09-19` -> `2026-03-30`


## 2. Distribuții clase

Clasele sunt indexate 0=Rest, 1=Normal, 2=Active.

### Distributie pe dataset complet (N=958)

| Clasa | N | % |
|---|---|---|
| Rest | **202** | **21,1%** |
| Normal | **710** | **74,1%** |
| Active | **46** | **4,8%** |
| **Total** | **958** | **100%** |

### Distributie train (pre-SMOTE, N=766)

| Clasa | N |
|---|---|
| Rest | **158** |
| Normal | **570** |
| Active | **38** |

### Distributie train (post-SMOTE k=3)

Echilibrat la clasa majoritara:

- Rest: **570**
- Normal: **570**
- Active: **570**

### Distributie test intern (N=192)

| Clasa | N |
|---|---|
| Rest | **44** |
| Normal | **140** |
| Active | **8** |


## 3. Rezultate modele principale

Rezultate pe 19 features lag-1, sursa JSON, test temporal 20%.

| Model | F1 Macro | Accuracy | F1 Rest | F1 Normal | F1 Active | CV F1 (+/-) |
|---|---|---|---|---|---|---|
| Random Forest | **0,5063** | 64,1% | 0,32 | 0,75 | 0,44 | 0,3369 (+/- 0,0377) |
| XGBoost | **0,4760** | 68,8% | 0,29 | 0,80 | 0,33 | 0,3343 (+/- 0,0261) |
| SVM (RBF) | **0,4347** | 63,0% | 0,30 | 0,75 | 0,25 | 0,3560 (+/- 0,0348) |

**Baseline DummyClassifier (most_frequent):** F1 Macro = 0,281, Accuracy = 72,9%

### Support pe test set

- Rest: **44**
- Normal: **140**
- Active: **8**
- Total: **192**


## 4. Ablation study

Configuratii RF, 19 features (toate cazurile folosesc lag-1, SMOTE k=3).

| Configuratie | N features | F1 Macro | Accuracy | F1 Rest | F1 Normal | F1 Active |
|---|---|---|---|---|---|---|
| Doar Oura (11 feat) | 11 | **0,428** | 62,0% | 0,247 | 0,752 | 0,286 |
| Doar Apple Watch (8 feat) | 8 | **0,324** | 47,4% | 0,269 | 0,604 | 0,100 |
| Ambele dispozitive (19 feat) | 19 | **0,506** | 64,1% | 0,322 | 0,753 | 0,444 |

### Delta F1 Macro vs combined

- Combined - Oura-only = +7,8 pp
- Combined - AW-only = +18,2 pp
- F1 Active: AW-only 0,100 -> Combined 0,444 (~4.4x)


## 5. Threshold sensitivity

Praguri pentru generarea labels (Rest/Normal/Active). RF, 19 features.

| Set | Rest readiness/sleep | Active readiness/sleep | Rest | Normal | Active | F1 Macro |
|---|---|---|---|---|---|---|
| T1: 55/55-80/70 | <55/<55 | >=80/>=70 | 97 | 703 | 158 | **0,325** |
| T2: 55/55-85/75 | <55/<55 | >=85/>=75 | 97 | 815 | 46 | **0,505** |
| T3: 60/60-85/75 *** | <60/<60 | >=85/>=75 | 202 | 710 | 46 | **0,506** |
| T4: 60/60-90/80 | <60/<60 | >=90/>=80 | 202 | 753 | 3 | **0,687** |
| T5: 65/65-85/75 | <65/<65 | >=85/>=75 | 360 | 552 | 46 | **0,456** |
| T6: 65/65-90/80 | <65/<65 | >=90/>=80 | 360 | 595 | 3 | **0,682** |
| T7: Q25/Q75 (64/66-77/77) | <64/<66 | >=77/>=77 | 356 | 450 | 152 | **0,385** |

**T3 selectat** ca optim. T4/T6 marcati cu `*` au F1 mai mare formal dar <5 eșantioane Active (inutile).


## 6. SMOTE comparison

5 strategii de gestionare a dezechilibrului. RF, 19 features.

| Strategie | F1 Macro | Accuracy | F1 Rest | F1 Normal | F1 Active |
|---|---|---|---|---|---|
| Baseline (nimic) | **0,280** | 72,4% | 0,000 | 0,840 | 0,000 |
| Cost-sensitive | **0,376** | 71,4% | 0,077 | 0,830 | 0,222 |
| UnderSampler | **0,325** | 46,9% | 0,248 | 0,620 | 0,108 |
| SMOTE k=3 * | **0,506** | 64,1% | 0,322 | 0,753 | 0,444 |
| SMOTE k=5 | **0,451** | 60,4% | 0,244 | 0,734 | 0,375 |

**SMOTE k=3 ales** ca optim. k=5 produce F1 inferior (0,451).


## 7. Confidence threshold

Safety layer: daca max(predict_proba) < threshold, fallback la Normal.

| Threshold | F1 Macro | Accuracy | Coverage | N overridden | Mean cal error (kcal) | Max cal error |
|---|---|---|---|---|---|---|
| 0.0 [baseline] | 0,520 | 62,0% | 100,0% | 0 | 76,0 | 200 |
| 0.45 | 0,511 | 62,0% | 93,8% | 12 | 76,0 | 200 |
| 0.5 [RECOMANDAT] | 0,522 | 65,6% | 80,2% | 38 | 68,8 | 200 |
| 0.55 | 0,400 | 67,7% | 56,2% | 84 | 64,6 | 200 |
| 0.6 | 0,376 | 71,4% | 29,2% | 136 | 57,3 | 200 |
| 0.65 | 0,351 | 71,4% | 17,2% | 159 | 57,3 | 200 |
| 0.7 | 0,278 | 71,4% | 8,9% | 175 | 57,3 | 200 |

**Threshold 0.5 ales in cod** (demo_end_to_end_v2 + streamlit).
Notat: flag-ul 'RECOMANDAT' in CSV la 0.6 este vestigial si ar trebui actualizat (vezi raport sanity check).


## 8. Hyperparameter tuning

RandomizedSearchCV, 50 iteratii, 5-fold TimeSeriesSplit CV, metrica = F1 Macro.

### Rezultate tuning

| Model | Default F1 CV | Tuned F1 CV | Delta (pp) |
|---|---|---|---|
| RF | 0,3369 | 0,3618 | **+2.49** |
| SVM | 0,3560 | 0,4008 | **+4.47** |
| XGBoost | 0,3343 | 0,4030 | **+6.87** |

**Interpretare:** Tuning post-fix CV leakage: RF +2,49pp, SVM +4,47pp, XGB +6,87pp. Dupa tuning, pe CV F1 Macro XGB (0,403) si SVM (0,401) depasesc RF (0,362), dar RF ramane best pe test F1 Macro (0,506) + QWK. Concluzia 'RF best overall' e nuantata: best pe metrica primara (test F1 Macro), competitiv pe CV.

### Best params gasiti

- **RF:** max_depth=20.0, max_features=log2, min_samples_leaf=7.0, min_samples_split=15.0, n_estimators=152.0
- **SVM:** C=6.111150117432088, gamma=0.01, kernel=rbf
- **XGBoost:** max_depth=6.0, n_estimators=210.0, colsample_bytree=0.6905983100791752, learning_rate=0.2035518371228349, min_child_weight=9.0, subsample=0.759028808435009


## 9. Modele extra

Extensie: LR liniar si LightGBM comparati cu RF/XGB.

| Model | F1 Macro | Accuracy | CV F1 mean | CV F1 std | Train time (s) |
|---|---|---|---|---|---|
| Logistic Regression | **0,342** | 44,3% | 0,3658 | 0,0517 | 0,024 |
| Random Forest (ref) | **0,506** | 64,1% | 0,3369 | 0,0377 | 0,270 |
| XGBoost (ref) | **0,476** | 68,8% | 0,3343 | 0,0261 | 0,191 |
| LightGBM | **0,471** | 70,3% | 0,3478 | 0,0157 | 0,109 |

**LR F1=0,342** confirma non-liniaritatea problemei. **LightGBM F1=0,471** similar cu XGBoost, RF ramane best.


## 10. MCC & QWK

Metrici robuste la dezechilibru si ordinalitate.

| Model | F1 Macro | MCC | QWK | Accuracy |
|---|---|---|---|---|
| Random Forest | **0,5063** | **0,1416** | **0,2551** | 0,6406 |
| XGBoost | **0,4760** | **0,1503** | **0,1805** | 0,6875 |
| SVM (RBF) | **0,4347** | **0,0956** | **0,1709** | 0,6302 |

**MCC** (Matthews Correlation Coefficient): robust la dezechilibru. **QWK** (Quadratic Weighted Kappa): captureaza ordinalitatea Rest<Normal<Active.


## 11. Calibration

Brier Score per clasa (mai mic = mai bine calibrat). RF, 19 features.

| Model | Rest | Normal | Active | Average |
|---|---|---|---|---|
| Random Forest | 0,200 | 0,259 | 0,050 | **0,170** |
| XGBoost | 0,207 | 0,233 | 0,035 | **0,158** |
| SVM (RBF) | 0,208 | 0,230 | 0,031 | **0,156** |

Brier < 0,20 considerat bine calibrat. Valori confirma confidence threshold 0.50.


## 12. Rolling features (respins)

Baseline lag-1 vs extensie cu rolling means (3, 7, 14 zile).

| Config | N features | F1 Macro | CV F1 mean |
|---|---|---|---|
| Baseline (19 feat) | 19 | **0,506** | 0,3369 |
| Extended (36 feat) | 36 | **0,330** | 0,3397 |

**Rolling features DAUNEAZA:** -17,6 pp pe test,
Justificare: pe N=958 (766 train) cele 36 features produc overfitting. Learning curves nu satureaza - inca cresc la ~131 zile (endpoint-ul range-ului TimeSeriesSplit pe train), deci semnalul e limitat de volumul de date, nu de numarul de features.


## 13. Circularity sanity check

Sanity check: cat contribuie features 'potential circulare' (derivate din Oura scores) la clasificare?

| Variant | N features | F1 Macro | CV F1 mean |
|---|---|---|---|
| A (BASELINE: toate 19 features) | 19 | **0,5063** | 0,3369 |
| B (SEMI: fara readiness+sleep_score) | 17 | **0,5148** | 0,3409 |
| C (STRICT: doar fiziologice brute) | 14 | **0,4697** | 0,3570 |

**Drop baseline -> strict (doar fiziologice brute):** -3,7 pp,
**Semi-strict (fara readiness+sleep) creste marginal:** +0,9 pp,

**Interpretare:** circularitatea nu e o problema majora. Modelul extrage semnal din features fiziologice brute (HRV, RHR, HR, pasi, calorii), nu doar din derivate.


## 14. Model config & features

### Hiperparametri finali (RF, model de productie)

- `algorithm` = **RandomForestClassifier**
- `n_estimators` = **100**
- `max_depth` = **10**
- `min_samples_leaf` = **5**
- `random_state` = **42**

### Preprocessing

- imputer: `SimpleImputer(strategy='median')`
- smote: `SMOTE(random_state=42, k_neighbors=3)`
- applied_on: `train only, after imputation`

### Scheme de etichetare T3

- Rest: `readiness_score < 60 OR sleep_score < 60` -> label **0**
- Active: `readiness_score >= 85 AND sleep_score >= 75` -> label **2**
- Normal: restul zilelor -> label **1**

### Features (19 lag-1)

**Oura (11):** `readiness_score_lag1`, `sleep_score_lag1`, `hrv_mean_lag1`, `hrv_balance_lag1`, `recovery_index_lag1`, `rhr_lowest_lag1`, `rhr_avg_lag1`, `activity_score_lag1`, `steps_oura_lag1`, `respiratory_rate_lag1`, `temp_deviation_lag1`

**Apple Watch (8):** `aw_hr_mean_lag1`, `aw_hrv_mean_lag1`, `aw_steps_lag1`, `aw_active_calories_lag1`, `aw_spo2_lag1`, `aw_sleep_deep_hours_lag1`, `aw_sleep_rem_hours_lag1`, `aw_resp_rate_lag1`


## 15. Reproducibilitate

Pentru dovada reproducibilitatii (util daca comisia AI/ML intreaba):

- **Frozen tag:** `20260331`
- **Frozen at (UTC):** `2026-04-19T12:26:59.968593Z`
- **Cutoff date** (data <= care s-au luat datele): `2026-03-31`
- **Last day in dataset** (efectiv): `2026-03-30`
- **Train internal end** (ultima zi din 80% train): `2025-09-18`
- **Test internal range:** `2025-09-19` -> `2026-03-30`
- **Dataset SHA256 (first 16 char):** `63bc3a2915bae864`
- **Metadata file:** `models/frozen_meta_20260331.json`

### Versiuni librarii

- python: `3.11.9`
- platform: `Windows-10-10.0.26200-SP0`
- sklearn: `1.8.0`
- imblearn: `0.14.1`
- numpy: `2.4.3`
- pandas: `2.3.3`

### Artefacte salvate

- model: `models\rf_frozen_20260331.joblib`
- imputer: `models\imputer_frozen_20260331.joblib`
- meta: `models/frozen_meta_20260331.json`


## 16. Deltas 16f vs 19f (Tabel 5.4)

Contributia celor 3 features suplimentare (Apple Watch sleep deep/REM + resp rate) la trecerea 16f -> 19f. Sursa JSON, split temporal 80/20, hiperparametri canonici (RF din frozen_meta, XGB/SVM defaults). Delta = 19f - 16f (puncte procentuale).

| Model | F1 Macro 16f | F1 Macro 19f | Delta F1 (pp) | F1 Active 16f | F1 Active 19f | Delta Active (pp) |
|---|---|---|---|---|---|---|
| Random Forest | 0,4697 | **0,5063** | **+3,66** | 0,353 | 0,444 | +9,15 |
| XGBoost | 0,4710 | **0,4760** | **+0,50** | 0,364 | 0,333 | -3,03 |
| SVM (RBF) | 0,4588 | **0,4347** | **-2,41** | 0,250 | 0,250 | +0,00 |

**Mapare in disertatie:** coloana 'Delta F1 (pp)' corespunde randului "Δ F1 Macro vs. 16f" din Tabel 5.4. Valorile canonice de mai sus inlocuiesc deltas stale (RF +3,6 / XGB +0,5 / SVM -3,1 pp), care fusesera calculate cand SVM 19f era 0,482 (stale). Doar SVM diverge semnificativ: 16f a scazut de la ~0,513 la 0,4588 post-fix rhr_avg, deci delta SVM trece de la -3,1pp la -2,41pp. RF (+3,66pp) si XGB (+0,50pp) raman practic neschimbate.


---

## Revision 1 (KBS, 2026)

Sectiune adaugata la T4.3 (2026-10-19), dupa planul `revision/KBS_Revision_Plan.md`. Sectiunile 1-16 de mai sus NU au fost modificate (byte-identice; verificat prin cmp la adaugare, `revision/scripts/_append_t43_canonice.py`). Toate cifrele de mai jos sunt copiate din `revision/CIFRE_REVIZIE.md` (randul indicat), cu punct zecimal ca in acel fisier; ele sunt aditive si NU schimba cifrele canonice ale modelului raportat (RF 100 / 10 / 5, seed 42: F1 macro 0.5063, accuracy 0.6406, MCC 0.1416).

### (a) Errata la sectiunea 7 (Confidence threshold)

Tabelul din sectiunea 7 NU provine din modelul canonic (RF 19 features, JSON). A fost produs de `confidence_threshold.main()` din pipeline-ul XML cu 16 features (commit f7cc4ca, 2026-05-26); evidenta: `revision/scripts/_check_t13_safety_layer.py` (exit 1 intentionat, 2026-10-19) si randurile T1.3 din CIFRE_REVIZIE.md. Auditul de provenienta din 19 oct 2026 arata ca celelalte 15 sectiuni sunt pe modelul canonic.

| Marime (prag 0.50) | Sectiunea 7 (gresit, pipeline XML 16f) | Corect (RF canonic, test 192 zile) | Sursa |
|---|---|---|---|
| Coverage | 80,2% (38 suprascrise) | 65.6% (126 / 192), 66 suprascrise | CIFRE_REVIZIE T1.3, `safety_layer_calibrated.py` |
| Eroare calorica medie la 0.50 | 68,8 kcal | 58.3 kcal | CIFRE_REVIZIE T1.3 |
| Eroare calorica medie la prag 0 (fara threshold) | 76,0 kcal | 71.9 kcal | CIFRE_REVIZIE T1.3 |
| Eroare calorica maxima | 200 kcal | 200 kcal (se mentine: nicio confuzie Active-Rest) | CIFRE_REVIZIE T1.3 |
| F1 macro / accuracy / F1 Active dupa fallback; stari emise R/N/A | 0,522 / 65,6% / nu era raportat | 0.5067 / 0.7083 / 0.4000; 28 / 162 / 2 | CIFRE_REVIZIE T1.3 |

Manuscrisul revizuit (sectiunea 4.12) foloseste cifrele corecte; corectia este dezvaluita in response letter (Reviewer 1, comentariul 9). Pragul 0.50 nu a fost reales. Eroarea calorica medie nu se raporteaza singura: sistemul "mereu Normal" are 54.2 kcal tocmai pentru ca nu emite nicio recomandare Rest / Active (CIFRE_REVIZIE T1.3).

### (b) Cifre noi care intra in manuscrisul revizuit (un rand per cifra)

| Cifra | Valoare | Rand CIFRE_REVIZIE | Loc in manuscris |
|---|---|---|---|
| Persistence F1 macro [CI 95%] | 0.3039 [0.2612, 0.3501] | T0.4 | Table 4, Table 6, 4.2, 4.3 |
| Persistence F1 Rest / F1 Normal / F1 Active | 0.205 / 0.707 / 0.0000 | T0.4 | Table 4 |
| Persistence accuracy / MCC [CI 95%] / QWK | 0.5625 / -0.0566 [-0.1632, 0.0597] / 0.0055 | T0.4, T1.1 | Table 4 |
| Persistence balanced accuracy [CI] / kappa [CI] / G-mean | 0.3039 [0.2602, 0.3520] / -0.0566 [-0.1626, 0.0593] / 0.0000 | T1.1 | Table 4 |
| Majority balanced accuracy / kappa / G-mean | 0.3333 / 0.0000 / 0.0000 | T1.1 | Table 4 |
| RF balanced accuracy [CI 95%] | 0.5227 [0.3851, 0.6634] | T1.1 | Table 4, 4.2 |
| RF Cohen's kappa [CI 95%] | 0.1415 [-0.0050, 0.2924] | T1.1 | Table 4 |
| RF G-mean [CI 95%] | 0.4923 [0.2960, 0.6228] | T1.1 | Table 4 |
| RF macro ROC-AUC OvR [CI 95%] | 0.5581 [0.4199, 0.6921] (include 0.5) | T1.1 | Table 4, 4.2 |
| RF macro PR-AUC [CI 95%] | 0.4983 [0.3716, 0.6359] | T1.1 | Table 4 |
| XGB balanced acc / kappa / G-mean / ROC-AUC / PR-AUC [CI] | 0.4500 [0.3439, 0.5763] / 0.1459 [0.0065, 0.2905] / 0.3759 [0.0000, 0.5156] / 0.5892 [0.4610, 0.7022] / 0.4598 [0.3616, 0.6082] | T1.1 | Table 4 |
| SVM balanced acc / kappa / G-mean / ROC-AUC / PR-AUC [CI] | 0.4342 [0.3288, 0.5683] / 0.0955 [-0.0395, 0.2310] / 0.3824 [0.0000, 0.5355] / 0.5595 [0.4493, 0.6733] / 0.4603 [0.3565, 0.6064] | T1.1 | Table 4 |
| RF F1 macro, CI pe matricea partajata (seed 42, 1000 resamples) | 0.5063 [0.3784, 0.6130] (CI canonic [0.373, 0.610] ramane in Table 5) | T1.1 | Table 6, Table 9, Table 11 |
| McNemar RF-XGB / RF-SVM / XGB-SVM, p Holm | 0.4004 / 0.8774 / 0.3818 | T1.2 | 4.3, Supplementary Table S1 |
| Cochran's Q (RF, XGB, SVM) | Q 3.3226, df 2, p 0.1899 | T1.2 | 4.3, Supplementary Table S1 |
| Paired bootstrap F1 macro RF-XGB / RF-SVM / XGB-SVM | 0.0303 [-0.0796, 0.1561] / 0.0716 [-0.0337, 0.1775] / 0.0413 [-0.1021, 0.1647] | T1.2 | 4.3, Table 6 |
| Paired bootstrap RF minus persistence, F1 macro / balanced accuracy | 0.2024 [0.0642, 0.3173] / 0.2188 [0.0728, 0.3630] | T1.2b | Abstract, 4.3, Table 6, 5 |
| McNemar RF-persistence (neajustat) | p_cc 0.1289 | T1.2 | 4.3 |
| ECE top-label RF test, raw / Platt / isotonic [CI] | 0.1377 [0.0815, 0.2074] / 0.0361 [0.0123, 0.0996] / 0.0405 [0.0189, 0.1077] | T1.3 | 4.5, Supplementary Table S2 |
| Brier multiclass [0, 2] raw / Platt / isotonic | 0.5091 / 0.4124 / 0.4095 | T1.3 | 4.5 |
| Log loss raw / Platt / isotonic | 0.8522 / 0.6969 / 0.7111 | T1.3 | 4.5 |
| Pante Platt (Rest / Normal / Active) | 0.3970 / 0.2210 / 0.2095 | T1.3 | 4.5 |
| p_active maxim pe test, raw / Platt / isotonic | 0.6312 / 0.1029 / 0.1896 | T1.3 | 4.5 |
| Argmax calibrat: F1 macro / F1 Active (toate 192 zile Normal) | 0.2811 / 0.0000 (= majority) | T1.3 | 4.5, Limitation (5) |
| Safety layer @0.50 raw (vezi errata) | 65.6% / 66 / 58.3 kcal (71.9 la prag 0) / 200; 0.5067 / 0.7083 / 0.4000; 28 / 162 / 2 | T1.3 | 4.12 |
| aw_hr_mean pe rangul 1 SHAP, bootstrap / seeds | 88 / 100 (top-3: 99 / 100) / 100 / 100 | T1.4 | Abstract, 4.11, 5 |
| Jaccard top-5 mediu, bootstrap / seeds | 0.4904 / 0.5276 | T1.4 | 4.11 |
| Kendall tau fata de referinta, bootstrap / seeds | 0.3909 [0.1813, 0.5912] / 0.7506 [0.6257, 0.8541] | T1.4 | 4.11 |
| Frecventa in top-5 (seeds): aw_active_calories / aw_sleep_deep_hours | 0.29 / 0.86 | T1.4 | 4.11 |
| Distanta intre locurile 3-7 (mean abs SHAP) | 0.00003 - 0.0005 | T1.4 (nota) | 4.11 |
| F1 macro al celor 100 modele pe seeds (dispersie seed-to-seed, NON-canonic) | 0.5142 [0.4682, 0.5677] | T1.4 | 4.6, 4.7 (referinta pentru "zgomot") |
| LOGO G1 composite scores: F1 macro / F1 Active; delta [CI] | 0.5148 / 0.5333; -0.0085 [-0.0893, 0.0640] | T1.5 | Table 8, 4.6 |
| LOGO G2 heart rate | 0.4467 / 0.4211; +0.0596 [-0.0150, 0.1274]; F1 Rest 0.1579 | T1.5 | Table 8, 4.6 |
| LOGO G3 HRV | 0.4730 / 0.4444; +0.0333 [-0.0415, 0.1091] | T1.5 | Table 8 |
| LOGO G4 sleep architecture | 0.5141 / 0.4706; -0.0077 [-0.0795, 0.0547] | T1.5 | Table 8 |
| LOGO G5 activity / energy (singurul CI care exclude 0) | 0.3780 / 0.1250; +0.1284 [0.0164, 0.2388] | T1.5 | Table 8, 4.6, 5 |
| LOGO G6 respiration / thermal / SpO2 | 0.5150 / 0.5000; -0.0087 [-0.0862, 0.0527] | T1.5 | Table 8 |
| KOGI: cel mai bun grup singur (heart rate) / G4 / G6 Active | 0.3589 / 0.2751 (sub majority) / 0 Active corecte | T1.5 | 4.6, Figure 3 |
| T2.1 defaults, F1 macro: CatBoost / XGB / SVM / FT-T / LightGBM / TabNet | 0.4980 / 0.4760 / 0.4347 / 0.4115 / 0.4089 / 0.2911 | T2.1 | Table 6, 4.4 |
| T2.1 defaults, F1 Active: CatBoost / FT-T / LightGBM / TabNet | 0.4615 / 0.1739 / 0.2222 / 0.0000 | T2.1 | Table 6 |
| T2.1 defaults, MCC: CatBoost / FT-T / LightGBM / TabNet | 0.1296 / 0.1218 / 0.0782 / -0.0204 | T2.1 | Table 6 |
| T2.1 delta RF minus model, F1 macro [CI]: CatBoost / FT-T / LightGBM / TabNet | +0.0083 [-0.0841, +0.1131] / +0.0948 [-0.0085, +0.1912] / +0.0974 [-0.0627, +0.2335] / +0.2152 [+0.0772, +0.3345] | T2.1 | Table 6, 4.4 |
| T2.1 McNemar TabNet vs RF | p 0.0006, Holm 0.0025 | T2.1 | 4.4 |
| T2.1 CatBoost pe 5 seed-uri (NON-canonic) | 0.4980 - 0.5388, peste RF pe 4 din 5 | T2.1 | 4.4 |
| T2.1 retele pe 5 seed-uri, maxim: FT-T / TabNet | 0.4194 / 0.3599 | T2.1 | 4.4 |
| T2.1 LightGBM default: zile prezise Normal | 165 / 192 | T2.1 | 4.4 |
| T2.2 CV: castigatori si scor (RF / XGB / SVM / LightGBM / CatBoost / TabNet / FT-T) | 0.3680 / 0.3925 / 0.4078 / 0.4013 / 0.3834 / 0.4162 / 0.4348; referinta RF rang 3 din 21 (0.3595) | T2.2 | Table A1, 4.4 |
| T2.2 tuned pe test, F1 macro [CI]: RF / XGB / SVM / LightGBM / CatBoost / TabNet / FT-T | 0.5683 [0.4172, 0.6761] / 0.5043 [0.3262, 0.6176] / 0.4386 [0.3463, 0.5307] / 0.5739 [0.4165, 0.6804] / 0.5368 [0.3938, 0.6443] / 0.3412 [0.2773, 0.4093] / 0.3698 [0.2987, 0.4522] | T2.2 | Table A1, 4.4, 5 |
| T2.2 tuned minus RF canonic, F1 macro [CI]: RF / LightGBM / CatBoost | +0.0619 [+0.0028, +0.1398] / +0.0676 [+0.0001, +0.1483] / +0.0305 [-0.0325, +0.1002] | T2.2 | Table A1, 4.4 |
| T2.2 tuned minus RF tuned: LightGBM / CatBoost | +0.0056 [-0.0305, +0.0450] / -0.0314 [-0.0810, +0.0100] | T2.2 | 4.4 |
| T2.2 retele tuned minus persistence: TabNet / FT-T | +0.0373 [-0.0486, +0.1204] / +0.0659 [-0.0208, +0.1655] (includ 0) | T2.2 | 4.4, 5 |
| T2.2 McNemar tuned vs RF canonic, Holm | niciuna < 0.05 (minim 0.0682 TabNet) | T2.2 | 4.4 |
| T2.2 dispersie pe seed-uri (NON-canonic): RF / LightGBM / CatBoost tuned | 0.5085 - 0.5816 / 0.4976 - 0.5739 / 0.5220 - 0.5642 | T2.2 | Table A1, 4.4 |
| T3.1 celule comparabile (6): interval F1 macro / F1 Active; RF > persistence si > majority | 0.3718 - 0.5063 / 0.2222 - 0.6154; 6 / 6 | T3.1 | Table 9, 4.7 |
| T3.1 R55_A85 (T2): F1 macro [CI]; persistence [CI]; delta | 0.5053 [0.3574, 0.6084]; 0.3211 [0.2771, 0.3679]; +0.1842 [+0.0281, +0.2919] | T3.1 | Table 9 |
| T3.1 R65_A85 (T5) | 0.4557 [0.3536, 0.5527]; 0.3089 [0.2581, 0.3592]; +0.1468 [+0.0288, +0.2656] | T3.1 | Table 9 |
| T3.1 R55_A80 / R65_A80: delta include 0 | +0.0514 [-0.0288, +0.1347] / +0.0804 [-0.0142, +0.1738]; F1 Active 0.2500 / 0.2222 | T3.1 | 4.7 |
| T3.1 R60_A80 | F1 macro 0.4129, F1 Active 0.2581, delta +0.1127 [+0.0207, +0.2053] | T3.1 | 4.7 (grila completa in Figure S1) |
| T3.1 celule A_r = 90 (T4 / T6, 1 Active in test, neinterpretate) | 0.6870 [0.6427, 0.7347] / 0.6822 [0.6364, 0.7237]; persistence 0.3199 / 0.3288; delta +0.3671 / +0.3534; 660 / 1000 resamples | T3.1 | Table 9 (asterisc) |
| T3.2 nivel A, aw_hr_mean: mediane R / N / A; eps2; Cliff A-R [CI] | 91.95 / 86.02 / 81.32 bpm; 0.1926; -0.840 [-0.912, -0.749] | T3.2 | Table 10, 4.7, 5 |
| T3.2 aw_resp_rate | 16.42 / 15.61 / 14.85; 0.1159; -0.732 [-0.817, -0.640] | T3.2 | Table 10 |
| T3.2 aw_hrv_mean | 22.82 / 24.87 / 23.85; 0.0191; +0.129 [-0.063, +0.308] (include 0) | T3.2 | Table 10 |
| T3.2 aw_spo2 | 94.36 / 94.70 / 94.81; 0.0060; +0.113 [-0.055, +0.306] | T3.2 | Table 10 |
| T3.2 aw_active_calories (DISCORDANT) | 883.6 / 809.4 / 579.3 kcal; 0.0201; -0.420 [-0.574, -0.246] | T3.2 | Table 10, 4.7, 5 |
| T3.2 aw_steps (DISCORDANT) | 8,057 / 7,079 / 4,499; 0.0132; -0.319 [-0.479, -0.129] | T3.2 | Table 10 |
| T3.2 nivel B: rhr_lowest / hrv_mean / temp_deviation / steps_oura, Cliff A-R | -0.879 / +0.603 / -0.270 / -0.260 | T3.2 | 4.7, Supplementary Table S3 |
| T3.2 lag-1: aw_active_calories / aw_steps / steps_oura, Cliff A-R | -0.267 / -0.196 / -0.202 | T3.2 | 4.7, Supplementary Table S3 |
| T3.3 fold 4 (test Dec 2023 - Jul 2024, train 190, 9 Active): F1 macro [CI] / F1 Active / persistence [CI] / delta [CI] | 0.3605 [0.3095, 0.4069] / 0.0000 / 0.3161 [0.2735, 0.3627] / +0.0444 [-0.0317, +0.1168] | T3.3 | Table 11, 4.8, 5 |
| T3.3 fold 3 (test Jul 2024 - Feb 2025, train 382, 14 Active) | 0.3771 [0.3127, 0.4456] / 0.0741 [0.0000, 0.2222] / 0.4120 [0.3253, 0.5136] / -0.0349 [-0.1600, +0.0841] (RF sub persistence) | T3.3 | Table 11, 4.8, 5 |
| T3.3 fold 2 (test Mar 2025 - Sep 2025, train 574, 22 Active) | 0.3124 [0.2561, 0.3749] / 0.1026 [0.0000, 0.2424] / 0.3070 [0.2559, 0.3682] / +0.0054 [-0.0768, +0.0930] | T3.3 | Table 11, 4.8, 5 |
| T3.3 fold 1 (canonic) | 0.5063 [0.3784, 0.6130] / 0.4444 [0.1176, 0.7000] / 0.3039 [0.2612, 0.3501] / +0.2024 [+0.0642, +0.3173] | T3.3 | Table 11 |
| T3.3 primul feature SHAP pe fold-urile 4 / 3 / 2 / 1 | aw_hr_mean / oura_readiness_score / oura_readiness_score / aw_hr_mean | T3.3 | Table 11, 4.8, 5 |
| T3.3 pooled (768 zile; 166 / 565 / 37): F1 macro [CI] / F1 Active [CI] | 0.3932 [0.3523, 0.4349] / 0.1386 [0.0522, 0.2316] (NON-canonic, nu inlocuieste 0.5063) | T3.3 | Table 11, 4.8, 5 |
| Durata dataset-ului | 10 iunie 2023 - 30 martie 2026, aproximativ 33 de luni (1,009 zile colectate / 958 utile) | KBS_Revision_Plan sectiunea 8.7 | 3.1, Abstract |

### (c) Decizia privind modelul raportat

RF canonic (100 arbori, depth 10, min leaf 5, seed 42; F1 macro 0.5063) ramane modelul raportat al sistemului (decizie Adrian, 19 oct 2026; de confirmat cu coautorul la draft-ul din 15 nov). Motive (CIFRE_REVIZIE T2.2): pe CV configuratia canonica este pe rangul 3 din 21 pentru RF, la 0.009 sub cea selectata; castigatorii T2.2 (LightGBM tuned 0.5739, RF tuned 0.5683, CatBoost tuned 0.5368) depasesc RF canonic la punct, dar CI-urile paired au limita inferioara la 0 (+0.0001 / +0.0028), niciun McNemar nu este semnificativ dupa Holm, iar cele trei ansambluri tuned nu sunt distinse intre ele; a alege modelul dupa rezultatul pe test ar contrazice protocolul de evaluare. Rezultatele tuned intra in manuscris ca analiza de sensibilitate (4.4, Appendix A, Table A1, Discussion "Choice of classifier"). Toate analizele de explicabilitate, calibrare si robustete temporala se refera la RF canonic.
