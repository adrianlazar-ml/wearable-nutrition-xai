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
