# SPRINT 4 RECONCILIATION REPORT

**Generat:** 2026-05-28  
**Scope:** Cross-check cifre canonice vs. .docx, identificare valori stale  
**Document auditat:** `_referinte_upb/Disertatie_Lazar_Adrian_2026_Mai_CB.docx`  
**Surse canonice:** `CIFRE_CANONICE.md`, `outputs/tables/mcc_qwk_results.csv`, `outputs/tables/valori_disertatie.csv`

---

## 1. Cifre canonice (din CIFRE_CANONICE.md)

### Rezultate modele principale (19 features, JSON, test temporal 20%)

| Metrica | RF | XGBoost | SVM (RBF) | Sursa fisier |
|---|---|---|---|---|
| F1 Macro | 0,5063 | 0,4760 | 0,4347 | CIFRE_CANONICE.md s3, mcc_qwk_results.csv |
| Accuracy | 64,1% | 68,8% | 63,0% | valori_disertatie.csv |
| F1 Rest | 0,3218 | 0,2933 | 0,3023 | mcc_qwk_results.csv |
| F1 Normal | 0,7527 | 0,8013 | 0,7518 | mcc_qwk_results.csv |
| F1 Active | 0,4444 | 0,3333 | 0,2500 | mcc_qwk_results.csv |
| MCC | 0,1416 | 0,1503 | 0,0956 | mcc_qwk_results.csv |
| QWK | 0,2551 | 0,1805 | 0,1709 | mcc_qwk_results.csv |
| CV F1 Macro (mean) | 0,3369 | 0,3343 | 0,3560 | valori_disertatie.csv |
| CV F1 Macro (std) | 0,0377 | 0,0261 | 0,0348 | valori_disertatie.csv |

### Dataset split

| Metrica | Valoare | Sursa |
|---|---|---|
| Total zile utile | 958 | CIFRE_CANONICE.md s1 |
| Train (80%) | 766 | CIFRE_CANONICE.md s1 |
| Test (20%) | 192 | CIFRE_CANONICE.md s1 |
| Rest (total) | 202 | CIFRE_CANONICE.md s2 |
| Normal (total) | 710 | CIFRE_CANONICE.md s2 |
| Active (total) | 46 | CIFRE_CANONICE.md s2 |
| Dataset hash | 63bc3a2915bae864 | CIFRE_CANONICE.md s15 |

### Hyperparameter tuning gains (CV F1 Macro)

| Model | Default CV | Tuned CV | Delta (pp) | Sursa |
|---|---|---|---|---|
| RF | 0,3369 | 0,3618 | +2,49 | CIFRE_CANONICE.md s8 |
| SVM | 0,3560 | 0,4008 | +4,47 | CIFRE_CANONICE.md s8 |
| XGBoost | 0,3343 | 0,4030 | +6,87 | CIFRE_CANONICE.md s8 |

---

## 2. Cross-check CIFRE_CANONICE.md vs CSV-uri

### mcc_qwk_results.csv

| Model | F1 Macro (CSV) | MCC (CSV) | QWK (CSV) | Match CIFRE_CANONICE? |
|---|---|---|---|---|
| Random Forest | 0.5063 | 0.1416 | 0.2551 | MATCH |
| XGBoost | 0.476 | 0.1503 | 0.1805 | MATCH |
| SVM (RBF) | 0.4347 | 0.0956 | 0.1709 | MATCH |

### valori_disertatie.csv

| Model | F1 Macro (CSV) | CV F1 mean (CSV) | CV F1 std (CSV) | Match CIFRE_CANONICE? |
|---|---|---|---|---|
| RF | 0.5063239 | 0.3368553 | 0.0376529 | MATCH (rounded) |
| XGBoost | 0.4760045 | 0.3342837 | 0.0260867 | MATCH (rounded) |
| SVM | 0.4346995 | 0.3560173 | 0.0347502 | MATCH (rounded) |

**Rezultat cross-check: ZERO discrepante intre CIFRE_CANONICE.md si CSV-uri.** Toate valorile sunt consistente (diferente doar de trunchiere la 4 decimale).

---

## 3. Audit .docx: ocurente valori cautate

### 3.1. SVM F1 Macro: 0,482 (STALE) vs 0,4347 (canonic)

| Locatie | Paragraf | Context | Verdict |
|---|---|---|---|
| Cap 5, Tabel 5.4 | para#988 | Celula "0,482" in coloana SVM (RBF) | **STALE - inlocuieste cu 0,4347** |
| Cap 6, Analiza | para#1105 | "F1 Macro 0,506 vs. 0,482 si 0,476" | **STALE - inlocuieste 0,482 cu 0,4347** |

**Nota:** Valoarea canonica 0,4347 NU apare nicaieri in .docx. Trebuie introdusa.

### 3.2. RF F1 Macro: 0,506 / 0,5063

| Locatie | Paragraf | Context | Verdict |
|---|---|---|---|
| Rezumat | para#131 | "F1 Macro de 0,506 pe un set de date de 958 de zile" | CORECT |
| Abstract | para#136 | "F1 Macro of 0.506 on a dataset spanning 958 days" | CORECT |
| Cap 5, Tabel 5.4 | para#986 | Celula "0,506" in coloana RF | CORECT |
| Cap 5, dupa Tabel 5.4 | para#997 | "cel mai bun F1 Macro (0,506)" | CORECT |
| Cap 5, Tabel 5.5 | para#1023 | Celula "0,506" (T3, threshold sensitivity) | CORECT |
| Cap 5, Tabel 5.5 | para#1046 | "T3 obtine cel mai bun F1 Macro (0,506)" | CORECT |
| Cap 5, Tabel 5.6 | para#1066 | Celula "0,506" (ablation, ambele disp.) | CORECT |
| Cap 5, Tabel 5.7 | para#1086 | Celula "0,506" (SMOTE k=3) | CORECT |
| Cap 5, dupa Tabel 5.7 | para#1088 | "obtine rezultatul cel mai bun (F1 Macro 0,506)" | CORECT |
| Cap 6, Analiza | para#1104 | "atinge un F1 Macro de 0,506" | CORECT |
| Cap 6, Analiza | para#1105 | "F1 Macro 0,506 vs. 0,482 si 0,476" | CORECT (RF value) |
| Cap 6, Analiza | para#1106 | "de la 0,470 la 0,506" | CORECT |
| Cap 6, RQ1 | para#1114 | "F1 Macro: 0,506 combinat vs. 0,428 Oura" | CORECT |
| Cap 6, RQ2 | para#1116 | "cel mai bun F1 Macro (0,506 pe 19 features)" | CORECT |
| Cap 6, RQ2 | para#1118 | "F1 Macro: 0,506 vs. 0,281 baseline" | CORECT |
| Cap 7, Concluzii | para#1189 | "F1 Macro variaza intre 0,325 si 0,506" | CORECT |
| Cap 7, Concluzii | para#1211 | "F1 Macro de 0,506 (Random Forest, 19 features)" | CORECT |
| Cap 7, Concluzii | para#1213 | "F1 Macro 0,506, near-optimal" | CORECT |

**Verdict:** Toate aparitiile RF F1=0,506 sunt corecte.

### 3.3. XGB F1 Macro: 0,476 / 0,4760

| Locatie | Paragraf | Context | Verdict |
|---|---|---|---|
| Cap 5, Tabel 5.4 | para#987 | Celula "0,476" in coloana XGBoost | CORECT |
| Cap 6, Analiza | para#1105 | "0,506 vs. 0,482 si 0,476" | CORECT |

### 3.4. F1 Active SVM: 0,38 (STALE in Tabel 5.4) vs 0,25 (canonic)

| Locatie | Paragraf | Context | Verdict |
|---|---|---|---|
| Cap 5, Tabel 5.2 (16f XML) | para#948 | Celula "0,38" in F1 Active SVM, 16 features | **NECLAR** (vezi nota) |
| Cap 5, Tabel 5.4 (19f JSON) | para#992 | Celula "0,38" in F1 Active SVM, 19 features | **STALE - inlocuieste cu 0,25** |

**Nota Tabel 5.2:** Valoarea 0,38 in Tabel 5.2 (16 features XML) este din experimentul cu sursa XML veche. CIFRE_CANONICE.md nu acopera explicit acest experiment (acoperit de un script separat). Nu pot confirma daca e stale fara re-rulare; flag ca "de verificat".

### 3.5. MCC RF (0,1416) si QWK RF (0,2551)

**NU APAR in .docx.** Zero ocurente gasite pentru 0,1416 sau 0,2551 (nici cu virgula, nici cu punct).

Ipoteza: aceste metrici sunt calculate in `mcc_qwk_results.csv` dar NU au fost inca incluse in textul disertatie. Cap 5 sau Cap 6 ar trebui sa le mentioneze (recomandat: un tabel dedicat MCC/QWK sau integrare in Tabel 5.4 extins).

### 3.6. Hash dataset (63bc3a2915bae864)

**NU APARE in .docx.** Zero ocurente.

Ipoteza: hash-ul nu a fost inclus in text. Recomandat: adaugare in Cap 5 la sectiunea de reproducibilitate sau in Anexe.

### 3.7. Dataset split (958, 766, 192, 202, 710, 46)

| Valoare | Gasit? | Locatie | Verdict |
|---|---|---|---|
| 958 | DA | Rezumat (para#131), Abstract (para#136), Anexe (para#1718, 1719), Cap 5 (para#1921) | CORECT |
| 766 | NU | - | **ABSENT din .docx** (doar "80%" menționat, fara cifra exacta) |
| 192 | NU* | para#1240 e o referinta bibliografica (vol. 192), nu split-ul | **ABSENT din .docx** |
| 202 | DA | Cap 3 (para#680: "Rest = 202"), Cap 4 (para#796), Cap 5 (para#894, #1024, #1046) | CORECT |
| 710 | PARTIAL | Doar para#1024 ("202/710/46" in Tabel 5.5) | **Inconsistent** (vezi 3.8) |
| 46 | DA | Cap 3 (para#680: "Active = 46"), Cap 5 Tabel 5.5 (para#1024) | CORECT (dar vezi erori mai jos) |

### 3.8. Distributie INCORECTA in .docx (Normal si Active)

| Locatie | Paragraf | Ce scrie | Ce ar trebui | Eroare |
|---|---|---|---|---|
| Cap 3, bias algoritmic | para#680 | "Normal = 708, Active = 46" | Normal = 710, Active = 46 | **Normal GRESIT (708 vs 710)** |
| Cap 4, distributie clase | para#796 | "Normal: 706 zile (74,1%), Active: 45 zile (4,7%)" | Normal: 710, Active: 46 | **Normal GRESIT (706), Active GRESIT (45)** |
| Cap 5, praguri T3 | para#894 | "Normal 706 zile (74,1%), Active 45 zile (4,7%)" | Normal: 710, Active: 46 | **Normal GRESIT (706), Active GRESIT (45)** |
| Cap 5, Tabel 5.5 | para#1024 | "202/710/46" | 202/710/46 | CORECT |

**Nota:** Procentul 74,1% este corect matematic pentru 710/958 = 74,11%. Dar textul scrie 706 si 708 in loc de 710. Probabil versiuni anterioare ale pipeline-ului (inainte de freeze) produceau aceste cifre.

### 3.9. Tuning gains STALE in Cap 6

| Locatie | Paragraf | Ce scrie | Canonic | Verdict |
|---|---|---|---|---|
| Cap 6 | para#1105 | "RF: +0,3pp dupa tuning" | RF: +2,49pp | **STALE - inlocuieste cu +2,49pp** |
| Cap 6 | para#1105 | "XGBoost: +8,2pp" | XGBoost: +6,87pp | **STALE - inlocuieste cu +6,87pp** |

---

## 4. Rezumat: ce trebuie inlocuit

### Inlocuiri OBLIGATORII (valori factual gresite)

| # | Locatie | Valoare curenta | Valoare corecta | Finding ref |
|---|---|---|---|---|
| 1 | Tabel 5.4, celula SVM F1 Macro | 0,482 | 0,4347 | FND-104 post-fix |
| 2 | Tabel 5.4, celula SVM F1 Active | 0,38 | 0,25 | FND-104 post-fix |
| 3 | Cap 6 para#1105, text | "0,482" | "0,435" (rotunjit) | Consecinta #1 |
| 4 | Cap 6 para#1105, text | "+0,3pp" (RF tuning) | "+2,5pp" | CIFRE_CANONICE s8 |
| 5 | Cap 6 para#1105, text | "+8,2pp" (XGB tuning) | "+6,9pp" | CIFRE_CANONICE s8 |
| 6 | Cap 3 para#680, text | "Normal = 708" | "Normal = 710" | CIFRE_CANONICE s2 |
| 7 | Cap 4 para#796, text | "Normal: 706 zile" | "Normal: 710 zile" | CIFRE_CANONICE s2 |
| 8 | Cap 4 para#796, text | "Active: 45 zile (4,7%)" | "Active: 46 zile (4,8%)" | CIFRE_CANONICE s2 |
| 9 | Cap 5 para#894, text | "Normal 706 zile" | "Normal 710 zile" | CIFRE_CANONICE s2 |
| 10 | Cap 5 para#894, text | "Active 45 zile (4,7%)" | "Active 46 zile (4,8%)" | CIFRE_CANONICE s2 |

### De verificat (nu pot confirma fara re-rulare)

| # | Locatie | Valoare | Problema |
|---|---|---|---|
| 11 | Tabel 5.2, F1 Active SVM (16f XML) | 0,38 | Poate fi stale daca fix-ul CV afecteaza si 16f |
| 12 | Tabel 5.4, Delta SVM "-3,1pp" | -3,1pp | Daca SVM 19f e acum 0,4347 (nu 0,482), delta trebuie recalculat vs 16f |
| 13 | Cap 5 para#997, "SVM scade cu 3,1pp" | 3,1pp | Consecinta delta recalculat |
| 14 | Tabel 5.4, Delta XGB "+0,5pp" | +0,5pp | Probabil corect (XGB 19f 0,476, XGB 16f ~0,471) dar de verificat |

### Valori ABSENTE din .docx (de adaugat)

| # | Valoare | Recomandare |
|---|---|---|
| 15 | MCC: RF=0,1416, XGB=0,1503, SVM=0,0956 | Adauga tabel MCC/QWK in Cap 5 (sau extinde Tabel 5.4) |
| 16 | QWK: RF=0,2551, XGB=0,1805, SVM=0,1709 | La fel |
| 17 | Hash dataset: 63bc3a2915bae864 | Adauga in Cap 5 sectiune reproducibilitate |
| 18 | N train = 766, N test = 192 | Adauga explicit in Cap 5 langa split 80%/20% |

---

## 5. Comentarii Word (w:comment)

Total comentarii in document: **9** (toate de la Catalin Buiu).

| ID | Autor | Data | Locatie (para#) | Capitol estimat | Text comentariu |
|---|---|---|---|---|---|
| c808 | Catalin Buiu (24503) | 2026-04-13 15:57 | para#155 | Cap 1 Introducere | "Dispozitive purtabile" |
| c812 | Catalin Buiu (24503) | 2026-04-13 15:58 | para#175 | Cap 1 (cuprins capitole) | "Starea actuala in domeniul abordat" |
| c837 | Catalin Buiu (24503) | 2026-04-13 16:07 | para#308 | Cap 2 (Tabel 5 zona) | "Ori in clar, ori deloc" |
| c856 | Catalin Buiu (24503) | 2026-05-18 08:56 | para#470 | Cap 3 (features lag-1) | "In romana" |
| c858 | Catalin Buiu (24503) | 2026-05-18 08:56 | para#527 | Cap 3 (clasificare) | "In romana" |
| c866 | Catalin Buiu (24503) | 2026-05-18 09:01 | para#630 | Cap 3 (SHAP formula) | "Scrie-o cu editorul de ecuatii, la fel peste tot unde mai ai ecuatii" |
| c896 | Catalin Buiu (24503) | 2026-05-18 10:12 | para#816 | Cap 5 (Streamlit UI) | "Un screenshot?" |
| c902 | Catalin Buiu (24503) | 2026-05-18 09:34 | para#874 | Cap 5 (Oura auth) | "Atentie la formatare" |
| c974 | Catalin Buiu (24503) | 2026-05-18 09:52 | para#1105 | Cap 6 (RF superioritate) | "Atentie formatare" |

### Interpretare comentarii Buiu

| ID | Actiune necesara |
|---|---|
| c808 | Probabil cere reformulare "wearables" -> "dispozitive purtabile" in Cap 1 |
| c812 | Cere redenumire Cap 2 -> "Starea actuala in domeniul abordat" (in loc de "State-of-the-Art") |
| c837 | Cere clarificare/explicitare a unui termen sau acronim (zona Tabel 5) |
| c856, c858 | Cere traducere in romana a unor pasaje tehnice din Cap 3 |
| c866 | Cere reformatare ecuatie SHAP cu Equation Editor Word |
| c896 | Cere adaugare screenshot Streamlit UI |
| c902 | Problema de formatare in sectiunea Oura OAuth |
| c974 | Problema de formatare in Cap 6 (paragraful cu cifre stale) |

---

## Rezumat executiv

**Discrepante critice gasite: 10 valori de inlocuit, 4 de verificat, 4 absent.**

Cele mai importante:
1. **SVM F1 Macro = 0,482 e STALE** in Tabel 5.4 si Cap 6. Canonic = 0,4347.
2. **SVM F1 Active = 0,38 e STALE** in Tabel 5.4. Canonic = 0,25.
3. **Tuning gains** in Cap 6 sunt stale (+0,3pp / +8,2pp vs canonic +2,49pp / +6,87pp).
4. **Distributia Normal** apare ca 706 sau 708 in 3 locatii. Corect = 710.
5. **Distributia Active** apare ca 45 in 2 locatii. Corect = 46.
6. **MCC, QWK, hash, N train/test** lipsesc complet din document.
