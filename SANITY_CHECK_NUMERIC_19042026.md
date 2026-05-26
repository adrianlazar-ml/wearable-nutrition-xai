# Sanity Check Numeric — Disertație Adrian Lazar
**Sesiunea 1 de review | 19.04.2026**

Sursa autoritară pentru cifre: rularea pipeline-ului `cap5_pipeline_json_extended.py` + CSV-urile din `outputs/tables/` + metadata `models/frozen_meta_20260331.json`. Dataset hash: `63bc3a2915bae864`.

---

## TL;DR

- **3 discrepanțe critice** (cifre greșite în text față de CSV-urile care au generat tabele)
- **5 discrepanțe de consistență internă** (același număr apare diferit în capitole diferite)
- **1 punct de decizie** (threshold-ul recomandat 0.5 vs 0.6)
- **Timp estimat de rezolvare:** 45-60 min (search & replace + verificări)

---

## 1. Discrepanțe CRITICE (text vs CSV canonic)

Acestea sunt cifre citate explicit în paragrafe de Cap.3, Cap.4 și Cap.5 care **nu corespund** cu rezultatele reale din experimente. Risc real ca membru al comisiei să verifice corespondența text ↔ figură/tabel și să găsească discrepanța.

### 1.1. Cifrele din ablation study, Cap.3 — GREȘITE

**Unde:** Cap.3, secțiunea "Integrarea datelor din dispozitive purtabile", paragraful "Validarea complementarității prin studiu ablativ".

**Text actual:**
> "F1 Macro semnificativ superior (0.517) față de oricare dispozitiv individual (Oura: 0,366, Apple Watch: 0,348), iar îmbunătățirea este deosebit de pronunțată pentru clasa Active (F1: 0,176 cu Apple Watch singur, 0,462 cu ambele dispozitive)."

**Cifre corecte (din `ablation_study_table.csv`):**
- Ambele: F1 Macro = **0,506** (nu 0,517)
- Doar Oura: F1 Macro = **0,428** (nu 0,366)
- Doar Apple Watch: F1 Macro = **0,324** (nu 0,348)
- F1 Active AW-only: **0,100** (nu 0,176)
- F1 Active Combined: **0,444** (nu 0,462)

**Ironic:** Tabel 5.6 din Cap.5 are cifrele corecte. Deci cele din Cap.3 sunt dintr-o iterație anterioară.

**Corectură propusă:**
> "F1 Macro semnificativ superior (0,506) față de oricare dispozitiv individual (Oura: 0,428, Apple Watch: 0,324), iar îmbunătățirea este deosebit de pronunțată pentru clasa Active (F1: 0,100 cu Apple Watch singur, 0,444 cu ambele dispozitive)."

---

### 1.2. Distribuția claselor T3, Cap.5 — GREȘITĂ (2 locații)

**Unde:** Cap.5.2 "Definirea claselor și strategia de etichetare" + Cap.4 "Integrarea datelor" (apare și acolo).

**Text actual:**
> "Distribuția rezultantă este: Rest 202 zile (21,2%), Normal **706** zile (74,1%), Active **45** zile (4,7%)."

**Cifre corecte (`sensitivity_threshold_table.csv`, T3):**
- Rest: 202 (21,1%) ✓
- Normal: **710** (74,1%)
- Active: **46** (4,8%)
- Total: **958** zile

**Contradictie internă:** **Tabel 5.5** din aceeași disertație (sensitivity analysis) afișează T3 = **202/710/46**, deci tabelul contrazice paragraful. O comisie care parcurge Cap.5.2 → Tabel 5.5 va vedea acest inconsistent.

**Corectură:**
> "Distribuția rezultantă este: Rest 202 zile (21,1%), Normal 710 zile (74,1%), Active 46 zile (4,8%)."

---

### 1.3. Split temporal, Cap.5 "Configurare experimentală" — GREȘIT

**Text actual:**
> "Setul de antrenare conține **762** de zile, iar setul de testare **191** de zile. Distribuția claselor în setul de antrenare (înainte de SMOTE) este: Rest 162, Normal 564, Active 36."
>
> "După aplicarea SMOTE, distribuția claselor în setul de antrenare devine echilibrată: **564** eșantioane per clasă."

**Cifre corecte (output pipeline real, 19.04.2026):**
- Train: **766** zile (`0.8 * 958 = 766.4` → `int() = 766`)
- Test intern: **192** zile
- Train pre-SMOTE: Rest **158**, Normal **570**, Active **38**
- Train post-SMOTE: **570** eșantioane per clasă

**Test distribution (pentru referință în alte paragrafe):** Rest 44, Normal 140, Active 8 — nu 9, nu 44/139/8.

**Corectură:**
> "Setul de antrenare conține 766 de zile, iar setul de testare 192 de zile. Distribuția claselor în setul de antrenare (înainte de SMOTE) este: Rest 158, Normal 570, Active 38. După aplicarea SMOTE, distribuția claselor în setul de antrenare devine echilibrată: 570 eșantioane per clasă."

---

## 2. Discrepanțe de consistență internă (text vs text)

Același număr apare cu valori diferite în capitole diferite. Comisia poate observa asta simplu citind liniar.

### 2.1. Total zile dataset

| Locație | Valoare |
|---|---|
| Cap.4 "Integrarea datelor" | "reduce setul de date la **953** de zile" |
| Anexa A2.2 și A2.4 | "N = **958**" + "958 zile utile" |
| Cap.5.5.1 Tabel 5.5 (toate rândurile) | **n_total = 958** |
| **Pipeline real** | **958** ✓ |

**Corectură:** În Cap.4 schimbă "953" cu "958". Să confirm textul exact: "Acest pas reduce setul de date la **953** de zile" → "Acest pas reduce setul de date la **958** de zile".

### 2.2. Total zile Oura

| Locație | Valoare |
|---|---|
| Cap.4 secțiunea Oura | "setul de date Oura rezultat conține **1.004** zile" |
| Cap.4 "Integrarea datelor" | "**1.004** zile în sursa Oura" |
| **Pipeline real** (run 19.04.2026) | **1.009** zile Oura |

**Notă:** Pipeline-ul rulat cu `train_end_date=2026-03-31` taie Oura la 2026-03-30 (ultima zi). Dataset-ul inițial a fost probabil mai mic la data scrierii textului. Dacă vrei cifră la 31.03, e 1.009. Dacă preferi să rotunjești la "cca. 1.000", zi "aproximativ 1.000 zile". Recomand valori exacte.

### 2.3. Total fișiere JSON Apple Watch

| Locație | Valoare |
|---|---|
| Cap.4 secțiunea Apple Watch | "procesează **1.082** de fișiere JSON individuale" |
| Cap.4 "Integrarea datelor" | "**1.082** de zile disponibile în sursa Apple Watch" |
| Anexa A2.2 | "procesează **1.078** de astfel de fișiere" |
| **Pipeline real** | **1.083** zile |

**Trei valori diferite pentru același număr.** Recomand să fixezi **1.083**.

### 2.4. Zile combinate (inner join)

| Locație | Valoare |
|---|---|
| Cap.4 "Integrarea datelor" | "producând **1.008** zile cu date din ambele dispozitive" |
| Cap.4 pipeline descriere | "**1.008** zile cu date complete" |
| **Pipeline real** | **1.009** zile (Dataset combinat) |

### 2.5. Număr Active în test set

**Text Cap.5 Tabel 5.2 discuție:**
> "din cauza numărului extrem de redus de eșantioane Active în setul de testare (**9** din **191**)"

**Valoare reală:** **8** din **192** (după fix rhr_avg + 19 features)

**Notă:** Aceasta era probabil cifra corectă pentru configurația XML 16 features, dar ai extrapolat-o. Pentru consistență cu configurația finală (JSON extended, 19 features), schimb-o.

---

## 3. Punct de decizie: confidence threshold 0.5 sau 0.6?

**În text (și `RECAPITULARE_EXPERIMENTE.md`) + codul integrat în demo/Streamlit:** threshold = **0.50**.

**În `confidence_threshold_analysis.csv` canonic:**
```
threshold=0.50: F1=0.522, coverage=80.2%, mean_cal_error=68.75 kcal
threshold=0.60: F1=0.376, coverage=29.2%, mean_cal_error=57.29 kcal  ← marcat "RECOMANDAT"
```

**Tradeoff:**
- La threshold=0.5: modelul rămâne util (coverage 80%), F1 se menține bun (0.52).
- La threshold=0.6: "siguranță" crescută (eroare calorică mai mică), dar pierzi 71% din predicții pe fallback.

**Contradicție aparentă:** codul curent folosește 0.50, dar flag-ul "RECOMANDAT" în CSV zice 0.60. Probabil flag-ul a fost gândit în termeni strict de "minimize calorie error" fără a ține cont de trade-off.

**Recomandarea mea:** păstrează **0.50** (cum e în codul integrat) și menționează explicit în Cap.5 sau Cap.6 de ce alegi 0.5 peste 0.6:
> "Deși threshold = 0,6 minimizează eroarea calorică reziduală (57 kcal vs 69 kcal), coverage-ul scade sub 30%, ceea ce face sistemul impractic. Threshold = 0,5 oferă echilibrul optim: coverage 80%, eroare calorică sub 70 kcal (sub zgomotul tipic al estimărilor nutriționale)."

Aceasta e exact genul de raționament pe care o comisie AI/ML îl apreciază: decizii informate pe tradeoff, nu pe metrică unică.

Apoi: **actualizează CSV-ul** `confidence_threshold_analysis.csv` să marcheze `0.5` ca "RECOMANDAT", nu `0.6`, altfel următoarea persoană care se uită (inclusiv tu la repetiții) va fi confuză. Sau elimină coloana asta din CSV.

---

## 4. Plan de execuție (ordine recomandată)

### Pas 1: Backup (2 min)
```powershell
Copy-Item "Disertatie_Lazar_Adrian_2026.docx" "Disertatie_Lazar_Adrian_2026_backup_19042026.docx"
```

### Pas 2: Search & replace pe cifrele critice din #1 (15 min)

Deschide Word. Ctrl+H. Înlocuiește în această ordine (de la cel mai specific la cel mai general, ca să nu suprascrii din greșeală):

| Caută | Înlocuiește | Așteaptă | Note |
|---|---|---|---|
| `0.517` sau `0,517` | `0,506` | 1 în Cap.3 | Cifra ablation veche |
| `Oura: 0,366` | `Oura: 0,428` | 1 în Cap.3 | |
| `Apple Watch: 0,348` | `Apple Watch: 0,324` | 1 în Cap.3 | |
| `0,176 cu Apple Watch` | `0,100 cu Apple Watch` | 1 în Cap.3 | F1 Active AW-only |
| `0,462 cu ambele` | `0,444 cu ambele` | 1 în Cap.3 | F1 Active combined |
| `Normal 706 zile` | `Normal 710 zile` | 1-2 locații | |
| `Active 45 zile` | `Active 46 zile` | 1-2 locații | |
| `(21,2%)` | `(21,1%)` | 1 locație | |
| `(4,7%)` | `(4,8%)` | 1 locație | |
| `762 de zile` | `766 de zile` | 1 locație | Train |
| `191 de zile` | `192 de zile` | 1 locație | Test |
| `Rest 162, Normal 564, Active 36` | `Rest 158, Normal 570, Active 38` | 1 locație | |
| `564 eșantioane per clasă` | `570 eșantioane per clasă` | 1 locație | |
| `9 din 191` | `8 din 192` | 1 locație | Support Active test |

### Pas 3: Fix consistență internă (#2) (10 min)

| Caută | Înlocuiește | Note |
|---|---|---|
| `953 de zile` | `958 de zile` | Cap.4 |
| `1.004 zile` | `1.009 zile` | Apare de 2 ori în Cap.4 |
| `1.082 de fișiere` | `1.083 de fișiere` | Cap.4 |
| `1.082 de zile` | `1.083 de zile` | Cap.4 |
| `1.078 de astfel de fișiere` | `1.083 de astfel de fișiere` | Anexa A2.2 |
| `1.008 zile` | `1.009 zile` | Cap.4 (apare de 2 ori) |

### Pas 4: Decide confidence threshold (5 min)

Alege 0.5 (recomandarea mea) sau 0.6. Actualizează:
- Textul relevant din Cap.5 și/sau Cap.6 (dacă apare explicit)
- `confidence_threshold_analysis.csv` (marchează corect "RECOMANDAT")

### Pas 5: Validare finală (15 min)

După toate modificările, fă un search pentru cifrele vechi ca să confirmi că nu au rămas:
- `706`, `45 zile`, `762`, `191`, `564`, `953`, `1.004`, `1.082`, `1.078`, `1.008`, `0,517`, `0,366`, `0,348`, `0,176`, `0,462`

Dacă vreuna dintre astea mai apare, verifică contextul: e citat vechi (ok), sau e alt context (fix).

### Pas 6 (opțional): Adaugă paragraf de reproducibilitate (10 min)

La Cap.5, secțiunea "Mediul de dezvoltare", adaugă un paragraf scurt:

> "Toate rezultatele experimentale raportate sunt reproductibile: modelul de producție a fost înghețat la data de 30 martie 2026 (dataset hash SHA256: `63bc3a2915bae864`), iar artefactele sunt disponibile în directorul `models/` al proiectului (`rf_frozen_20260331.joblib`, `imputer_frozen_20260331.joblib`, `frozen_meta_20260331.json`). Metadata include versiunile exacte ale librăriilor, configurația modelului și distribuția claselor, permițând validarea independentă a setup-ului experimental."

Asta răspunde preventiv unei întrebări potențiale de la o comisie AI/ML riguroasă: "Cum dovediți reproducibilitatea?"

---

## 5. Ce NU trebuie schimbat

Cifrele care **sunt deja corecte** și nu le atingi:

- **F1 Macro RF = 0,506** (match exact cu `cap5_compare_algorithms_json_ext`)
- **F1 Macro SVM = 0,482**, **XGB = 0,476**
- **Tabel 5.5** (threshold sensitivity) — toate rândurile corect
- **Tabel 5.6** (ablation Cap.5) — toate cifrele corect
- **Hyperparameter tuning** din Tabel 5.X — corect (RF 0,4174 → 0,4204)
- **LR F1 = 0,34** + **LightGBM F1 = 0,47** — corect
- **aw_hr_mean = feature #1** cross-model — corect
- **Pragurile T3**: Rest<60/60, Active≥85/75 — corect
- **SMOTE k=3** — corect

---

## 6. Sugestie pentru a evita recurența

După ce faci fix-urile, consideră să adaugi în `cap5_pipeline_json_extended.py` un **header summary** care print-uiește toate cifrele cheie la sfârșit, într-un format copy-pasteable. Ceva de genul:

```python
print("\n=== CIFRE CANONICE PENTRU DISERTATIE ===")
print(f"Total zile Oura: {len(oura)}")
print(f"Total zile Apple Watch: {len(aw)}")
print(f"Dataset combinat (inner join): {len(df_merged)}")
print(f"Dataset final (dupa lag-1 + dropna label): {len(df_model)}")
print(f"Distributie clase (full): {y_full.value_counts().sort_index().to_dict()}")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Train pre-SMOTE: {y_train.value_counts().sort_index().to_dict()}")
print(f"Test distributie: {y_test.value_counts().sort_index().to_dict()}")
```

Deja ai asta în `frozen_meta_20260331.json` — mai simplu, poți doar să zici: pentru orice update la disertație, deschide acest JSON și ia cifrele de acolo.
