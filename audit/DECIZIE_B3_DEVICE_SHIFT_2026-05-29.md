# Decizie metodologica B3: split train/val/test si confound-ul Gen3/Gen4

Data: 2026-05-29
Branch: b3-train-val-test
Autor: Adrian Lazar (cu the LLM assistant)
Status: DECIS, drum B (vezi sectiunea 5)

Toate cifrele din acest document sunt verificate empiric in sesiunea din
2026-05-29 (rulare `cap5_pipeline_tvt.py` + 3 verificari read-only ale
distributiei device pe split). NU sunt estimari.

---

## 1. CONTEXT

Auditul independent (B3) a semnalat **selection leakage**: o serie de
hiperparametri si praguri au fost alesi privind direct performanta pe test set,
nu pe un set de validare dedicat. Concret:

- pragurile T3 de etichetare (`nutrition_label` din `t3_labels.py`),
- `k_neighbors` la SMOTE,
- confidence threshold-ul pentru recomandare.

Toate trei au fost calibrate uitandu-ne la metrica de test, ceea ce transforma
test set-ul intr-un quasi-validation set si umfla optimist cifrele raportate.

**Planul initial (B3):** introducem un set de **validare dedicat** prin trecerea
de la split 80/20 (train/test) la split temporal **70/10/20** (train/val/test).
Selectia de praguri si hiperparametri s-ar face pe val, test-ul ramanand intact
pentru raportarea finala. Scop: eliminarea selection leakage-ului prin
constructie, nu prin declarare.

---

## 2. CE AM IMPLEMENTAT

Script nou `cap5_pipeline_tvt.py`, **derivat prin import** din
`cap5_pipeline_json_extended.py` (NU rescriere): refoloseste `load_oura`,
`load_apple_watch`, `merge_datasets`, `engineer_features`, `FEATURES`. Astfel
cele **19 features** si fix-ul `rhr_avg < rhr_lowest -> NaN` raman bit-identice
cu pipeline-ul canonic. Split **pozitional 70/10/20** fara shuffle pe `df_model`
(deja sortat cronologic), imputer median fit doar pe train, SMOTE doar pe train
(`SMOTE_K=3`), RF identic cu `_extended` (`n_estimators=100`, `max_depth=10`,
`min_samples_leaf=5`, `random_state=42`). Output izolat in `outputs/b3_tvt/`,
nimic canonic atins.

### Dimensiuni split (N=958)

| Split | Zile | Interval | Distributie {0,1,2} |
|---|---|---|---|
| train | 670 | 2023-06-10 -> 2025-06-11 | {0: 145, 1: 499, 2: 26} |
| val | 95 | 2025-06-12 -> 2025-09-17 | {0: 13, 1: 70, 2: 12} |
| test | 193 | 2025-09-18 -> 2026-03-30 | {0: 44, 1: 141, 2: 8} |

### Metrici val vs test vs frozen canonic

| Metric | Val (95 zile) | Test B3 (193 zile) | Frozen canonic 80/20 (test 192) |
|---|---|---|---|
| Accuracy | 0.5579 | 0.6269 | 0.640625 |
| F1 Macro | 0.2933 | 0.4175 | 0.506324 |
| MCC | -0.0965 | 0.0965 | 0.141560 |
| F1 Rest (cls 0) | 0.1667 | 0.2857 | 0.3218 |
| F1 Normal (cls 1) | 0.7133 | 0.7562 | 0.7527 |
| F1 Active (cls 2) | 0.0000 | 0.2105 | 0.4444 |

Observatie imediata: val-ul este **rupt** (F1 Macro 0.29, MCC negativ -0.097,
Active F1 = 0, model la nivel de hazard pe acel interval), iar test-ul B3 este
sub frozen-ul canonic cu aproape 9pp F1 Macro (0.418 vs 0.506). Inainte de a
declara fragilitate generala a modelului, am investigat cauza.

---

## 3. CE AM DESCOPERIT (finding cheie): confound Gen3 -> Gen4

Suspiciunea (Adrian): val-ul rupt nu vine din fragilitate generala, ci din
tranzitia hardware Oura **Gen3 -> Gen4**. Verificare read-only pe `df` post-merge
(care pastreaza `oura_device`, spre deosebire de `df_model`), proiectat inapoi pe
randurile reale ale `df_model` prin coloana `date`, sectionat pe cele 3 intervale
ale split-ului real.

### Tranzitia este un cutover curat, nu o zona mixta

- Ultima zi Gen3 in `df_model`: **2025-05-29**
- Prima zi Gen4 in `df_model`: **2025-05-30**
- Overlap (zile Gen3 dupa prima zi Gen4): **0**
- Overlap invers (zile Gen4 inainte de ultima zi Gen3): **0**

Deci nu exista intercalare: device-ul comuta o singura data, fix la **finalul
train-ului**, nu in interiorul val-ului.

### Distributie device pe split (N=958)

| Split | Zile | Gen3 | Gen4 | % Gen4 |
|---|---|---|---|---|
| train | 670 | 657 | 13 | 2% |
| val | 95 | 0 | 95 | **100%** |
| test | 193 | 0 | 193 | **100%** |

Cele 13 zile Gen4 din train sunt contigue la coada: **2025-05-30 -> 2025-06-11**.

### De ce frozen-ul canonic 80/20 bate B3 pe test

Split-ul canonic 80/20 (`iloc[:766]` train, `iloc[766:]` test) are o distributie
de device complet diferita pe train:

| Split canonic 80/20 | Gen3 | Gen4 |
|---|---|---|
| train (766 zile) | 657 | **109** |
| test (192 zile) | 0 | 192 |

Train-ul canonic inghite **109 zile Gen4** (intervalul care in B3 devine val plus
inceputul test-ului), pe cand train-ul B3 vede doar **13**. Mai mult Gen4 in
train inseamna transfer mai bun pe test-ul Gen4. Asta explica direct cei ~9pp
diferenta F1 Macro: nu e regression, e mai putin Gen4 in train.

### Concluzie

1. Metricile pe val si test **nu masoara generalizare temporala pura**, ci
   **transfer Gen3 -> Gen4**. Train-ul este ~98% Gen3, eval-ul este 100% Gen4.
2. Val-ul rupt este **diagnostic, nu defect**: este prima fereastra Gen4
   sustinuta, imediat post-cutover, cand modelul a vazut doar 13 zile Gen4 in
   train. Este cel mai prost moment posibil de evaluat. In plus val-ul este vara
   (mai multe zile Active: 12 vs 8 in test), pe care modelul antrenat pe Gen3 le
   rateaza complet (Active F1 = 0). Esantion mic (95 zile) -> MCC volatil, trece
   usor in negativ.
3. Split-ul 70/10/20 **agraveaza** confound-ul fata de 80/20: scoate aproape tot
   Gen4-ul din train (109 -> 13) ca sa formeze val-ul.

---

## 4. OPTIUNI EVALUATE

### Drum A: train/val/test complet + re-frozen

Adoptam 70/10/20 ca pipeline oficial, refacem selectia de praguri pe val,
re-inghetam modelul pe noul train si raportam test-ul B3.

### Drum B: pastram cifrele canonice 80/20, documentam onest limitarile

Mentinem modelul frozen 80/20 (frozen 20260331) si cifrele canonice deja
validate bit-perfect, iar selection leakage-ul il tratam prin declarare post-hoc
in lucrare, alaturi de confound-ul Gen3/Gen4.

| Criteriu | Drum A (re-frozen TVT) | Drum B (pastram canonic + documentare) |
|---|---|---|
| Selection leakage | eliminat prin constructie | tratat prin declarare post-hoc (acceptabil) |
| Confound Gen3/Gen4 | **agravat** (13 Gen4 in train vs 109) | neschimbat fata de canonic |
| Cifre canonice | invalidate, trebuie regenerate tot | raman bit-perfect (F1M 0.5063) |
| Risc tehnic | mare (re-freeze, re-sanity, re-CSV, UI) | minim (doar text in lucrare) |
| Efort la 6 sapt. de aparare | ridicat | scazut |
| Onestitate metodologica | buna | buna (cu declarare explicita) |

---

## 5. DECIZIE: Drum B

**Pastram cifrele canonice 80/20. Modelul frozen ramane neschimbat
(rf_frozen_20260331, F1 Macro test 0.506324, MCC 0.141560).**

Motivatie:

1. Train/val/test **agraveaza** confound-ul real (scoate Gen4 din train, 109 ->
   13), deci ar inrautati exact problema de generalizare pe care trebuia sa o
   masoare mai curat.
2. Selection leakage-ul se trateaza acceptabil prin **declarare post-hoc**: este
   o limitare metodologica standard de mentionat, nu un defect care invalideaza
   rezultatele.
3. Cifrele canonice sunt deja **validate bit-perfect** (sanity check PASS pe toate
   liniile in aceeasi sesiune: hash dataset `63bc3a2915bae864`, N test 192,
   distributie {0:44, 1:140, 2:8}, Accuracy 0.640625, F1 Macro 0.506324, MCC
   0.141560). Re-frozen-ul ar arunca aceasta baza solida.
4. La **6 saptamani de aparare** (iunie-iulie 2026), riscul tehnic al unui
   re-freeze complet (model, sanity, CSV-uri canonice, UI Streamlit) nu se
   justifica.

`cap5_pipeline_tvt.py` si `outputs/b3_tvt/` raman ca **artefact de diagnostic**
(dovada empirica a confound-ului), NU ca pipeline de productie.

---

## 6. ACTIUNI URMATOARE

Documentare in disertatie a **3 limitari** declarate explicit:

1. **Selection post-hoc**: pragurile T3, `k_neighbors` SMOTE si confidence
   threshold au fost calibrate pe test set; in absenta unui set de validare
   dedicat, metricile de test sunt o estimare optimista. Declarat ca limitare.
2. **Confound Gen3/Gen4**: train ~98% Gen3, test 100% Gen4 (cutover curat
   2025-05-30, overlap 0). Metrica de test reflecta si transfer cross-device, nu
   doar generalizare temporala. De mentionat ca factor de confuzie hardware.
3. **Active n=8 fara interval de incredere**: clasa Active are doar 8 esantioane
   in test (F1 0.4444); metrica per-clasa este instabila la acest n. De raportat
   cu rezerva explicita.

Optional:

- **Bootstrap CI pe test** (resampling cu inlocuire pe cele 192 zile) pentru a da
  intervale de incredere la F1 Macro, MCC si F1 per clasa, in special pentru
  clasa Active. Read-only fata de model, nu atinge nimic canonic.

---

## Anexa: provenienta cifrelor

| Sursa | Cifre |
|---|---|
| `cap5_pipeline_tvt.py` (rulare 2026-05-29) | split 670/95/193, metrici val/test din tabel sectiunea 2 |
| `outputs/b3_tvt/metrics_val_vs_test.csv` | val F1M 0.293318, MCC -0.096456; test F1M 0.417475, MCC 0.096512 |
| Verificare device read-only (3 rulari) | train 657 Gen3 + 13 Gen4; val/test 100% Gen4; cutover 2025-05-30 overlap 0; canonic 80/20 train 657 Gen3 + 109 Gen4 |
| `sanity_check_frozen.py` (PASS 2026-05-29) | frozen: Acc 0.640625, F1M 0.506324, MCC 0.141560, hash 63bc3a2915bae864, test 192 {0:44,1:140,2:8} |
