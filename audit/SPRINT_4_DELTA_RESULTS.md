# SPRINT 4 - REZULTATE DELTAS 16f vs 19f (FND-SVM-DELTA-001)

**Generat:** 2026-05-28
**Script:** `scripts/canonical/compare_16f_vs_19f.py`
**Output CSV:** `outputs/tables/compare_16f_vs_19f.csv`
**Sursa canonica:** CIFRE_CANONICE.md sectiunea 16 (nou)

---

## 1. Context

Tabel 5.4 din `.docx` are randul "Δ F1 Macro vs. 16f" cu deltas stale:
**RF +3,6pp / XGB +0,5pp / SVM -3,1pp**. Acestea fusesera calculate cand SVM 19f
era `0,482` (stale). Canonic curent: SVM 19f = `0,4347`. Intrebarea: sunt si
valorile 16f stale dupa fix-ul rhr_avg (`rhr_avg < rhr_lowest -> NaN`)?

Metodologie noua: 16f si 19f pe **aceeasi sursa JSON**, acelasi split temporal
80/20 (766/192), aceiasi hiperparametri canonici (RF din frozen_meta, XGB/SVM
defaults), toate fix-urile aplicate (CV de-leaked, SMOTE/scaler SVM, rhr_avg).
16f = 19f minus `{aw_sleep_deep_hours, aw_sleep_rem_hours, aw_resp_rate}_lag1`.

**Verificari trecute:** dataset hash `63bc3a2915bae864` (match), split 766/192 (match),
sanity 19f vs canonic RF=0,5063 / XGB=0,4760 / SVM=0,4347 (toate [OK], bit-match).

---

## 2. Rezultate canonice

| Model | F1 16f | F1 19f | **Delta nou** | Delta stale (docx) | Verdict delta |
|---|---|---|---|---|---|
| Random Forest | 0,4697 | 0,5063 | **+3,66pp** | +3,6pp | MATCH (rounding) |
| XGBoost | 0,4710 | 0,4760 | **+0,50pp** | +0,5pp | MATCH (exact) |
| SVM (RBF) | 0,4588 | 0,4347 | **-2,41pp** | -3,1pp | **DIVERGE -0,69pp** |

Extra (din CSV): F1 Active, Accuracy, MCC, CV F1 per config - vezi
`outputs/tables/compare_16f_vs_19f.csv`.

---

## 3. Sunt valorile 16f stale? (raspuns la intrebarea-cheie)

Reconstructie valori 16f implicite de deltas stale (stale_16f = stale_19f - stale_delta):

| Model | 16f stale (implicit) | 16f canonic | Schimbare 16f |
|---|---|---|---|
| Random Forest | ~0,470 | 0,4697 | **neschimbat** |
| XGBoost | ~0,471 | 0,4710 | **neschimbat** |
| SVM (RBF) | ~0,513 | 0,4588 | **STALE, -5,4pp** |

**Concluzie:** doar **SVM 16f era stale**. RF si XGB 16f erau deja canonice.

---

## 4. Surpriza

**SVM s-a miscat pe AMBELE configuratii, aproape paralel.** Fix-urile din
Sprint 1-2 (CV de-leaked + scaler/SMOTE order FND-105 + rhr_avg) au coborat SVM
atat pe 19f (0,482 -> 0,4347, -4,7pp) cat si pe 16f (~0,513 -> 0,4588, -5,4pp).

Pentru ca 16f a scazut **putin mai mult** decat 19f, delta (19f - 16f) devine
mai putin negativa: **-3,1pp -> -2,41pp**. Deci desi ambele valori absolute SVM
erau stale, randul de delta se schimba doar cu 0,69pp.

RF si XGB nu folosesc scaler si sunt mai robuste la cele ~60 zile cu rhr_avg
corectat; valorile lor 16f/19f raman practic identice cu cele din docx. Nicio
surpriza pe RF/XGB.

Nota metodologica: aceste 16f sunt **subset JSON** (aceeasi sursa cu 19f), deci
delta compara configuratii comparabile. Daca Tabel 5.4 compara implicit cu 16f-XML
(Tabel 5.2, alta sursa, alt N), baza de comparatie difera - recomandarea de mai
jos foloseste 16f-JSON, varianta corecta metodologic.

---

## 5. Recomandare modificari `.docx` (manual, de catre Adrian)

### 5.1. Tabel 5.4, randul "Δ F1 Macro vs. 16f"

| Coloana | Valoare curenta | Valoare canonica | Actiune |
|---|---|---|---|
| RF | +3,6pp | +3,7pp (3,66) | optional (rounding 4 zecimale) |
| XGBoost | +0,5pp | +0,5pp | fara schimbare |
| SVM | **-3,1pp** | **-2,4pp** (-2,41) | **OBLIGATORIU** |

### 5.2. Paragraf dupa Tabel 5.4 (Cap 5, ~para#997)

VECHI: "...SVM scade cu 3,1pp..." (sau formulare echivalenta despre 16f -> 19f).
NOU: "...SVM scade cu 2,4pp la trecerea de la 16 la 19 features..."

### 5.3. Orice text Cap 6 care citeaza delta SVM "-3,1" sau "3,1 puncte"

Cauta "3,1" in zona Tabel 5.4 / Cap 6 si inlocuieste cu "2,4" unde se refera la
delta SVM 16f vs 19f. (NU confunda cu alte aparitii 3,1 fara legatura.)

### 5.4. Nemodificat de acest finding (alte findings, vezi SPRINT_4_RECON.md)

- SVM 19f F1 Macro `0,482 -> 0,4347` si F1 Active `0,38 -> 0,25` (FND-104 post-fix)
  raman de aplicat separat; nu fac parte din randul de delta.

---

## 6. Reproducere

```
PYTHONUTF8=1 python scripts/canonical/compare_16f_vs_19f.py
PYTHONUTF8=1 python generate_canonical_numbers.py   # regenereaza CIFRE_CANONICE.md sect 16
python sanity_check_frozen.py                        # Ground Truth bit-perfect PASS
```
