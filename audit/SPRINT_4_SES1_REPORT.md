# SPRINT 4 Sesiunea 1: Raport rulare scripturi

**Data:** 2026-05-28  
**Scope:** Validare cifre canonice prin rulare scripturi pipeline  
**Mediu:** Linux sandbox (sklearn 1.7.2, imblearn 0.14.1, pandas 2.3.3, numpy 2.2.6)

---

## Status per script

| # | Script | Status | Nota |
|---|---|---|---|
| 1 | `experiments_new.py --experiment 1` | **SKIP** | xgboost (125MB) nu s-a putut instala (timeout sandbox 45s). CSV-ul din Sprint 2 validat manual. |
| 2 | `ablation_study.py` | **PASS** | Rulat complet, output consistent cu CIFRE_CANONICE.md |
| 3 | `smote_comparison.py` | **PASS** | Rulat complet, output consistent cu CIFRE_CANONICE.md |
| 4 | `generate_canonical_numbers.py` | **PASS** | CIFRE_CANONICE.md regenerat bit-identical (md5: b3e4b1fbb9ecc2391ddbf996e0967fcb) |
| 5 | `verify_disertatie_numbers.py` | **NOT FOUND** | Fisierul nu exista. Rulat `sanity_check_frozen.py` in loc. |
| 5b | `sanity_check_frozen.py` | **PASS** | Toate metricile reproduse identic vs canonic |

---

## Cifre extrase din output

### Script 2: ablation_study.py

| Configuratie | F1 Macro | Accuracy | F1 Rest | F1 Normal | F1 Active |
|---|---|---|---|---|---|
| Doar Oura (11 feat) | 0,428 | 0,620 | 0,247 | 0,752 | 0,286 |
| Doar Apple Watch (8 feat) | 0,324 | 0,474 | 0,269 | 0,604 | 0,100 |
| Ambele dispozitive (19 feat) | **0,506** | 0,641 | 0,322 | 0,753 | 0,444 |

Gain combinat vs Oura: +7,8pp. Gain combinat vs AW: +18,2pp.

### Script 3: smote_comparison.py

| Strategie | F1 Macro | F1 Active |
|---|---|---|
| Baseline (nimic) | 0,280 | 0,000 |
| Cost-sensitive | 0,376 | 0,222 |
| UnderSampler | 0,325 | 0,108 |
| **SMOTE k=3** | **0,506** | **0,444** |
| SMOTE k=5 | 0,451 | 0,375 |

### Script 1: tuning_comparison.csv (din Sprint 2, validat)

| Model | Default CV F1 | Tuned CV F1 | Delta (pp) |
|---|---|---|---|
| RF | 0,3369 | 0,3618 | **+2,49** |
| SVM | 0,3560 | 0,4008 | **+4,47** |
| XGBoost | 0,3343 | 0,4030 | **+6,87** |

### Script 5b: sanity_check_frozen.py

| Metric | Reprodus | Canonic | Status |
|---|---|---|---|
| Dataset hash | 63bc3a2915bae864 | 63bc3a2915bae864 | OK |
| N test | 192 | 192 | OK |
| Distributie test | {0:44, 1:140, 2:8} | {0:44, 1:140, 2:8} | OK |
| Accuracy | 0,640625 | 0,640625 | OK |
| F1 Macro | 0,506324 | 0,506324 | OK |
| MCC | 0,141560 | 0,141560 | OK |

---

## Comparatie cifre noi vs CIFRE_CANONICE.md anterior

### CIFRE_CANONICE.md: regenerat identic

`generate_canonical_numbers.py` a produs un fisier **bit-identical** cu cel din git (md5 b3e4b1fbb9ecc2391ddbf996e0967fcb, 0 linii diff). Nu exista drift.

### Ablation study vs CIFRE_CANONICE sectiunea 4

| Config | CIFRE_CANONICE | Script output | Match? |
|---|---|---|---|
| Oura-only F1 Macro | 0,428 | 0,428 | MATCH |
| AW-only F1 Macro | 0,324 | 0,324 | MATCH |
| Combined F1 Macro | 0,506 | 0,506 | MATCH |
| Combined F1 Active | 0,444 | 0,444 | MATCH |

### SMOTE comparison vs CIFRE_CANONICE sectiunea 6

| Strategie | CIFRE_CANONICE | Script output | Match? |
|---|---|---|---|
| Baseline | 0,280 | 0,280 | MATCH |
| Cost-sensitive | 0,376 | 0,376 | MATCH |
| UnderSampler | 0,325 | 0,325 | MATCH |
| SMOTE k=3 | 0,506 | 0,506 | MATCH |
| SMOTE k=5 | 0,451 | 0,451 | MATCH |

### Tuning gains vs CIFRE_CANONICE sectiunea 8

| Model | CIFRE_CANONICE Delta | CSV Delta | Match? |
|---|---|---|---|
| RF | +2,49pp | +2,49pp | MATCH |
| SVM | +4,47pp | +4,47pp | MATCH |
| XGBoost | +6,87pp | +6,87pp | MATCH |

### Frozen model sanity check vs Ground Truth (the tooling context file)

| Metric | Ground Truth | Reprodus | Match? |
|---|---|---|---|
| Hash dataset | 63bc3a2915bae864 | 63bc3a2915bae864 | MATCH |
| N test | 192 | 192 | MATCH |
| Distributie | {0:44,1:140,2:8} | {0:44,1:140,2:8} | MATCH |
| Accuracy | 0.640625 | 0.640625 | MATCH |
| F1 Macro | 0.506324 | 0.506324 | MATCH |
| MCC | 0.141560 | 0.141560 | MATCH |

**ZERO DIFERENTE detectate intre output-urile scripturilor si CIFRE_CANONICE.md.**

---

## Erori si warnings

### Warnings (non-blocking)

```
InconsistentVersionWarning: Trying to unpickle estimator from version 1.8.0 
when using version 1.7.2.
```

Cauza: modelul frozen a fost creat cu sklearn 1.8.0, sandbox-ul are 1.7.2. Warning only, nu afecteaza rezultatele (toate metricile reproduse identic).

### Erori

1. **`verify_disertatie_numbers.py` nu exista.** Fisierul nu este in repo. Probabil nu a fost creat inca sau are alt nume. `sanity_check_frozen.py` acopera partial aceeasi functionalitate.

2. **xgboost nu s-a putut instala.** Pachetul (125MB wheel) depaseste timeout-ul de 45s al sandbox-ului. `experiments_new.py` nu s-a putut rula direct, dar CSV-ul generat in Sprint 2 exista si confirma valorile canonice.

---

## Concluzii

### E safe sa pornesti Sesiunea 2 (editare .docx)?

**DA, cu urmatoarele conditii:**

1. **Cifrele canonice sunt stabile si verificate.** 4 din 5 scripturi au rulat cu succes. Al 5-lea (experiments_new.py) are output-ul salvat din Sprint 2 si valorile sunt consistente.

2. **CIFRE_CANONICE.md e regenerat identic.** Nu exista drift intre surse (CSV-uri, frozen_meta, script outputs).

3. **Ground truth intact:** sanity_check_frozen.py confirma reproducibilitate bit-perfect pe toate metricile.

4. **Singura precautie:** `verify_disertatie_numbers.py` nu exista. Daca intentioneazi un cross-check automat intre .docx si cifre canonice, trebuie creat (sau facut manual cu SPRINT_4_RECON.md ca ghid).

### Recomandare

Poti edita .docx-ul folosind `audit/SPRINT_4_RECON.md` ca checklist. Inlocuirile sunt clare (10 valori obligatorii, 4 de verificat). Dupa editare, o a doua rulare a auditului .docx va confirma ca toate cifrele sunt aliniate.
