# Sprint 4 Sesiunea 2, Phase 1: Raport inlocuiri text

**Data:** 2026-05-28  
**Tool:** python-docx 1.2.0  
**Working file:** `_referinte_upb/Disertatie_Lazar_Adrian_2026_Sprint4_WIP.docx`  
**Original file:** NEMODIFICAT (verificat: nu s-a atins)

---

## Integritate comentarii Word

| Moment | N comentarii | N commentRangeStart | Status |
|---|---|---|---|
| INAINTE de editare | 9 | 9 | OK |
| DUPA editare | 9 | 9 | OK |

Toate 9 comentarii Buiu (c808, c812, c837, c856, c858, c866, c896, c902, c974) sunt intacte, cu text si range-uri neschimbate.

---

## Status per inlocuire

| Step | Status | Locatie | Find | Replace | Verificare |
|---|---|---|---|---|---|
| 1 | **SUCCESS** | para index 281 (Cap.3) | "Normal = 708" | "Normal = 710" | OLD absent, NEW prezent |
| 2 | **SUCCESS** | para index 317 (Cap.4) | "Normal: 706 zile (74,1%)" | "Normal: 710 zile (74,1%)" | OLD absent, NEW prezent |
| 3 | **SUCCESS** | para index 317 (Cap.4) | "Active: 45 zile (4,7%)" | "Active: 46 zile (4,8%)" | OLD absent, NEW prezent |
| 4 | **SUCCESS** | para index 371 (Cap.5) | "Normal 706 zile (74,1%)" | "Normal 710 zile (74,1%)" | OLD absent, NEW prezent |
| 5 | **SUCCESS** | para index 371 (Cap.5) | "Active 45 zile (4,7%)" | "Active 46 zile (4,8%)" | OLD absent, NEW prezent |
| 6 | **SKIPPED** | Tabel 5.4 | - | - | Manual (risc fragmentare bold) |
| 7 | **SKIPPED** | Tabel 5.4 | - | - | Manual (risc fragmentare bold) |
| 8 | **SKIPPED** | Tabel 5.4 | - | - | Manual (delta SVM necunoscut) |
| 9 | **SKIPPED** | para index 411 (Cap.5) | "SVM scade cu 3,1pp" | ? | Depinde de step 8 (vezi nota) |
| 10 | **SUCCESS** | para index 451 (Cap.6) | "0,506 vs. 0,482 și 0,476" | "0,506 vs. 0,435 și 0,476" | OLD absent, NEW prezent |
| 11 | **SUCCESS** | para index 451 (Cap.6) | "+0,3pp" | "+2,5pp" | OLD absent, NEW prezent |
| 12 | **SUCCESS** | para index 451 (Cap.6) | "+8,2pp" | "+6,9pp" | OLD absent, NEW prezent |

**Total: 8 SUCCESS, 4 SKIPPED (3 by design + 1 dependency)**

---

## Nota Step 9 (delta SVM 16f -> 19f)

Textul curent: "SVM scade cu 3,1pp de la 16 la 19 features"

Problema: delta-ul depinde de SVM F1 Macro pe 16 features (sursa JSON/XML), care:
- Nu e in CIFRE_CANONICE.md (acoperit doar modelul pe 19 features)
- Nu se poate calcula fara re-rularea experimentului pe 16 features cu pipeline-ul fixat
- Pasul 8 (care ar fi calculat delta) e skip

Calcul ipotetic (daca SVM 16f a ramas neschimbat la ~0.513):
- Nou delta = 0.4347 - 0.513 = -0.0783 = -7.8pp
- Dar ipoteza "SVM 16f neschimbat" e nesigura

**Recomandare manuala pentru Step 9:**
1. In Word, deschide Tabel 5.2 (sau Tabel 5.3) si noteaza valoarea SVM F1 Macro pe 16 features
2. Calculeaza: delta = 0.4347 - SVM_16f
3. In para#997 (index 411), Find "SVM scade cu 3,1pp" si Replace cu "SVM scade cu X,Xpp"
4. ATENTIE: daca SVM 16f din Tabel 5.2 e si el stale (probabil e), intreaga comparatie 16f vs 19f trebuie re-evaluata dupa re-rulare. Poate fi mai safe sa reformulezi paragraful fara cifra exacta.

---

## Verificare post-save

Textul paragrafului Cap.6 (para index 451) dupa toate inlocuirile:

> "Superioritatea Random Forest pe date extinse. Random Forest depășește SVM și XGBoost pe configurația cu 19 features (F1 Macro 0,506 vs. 0,435 și 0,476). Acest rezultat este consistent cu literatura de specialitate privind robustețea modelelor tree-based pe date tabulare de dimensiune medie. Confirmarea prin hyperparameter tuning sistematic (RandomizedSearchCV) arată că RF este near-optimal din parametrii default (+2,5pp după tuning), în timp ce XGBoost beneficiază cel mai mult de tuning (+6,9pp), fără a depăși însă RF."

---

## Ce ramane de facut manual (checklist)

### Tabel 5.4 (steps 6, 7, 8)
1. Celula SVM F1 Macro: schimba "0,482" -> "0,4347"
2. Celula SVM F1 Active: schimba "0,38" -> "0,25"  
3. Celula Delta SVM: schimba "-3,1pp" -> valoarea recalculata (sau "n/a" daca SVM 16f e stale)

### Step 9 (para#997 / index 411)
4. Find "SVM scade cu 3,1pp" -> replace cu valoarea corecta (vezi nota mai sus)

### Phase 2 (adaugari)
5. Paragraf reproducibilitate (766 train / 192 test / hash)
6. Randuri MCC/QWK in Tabel 5.4
7. Paragraf explicativ MCC/QWK

### Phase 3 (Cap.7 extensii)
8. Paragrafe B1, B2, D1c, D11a, D14c

### Phase 4 (comentarii Buiu)
9. c808, c812, c837, c856, c858, c866, c896, c902, c974

---

## Concluzii

- **8 din 9 inlocuiri text aplicate cu succes** prin python-docx
- **Integritate document confirmata**: 9 comentarii + 9 commentRangeStart intacte
- **Zero riscuri detectate**: inlocuirile au fost in body paragraphs (nu tabele), single-occurrence, cu verificare bidirectionala (old absent + new present)
- **1 step FLAG (step 9)**: necesita recalculare delta SVM care depinde de re-rulare experiment 16f
