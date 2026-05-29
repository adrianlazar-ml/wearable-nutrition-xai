# Sprint 4 Polish Audit

**Data:** 2026-05-28  
**Working file:** `_referinte_upb/Disertatie_Lazar_Adrian_2026_Sprint4_WIP.docx` (READ-ONLY)  
**Scope:** Pregatire Phase 4 (comentarii Buiu) + Sprint 5 polish

---

## Sectiunea 1: Inventar ecuatii (pentru c866)

### Constatare principala

**ZERO ecuatii OMML (Equation Editor nativ)** in intregul document. Toate formulele sunt scrise ca text plat cu simboluri Unicode.

### Inventar formule

| # | Para# | Capitol | Continut formula | Format | Categorie | Effort (min) |
|---|---|---|---|---|---|---|
| 1 | 630 | Cap.3 | `SHAP_ij = Σ_{S⊆F\{j}} [\|S\|! · (\|F\|−\|S\|−1)! / \|F\|!] · [f(x_{i,S∪{j}}) − f(x_{i,S})]` | Text plat Unicode | **MUST** | 10 |
| 2 | 631 | Cap.3 | `f(x_{i,S∪{j}})`, `f(x_{i,S})` (definitii variabile cu subscripte) | Text plat Unicode | **SHOULD** | 5 |
| 3 | 644 | Cap.3 | `TDEE = BMR + TEF + PA` (mentionata in text, nu izolata) | Text plat inline | SKIP | - |
| 4 | 645 | Cap.3 | `25 kcal/kg/zi` (formula calorica simplificata) | Text plat inline | SKIP | - |
| 5 | 891 | Cap.5 | `Rest: readiness_score < 60 SAU sleep_score < 60` | Text plat | **SHOULD** | 3 |
| 6 | 892 | Cap.5 | `Active: readiness_score ≥ 85 ȘI sleep_score ≥ 75` | Text plat | **SHOULD** | 3 |
| 7 | 632 | Cap.3 | Proprietati SHAP: `Σ_j SHAP_ij = f(x_i) - E[f(x)]` (implicita in text) | Text plat inline | **SHOULD** | 5 |

### Ecuatii din Tabel 3.2 (headers tehnice)

Tabel 3.2 contine coloane cu formule/metrici (SDNN, RMSSD, pNN50). Acestea sunt headers de tabel, NU ecuatii de convertit.

### Rezumat c866

| Categorie | Count | Effort total |
|---|---|---|
| MUST convert | 1 | ~10 min |
| SHOULD convert | 4 | ~16 min |
| SKIP | 2+ | - |
| **Total effort c866** | | **~26 min** |

**Actiune principala:** Convertirea formulei SHAP (para#630) la Equation Editor. In Word: Insert > Equation, scrie formula LaTeX-like. Restul SHOULD sunt formule simple (2-3 variabile cu subscript).

**Sugestie:** Pentru para#630, in Word Equation Editor tastezi:
```
SHAP_{ij} = ∑_{S⊆F\{j}} (|S|! · (|F|-|S|-1)!)/(|F|!) · [f(x_{i,S∪{j}}) - f(x_{i,S})]
```

---

## Sectiunea 2: Audit em dashes (—)

### Constatare

**6 em dashes gasite** in document.

| # | Para# | Capitol | Context | Tip | Recomandare |
|---|---|---|---|---|---|
| 1 | 466 | Cap.3 | "fizică (PA — physical activity) din" | Explicatie inline | Inlocuieste cu virgula: "fizică (PA, physical activity) din" |
| 2 | 471 | Cap.3 | "Tabel 3.2 — Indicatorii fiziologici" | Caption tabel | Inlocuieste cu punct: "Tabel 3.2. Indicatorii fiziologici" |
| 3 | 534 | Cap.3 | "Tabel 3.3 — Comparație sintetică" | Caption tabel | Inlocuieste cu punct: "Tabel 3.3. Comparație sintetică" |
| 4 | 651 | Cap.3 | "Tabel 3.5 — Distribuția macronutrienților" | Caption tabel | Inlocuieste cu punct: "Tabel 3.5. Distribuția macronutrienților" |
| 5 | 950 | Cap.5 | "—" (singur in celula tabel, inseamna "N/A") | Valoare tabel | **PASTREAZA** (conventie tabel pentru "nu se aplica") |
| 6 | 1608 | Anexe | "A2.3. Setul de date integrat — exemple per clasă" | Caption anexa | Inlocuieste cu doua puncte: "A2.3. Setul de date integrat: exemple per clasă" |

### Rezumat

- **5 de inlocuit** (4 caption-uri de tabele + 1 explicatie inline)
- **1 de pastrat** (valoare "N/A" in celula de tabel)
- Nota: Tabelele Cap.5 (Tabel 5.1 - 5.7) folosesc deja punct in caption ("Tabel 5.4. Comparația..."). Inconsistenta este doar in Cap.3 (3.2, 3.3, 3.5 folosesc —, iar 3.1 si 3.4 folosesc ": " sau punct).

**Actiune:** Ctrl+H global: Find "—" Replace cu ". " DAR manual, nu Replace All (risc de a schimba celula tabel de la para#950).

---

## Sectiunea 3: Audit cross-references

### Constatare generala

- **Referinte de tip text plat** (NU Word REF fields). NU se auto-actualizeaza cu F9.
- **TOC** (Table of Contents) foloseste PAGEREF fields = OK, se actualizeaza cu F9.
- **79 PAGEREF fields** in TOC.
- **0 REF/BOOKMARK cross-references** in body text.

### Status referinte

| Tip referinta | Total | OK | Potentiale probleme | Note |
|---|---|---|---|---|
| Tabel X.Y | 13 | 13 | 0 | Toate tabelele referentiate exista ca captions |
| Figura X.Y | 2 | AMBIGUOUS | - | "Figura 4.1" referita in 2 locuri, dar caption-ul Figura 4.1 nu e detectat (posibil imagine fara caption text sau caption in alt format) |
| Subcapitol X.Y | 18 | Probabil OK | - | Numerotare generata de Word multilevel list; sectiunile exista dar fara numerotare literala in XML |
| Anexa X | 0 ref | - | - | Nicio referinta la anexe din body |

### Detaliu Figura 4.1

Para#700 si para#870 refera "Figura 4.1" dar captionul nu a fost detectat in formatul standard. Posibilitati:
- Caption-ul e in alt format (nu incepe cu "Figura")
- Imaginea e inserata fara caption formal
- **FLAG: verifica manual daca Figura 4.1 exista si are caption.**

### Detaliu Subcapitol 6.4

Referenciat de 4 ori (para#702, 807, 976, 1122). Verifica manual ca exista o sectiune 6.4 in Cap.6.

### Recomandare

Referintele sunt text plat, nu Word fields. Conversia la Word REF fields ar necesita:
1. Inserare bookmarks pe fiecare caption/heading
2. Inlocuire text "Tabel 5.4" cu Insert > Cross-reference
3. Effort estimat: ~45 min pentru toate 33 referintele

**Recomandare: NU converti acum.** Text plat e acceptabil pentru disertatie. Doar asigura-te ca actualizezi manual daca reordonezi tabele/sectiuni.

---

## Sectiunea 4: Acronime nedefinite la prima aparitie

### Cap.1-3 (cele mai vizibile la defense)

| Acronim | Prima aparitie | Expansiune gasita | Status | Actiune |
|---|---|---|---|---|
| HRV | para#59 (cuprins) | para#148 (Cap.1) | **WARN** | Expandeaza la prima mentiune body: "Heart Rate Variability (HRV)" |
| RHR | para#59 (cuprins) | para#426 (Cap.3) | **WARN** | Expandeaza la prima mentiune body: "Resting Heart Rate (RHR)" |
| SVM | para#60 (cuprins) | para#131 (Rezumat) | OK | Expanded in Rezumat |
| SMOTE | para#61 (cuprins) | para#581 (Cap.3) | **WARN** | Expandeaza mai devreme: "Synthetic Minority Over-sampling Technique (SMOTE)" |
| RBF | para#527 (Cap.3) | para#532 (Cap.3) | OK | Expanded 5 paragrafe mai tarziu, acceptabil |
| SDNN | para#438 (Cap.3) | para#770 (Cap.5) | **WARN** | Expandeaza la prima mentiune Cap.3: "Standard Deviation of NN intervals (SDNN)" |
| SHAP | para#62 (cuprins) | para#628 (Cap.3) | OK | Expanded complet: "SHapley Additive exPlanations (SHAP)" |
| XAI | para#62 (cuprins) | para#363 (Cap.2) | OK | Expanded in context |
| TDEE | para#507 (Cap.3) | para#644 (Cap.3) | OK | Expanded: "total daily energy expenditure" |
| BMR | para#507 (Cap.3) | para#644 (Cap.3) | OK | Expanded: "basal metabolic rate" |
| MCC | para#585 (Cap.3) | para#585 (Cap.3) | OK | "Matthews Correlation Coefficient (MCC)" |

### Focus c837: para#308 "Ori in clar, ori deloc"

Heading-ul la para#308 este: **"Intervenții și feedback în timp real (RL)"**

Comentariul Buiu cere clarificarea. Problema: **"RL"** este ambiguu:
- Poate insemna "Reinforcement Learning" (invatare prin intarire)
- Nu este expandat nicaieri in document

**Actiune recomandata:** Schimba heading-ul la:
- Varianta A: "Intervenții și feedback în timp real (Reinforcement Learning)" - explicit
- Varianta B: "Intervenții și feedback în timp real" - elimina abrevierea complet (Buiu: "ori deloc")

**Recomandare: Varianta B** (elimina "(RL)"). Motivul: sectiunea NU discuta RL propriu-zis, ci feedback general. "RL" in titlu e misleading.

### Acronime NEVER EXPANDED (de interes)

| Acronim | Prima aparitie | Context | Propunere expansiune |
|---|---|---|---|
| RL | para#308 | Heading Cap.2 | Reinforcement Learning (sau elimina) |
| LIME | para#640 | Cap.3 text | Local Interpretable Model-agnostic Explanations |
| KNN | para#1147 | Cap.6 tabel | K-Nearest Neighbors |
| CVD | para#1160 | Cap.6 tabel | Cardiovascular Disease |
| NLP | para#1319 | Anexe/tabel | Natural Language Processing |

Nota: acronimele din tabelele de comparatie cu literatura (Cap.6, Anexe) sunt mai putin critice; comisia le cunoaste. Prioritate: RL (c837), LIME (Cap.3).

### para#871: "FORMATUL ESTE GRESIT DE AICI IN JOS"

**FLAG IMPORTANT:** La para#871 (Cap.5) exista textul: "FORMATUL ESTE GRESIT DE AICI IN JOS". Aceasta pare o nota de lucru ramasa in document. **TREBUIE STEARSA** inainte de defense.

---

## Sumar executiv

| Metrica | Valoare | Urgenta |
|---|---|---|
| Ecuatii MUST convert (c866) | **1** (formula SHAP) | Medie (10 min) |
| Ecuatii SHOULD convert | **4** | Low (16 min) |
| Total effort c866 | **~26 min** | - |
| Em dashes de inlocuit | **5** (din 6 total) | Low (5 min cu Ctrl+H) |
| Cross-references BROKEN | **1 FLAG** (Figura 4.1 caption nedetectat) | Verifica manual |
| Cross-references text plat (nu fields) | **33** | Accept as-is |
| Acronime UNDEFINED la prima aparitie | **5** prioritare (HRV, RHR, SMOTE, SDNN, RL) | Medie |
| Nota de lucru ramasa in document | **1** ("FORMATUL ESTE GRESIT...") | **CRITICA - sterge** |

### Prioritati (ordinea de executie recomandata)

1. **CRITICA:** Sterge "FORMATUL ESTE GRESIT DE AICI IN JOS" (para#871)
2. c837: Schimba heading "(RL)" -> elimina sau expandeaza
3. c866: Converteste formula SHAP (para#630) la Equation Editor
4. Em dashes: 5 inlocuiri rapide (Ctrl+H one-by-one)
5. Acronime: expandeaza HRV, RHR, SMOTE, SDNN la prima aparitie in body
6. Optional: converteste restul SHOULD ecuatii
7. Optional: verifica Figura 4.1 caption
