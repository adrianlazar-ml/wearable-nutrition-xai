# SPRINT 4 BRIEFING

**Generat:** 28 mai 2026, după validare raport `SPRINT_4_RECON.md` (Cowork)
**Scope:** Corectare `.docx` disertație: cifre stale, distribuție clase, comentarii Buiu, paragrafe noi Cap.7
**Documentul activ:** `_referinte_upb/Disertatie_Lazar_Adrian_2026_Mai_CB.docx` (modificat 25.05.2026)
**Buget timp estimat:** 9-11h în 3-4 sesiuni (29 mai - 4 iunie)
**Deadline upload:** 15 iunie 2026, 23:59 (18 zile disponibile)

---

## 0. Pre-flight checklist (BLOCANTE pentru pornire)

- [ ] **Backup `.docx`** efectuat local + cloud personal (NU OneDrive UPB)
- [ ] **Email Buiu** trimis pentru disponibilitate semnătură cerere susținere 5-11 iunie
- [ ] **Working copy** creată: `Disertatie_Lazar_Adrian_2026_Sprint4_WIP.docx` (lucrezi pe asta, păstrezi `_CB.docx` ca master)
- [ ] **Tag git** pre-Sprint 4 pe repo proiect: `git tag pre-sprint-4-2026-05-28`

---

## 1. Decizii confirmate (din conversația 28.05.2026)

| Decizie | Confirmată | Implicații |
|---|---|---|
| **A. Re-rulare scripturi 16f/19f** | DA | +1,5h efort. Acoperă deltas Tabel 5.4 + Tabel 5.2 SVM 16f |
| **B. Adăugare MCC, QWK, hash, N train/test** | DA | +1h efort. Extinde Tabel 5.4 cu 2 coloane noi + secțiune reproducibilitate Cap.5 |
| **C. Editare manuală în Word, NU Cowork** | DA | Cowork = recon only, editarea o face Adrian în Word cu acest briefing ca ghid |

---

## 2. Backlog prioritizat pe 4 sesiuni

### Sesiunea 1: Re-rulări scripturi (3-4h, 29-30 mai)

Obiectiv: regenerare CSV-uri pentru a valida toate cifrele canonice și a obține valorile 16f curate.

```powershell
# 1. Hyperparameter tuning (validare +2,49pp / +6,87pp)
python experiments_new.py --experiment 1
# Output: outputs/tables/tuning_comparison.csv

# 2. Comparație 16f vs 19f (Tabel 5.4 deltas)
python ablation_study.py
python compare_16f_vs_19f.py  # sau scriptul echivalent care produce Tabel 5.4

# 3. SMOTE comparison post-fix (validare F1 0,506)
python smote_comparison.py

# 4. Regenerare CIFRE_CANONICE.md
python generate_canonical_numbers.py

# 5. Sanity check final
python verify_disertatie_numbers.py
```

**Checkpoint după Sesiunea 1:**
- [ ] Confirmă tuning gains: RF +2,49pp, XGB +6,87pp, SVM +4,47pp
- [ ] Confirmă SVM 16f baseline (pentru recalcul delta -3,1pp Tabel 5.4)
- [ ] Confirmă F1 Active SVM 16f (pentru Tabel 5.2)
- [ ] Toate cifrele din CIFRE_CANONICE.md match cu CSV-urile regenerate

### Sesiunea 2: Corectări `.docx` cifre stale + adăugări (3-4h, 31 mai - 1 iunie)

Toate în ordinea de mai jos (Cap.3 → Cap.4 → Cap.5 → Cap.6). Vezi Sec. 3 pentru tabel detaliat.

1. Cap.3 para#680: distribuție clase (Normal 708 → 710)
2. Cap.4 para#796: distribuție clase (Normal 706 → 710, Active 45 → 46, procent 4,7% → 4,8%)
3. Cap.5 para#894: distribuție clase (idem)
4. Cap.5 Tabel 5.4: SVM F1 Macro (0,482 → 0,4347), SVM F1 Active (0,38 → 0,25), delta SVM (recalculat după Sesiunea 1)
5. Cap.5 secțiune reproducibilitate: adăugare hash, N train (766), N test (192)
6. Cap.5 Tabel 5.4 extins: 2 coloane noi (MCC, QWK) pentru toate cele 3 modele
7. Cap.6 para#1105: SVM (0,482 → 0,4347), tuning gains (+0,3pp → +2,49pp, +8,2pp → +6,87pp)
8. Cap.6 para#997: delta SVM scăderea 16f→19f recalculată

### Sesiunea 3: Comentarii Buiu c808-c902 (2-3h, 2-3 iunie)

8 comentarii fără c974 (acoperit în Sesiunea 2). Vezi Sec. 5 pentru fiecare cu răspuns operativ.

Ordine recomandată:
1. c808, c812 (Cap.1, terminologie + titlu cap.2): 10 min
2. c896 (screenshot Streamlit): 15 min (rulează Streamlit, fă captură, plasează la para#816)
3. c902 (formatare Oura OAuth): 10 min
4. c856, c858 (traducere română Cap.3): 30 min
5. c837 (explicit termen Cap.2): 10 min
6. **c866 (ecuații Equation Editor)**: 60-90 min — singurul time-sink. Audit toate ecuațiile din Cap.3, Cap.5; convertește la Equation Editor Word

### Sesiunea 4: Paragrafe noi Cap.7 (2-3h, 3-4 iunie)

5 paragrafe noi/extinse. Texte template în Sec. 6.

1. B1 (extinde paragraf N=1 existent cu argumente metodologice consolidate): 30 min
2. B2 (paragraf NOU despre pseudo-etichetă și ablation A/B/C): 45 min
3. D1c (extinde paragraf calorie cu disclaimer non-medical): 20 min
4. D11a (extinde paragraf feedback loop cu RL future direction): 20 min
5. D14c (extinde paragraf praguri T3 cu sensitivity analysis reference): 20 min

---

## 3. Tabel cifre stale → canonice (obligatoriu)

Ordonate pe capitole pentru execuție secvențială.

| # | Capitol | Para# | Locație | Stale | Canonic | Sursa canonic |
|---|---|---|---|---|---|---|
| 1 | Cap.3 | #680 | Bias algoritmic, distribuție | "Normal = 708" | "Normal = 710" | CIFRE_CANONICE.md s2 |
| 2 | Cap.4 | #796 | Distribuție clase | "Normal: 706 zile (74,1%)" | "Normal: 710 zile (74,1%)" | CIFRE_CANONICE.md s2 |
| 3 | Cap.4 | #796 | Distribuție clase | "Active: 45 zile (4,7%)" | "Active: 46 zile (4,8%)" | CIFRE_CANONICE.md s2 |
| 4 | Cap.5 | #894 | Praguri T3 | "Normal 706 zile (74,1%)" | "Normal 710 zile (74,1%)" | CIFRE_CANONICE.md s2 |
| 5 | Cap.5 | #894 | Praguri T3 | "Active 45 zile (4,7%)" | "Active 46 zile (4,8%)" | CIFRE_CANONICE.md s2 |
| 6 | Cap.5 | #988 | Tabel 5.4 celulă SVM F1 Macro | "0,482" | "0,4347" | mcc_qwk_results.csv |
| 7 | Cap.5 | #992 | Tabel 5.4 celulă SVM F1 Active | "0,38" | "0,25" | mcc_qwk_results.csv |
| 8 | Cap.5 | #997 | Text "SVM scade cu 3,1pp" | "3,1pp" | recalculat după Sesiunea 1 | derivat ablation_study.csv |
| 9 | Cap.6 | #1105 | Text "0,506 vs 0,482 si 0,476" | "0,482" | "0,435" (3 zecimale) sau "0,4347" | mcc_qwk_results.csv |
| 10 | Cap.6 | #1105 | Text "RF tuning +0,3pp" | "+0,3pp" | "+2,5pp" (sau "+2,49pp" precis) | tuning_comparison.csv post-Sesiunea 1 |
| 11 | Cap.6 | #1105 | Text "XGBoost +8,2pp" | "+8,2pp" | "+6,9pp" (sau "+6,87pp" precis) | tuning_comparison.csv post-Sesiunea 1 |

**Notă pe rotunjire**: păstrează 3 zecimale (0,435 / +2,5pp) în text discursiv pentru lizibilitate. Folosește 4 zecimale (0,4347) doar în celulele de tabel.

---

## 4. Cifre de adăugat (NU existau în `.docx`)

| # | Valoare | Locație recomandată | Format propus |
|---|---|---|---|
| 12 | MCC RF=0,1416, XGB=0,1503, SVM=0,0956 | Tabel 5.4 extins, 2 coloane noi | adaugă rând „MCC" sub „F1 Active" |
| 13 | QWK RF=0,2551, XGB=0,1805, SVM=0,1709 | Tabel 5.4 extins | adaugă rând „QWK" sub „MCC" |
| 14 | Hash dataset `63bc3a2915bae864` | Cap.5 secțiune reproducibilitate (după Tabel 5.4) | „Dataset-ul utilizat are hash SHA256 truncat la 16 caractere `63bc3a2915bae864`, calculat cu lineterminator LF pentru reproductibilitate cross-platform Windows/Linux." |
| 15 | N train = 766, N test = 192 | Cap.5 split section (înainte de Tabel 5.4) | „Setul de date este împărțit cronologic 80%/20% (766 zile train / 192 zile test), păstrând ordinea temporală pentru a evita data leakage." |

**Justificare adăugări (pentru defense)**:
- MCC și QWK sunt metrici robuste pentru clasificare imbalanced (Rest 21%, Normal 74%, Active 5%). F1 Macro singur poate fi argumentat ca insuficient de o comisie AI/ML focus.
- Hash + N train/test sunt argumente directe pentru reproductibilitate, anticipând întrebări de tip „cum dovediți?".

---

## 5. Comentarii Buiu cu răspuns operativ

| ID | Para# | Cap | Text Buiu | Acțiune recomandată | Effort |
|---|---|---|---|---|---|
| c808 | #155 | 1 | „Dispozitive purtabile" | Înlocuiește „wearables" cu „dispozitive purtabile" la para#155 și verifică consistența în Cap.1. Nu schimba global, doar Cap.1 (în secțiunile tehnice „wearable data" rămâne dacă e tehnic) | 5 min |
| c812 | #175 | 1 | „Starea actuala in domeniul abordat" | Redenumește titlul Cap.2 din actualul (probabil „Stadiul actual" sau „State of the Art") în „Starea actuală în domeniul abordat". Actualizează și TOC | 5 min |
| c837 | #308 | 2 | „Ori in clar, ori deloc" | Buiu cere clarificare la un termen/acronim la para#308 (zona Tabel 5). **Trebuie să vezi tu ce e exact la para#308** — probabil un acronim introdus fără explicare. Acțiune: expandare acronim la prima apariție sau eliminare dacă nu e esențial | 10 min |
| c856 | #470 | 3 | „In romana" | Tradu în română un pasaj la para#470 (probabil termen tehnic engleză uitat: „feature engineering", „lag-1 features" etc.) | 15 min |
| c858 | #527 | 3 | „In romana" | Idem para#527 (secțiunea de clasificare) | 15 min |
| c866 | #630 | 3 | „Scrie-o cu editorul de ecuatii, la fel peste tot unde mai ai ecuatii" | **TIME SINK**. Convertește ecuația SHAP la para#630 + audit toate ecuațiile din document (probabil ~5-10 ecuații). Folosește Word Equation Editor (Insert → Equation). Stilul standard academic | 60-90 min |
| c896 | #816 | 5 | „Un screenshot?" | Rulează `streamlit run streamlit_app_v2.py`, fă captură (Win+Shift+S), inserează ca Figure 5.X la para#816 cu caption Word auto | 15 min |
| c902 | #874 | 5 | „Atentie la formatare" | Inspectează para#874 (Oura OAuth) pentru: stiluri inconsistente (font, spacing), code blocks neformatate ca „Code" style, link-uri rupte | 10 min |
| c974 | #1105 | 6 | „Atentie formatare" | Acoperit în Sesiunea 2 (paragraful e oricum rescris pentru cifre stale). Verifică în plus: bold-uri inconsistente, font Calibri uniform, linie spacing 1,15 | 0 min |

**Total Sesiunea 3:** 2h15min - 3h conform timing-urilor.

---

## 6. Texte ready-to-paste pentru Cap.7 (5 paragrafe)

Nota: paragrafele B1, D11a, D14c există deja în formă scurtă în Cap.7. Le **extinzi** cu textul de mai jos. B2 și D1c sunt **paragrafe NOI**, le inserezi la pozițiile indicate.

### 6.1 B1 — Extindere paragraf N=1 (existent)

**Locație**: după paragraful actual care începe cu „Rezultatele reflectă pattern-urile fiziologice ale unui singur individ..."

**Text de adăugat (consolidare metodologică)**:

```
Limitarea N=1 este atenuată metodologic prin trei elemente: (1) durata extinsă a colectării
(>1.000 de zile efective, 958 zile utilizabile după agregare lag-1), care produce un volum
de observații temporale suficient pentru analiza variabilității intra-individuale; (2) split-ul
temporal strict 80%/20% (766 train, 192 test, fără overlap), aliniat cu best practice din
literatura time-series ML (Bergmeir si Benitez, 2012; Saeb et al., 2017; Sherman et al., 2018),
care evită inflația artificială a performanței asociată cu random k-fold (Kapoor si Narayanan,
2023); (3) modelul produs (`rf_frozen_20260331.joblib`, hash dataset `63bc3a2915bae864`)
este înghețat și reproductibil cross-platform, permițând validare prospectivă pe date noi
(propusă ca direcție viitoare imediată).

Aceste măsuri nu elimină limitarea de generalizabilitate populațională, dar transformă
studiul dintr-un single-case anecdotic într-un proof-of-concept reproductibil pentru
pipeline-ul wearable-only de clasificare a stării fiziologice zilnice.
```

### 6.2 B2 — Paragraf NOU despre pseudo-etichetă (de inserat înainte de B1)

**Locație recomandată**: în Cap.7 secțiunea „Limitări metodologice", înainte de paragraful N=1.

**Text COMPLET de inserat**:

```
- Pseudo-etichetă vs. adevăr fiziologic

Eticheta clasă (Rest/Normal/Active) este derivată din scorurile Oura Readiness și Sleep,
care sunt construcții proprietare ale producătorului, nu măsurători fiziologice gold standard
validate clinic. Această caracteristică, comună sistemelor de clasificare bazate pe metrici
compozite din wearables consumer (Doherty et al., 2025; Hagmann et al., 2023), introduce
trei limitări metodologice care trebuie declarate explicit.

În primul rând, modelul prezice un construct operațional (cum codifică Oura starea fiziologică
zilnică pe baza propriilor algoritmi proprietari), nu o stare biologică latentă independentă.
Termenul folosit consistent în această lucrare este „pseudo-etichetă", nu „ground truth".

În al doilea rând, există un risc teoretic de circularitate: features-uri derivate din aceleași
date pe baza cărora Oura calculează scorurile (rhr_avg, hrv_balance, sleep_efficiency) ar
putea genera o suprapunere parțială între input și target. Pentru a testa empiric această
ipoteză, am realizat trei variante experimentale de ablation (A: toate features, B: fără
features Oura derivate direct, C: doar features Apple Watch). Diferența de performanță
între variantele A și B este de 3,6pp F1 Macro, ceea ce indică suprapunere parțială dar
nu colaps al modelului. Pentru comparație, regimul de reconstrucție pură descris de Hagmann
et al. (2023) ar produce performanță aproape perfectă cu modele simple, ceea ce nu se
observă în acest studiu (RF F1 Macro = 0,506).

În al treilea rând, autocorelația de ordin 1 a etichetei (lag-1: 0,234, măsurată pe seria
temporală a clasei) implică o componentă de persistență care contribuie la performanța
modelului. Această contribuție este moderată, nu dominantă, dar trebuie raportată explicit.
Un test de persistence baseline (`y_pred[t] = y_true[t-1]`) este propus ca direcție de
validare suplimentară.

Aceste limitări nu invalidează contribuția lucrării, dar îngustează claim-ul: sistemul
clasifică starea zilnică relativ la construcția operațională Oura, nu validează independent
fiziologia recuperării.
```

### 6.3 D1c — Extindere paragraf deficit caloric

**Locație**: după paragraful existent despre formula 25 kcal/kg/zi din AI4Food.

**Text de adăugat (disclaimer)**:

```
Recomandările calorice generate de sistem sunt sugestii informative bazate pe formule
evidence-based din literatura nutrițională (Mifflin-St Jeor pentru BMR, formula 25 kcal/kg
din studiul clinic AI4Food (Lacruz-Pleguezuelos et al., 2025)). Acestea NU constituie
prescripții medicale și nu înlocuiesc consultarea unui medic nutriționist sau a unui
profesionist din domeniul sănătății. Utilizatorii cu condiții medicale (diabet, boli
cardiovasculare, tulburări metabolice, sarcină, lactație) trebuie să consulte un specialist
înainte de aplicarea oricăror recomandări nutriționale automate. Acest disclaimer este
afișat explicit în interfața Streamlit a sistemului, conform recomandărilor de proiectare
pentru sisteme de sănătate AI (Saad si Islam, 2025).
```

### 6.4 D11a — Extindere paragraf feedback loop

**Locație**: după paragraful existent „Absența feedback loop-ului nutritional".

**Text de adăugat (RL future direction)**:

```
Închiderea bucle de feedback ar transforma sistemul dintr-un clasificator static într-un
agent adaptiv. Două direcții tehnice sunt promițătoare. Prima este integrarea unui modul
de tracking nutrițional (jurnal alimentar manual, recunoaștere automată din imagini cu
modele vision-language, sau date CGM pentru utilizatorii cu acces) care să producă semnal
de aderență la recomandări. A doua este Reinforcement Learning offline pe date longitudinale,
unde recomandarea zilnică devine acțiune și starea fiziologică din ziua t+1 devine reward,
permițând ajustarea politicii fără explorare riscantă pe utilizator (Tripathy et al., 2024).
Implementarea acestor extensii depășește scopul prezentei lucrări dar constituie o direcție
naturală de cercetare doctorală sau de proiect post-master.
```

### 6.5 D14c — Extindere paragraf praguri T3

**Locație**: după paragraful existent „Definirea claselor prin praguri".

**Text de adăugat (sensitivity analysis cross-reference)**:

```
Robustețea pragurilor T3 este evaluată empiric prin analiza de sensibilitate prezentată în
secțiunea 5.X (Tabel 5.5), care testează patru seturi alternative de praguri (T1 stricte,
T2 quantile-based, T3 conform documentației Oura, T4 relaxate). Rezultatele arată că F1
Macro variază între 0,325 (T1 stricte, distribuție puternic dezechilibrată) și 0,506 (T3),
cu T3 obținând cel mai bun rezultat dintre toate variantele testate. T3 sunt fundamentate
pe pragurile oficiale Oura („Pay Attention" sub 60, „Optimal" peste 85), ceea ce le acordă
validitate semantică pe lângă cea empirică. Nu există însă o validare clinică externă a
acestor praguri, iar transferul la alți utilizatori cu baseline-uri fiziologice diferite
ar necesita re-calibrare specifică individului.
```

---

## 7. Sanity check final (Sesiunea 4 încheiere)

După toate modificările, rulează în această ordine:

1. **Word find & replace audit**: caută în `.docx` (Ctrl+F):
   - „0,482" → trebuie ZERO ocurențe (toate înlocuite cu 0,4347 sau 0,435)
   - „708" în context distribuție → ZERO ocurențe
   - „706 zile" → ZERO ocurențe
   - „Active: 45" sau „Active 45 zile" → ZERO ocurențe
   - „+0,3pp" sau „+8,2pp" → ZERO ocurențe
   - „wearables" în Cap.1 → ZERO ocurențe (după c808)

2. **Rulare script verificare**:
   ```powershell
   python verify_disertatie_numbers.py
   ```
   Trebuie 100% verzi. Dacă vreo cifră e roșie, întoarce-te la Sesiunea 2.

3. **Verificare comentarii Buiu**:
   - Toate 9 comentariile rezolvate (poți marca „Resolved" în Word după ce ai aplicat fixul)
   - NU șterge comentariile încă, lasă-le pe „Resolved" — Buiu le poate revedea

4. **Captions auto**: verifică Tabel 5.4 extins are caption Word auto, nu hardcoded. La fel pentru orice tabel/figură nouă.

5. **Backup nou**: după sanity check pass, salvează `Disertatie_Lazar_Adrian_2026_Sprint4_DONE.docx` ca milestone.

---

## 8. Riscuri și mitigation

| Risc | Probabilitate | Impact | Mitigare |
|---|---|---|---|
| Re-rularea scripturilor produce cifre diferite de CIFRE_CANONICE.md | Mică | Mare (rescriere parțială) | Verifică hash dataset ÎNAINTE de re-rulare. Dacă match, cifrele trebuie să fie identice |
| c866 ecuații durează >90 min | Medie | Mediu (consumă din buffer) | Dacă depășești 90 min, oprește, fac doar ecuația SHAP de la para#630 și flag-uiește restul ca „TODO post-Sprint 4" |
| Comentarii Buiu noi după review-ul lui pe versiunea Sprint 4 | Mare | Mediu | Anticipează 1 rundă suplimentară de corecții în buffer săpt. 3 (12-15 iunie) |
| Corupere `.docx` (formatare, captions, cross-refs) | Mică | Mare | Backup-uri la final de sesiune. Test deschidere fișier în Word + LibreOffice (verificare cross-tool) |

---

## 9. Pași următori imediați pentru Adrian

1. **Acum (28 mai seara/29 mai dimineață):**
   - Trimite email Buiu pentru disponibilitate iunie
   - Fă backup `.docx` local + cloud personal
   - Tag git `pre-sprint-4-2026-05-28` pe repo

2. **Sesiunea 1 (29-30 mai, 3-4h):**
   - Rulează scripturile din Sec. 2
   - Confirmă cifrele din checkpoint Sesiunea 1
   - Trimite-mi summary cu output-urile (tuning_comparison.csv + CIFRE_CANONICE.md regenerat) pentru validare înainte de Sesiunea 2

3. **Sesiunile 2-4:** execută conform Sec. 2 cu acest briefing ca ghid permanent

---

## ANEXA A: Referință raport Cowork

Raportul complet `SPRINT_4_RECON.md` (generat de Cowork 28.05.2026) conține toate ocurențele de cifre din `.docx` cu paragraf# exact. Folosește-l ca lookup când lucrezi în Word.

## ANEXA B: Contradicție surse rezolvată

**Conflict identificat**: tuning gains diferă între CIFRE_CANONICE.md și RECAPITULARE_EXPERIMENTE.md.

| Sursă | RF gain | XGB gain | SVM gain |
|---|---|---|---|
| CIFRE_CANONICE.md (canonic curent) | +2,49pp | +6,87pp | +4,47pp |
| RECAPITULARE_EXPERIMENTE.md (snapshot istoric) | +0,30pp | +8,21pp | +1,77pp |
| `.docx` Cap.6 para#1105 (stale) | +0,3pp | +8,2pp | nu apare |

**Decizie**: CIFRE_CANONICE.md prevalează (memoria #12, handoff Sprint 6). Cifrele din .docx sunt stale și au fost preluate din RECAPITULARE când acesta era cel mai recent. Re-rularea `experiments_new.py --experiment 1` în Sesiunea 1 va confirma cifrele canonice.

Dacă re-rularea produce o a treia versiune, oprește execuția Sprint 4 și flag-uiește pentru investigare (improbabil cu hash dataset identic).
