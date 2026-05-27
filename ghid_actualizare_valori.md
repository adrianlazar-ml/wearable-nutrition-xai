# ================================================================
# SUPERSEDED. Sursa unica de adevar pentru cifre: CIFRE_CANONICE.md
# ================================================================
# Cifrele "NOU" de mai jos au fost redactate cu valori intermediare
# post-fix rhr_avg. O parte au fost RECALCULATE in Sprint 2 (regenerare
# pipeline de-leaked) si NU mai sunt canonice. Divergente confirmate
# (verificate pe outputs/tables/ + CIFRE_CANONICE.md):
#   Ablation Combined:    ghid 0,482        ->  canonic 0,506
#   Ablation Oura / AW:   ghid 0,380/0,358  ->  canonic 0,428/0,324
#   SVM test F1 Macro:    ghid 0,482        ->  canonic 0,4347
#   SMOTE k=3:            ghid 0,482        ->  canonic 0,506
#   Threshold T3:         ghid 0,482        ->  canonic 0,506
# Foloseste acest ghid DOAR pentru locatiile/structura din docx, NU pentru
# valori. Verifica fiecare cifra in CIFRE_CANONICE.md inainte de a o scrie.
# ================================================================

# GHID ACTUALIZARE VALORI — Disertație Lazar Adrian
# Post-fix rhr_avg, date curate, 31.03.2026
# ================================================================
# INSTRUCȚIUNI: Deschide _referinte_upb/Disertatie_Lazar_Adrian_2026_Mai_CB.docx în Word.
# Parcurge fiecare secțiune de mai jos și fă Ctrl+H (Find & Replace).
# Bifează "Match case" unde e cazul.
# ================================================================


# ============================================================
# A. SCHIMBĂRI GLOBALE (afectează tot documentul)
# ============================================================

# A1. Best model: SVM → RF (text narativ)
#   ATENȚIE: NU face replace-all orbește! Caută fiecare instanță manual.
#   Unele referințe la SVM sunt corecte (ex: în Tabel 5.2 XML, SVM chiar
#   e bun pe 16 features XML). Schimbă doar unde SVM e prezentat ca
#   "cel mai bun model" pe 19 features / pipeline final.

LOCAȚII DE SCHIMBAT (7 locații):

1. SINOPSIS (~pag 1):
   VECHI: "cel mai bun model (SVM cu kernel RBF) obținând un F1 Macro de 0,517 pe un set de date de 957 de zile"
   NOU:  "cel mai bun model (Random Forest) obținând un F1 Macro de 0,506 pe un set de date de 958 de zile"

   VECHI: "îmbunătățire de 15,2 puncte procentuale"
   NOU:  "îmbunătățire de 10,2 puncte procentuale"

2. ABSTRACT (~pag 1):
   VECHI: "with the best model (SVM with RBF kernel) achieving an F1 Macro of 0.517 on a dataset spanning 957 days"
   NOU:  "with the best model (Random Forest) achieving an F1 Macro of 0.506 on a dataset spanning 958 days"

   VECHI: "a 15.2 percentage point improvement"
   NOU:  "a 10.2 percentage point improvement"

3. STREAMLIT descriere (Cap 5.6, ~pag 60):
   VECHI: "cel mai bun model (SVM pe 19 features)"
   NOU:  "cel mai bun model (Random Forest pe 19 features)"

4. Cap 6.1 INTERPRETARE:
   VECHI: "Cel mai bun model (SVM pe 19 features, SMOTE k=3, praguri T3) atinge un F1 Macro de 0,517"
   NOU:  "Cel mai bun model (Random Forest pe 19 features, SMOTE k=3, praguri T3) atinge un F1 Macro de 0,506"

   VECHI: "un F1 Macro de 0,517 demonstrează fezabilitatea"
   NOU:  "un F1 Macro de 0,506 demonstrează fezabilitatea"

   VECHI: "Superioritatea SVM pe date extinse. SVM cu kernel RBF depășește RF și XGBoost pe configurația cu 19 features, inversând ierarhia observată pe 16 features"
   NOU:  "Superioritatea Random Forest pe date extinse. Random Forest depășește XGBoost și SVM pe configurația cu 19 features (F1 Macro 0,506 vs. 0,476 și 0,482). Acest rezultat este consistent cu observațiile din literatura de specialitate privind robustețea RF pe date tabulare de dimensiune medie"
   NOTĂ: Rescrie tot paragraful despre "inversarea ierarhiei" — nu mai e cazul, RF e best pe ambele config.

5. Cap 6.2 RQ2:
   VECHI: "nu există un algoritm universal superior: SVM obține cel mai bun F1 Macro (0,517 pe 19 features), XGBoost cel mai mic CV std (2,8%, indicând stabilitate), iar RF oferă cel mai bun compromis între performanță și interpretabilitate"
   NOU:  "Random Forest obține cel mai bun F1 Macro (0,506 pe 19 features), combinând performanța absolută cu interpretabilitate nativă (feature importance + TreeExplainer rapid). XGBoost oferă cel mai mic CV std (3,7%), iar SVM cel mai bun scor de calibrare (Brier avg 0,156)"

   VECHI: "SMOTE cu k=3 s-a dovedit cea mai eficientă strategie de gestionare a dezechilibrului de clase (F1 Macro: 0,517 vs. 0,281 baseline)"
   NOU:  "SMOTE cu k=3 s-a dovedit cea mai eficientă strategie de gestionare a dezechilibrului de clase (F1 Macro: 0,482 vs. 0,281 baseline)"

6. Cap 7.1 CONCLUZII GENERALE:
   VECHI: "cu un F1 Macro de 0,517 (SVM, 19 features)"
   NOU:  "cu un F1 Macro de 0,506 (Random Forest, 19 features)"

   VECHI: "F1 Macro crește cu 15,2 puncte procentuale față de Oura singur și cu 16,9 puncte procentuale față de Apple Watch singur, iar pentru clasa Active îmbunătățirea este de aproximativ 2,6x"
   NOU:  "F1 Macro crește cu 10,2 puncte procentuale față de Oura singur și cu 12,4 puncte procentuale față de Apple Watch singur, iar pentru clasa Active îmbunătățirea este de aproximativ 2,6x"

   VECHI: "nu există un algoritm universal superior pe acest tip de date: alegerea optimă depinde de configurația features (SVM pe 19 features vs. XGBoost pe 16 features)"
   NOU:  "Random Forest oferă cel mai bun compromis pe acest tip de date (F1 Macro 0,506, near-optimal din parametrii default, confirmat prin hyperparameter tuning sistematic), iar rezultatul este consistent cross-configurație"

7. ORICE altă referință la "0,517" care nu e în tabelul threshold sensitivity T4/T6
   → Caută "0,517" în tot documentul. Fiecare instanță trebuie verificată individual.


# ============================================================
# B. TABELE DE ACTUALIZAT
# ============================================================

# B1. TABEL 5.4 — Comparația algoritmilor pe 19 features (sursa JSON)
# ----------------------------------------------------------------
VECHI:                        RF        XGBoost      SVM
  F1 Macro                   0,474      0,482       0,517
  F1 Active                  0,42       0,40        0,46
  Δ F1 Macro vs. 16f         +6,8pp     -1,5pp      +2,3pp

NOU:                          RF        XGBoost      SVM
  Accuracy                   64,1%      68,8%       64,6%
  F1 Macro                   0,506      0,476       0,482
  F1 Weighted                0,641      0,665       0,641
  F1 Rest                    0,32       0,29        0,31
  F1 Normal                  0,75       0,80        0,76
  F1 Active                  0,44       0,33        0,38
  CV F1 Macro (std)          41,7%(±8,1%) 36,1%(±3,7%) 40,3%(±5,1%)

TEXTUL DUPĂ TABEL — RESCRIE COMPLET:
  VECHI: "SVM pe 19 features atinge cel mai bun F1 Macro din toate configurațiile testate (0,517)..."
  NOU:  "Random Forest pe 19 features atinge cel mai bun F1 Macro (0,506), depășind XGBoost (0,476) și SVM (0,482). RF obține cel mai bun F1 Active (0,44), ceea ce sugerează că modelele bazate pe arbori valorifică mai eficient informația din features-urile suplimentare de somn."

  VECHI: "Îmbunătățirea de la 0,494 la 0,517 ... F1 Active crește de la 0,38 la 0,46"
  → Șterge sau rescrie cu valorile noi (comparația 16→19 features necesită re-calculare).


# B2. TABEL 5.5 — Threshold sensitivity
# ----------------------------------------------------------------
# Doar T3 se schimbă (era 0,517, acum 0,482). Restul se schimbă și ele:
VECHI:                                           F1 Macro
  T1: 55/55-80/70                                0,359
  T2: 55/55-85/75                                0,449
  T3: 60/60-85/75 *                              0,517
  T4: 60/60-90/80                                0,697*
  T5: 65/65-85/75                                0,434
  T6: 65/65-90/80                                0,637*
  T7: Q25/Q75                                    0,282

NOU:                                             F1 Macro    Distrib (R/N/A)
  T1: 55/55-80/70                                0,361       97/703/158
  T2: 55/55-85/75                                0,446       97/815/46
  T3: 60/60-85/75 *                              0,482       202/710/46
  T4: 60/60-90/80                                0,702*      202/753/3
  T5: 65/65-85/75                                0,434       360/552/46
  T6: 65/65-90/80                                0,679*      360/595/3
  T7: Q25/Q75 (64/66-77/77)                      0,294       356/450/152

TEXTUL DUPĂ TABEL:
  VECHI: "T3 obține cel mai bun F1 Macro (0,517)"
  NOU:  "T3 obține cel mai bun F1 Macro (0,482)"


# B3. TABEL 5.6 — Ablation study
# ----------------------------------------------------------------
VECHI:                        Nr feat   F1 Macro    F1 Active
  Doar Oura                    11       0,366       0,154
  Doar Apple Watch              8       0,348       0,176
  Ambele dispozitive           19       0,517       0,462

NOU:                          Nr feat   F1 Macro    F1 Active
  Doar Oura                    11       0,380       0,143
  Doar Apple Watch              8       0,358       0,188
  Ambele dispozitive           19       0,482       0,375

TEXTUL DUPĂ TABEL:
  VECHI: "cu 15,2 puncte procentuale superior față de Oura singur și 16,9 puncte procentuale superior față de Apple Watch singur"
  NOU:  "cu 10,2 puncte procentuale superior față de Oura singur și 12,4 puncte procentuale superior față de Apple Watch singur"

  VECHI: "F1 crește de la 0,176 (Apple Watch singur) la 0,462 (ambele dispozitive), o îmbunătățire de aproximativ 2,6x"
  NOU:  "F1 crește de la 0,188 (Apple Watch singur) la 0,375 (ambele dispozitive), o îmbunătățire de aproximativ 2,0x"


# B4. TABEL 5.7 — SMOTE comparison
# ----------------------------------------------------------------
VECHI:                                           F1 Macro
  Baseline (fără balansare)                      0,281
  RandomUnderSampler                             0,282
  Cost-sensitive (class_weight='balanced')       0,401
  SMOTE k=5                                      0,496
  SMOTE k=3                                      0,517

NOU:                                             F1 Macro
  Baseline (fără balansare)                      0,281
  RandomUnderSampler                             0,325
  Cost-sensitive (class_weight='balanced')       0,373
  SMOTE k=5                                      0,467
  SMOTE k=3                                      0,482

NOTĂ: Ordinea s-a schimbat! Cost-sensitive (0,373) < UnderSampler (0,325)?
      Nu — UnderSampler (0,325) < Cost-sensitive (0,373) < SMOTE k=5 (0,467) < SMOTE k=3 (0,482).
      Rescrie textul: "SMOTE k=3 rămâne best, dar marja față de k=5 s-a redus."


# ============================================================
# C. SECȚIUNI SPECIFICE
# ============================================================

# C1. CONFIDENCE THRESHOLD (Cap 5.5.4)
# ----------------------------------------------------------------
VECHI: "eroarea calorică medie de la 78 kcal la 71 kcal, cu o acoperire (coverage) de 74%"
NOU:  "eroarea calorică medie de la 76 kcal la 69 kcal, cu o acoperire (coverage) de 80%"

# C2. Cap 6.2 RQ1 (referințe ablation)
# ----------------------------------------------------------------
VECHI: "F1 Macro: 0,517 combinat vs. 0,366 Oura singur vs. 0,348 Apple Watch singur"
NOU:  "F1 Macro: 0,482 combinat vs. 0,380 Oura singur vs. 0,358 Apple Watch singur"

VECHI: "F1: 0,462 combinat vs. 0,176 Apple Watch singur"
NOU:  "F1: 0,375 combinat vs. 0,188 Apple Watch singur"

# C3. Cap 6.3 — eroarea calorica in discutie
VECHI: "reducând eroarea medie de la 78 la 71 kcal"
NOU:  "reducând eroarea medie de la 76 la 69 kcal"

# C4. ANEXE — Tabel A2.2 (statistici descriptive)
# ----------------------------------------------------------------
VECHI: "N = 957"
NOU:  "N = 958"

VECHI: "957 de zile utile" (apare de 2 ori în Anexa 2)
NOU:  "958 de zile utile"

VECHI: "957" în fiecare rând al tabelului de statistici (coloana N)
NOU:  "958" (ATENȚIE: steps_oura are N=1.007, NU schimba acel rând)

# C5. Referință "26 de scripturi" → "29 de scripturi" (dacă apare)
# Verifică dacă Cap 5.1 sau Cap 4 menționează "26 scripturi"
# NOU: "29 de scripturi"


# ============================================================
# D. VALORI NOI DE ADĂUGAT (experimente noi, Cap 5)
# ============================================================
# Aceste 5 experimente sunt NOI și trebuie integrate în Cap 5
# ca secțiuni noi sau extinse:
#
# 1. Hyperparameter tuning (RF +0.3pp, XGB +8.2pp → RF rămâne best)
# 2. LR + LightGBM (LR: 0.34 → confirmă non-liniaritate)
# 3. Learning curves (saturare ~160 zile, argument Cap.6)
# 4. Calibration analysis (Brier <0.20, validează threshold 0.50)
# 5. Rolling features (36 feat → -5pp, 19 feat = optim)
#
# Sugerez o secțiune nouă 5.5.6 "Experimente suplimentare de validare"
# sau subsecțiuni separate. Discutăm structura separat.


# ============================================================
# E. CHECKLIST FINAL
# ============================================================
# [ ] Caută "0,517" în tot documentul — fiecare instanță verificată
# [ ] Caută "957" — actualizat la 958 unde e cazul
# [ ] Caută "15,2" — actualizat la 10,2
# [ ] Caută "SVM" — verificat contextul fiecărei instanțe
# [ ] Caută "cel mai bun model" — verificat că e RF, nu SVM
# [ ] Tabele 5.4, 5.5, 5.6, 5.7 actualizate complet
# [ ] Confidence threshold: 78→76, 71→69, 74%→80%
# [ ] Sinopsis + Abstract actualizate
# [ ] Concluzii Cap 7 actualizate
# [ ] Anexa 2: N=957→958
# [ ] Nr scripturi: 26→29 (dacă apare)
