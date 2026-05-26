# PLAN_REMEDIERE 2026-05-26

Plan ordonat de remediere bazat pe AUDIT_NEW v0.4 (107 finding-uri, 15 BLOCKER-uri active, 13 decizii deschise).

**Versiune document:** 0.5 (Faza 2 Sinteză plan remediere)
**Data:** 26 mai 2026
**Predecesor:** AUDIT_NEW_2026-05-26.md v0.4
**Disciplina Faza 2:** PLAN scris, zero execuții fix. Faza 3 (execuție) începe DOAR după validarea acestui document.

---

## 1. Sumar executiv

**Context.** Disertația UPB a lui Adrian (apărare iunie-iulie 2026) are 107 finding-uri catalogate prin audit zero-bias în Faza 1. 15 BLOCKER-uri active afectează validitatea metodologică sau lizibilitatea defendabilă a lucrării. Restul (MAJOR, MINOR, INFO) afectează consistența sau sunt cosmetice. Comisia este AI/ML focused (echivalent AI MSC sau ISI), deci accentul fix-urilor cade pe rigoare metodologică ML, nu pe nutriție clinică.

**Goal.** Stare „defendabilă" la apărare: zero contradicții interne între cod, CSV-uri, CIFRE_CANONICE.md, UI Streamlit și textul disertației. Modelul înghețat rămâne EXACT același (verificat bit-perfect Pas 5 cu FND-146).

**Approach.** 6 sprint-uri secvențiale în cascadă strictă upstream → downstream: cod ML → CSV outputs → CIFRE_CANONICE.md → UI Streamlit → lucrare docx → BLOCKER lucrare reziduale → cleanup MINOR. Inversarea ordinii produce drift și rework dublu.

**Cifre cheie de așteptat post-fix.** CV F1 Macro va scădea de la 0,417 la aproximativ 0,378 (drop ~3-4 pp). Aceasta este expected, nu bug. Reprezintă valoarea CV CORECTĂ (fără leakage). Test F1 Macro rămâne 0,506 (verificat Pas 5, nu se schimbă pentru că folosește X_train și X_test stricte). XGB tuning gain estimat va scădea de la +8,2 pp la aproximativ +3-5 pp. Concluzia „RF best overall" rămâne validă, dar argumentul „near-optimal din default" devine mai nuanțat.

**Effort estimat.** 20-30 ore-om cumulat, plus aproximativ 2h compute. Distribuit pe 3-4 săptămâni calendar pre-apărare. Cel mai mare sprint este Sprint 4 (update lucrare), cu 8-12h.

**Out-of-scope explicit.** Niciun fix nu invalidează modelul înghețat. Niciun fix nu adaugă DXA, deep learning, sau alte ML upgrade-uri. Niciun fix nu rescrie Cap.7-8 substanțial. Validarea prospectivă REALĂ (date noi 31.03+) rămâne separat pentru iunie-iunie 2026.

**Disciplina.** Faza 2 = PLAN scris, validat de Adrian. Faza 3 (execuție) începe doar la green light explicit, cu commit-uri atomice pe sprint și posibilitate de rollback între sprint-uri.

---

## 2. Cascada de propagare

Fluxul fix-urilor respectă ordinea strictă upstream → downstream. Aplicarea în ordine inversă produce inconsistențe (cifre noi în lucrare bazate pe CSV vechi, etc.).

```
SPRINT 1 (cod ML)                       SPRINT 2 (date generate)
─────────────────                       ──────────────────────────
cap5_pipeline_json_extended.py    →     valori_disertatie.csv
   ↓ D4 fix CV leakage (4 puncte)        ablation_study_table.csv
                                          smote_comparison_table.csv
cap5_compare_algorithms_json_ext.py →   confusion_matrices.png
   ↓ D4 + D2 fix CV + SHAP loop RF      feature_importance_comparison.png
                                          shap_bar_rf.png (NOU)
experiments_new.py                →     tuning_comparison.csv
   ↓ D4 + D9 fix CV + SMOTE/scaler      new_models_comparison.csv
                                          learning_curves.png
                                          rolling_features_comparison.csv
compute_mcc_qwk.py                →     [git mv în scripts_archive/]
   ↓ D3 mutare archive
frozen_meta patch utility (NOU)    →   frozen_meta_20260331.json (updated)
   ↓ D10 cifre F1 per clasa + CM        (modelul .joblib NU se atinge)


SPRINT 3 (docs canonice + UI)
──────────────────────────────
generate_canonical_numbers.py
   ↓ FND-405 fix "inner→left join"
   ↓ FND-406, 408 fix hardcoded
   ↓ rerulare cu CSV-uri actualizate
→ CIFRE_CANONICE.md (versiune nouă, post-fix)

streamlit_app_v2.py
   ↓ D13 elimină sub-taburi pre-canonice
   ↓ D2 SHAP RF în Tab 1
   ↓ D14 cifre dinamice din CSV
   ↓ FND-201/202/203 fix cod duplicat + lean_mass + deficit
   ↓ FND-207-225 fix UI noi pagini
→ UI consistent cu CIFRE_CANONICE


SPRINT 4 (lucrare docx)
────────────────────────
Cap.4 update cifre dataset (1.009/1.083/958)
Cap.5 update split temporal (766/192) + distribuție (202/710/46)
Cap.6 update CV F1 + tuning + interpretări (post-fix)
B1, B2 propuneri text adăugate Cap.4 + Cap.7
D6 echivalente RO aplicate
9 comentarii Buiu rezolvate


SPRINT 5 (BLOCKER lucrare reziduale)
─────────────────────────────────────
Verificare reziduale FND-001/002/003 (majoritatea rezolvate prin Sprint 4)
Final review comentarii Buiu
Grep regex pe cifre vechi


SPRINT 6 (cleanup MINOR)
─────────────────────────
FND-005, FND-124, FND-131, FND-410, FND-412, etc.
README.md update (FND-402)
ghid_actualizare_valori.md cleanup (FND-403)
```

**Principiul cascadei:** fiecare sprint depinde de outputs-urile sprint-ului anterior. Niciun sprint nu poate începe înainte de finalizarea celui precedent. Sprint 5 + Sprint 6 pot rula în paralel (sau Sprint 6 după Sprint 5).

---

## 3. Recomandări per decizie

Pentru fiecare decizie deschisă (13 total), opțiunea recomandată plus rationale concis. Decizia D7 a fost închisă în v0.3.

### D1. Strategie deficit caloric pentru obiectiv fat_loss (FND-202, FND-208)

**Recomandare: (c) Documentare ca limitare în Cap.7.**

Rationale: Formula AI4Food (25 kcal/kg) este folosită pe scară largă în literatura nutrițională recentă (Lacruz-Pleguezuelos 2025). Recalibrarea ar invalida comparația cu literatura citată. Pentru subiectul prezent (obez, BMI 34.3, activ ușor), formula produce TDEE-echivalent, ceea ce e onestă limitare a formulei aplicate la o populație non-țintă. Cap.7 deja conține paragrafele B1 + B2 propuse (limitări BIA), perfect loc pentru paragraf scurt despre limitare formulă calorică. Cost: 15 min draft text. Fix alternativ (a) Recalibrare TDEE necesită justificare științifică suplimentară pe care nu o avem (Mifflin-St Jeor vs Harris-Benedict vs Katch-McArdle, ar deschide alt subject de critică).

### D2. Strategie SHAP RF (titlu vs grafice) (FND-110, FND-204, FND-217, FND-222)

**Recomandare: (b) Generare SHAP pentru AMBELE modele și salvare separată.**

Rationale: Random Forest cu 100 trees și depth=10 este compatibil cu TreeExplainer SHAP, cost compute aproximativ 5-15 min pentru summary global + 30s/sample pentru waterfall. Generarea pentru AMBELE modele (RF + XGB) păstrează valoarea comparativă XGB (interesant ca decizie boosting vs RF baggin) și aliniază titlul cu graficul principal. Opțiunea (a) Doar RF elimină valoare informativă. Opțiunea (c) Schimbare subheader e cosmetic, nu rezolvă cauza fundamentală. Costul implementării: 1 linie schimbată în loop cap5_compare (linia 387, adăugare „Random Forest" ca prim element) plus rerulare 15 min compute. Fișiere noi: `ext_shap_bar_rf.png`, `ext_shap_waterfall_rf_clean.png`.

### D3. compute_mcc_qwk.py orfan (FND-401, FND-112)

**Recomandare: (b) Mutare în scripts_archive/.**

Rationale: FND-112 a confirmat empiric că cifrele MCC/QWK canonice provin din `cap5_compare_algorithms_json_ext.py` (export în `outputs/tables/mcc_qwk_results.csv`), NU din `compute_mcc_qwk.py`. Acest script este demonstrabil orfan (folosește inner join + kJ brut, produce MCC 0.1156 vs canonic 0.1416). Mutarea în archive elimină ambiguitatea fără rework. Opțiunea (a) Aliniere completă cu pipeline canonic ar consuma 2-3h pentru zero beneficiu (output-ul nu este folosit nicăieri). Cost recomandat: 2 min (`git mv compute_mcc_qwk.py scripts_archive/old_versions/`).

### D4. Patch CV leakage (scope amplificat) (FND-101, FND-104, FND-119, FND-120, FND-121, FND-122)

**Recomandare: (a) Fix X_train/y_train în loc de X/y în 8 puncte + regenerare cascadă completă.**

Rationale: BLOCKER metodologic critic. Comisia AI/ML poate detecta în 5 linii de cod. Fix-ul este conceptually trivial: înlocuire `cross_val_score(pipe, X, y, ...)` cu `cross_val_score(pipe, X_train_t, y_train_t, ...)`. Locațiile sunt 4 în cap5_pipeline + cap5_compare + 4 în experiments_new. Opțiunea (b) Documentare ca limitare nu e suficientă: dacă comisia întreabă „de ce nu fix simplu?", răspunsul „prefer să las leakage" este indefendabil. Costul total: ~2h fix manual + 3h compute regenerare CSV + 1h verificare. Cifrele post-fix sunt mai mici dar VALIDE.

### D5. Versiune ENG paralel (n/a)

**Recomandare: (b) După RO final.**

Rationale: Traducerea paralelă consumă ciclu de validare la fiecare schimbare în RO. Versiune RO finală este post-Sprint 4. Traducerea poate începe în paralel cu Sprint 5/6 (BLOCKER reziduale + cleanup), deoarece acelea nu afectează cifre. Cost estimat ENG: 2-3 zile pentru aproximativ 80-100 pagini. Beneficiu apărare: comisia poate cere versiune ENG opțional, dar nu este obligatorie pentru UPB ACS.

### D6. Echivalente RO pentru wearables/features/RL (FND-007)

**Recomandare: aplicare în Sprint 4 cu lista canonică de mai jos.**

Lista propusă (per comentariile Buiu + standarde academice românești):

| Engleză | Română |
|---|---|
| wearables | dispozitive purtabile |
| features | caracteristici |
| RL (Reinforcement Learning) | Învățare prin Întărire (în clar) |
| State-of-the-Art | Starea actuală în domeniu |
| pipeline | pipeline (păstrat, termen tehnic standard) |
| baseline | linie de bază |
| training set | set de antrenament |
| test set | set de testare |
| lag-1 features | caracteristici cu decalaj 1 zi |
| confusion matrix | matrice de confuzie |
| classification | clasificare |
| label | etichetă |

Rationale: Lista respectă convenția uzuală academică românească (păstrare termeni tehnici standard ca „pipeline", „SHAP", traducere termeni descriptivi). Aplicare prin find-and-replace cu confirmare manuală per ocurență, NU global automat. Cost: 1-2h în Sprint 4.

### D8. Verificare pierdere 51 zile (FND-505)

**Recomandare: (a) Re-citire cap5_pipeline_json_extended pas drop NA + adăugare paragraf metodologic Cap.4.**

Rationale: Ipoteza primară (drop pe lipsa readiness_score sau sleep_score, 1009-51=958) este ușor verificabilă prin printare intermediară în pipeline canonic. Confirmarea oferă paragraf metodologic defendabil pentru Cap.4 explicând „de la 1.009 zile Oura brut la 958 zile utile pentru model". Cost: 30 min verificare + 15 min draft text. Paragraf intră în Cap.4 alături de B1 (propunerea Xiaomi pipeline).

### D9. Propagare fix B1 (SMOTE/scaler) + CV leakage în experiments_new.py (FND-105, FND-602, FND-123)

**Recomandare: (b) Apply identic, propagare CONFIRMATĂ Pas 1.**

Rationale: Audit Pas 1 a confirmat că experiments_new.py conține EXACT aceleași bug-uri ca cap5_compare (cele 2 mamă). Patch-urile sunt copy-paste din D4. 3 locații SMOTE/scaler pentru SVM (linii 229-234, 292-298, 434-439) plus 4 locații CV leakage (linii 214, 241, 272, 312 + 448, 561, 772). Total 7 puncte unice. Cost aplicare: ~1h. Inclus în Sprint 1 alături de D4. Decizia (c) Documentare nu funcționează aici, este propagare clară.

### D10. Patch frozen_meta.json cu f1_per_class + CM intern (FND-116, FND-139)

**Recomandare: (a) Script utilitar one-shot fără re-train.**

Rationale: Sanity check Pas 5 a EXPUS deja cifrele de adăugat în meta (F1 Rest 0.3218, F1 Normal 0.7527, F1 Active 0.4444, CM completă). Script utilitar de aproximativ 15 linii: load model frozen + imputer + dataset snapshot, replicate split 80/20, apel classification_report cu output_dict=True + confusion_matrix, merge în meta JSON existent, suprascrie. Modelul .joblib NU se atinge, hash unchanged. Verificat reproductibilitate Pas 5. Opțiunea (b) Re-rulare freeze este nenecesar riscantă (rerun ar putea produce hash diferit dacă mediul software diferă marginal).

### D11. Strategie pentru Experiment 3 learning_curves XGBoost (FND-128)

**Recomandare: (a) Acceptă inechitatea și documentează.**

Rationale: XGBoost multiclass nu suportă scale_pos_weight nativ. Workaround sample_weight (opțiunea b) este fragil cross-platform și introduce alt risc metodologic. Cleanest: paragraf documentar în Cap.5 explicând că learning_curve XGBoost folosește defaults, comparabilitatea cu RF/SVM (class_weight='balanced') este cu limitare metodologică minoră. Eliminarea XGBoost din learning curves (opțiunea c) ar pierde informație utilă. Cost: 10 min draft paragraf pentru Cap.5.

### D12. Re-rulare RandomizedSearchCV cu fix CV leakage (FND-119)

**Recomandare: (a) Re-rulare cu fix.**

Rationale: Tuning este unul dintre argumentele cheie din Cap.6 („RF near-optimal +0.30pp"). Post-fix CV leakage, cifrele se schimbă. Estimare: XGB tuning gain poate scădea de la +8.2pp la +3-5pp. Re-rulare validează defendabilitatea concluziei „RF best overall". Cost compute: aproximativ 30 min (RandomizedSearchCV 50 iter × 3 modele × 5-fold TSCV). Rezultate intră direct în Sprint 2 regenerare CSV. Opțiunea (b) Documentare ca limitare nu funcționează: concluzia „RF best" trebuie să rămână validă post-fix.

### D13. Strategie UI pentru figuri pre-canonice (FND-216, FND-218)

**Recomandare: (c) Eliminare sub-taburi JSON 16 + XML 16 din Tab 2 (păstrare doar Extended).**

Rationale: Figurile pre-canonice din sub-taburile JSON 16 feat și XML 16 feat NU mai oferă valoare comparativă validă (toate sunt PRE-fix rhr_avg). Eliminarea simplifică UI și elimină risc apărare („de ce afișați comparativ cu cifre vechi?"). Comparația XML vs JSON quality rămâne în Tab 3 (Calitate Date), care e scop diferit (data quality, nu evaluare metrică). Opțiunea (b) Regenerare figuri post-fix pentru toate 3 surse este disproporționat (cost 2-3h pentru valoare comparativă redusă). Opțiunea (a) Disclaimer „legacy" e cosmetic dar nu elimină riscul. Cost: 30 min refactor Tab 2.

### D14. Sincronizare cifre hardcoded UI ↔ Cap.6 (FND-219, FND-220, FND-221)

**Recomandare combinată: (a) + (c).**

Rationale: 
- Pentru cifre disponibile în `outputs/tables/*.csv`: opțiunea (a) Extragere dinamică în UI. Elimină drift silențios. Implementare: replace dict hardcoded cu citire CSV (similar cu pattern-ul deja folosit în Tab 1 metrici comparative).
- Pentru cifre din scripturi arhivate (FND-220 MAE 9106 pași): opțiunea (c) Acceptare ca limitare în Cap.7. Re-rularea scripturilor arhivate (compare_apple_xml_vs_json.py) ar fi 2-3h pentru zero beneficiu metodologic. Paragraf onestă în Cap.7: „comparația XML vs JSON a fost efectuată în iterație anterioară (martie 2026), cifrele sunt stocate ca text static; reproducibilitatea exactă este compromisă pentru această analiză comparativă, dar finding-ul calitativ (XML dublu-contorizează pași) este robust și confirmat în 4 surse independente".

Cost combinat: 1h pentru UI dynamic + 15 min paragraf Cap.7.

---

## 4. Plan ordonat pe sprint-uri

### Sprint 0: Backup pre-remediere (mandatoriu)

**Scop:** snapshot complet baseline pre-Sprint 1, pentru rollback emergency.

**Tasks:**
1. Verificare working directory clean: `git status` returnează „nothing to commit, working tree clean". Dacă există modificări netracked, decide commit sau stash.
2. Commit baseline: `git add -A && git commit -m "baseline pre-Sprint 1 remediere, AUDIT_NEW v0.4 validat"`.
3. Tag baseline: `git tag baseline-pre-remediere-2026-05-26 -m "Pre-Faza 3 baseline. 107 finding-uri, 15 BLOCKER active, model bit-perfect verified Pas 5."`.
4. ZIP backup lucrare docx: `Compress-Archive -Path Disertatie_Lazar_Adrian_2026_Mai_CB.docx -DestinationPath backups/docx_baseline_2026-05-26.zip` (asigură că folderul `backups/` există).
5. Verificare integrală artefacte: `python sanity_check_frozen.py` → confirmă PASS cu cifrele canonice (accuracy 0.640625, F1 Macro 0.506324, MCC 0.141560, hash 63bc3a2915bae864).

**Verificare pre-Sprint 1:** `git log --oneline | head -1` afișează commit baseline. `git tag --list` afișează `baseline-pre-remediere-2026-05-26`. ZIP docx există în `backups/`. Sanity check PASS.

**Acceptance criteria:** rollback la baseline e posibil oricând cu `git reset --hard baseline-pre-remediere-2026-05-26` + unzip ZIP docx.

**Cost:** 10-15 min.

---

### Sprint 1: Fix BLOCKER metodologic (cod ML)

**Scop:** elimină bug-urile-mamă (CV leakage + SMOTE/scaler) și ROOT CAUSE-urile (SHAP loop RF, frozen_meta).

**Tasks:**
1. **D4 (a)** Fix CV leakage în `cap5_pipeline_json_extended.py` (linii 207, 219) + `cap5_compare_algorithms_json_ext.py` (linii 263-264). Înlocuire `cross_val_score(pipe, X, y, cv=...)` cu `cross_val_score(pipe, X_train_t, y_train_t, cv=...)`.
2. **D4 + D9 (b)** Fix CV leakage în `experiments_new.py` (4 locații: `rf_search.fit`, `svm_search.fit`, `xgb_search.fit` în experiment_1, plus `cross_val_score` în experiment_2 + learning_curve experiment_3 + experiment_5). Total 7 puncte între experiments_new și pipeline canonic.
3. **D9 (b)** Fix SMOTE/scaler ordering în `experiments_new.py` pentru SVM (3 locații: linii 229-234, 292-298, 434-439). Schimbare ordine `imputer→smote→scaler→SVC` în `imputer→scaler→smote→SVC` (sau echivalent ImbPipeline cu scaler înainte de SMOTE).
4. **D2 (b)** Adăugare Random Forest în SHAP loop la `cap5_compare_algorithms_json_ext.py` linia 387. Generare `ext_shap_bar_rf.png` + `ext_shap_waterfall_rf_clean.png`.
5. **D3 (b)** `git mv compute_mcc_qwk.py scripts_archive/old_versions/`. Update README dacă referențiat.
6. **D10 (a)** Creează `patch_frozen_meta.py` (script utilitar 15 linii) care injectează f1_per_class + CM în `frozen_meta_20260331.json`. Modelul .joblib intact, hash unchanged.

**Notă paralelism D10:** Task 6 este INDEPENDENT de task 1-5. Pentru că modelul înghețat nu se atinge în Sprint 1 (doar metadata se actualizează), `patch_frozen_meta.py` poate fi creat și rulat în paralel cu fix-urile cod (sau chiar înainte de ele, imediat după Sprint 0). Cifrele de injectat sunt deja cunoscute din sanity check Pas 5 (F1 Rest 0.3218, F1 Normal 0.7527, F1 Active 0.4444, CM `[[14,30,0],[29,105,6],[0,4,4]]`). Acceptance criteria: `frozen_meta_20260331.json` conține câmpurile noi, hash model neschimbat, sanity check rerulat continuă să producă PASS.

**Verificare pre-Sprint 2:** scripturile rulează clean sintactic (`python -c "import cap5_pipeline_json_extended"` fără erori).

**Atenție:** sanity check (FND-146) trebuie rerulat după fix-uri pentru a confirma că F1 Macro test rămâne 0.5063. Dacă scade, atunci patch-ul a introdus regression neașteptată.

---

### Sprint 2: Re-rulare pipeline + regenerare CSV

**Scop:** generează toate CSV-urile cu cifrele POST-fix din Sprint 1.

**Tasks (rulate secvențial):**
1. `python cap5_pipeline_json_extended.py` → noi outputs (valori_disertatie.csv, ablation_*, mcc_qwk_results.csv parțial). Timp aproximativ 2 min.
2. `python cap5_compare_algorithms_json_ext.py` → confusion_matrices.png + SHAP RF + SHAP XGB + feature_importance. Timp aproximativ 10-15 min compute (datorită SHAP).
3. `python experiments_new.py` → tuning_comparison.csv (cu D12), new_models_comparison.csv, learning_curves.png, calibration_brier_scores.csv, rolling_features_comparison.csv. Timp aproximativ 30-40 min compute (datorită RandomizedSearchCV).
4. `python patch_frozen_meta.py` → frozen_meta_20260331.json updated cu f1_per_class + CM.
5. Sanity check rerulat: `python sanity_check_frozen.py` → trebuie să mențină F1 Macro test = 0.5063 (model neschimbat).

**Verificare pre-Sprint 3:** toate CSV-urile au valori noi, dataset hash rămâne `63bc3a2915bae864`, frozen_meta.json conține câmpurile noi.

**Risc:** dacă vreun script aruncă erori după Sprint 1 fix, înseamnă bug introdus. Rollback Sprint 1 specific (cu git diff) și re-investigare.

---

### Sprint 3: Update CIFRE_CANONICE + UI Streamlit

**Scop:** propagă cifrele noi în documentul canonic și UI utilizator.

**Tasks A: generate_canonical_numbers.py**
1. Fix **FND-405**: linia 68, schimbare „inner join" → „left join". 
2. Fix **FND-406**: convertire cifre hardcoded 1.009/1.083 la citire din audit_csv outputs sau adăugare în meta JSON.
3. Fix **FND-408**: interpretări hardcoded → derivare din CSV (pentru cifre numerice) + marcaj „snapshot la X.X.2026" (pentru texte interpretative).
4. Fix **FND-409**: condiționare LightGBM (dacă există în CSV).
5. Rulare `python generate_canonical_numbers.py` → noua `CIFRE_CANONICE.md`.

**Tasks B: streamlit_app_v2.py**
6. **D13 (c)**: Elimină sub-taburi JSON 16 + XML 16 din Tab 2. Păstrează doar sub-tab Extended. Reduce Tab 2 la o singură vedere.
7. **D2 (b) follow-up**: Tab 1 Explicabilitate, schimbare titlu și grafic afișat: `ext_shap_bar_rf.png` + `ext_shap_waterfall_rf_clean.png` (nu XGB).
8. **D14 (a)**: Tab 3 tabel cifre hardcoded → citire dinamică din CSV (cele disponibile). Cifrele din scripturi arhivate rămân hardcoded cu disclaimer „snapshot martie 2026".
9. **FND-211**: linia 498, schimbare `demo_end_to_end.py` → `demo_end_to_end_v2.py`.
10. **FND-212**: linia 512, eliminare referință `cap5_pipeline.py (XML legacy)`.
11. **FND-214**: clarificare warning Dashboard „predicție pentru ziua X, nu pentru azi".
12. **FND-207**: default body_fat_pct 36.2 → 36.3.
13. **FND-208**: text deficit hardcoded → calcul dinamic per profil utilizator (sau text neutru fără cifră specifică).
14. **FND-201**: eliminare lean_mass hardcoded din `nutrition_recommender.py`, înlocuire cu apel `get_lean_mass(p)` ovunde.
15. **FND-202**: rationale text dinamic per utilizator (sau marcaj „limitare formula AI4Food pe obezitate, vezi Cap.7").
16. **FND-203**: consolidare cod recomandare. Opțiune simplă: streamlit_app_v2.py să importe din nutrition_recommender.py în loc de reimplementare inline. Cost: 30 min refactor.
17. **FND-209**: eliminare duplicat base_cal intra-streamlit (folosire helper function unică).

**Verificare pre-Sprint 4:** UI rulează (`streamlit run streamlit_app_v2.py`), niciun grafic XGB titrat ca RF, niciun lean_mass hardcoded, CIFRE_CANONICE conține cifre noi.

---

### Sprint 4: Update Cap.4-6 lucrare

**Scop:** sincronizează textul docx cu CIFRE_CANONICE v2.

**Tasks:**
1. **Cap.4 cifre dataset:**
   - 1.004 → 1.009 (Oura)
   - 1.082 → 1.083 (AW JSON)
   - 1.078 → 1.083 (Anexa A2.2)
   - 1.008 → 1.009 inner join → left join (FND-004 + FND-405 propagate)
   - 953 → 958 (post-lag-1)
   - Date interval „28 martie 2026" → „30 martie 2026"
   - Sumă Gen3 + Gen4 = 708 + 300 = 1008 → 1009 (verificare aritmetică)

2. **B1 (FND-008)**: Inserare paragraf Cap.4 despre pipeline antropometric Xiaomi (text deja draftat în AUDIT_NEW v0.4 Anexa B1).

3. **D8 (a)**: Inserare paragraf Cap.4 despre pierdere 51 zile (de la 1.009 la 958).

4. **Cap.5 split temporal:**
   - 762 → 766 (train)
   - 191 → 192 (test)
   - 162/564/36 → 158/570/38 (pre-SMOTE)
   - 564 → 570 (per clasă post-SMOTE)

5. **Cap.5.2 distribuție T3:**
   - Rest: 21,2% → 21,1%
   - Active: 4,7% → 4,8%
   - Normal: 706 → 710
   - Active: 45 → 46
   - Verificare aritmetică: 202 + 710 + 46 = 958 ✓

6. **Cap.5 / Cap.6 cifre CV F1 Macro:**
   - 41,7% → 37-38% (CV post-fix, valoare exactă din Sprint 2 outputs)
   - XGB tuning gain +8,2pp → valoare exactă post-fix
   - Test F1 Macro 0,506 rămâne unchanged

7. **Cap.6 paragraf #712:**
   - 0.517 → 0,506
   - 0,366 → 0,428 (Oura-only)
   - 0,348 → 0,324 (AW-only)
   - 0,176 → 0,100 (F1 Active AW-only)
   - 0,462 → 0,444 (F1 Active Combined)

8. **Cap.7 propuneri text:**
   - B2 (FND-009): paragraf BIA vs DXA (deja draftat)
   - D1 (c): paragraf limitare formula AI4Food pe obezitate (~50 cuvinte)
   - D11 (a): paragraf limitare learning_curve XGB unbalanced (~30 cuvinte)
   - D14 (c): paragraf limitare reproducibilitate XML vs JSON arhivat (~50 cuvinte)

9. **Comentarii Buiu (FND-007):**
   - #808: wearables → dispozitive purtabile (toate ocurențele Cap.1)
   - #812: State-of-the-Art → Starea actuală în domeniul abordat
   - #837: RL → Învățare prin Întărire (sau dezvoltat în clar)
   - #856: features → caracteristici (Cap.3)
   - #858: wearables → dispozitive purtabile (Cap.3)
   - #866: Formula SHAP cu editorul de ecuații
   - #896: screenshot Streamlit
   - #902: formatare oura_auth.py
   - #974: formatare paragraf „Superioritatea RF" Cap.6

10. **D6 lista RO**: aplicare consistentă cu lista canonică propusă în Secțiunea 3.

**Verificare pre-Sprint 5:** find-and-replace verificat (atenție la cifre cu punct vs virgulă: „0.517" și „0,517"). Grep pe cifre vechi: `0\.517|0,366|0,348|0,176|0,462|21,2%|4,7%|762|191|953|1\.004|1\.078|1\.008`.

---

### Sprint 5: BLOCKER lucrare reziduale + comentarii

**Scop:** verifică că BLOCKER-urile FND-001, FND-002, FND-003 sunt complet rezolvate post-Sprint 4. Majoritatea ar trebui să fie automat rezolvate prin update cifre Sprint 4.

**Tasks:**
1. **FND-001 verificare:** grep regex pe paragraf #712 Cap.3. Cifrele vechi (0.517, 0,366, etc.) NU trebuie să mai apară.
2. **FND-002 verificare:** Cap.5.2 distribuție = Tabel 5.5 distribuție = 202/710/46. Verificare două locuri sync.
3. **FND-003 verificare:** Cap.5 split = canonic = 766/192.
4. **FND-006 verificare:** paragraful #876 cifra 0,482. Inspecție vizuală în Word - verifică context (tabel SMOTE k=3 vs interpretare). Decide actualizare la 0,506 sau păstrare cu marcaj.
5. **Final review comentarii Buiu:** verifică toate 9 marcate rezolvate. Eventual mark-as-resolved în Word.

**Acceptance criteria:** grep regex pe toate cifrele vechi documentate în FND-001 până FND-005 returnează zero match.

---

### Sprint 6: Cleanup MINOR + cosmetic

**Scop:** rezolvă MINOR-urile rămase și actualizează documentația.

**Tasks:**
1. **FND-005**: procente + support fine-tuning (21,2% → 21,1%, 4,7% → 4,8%, support 9→8 din 191→192).
2. **FND-124**: dict cheie duplicată în `experiments_new.py` `prepare_train_test`.
3. **FND-131**: dead code „SVM in name" în experiment_2.
4. **FND-141**: classification_report cu `output_dict=True` în freeze_production_model (cuplat cu D10 patch utility).
5. **FND-143**: adăugare model_hash + imputer_hash în meta JSON la rulare freeze viitoare.
6. **FND-410**: load_meta selecție prin tag în loc de mtime (în generate_canonical_numbers.py).
7. **FND-412**: replace global „." → "," doar pe cifre, nu pe puncte finale (regex mai precis).
8. **FND-215**: consolidare label_names/label_colors într-o singură constantă globală.
9. **FND-402**: README.md update (lista 31 scripturi, cifre canonice noi).
10. **FND-403**: ghid_actualizare_valori.md → mutare în archive sau actualizare cu cifre post-Sprint 2.
11. **D5 (b)**: Start traducere ENG paralel (poate începe în paralel cu Sprint 6, nu blochează nimic).

**Acceptance criteria:** lint cleanup, README updated, ghid arhivat sau actualizat.

---

## 5. Estimări efort

### Tabel rezumativ

| Sprint | Efort uman | Compute | Dependențe | Risc |
|---|---|---|---|---|
| Sprint 0 | 10-15 min | 30 sec sanity | nimic | scăzut (backup mecanic) |
| Sprint 1 | 4-5 h | 5 min | Sprint 0 complet | scăzut (fix mecanic), D10 task 6 paralel cu task 1-5 |
| Sprint 2 | 1 h | 1-1.5 h | Sprint 1 complet | mediu (rerulare poate descoperi alte bug-uri) |
| Sprint 3 | 4-5 h | 5 min | Sprint 2 complet | scăzut (refactor UI direct) |
| Sprint 4 | 8-12 h | 0 | Sprint 3 complet (CIFRE_CANONICE) | mediu-mare (manual docx, git commits 2h în timpul sprint-ului) |
| Sprint 5 | 1-2 h | 0 | Sprint 4 complet | scăzut (verificare grep + sample) |
| Sprint 6 | 2-3 h | 0 | Sprint 5 complet, sau paralel | scăzut |
| **TOTAL** | **20-28 h** | **~2 h** | secvențial cu Sprint 5+6 paraleli posibili + D10 paralel intra-Sprint 1 | mediu cumulat |

### Riscuri identificate

**Risc Sprint 1: regression în model.** Fix CV leakage modifică doar metricile CV, nu modelul. Sanity check post-fix trebuie să confirme F1 Macro test = 0.5063 unchanged. Dacă scade, înseamnă bug introdus în patch.

**Risc Sprint 2: scripturi pică la rulare.** Posibil dacă fix-ul Sprint 1 introduce sintax greșit. Mitigare: lint check + dry run pe sample mic înainte de full pipeline run.

**Risc Sprint 3: UI break.** Refactor major Tab 2 (eliminare sub-taburi) poate sparge naivigarea. Mitigare: test local interactiv post-fix.

**Risc Sprint 4: typo în docx sau pierdere/corupere fișier.** Manual edits pe aproximativ 20 cifre numerice plus 9 comentarii Buiu plus 4 paragrafe propuse. Risc cumulat pe sesiune lungă (8-12h) de pierdere prin crash Word sau Ctrl+Z dincolo de history. Mitigare în două straturi:
- **Stratul 1 (verificare):** grep regex pe cifre vechi după edit + cross-check cu CIFRE_CANONICE.md final.
- **Stratul 2 (backup periodic):** commit git pentru docx la fiecare 2h în timpul Sprint 4. Pattern: `git add Disertatie_Lazar_Adrian_2026_Mai_CB.docx && git commit -m "Sprint 4 progress, HH:MM, X capitole updated"`. Permite rollback la orice checkpoint intermediar dacă apare corupere. Plus ZIP suplimentar la trecere cap.4 → cap.5 → cap.6.

**Risc Sprint 5/6: omisiune.** Posibil să rămână cifre vechi în text nedetectate. Mitigare: regex comprehensiv + verificare unanimitate Tabel 5.5 ↔ text Cap.5.2.

**Risc cumulat: timeline.** 20-28h cumulat × 4 săptămâni efort = 5-7h/săptămână. Realist dacă Adrian dedică timp constant. Stretch dacă apar interruperi (job, alte tasks).

### Dependențe critice

- **Sanity check `sanity_check_frozen.py`** trebuie disponibil pentru rerun după Sprint 1 și după Sprint 2.
- **Git backup** trebuie făcut înainte de Sprint 1 (commit clean baseline). Permite rollback per sprint.
- **disertatie_audit.zip** (backup 25.05) trebuie păstrat ca emergency rollback.

---

## 6. Acceptance criteria per fix BLOCKER

Pentru fiecare BLOCKER, criteriul cu care Adrian verifică post-fix că remedierea a funcționat.

### BLOCKER-uri Lucrare (3)

**FND-001** (paragraf #712 cifre vechi Cap.3): 
- Criteriu primar: grep regex `0\.517|0,366|0,348|0,176|0,462` pe docx returnează zero match.
- **Criteriu secundar (verificare false positive):** pentru fiecare cifră canonică nouă (0,506 / 0,428 / 0,324 / 0,100 / 0,444), verificare manuală a sample-urilor de apariție în docx. Cifrele apar în multiple contexte (paragraf #712, Tabel 5.5, Anexă, captions figuri). Risc real: cifră nouă cu valoare similară (de exemplu 0,506 apare și în Cap.6 paragraf #982 pentru o comparație diferită) poate genera fals pozitiv la grep. Verificare manuală asigură că semantica este corectă, nu doar pattern-ul numeric.
- Cifre post-fix: 0,506 (combined), 0,428 (Oura-only), 0,324 (AW-only), 0,100 (F1 Active AW), 0,444 (F1 Active combined).

**FND-002** (distribuție T3 inconsistentă):
- Criteriu: Tabel 5.5 + text Cap.5.2 + Anexă afișează unanim 202/710/46. Total = 958.
- Verificare aritmetică: 202 + 710 + 46 = 958, procente 21,1% + 74,1% + 4,8% = 100%.

**FND-003** (split temporal Cap.5):
- Criteriu: text Cap.5 menționează 766/192 train/test, distribuție pre-SMOTE 158/570/38, post-SMOTE 570 per clasă.

### BLOCKER-uri Pipeline ML cap5_* (4)

**FND-101** (CV leakage canonic):
- Criteriu: în `cap5_pipeline_json_extended.py` liniile 207 și 219, apel `cross_val_score(cv_pipeline, X_train_t, y_train_t, cv=tscv, ...)`. Inspecție vizuală cod + dry run.
- Post-fix CV F1 Macro: estimat ~0.378 (drop ~3-4pp de la 0.417). Valoare expected, marchează ca rezultat corect.

**FND-104** (CV leakage cap5_compare):
- Criteriu: idem cu FND-101 pentru `cap5_compare_algorithms_json_ext.py` liniile 263-264.
- Toate cele 3 modele (RF, XGB, SVM) reflectă cifre CV corecte post-fix.

**FND-105** (SMOTE/scaler SVM):
- Criteriu: în `cap5_compare_algorithms_json_ext.py` ordine pipeline SVM = `imputer → scaler → SMOTE → SVC` (sau ImbPipeline echivalent).
- Post-fix: SVM F1 Macro test poate crește 1-2pp (estimat 0.482 → 0.494-0.502). Validează că SMOTE pe date scalate produce puncte sintetice mai bune.

**FND-110** (SHAP loop preferă RF):
- Criteriu: fișiere noi în `outputs/figures/json_extended_pipeline/`: `ext_shap_bar_rf.png`, `ext_shap_waterfall_rf_clean.png`.
- Plus fișiere existente XGB rămân (păstrate pentru valoare comparativă).

### BLOCKER-uri Pipeline ML experiments_new (5)

**FND-119** (CV leakage RandomizedSearchCV):
- Criteriu: în `experiments_new.py` liniile 214, 241, 272 apeluri devin `rf_search.fit(X_train, y_train)`, `svm_search.fit(X_train, y_train)`, `xgb_search.fit(X_train, y_train)` (unde X_train = X.iloc[:split_idx]).
- Post-fix: XGB tuning gain estimat scade de la +8.2pp la +3-5pp. Concluzia „RF best overall" rămâne validă.

**FND-120** (CV leakage experiment_2):
- Criteriu: linia 448 devine `cross_val_score(cv_pipe, X_train, y_train, cv=tscv, ...)`.

**FND-121** (CV leakage learning_curves):
- Criteriu: linia 561 devine `learning_curve(pipe, X_train, y_train, train_sizes=..., cv=tscv, ...)`.
- Atenție: train_sizes_frac=np.linspace(0.2, 1.0, 9) trebuie reinterpretat în context X_train (766 zile), nu X complet (958 zile). Saturarea „~160 zile" se va recalcula.

**FND-122** (CV leakage rolling features):
- Criteriu: linia 772 devine `cross_val_score(rf_pipe, X_train_c, y_train_c, cv=tscv, ...)`. Splituirea pe `X_c` în experiment_5 trebuie să respecte același 80/20 cronologic.

**FND-123** (SMOTE/scaler propagat 3 locuri):
- Criteriu: 3 locații (linii 229-234, 292-298, 434-439) reflectă ordine `imputer → scaler → SMOTE → SVC` (sau ImbPipeline echivalent).

### BLOCKER-uri Streamlit (3)

**FND-201** (lean_mass hardcoded):
- Criteriu: în `nutrition_recommender.py` USER_PROFILE nu mai conține „lean_mass_kg": 72.49. Funcția folosește `get_lean_mass(profile)` definită similar cu Streamlit.
- Demo CLI și Streamlit produc cifre IDENTICE pentru același label (Rest, Normal, Active).

**FND-202** (deficit caloric fals):
- Criteriu: rationale text fie dinamic per utilizator (calculat din TDEE și base_cal), fie marcat cu disclaimer „limitare formula AI4Food pe obezitate, vezi Cap.7".
- Cap.7 conține paragraf nou (D1 (c)) explicând limitarea.

**FND-204** (SHAP XGB pe titlu RF):
- Criteriu: pagina Explicabilitate Tab 1 afișează titlu „Random Forest — 19 features — F1 Macro: 50.6%" + grafic `ext_shap_bar_rf.png` (RF, nu XGB).
- Graficele XGB pot rămâne ca expandable secundare pentru comparație.

---

## 7. Out-of-scope

### Ce NU se schimbă în Faza 3

**Modelul înghețat.** `rf_frozen_20260331.joblib` și `imputer_frozen_20260331.joblib` NU se ating. Hash bit-perfect verified Pas 5 (FND-146). Sanity check rerulat post-Sprint 1 confirmă F1 Macro test = 0.5063 unchanged.

**Pipeline arhitectural.** Rămâne cu LAG_FEATURES + lag-1 shift, T3 label scheme, 80/20 chronological split, SMOTE k=3, RF 100 trees depth 10. Nicio schimbare conceptuală.

**Label scheme T3.** Praguri rămân: Rest (readiness<60 OR sleep<60), Active (readiness≥85 AND sleep≥75), Normal (rest). Documentat în 3 locuri (FND-206, FND-210), va deveni un singur loc canonic în Faza 3 dar logica rămâne.

**Cap.7 + Cap.8 structurale.** Doar adăugări (B1, B2, D1 (c), D11 (a), D14 (c)) ca paragrafe noi în Cap.7. Cap.8 viitor rămâne identic.

**Validare prospectivă REALĂ.** Date noi 31.03+ → rulare `prospective_validation.py --tag <nou>` pe date noi va fi efectuată separat în iunie 2026 (cu 30+ zile prospective). Nu este parte din Faza 3 actuală.

**Production deployment.** Niciun FastAPI, niciun cloud, niciun container. Streamlit local + scripturi rulate manual rămâne arhitectura.

**ML upgrade.** Niciun deep learning, niciun transformer, niciun ensemble nou. Modelul curent este sufficient pentru disertație.

**Re-design Streamlit UI complet.** Doar ajustări punctuale conform fix-urilor (D13, D14, FND-207-225). Estetica/layout-ul general rămâne.

**Comentarii Buiu reziduale dificile.** Comentariul #866 (formula SHAP cu editorul de ecuații) poate necesita re-export PDF; depinde de format final lucrare. Comentariul #896 (screenshot Streamlit) depinde de UI final post-Sprint 3.

### Limitări care rămân documentate în Cap.7

Lista de limitări care intră în Cap.7 ca paragrafe documentare (nu fix), parte din strategia onestă „defendabil la apărare":

1. **BIA consumer-grade vs DXA gold-standard** (FND-009 / B2 propunere). Bias 2-5pp pe %BF pentru obezitate.

2. **N=1 longitudinal design.** Limitare metodologică deja documentată în lucrare. Singur subiect, generalizare limitată.

3. **Threshold 0.50 fără tuning empiric.** Aleasă ca limită naturală interpretabilă, nu optimizată empiric pe test set. Decizie defendabilă: optimizare pe test ar fi forma subtilă de leakage.

4. **Single-reviewer pe literatură (Cap.2).** Limitare PRISMA-like documentată. Lipsește peer-review independent pe selecția literaturii.

5. **Formula AI4Food (25 kcal/kg) pe obezitate** (D1 (c)). Formula a fost dezvoltată pentru subiecți cu greutate normală. Pentru obez (BMI 34.3) produce TDEE-echivalent, nu deficit. Limitare onestă.

6. **Learning curve XGBoost unbalanced** (D11 (a)). XGBoost multiclass nu suportă class_weight nativ. Comparabilitate cu RF/SVM (balanced) este cu limitare metodologică minoră.

7. **Reproducibilitate parțială pentru cifre din scripturi arhivate** (D14 (c)). Comparația XML vs JSON a fost efectuată în iterație anterioară (martie 2026), cifrele sunt stocate ca text static în UI; reproducibilitatea exactă este compromisă, dar finding-ul calitativ (XML dublu-contorizează pași) este robust.

8. **Lipsă feedback loop nutrițional** (preexistent în lucrare). Recomandările nutriționale nu au fost testate empiric pe adherence sau outcome metabolic.

9. **Absența validării prospective la momentul scrierii.** Modelul este înghețat la 31.03.2026; validarea prospectivă reală pe date noi este planificată dar nefinalizată la momentul apărării (poate fi „in progress" în iunie).

---

## 8. Următoarea acțiune

**Pentru Adrian:**
1. Citește integral `PLAN_REMEDIERE_v0.5.md`.
2. Verifică recomandările per decizie (Secțiunea 3). Pentru fiecare D, accept/refuz/ajustare.
3. Verifică ordinea sprint-urilor (Secțiunea 4). Acceptă cascada sau propune reordonare.
4. Verifică estimările efort (Secțiunea 5). Adjustează dacă sunt nerealiste pentru calendarul tău.
5. Verifică out-of-scope (Secțiunea 7). Confirmă că tot ce e exclus este OK să rămână ne-atins.

**Output așteptat de la Adrian:**
- GREEN LIGHT integral, sau
- ACCEPT cu modificări specifice (per secțiune), sau
- REFUZ / re-discussion pentru decizii specifice.

**Disciplină Faza 2:** Niciun fix nu se aplică încă. Faza 3 (execuție Sprint 1) începe DOAR după validarea explicită a acestui plan de către Adrian.

---

**Sfârșit document v0.5.**

Faza 2 Sinteză COMPLETĂ. Următoarea actualizare: după validare plan, începere Faza 3 cu sprint-uri atomice și commit-uri per fix.
