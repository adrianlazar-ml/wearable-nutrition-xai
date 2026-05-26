# the tooling context file - Disciplina executie Faza 3 disertatie UPB

## Context
Adrian Lazar, disertatie UPB ACS, aparare iunie-iulie 2026.
Document master audit: AUDIT_NEW_2026-05-26.md v0.4 (107 finding-uri).
Document plan: PLAN_REMEDIERE_v0.5.md (6 sprint-uri, 20-28h cumulat).

CITESTE OBLIGATORIU LA START:
1. AUDIT_NEW_2026-05-26.md (cunoaste finding-urile si severitatile)
2. PLAN_REMEDIERE_v0.5.md (cunoaste planul de executie)

## Status actual
- Faza 1 Discovery COMPLETA (v0.4)
- Faza 2 Plan VALIDAT (v0.5 cu 4 ajustari aplicate)
- Sprint 0 DONE: commit d78deb3, tag baseline-pre-remediere-2026-05-26
- Faza 3 EXECUTIE: porneste cu Sprint 1 la confirm Adrian

## Ground Truth (NU SE SCHIMBA, verificat Pas 5 bit-perfect)
Aceste cifre trebuie sa ramana EXACT identice dupa orice fix.
Daca se schimba = regression, rollback immediate.

| Verificare | Valoare canonica |
|---|---|
| Hash dataset (DataFrame normalizat) | 63bc3a2915bae864 |
| Model hash (rf_frozen_20260331.joblib) | bit-identic pre/post |
| Imputer hash (imputer_frozen_20260331.joblib) | bit-identic pre/post |
| N test intern | 192 |
| Distributie test | {0: 44, 1: 140, 2: 8} |
| Accuracy test | 0.640625 |
| F1 Macro test | 0.506324 |
| MCC test | 0.141560 |
| F1 Rest | 0.3218 |
| F1 Normal | 0.7527 |
| F1 Active | 0.4444 |

Verificare via: python sanity_check_frozen.py
Trebuie sa returneze PASS pe toate liniile dupa fiecare sprint.

## Cifre post-fix REALE (FIXED canonic post-Sprint 2, commit b15862a)
Valorile REALE post-fix (CV de-leaked pe X_train 766), NU estimari. Referinta de
Ground Truth pentru Sprint 3-5. Consistente in cele 5 scripturi care le calculeaza
(cap5_pipeline, cap5_compare, experiments_new, circularity, extract_all_results).

| Metric | Pre-fix | Post-fix REAL | Nota |
|---|---|---|---|
| CV F1 Macro RF (TSCV 5-fold) | 0.417 | 0.337 | drop 8pp (estimarea fusese 0.378) |
| CV F1 Macro SVM | similar drop | 0.356 | |
| CV F1 Macro XGB tuned | +8.2pp gain | +6.87pp | actual |
| Tuning gain RF | ~+0.3pp estimat | +2.49pp | actual |
| SVM F1 Macro test (post SMOTE/scaler fix) | 0.482 | 0.4347 | a SCAZUT 4.7pp, NU a crescut |
| Learning curves saturare | ~160 zile | "inca cresc la 131 zile" | "160" era artefact al range-ului TSCV pe 958 |

RF F1 Macro test ramane 0.506324 (bit-perfect, vezi tabelul Ground Truth de mai sus).
Pragul vechi ">5pp scadere = stricat" a fost depasit legitim de RF (8pp): test bit-perfect
+ sanity PASS dupa fiecare script dovedesc ca e efectul real al eliminarii leakage-ului,
nu regression. Ratificat de Adrian in Sprint 2.

NOTA RULARE (Windows): scripturile pipeline cer `PYTHONUTF8=1 python <script>.py`, altfel
UnicodeEncodeError pe stdout cp1252 (caractere ca →). sanity_check_frozen.py si
patch_frozen_meta.py nu au nevoie (doar ASCII).

## Disciplina stricta
Executia se face PE SPRINT-URI ATOMICE. Niciun sprint nu incepe
fara confirmarea explicita a lui Adrian.

Regula: spune ce vei face, asteapta confirm, apoi executa.

Per sprint:
1. Citeste task-urile din PLAN_REMEDIERE_v0.5.md sectiunea 4
2. Anunta task-urile pe care le vei face
3. Asteapta confirm de la Adrian
4. Executa SECVENTIAL, anuntand fiecare task inainte
5. Pentru modificari cod multi-linie sau logica schimbata: 
   arata diff inainte de write. Pentru one-liner trivial 
   (de exemplu rename variabila in arg): mentionare scurta 
   suficienta.
6. La final task major: git commit atomic cu mesaj descriptiv
7. Ruleaza sanity check (python sanity_check_frozen.py) dupa 
   orice sprint care afecteaza pipeline (Sprint 1, 2, 3)
8. Verifica Ground Truth tabel: toate cifrele match
9. Asteapta green light inainte de Sprint urmator

## Actiuni permise
- Read files (orice fisier din proiect)
- Edit files (cu diff aratat pentru schimbari multi-linie)
- Grep / find / list directoare
- Git: status, diff, log, add, commit, tag --list
- Bash: rulare scripturi Python listate in PLAN_REMEDIERE.md
  (cap5_pipeline_json_extended.py, cap5_compare_algorithms_json_ext.py,
   experiments_new.py, freeze_production_model.py, generate_canonical_numbers.py,
   sanity_check_frozen.py, prospective_validation.py, patch_frozen_meta.py)

## Actiuni INTERZISE
- Modificare rf_frozen_20260331.joblib sau imputer_frozen_20260331.joblib
  (must remain bit-identical, FND-146 verified Pas 5)
- Re-rulare freeze_production_model.py FARA confirm explicit
  (risc suprascriere model inghetat)
- Modificari in .docx (manual de catre Adrian, Sprint 4)
- npm install / pip install fara confirm
- git push / git rebase / git reset --hard fara confirm
- Editari in the tooling context file (acest fisier) fara confirm

## Out-of-scope (referinta PLAN sectiunea 7)
NU schimba:
- Modelul inghetat (FND-146 verified)
- Pipeline arhitectural (LAG_FEATURES, T3 labels, 80/20 split)
- Cap.7 + Cap.8 structurale (doar adaugari, nu rescriere)
- Streamlit UI design general (doar fix-uri punctuale)
- Validare prospectiva reala (separat iunie 2026)

## Sprint-uri the LLM assistant
In scope pentru tine (the LLM assistant):
- Sprint 0: DONE in chat anterior (commit d78deb3 + tag)
- Sprint 1: fix cod ML (CV leakage, SMOTE/scaler, SHAP loop, frozen_meta patch)
- Sprint 2: re-rulare pipeline + regenerare CSV
- Sprint 3: update generate_canonical_numbers + UI Streamlit
- Sprint 6: cleanup MINOR (partial, doar cod)

Out of scope pentru tine (manual de catre Adrian):
- Sprint 4: update Cap.4-6 lucrare (.docx editare manuala)
- Sprint 5: verificare grep regex in Word + comentarii Buiu (manual)

## Conventii git commit
Mesaj per task major:
- "Sprint 1 task 1: fix CV leakage cap5_pipeline (FND-101)"
- "Sprint 1 task 2: fix CV leakage cap5_compare (FND-104)"
- "Sprint 1 task 3: fix CV leakage experiments_new (FND-119-122)"
- "Sprint 1 task 4: fix SMOTE/scaler experiments_new (FND-123)"
- "Sprint 1 task 5: SHAP loop RF cap5_compare (FND-110, D2)"
- "Sprint 1 task 6: patch frozen_meta f1_per_class + CM (D10)"
- "Sprint 1 task 7: git mv compute_mcc_qwk archive (FND-401, D3)"

Commit-uri atomice per task major, NU mega-commit per sprint.
Permite rollback granular cu git reset HEAD~1 daca un task pica.

## Rollback strategy
Baseline tag: baseline-pre-remediere-2026-05-26 (commit d78deb3)

Daca un sprint produce eroare:
1. NU continua. OPRESTE.
2. Raporteaza Adrian: eroarea exacta + ultimul commit reusit
3. Asteapta decizie:
   (a) Debug task curent
   (b) Rollback task curent: git reset --hard HEAD~1
   (c) Rollback complet sprint: git reset --hard <commit_pre_sprint>
   (d) Rollback Faza 3 complet: git reset --hard baseline-pre-remediere-2026-05-26

## Reguli stilistice (de la Adrian)
- Raspunde in romana, termeni tehnici in engleza
- Fara em dashes. Foloseste virgula, punct, paranteze
- Direct, no-bullshit, fara fluff
- Lungime medie, mai scurt pentru factual
- Liste numerotate doar pentru pasi/proceduri/checklists
- Tabele doar cand compari optiuni
