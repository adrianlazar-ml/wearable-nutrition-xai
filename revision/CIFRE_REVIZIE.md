# CIFRE_REVIZIE.md - Revision 1, Knowledge-Based Systems

Regula: nicio cifra nu intra in manuscris fara sa fie aici, cu script, seed si data rularii.
Cifrele canonice ale modelului raportat (CIFRE_CANONICE.md) NU se schimba; tot ce e aici e aditiv.

| ID | Cifra | Valoare | Script | Seed | Data rularii | Observatii |
|---|---|---|---|---|---|---|
| T0.1 | RF F1 macro (rerun) | 0.5063 (0.506324) | sanity_check_frozen.py | 42 | 2026-10-14 | pass = 0.5063; PASS, hash match 63bc3a2915bae864 |
| T0.1 | RF accuracy (rerun) | 0.6406 (0.640625) | sanity_check_frozen.py | 42 | 2026-10-14 | pass = 0.6406; PASS |
| T0.1 | RF MCC (rerun) | 0.1416 (0.141560) | sanity_check_frozen.py | 42 | 2026-10-14 | pass = 0.1416; PASS |
| T0.1 | Versiuni: python / sklearn / numpy / pandas | 3.11.9 / 1.8.0 / 2.4.3 / 2.3.3 | python -c (mediul rularii) | | 2026-10-14 | identic cu frozen_meta: 3.11.9 / 1.8.0 / 2.4.3 / 2.3.3 |
| T0.2 | RF / XGB / SVM F1, acc, MCC (rerun din revision/) | RF 0.5063 / 0.6406 / 0.1416; XGB 0.4760 / 0.6875 / 0.1503; SVM 0.4347 / 0.6302 / 0.0956 | revision/scripts/save_test_predictions.py | 42 | 2026-09-14 | toate PASS (results/test_predictions_metrics.csv); predictii + probabilitati in results/ |
| T0.2 | SVM: acord argmax(proba) vs predict() | 0.9115 | save_test_predictions.py | 42 | 2026-09-14 | Platt scaling poate diferi de predict(); relevant pentru T1.3 |
| T0.3 | t3_labels parametrizat, identitate | ALL PASS | revision/scripts/test_t3_labels.py | 42 | 2026-09-14 | 10,201 perechi + 29 configuratii |
| T0.4 | Persistence F1 macro [CI 95%] | 0.3039 [0.2612, 0.3501] | revision/scripts/persistence_baseline.py | 42 | 2026-09-14 | 1000 resamples (results/persistence_baseline.csv) |
| T0.4 | Persistence F1 Active | 0.0000 | persistence_baseline.py | 42 | 2026-09-14 | |
| T0.4 | Persistence MCC [CI 95%] | -0.0566 [-0.1632, 0.0597] | persistence_baseline.py | 42 | 2026-09-14 | |
| T0.4 | Persistence accuracy | 0.5625 | persistence_baseline.py | 42 | 2026-09-14 | |
| T0.4 | Majority F1 macro / accuracy | 0.2811 / 0.7292 | persistence_baseline.py | 42 | 2026-09-14 | reproduce 0.281 / 72.9% din CIFRE_CANONICE |

## Note metodologice

- T0.4: persistence = eticheta zilei precedente din seria de 958 zile (randul anterior, nu neaparat ziua calendaristica anterioara; scriptul raporteaza cate randuri din test au gap > 1 zi). Prima zi de test foloseste ultima zi de train.
- T0.4: pe acest dataset persistence e o functie determinista a doua dintre cele 19 features (readiness_score_lag1, sleep_score_lag1); RF are strict mai multa informatie decat baseline-ul. De folosit in response letter.

- T0.2: `predict()` la SVC(probability=True) foloseste functia de decizie, iar `predict_proba` foloseste Platt scaling (CV intern); pot diferi pe cateva zile. Faza 1 foloseste `y_pred` (= predict, ca in scriptul canonic) pentru metrici si McNemar, si probabilitatile doar pentru calibrare.

- T0.4: balanced_accuracy == f1_macro pentru persistence (0.30389610...) NU e bug. Verificat in `revision/scripts/_check_t04_coincidence.py` (rulat 2026-10-14): numarul de predictii per clasa (44/140/8) este exact egal cu suportul per clasa, deci precision = recall per clasa, deci F1 per clasa = recall si macro F1 = macro recall = balanced accuracy. Structural: predictiile persistence sunt etichetele de test shiftate cu o zi, iar ultima eticheta din train (Normal) coincide cu ultima din test (Normal), deci multiset-ul predictiilor = multiset-ul adevarului. Acelasi fapt explica si kappa == MCC. Scriptul persistence_baseline.py e corect.

## Status Faza 0

Inchisa: 2026-10-14.

- T0.1 PASS: rerun sanity_check_frozen.py reproduce exact cifrele canonice RF (F1 0.5063, acc 0.6406, MCC 0.1416), hash dataset match; versiuni mediu identice cu frozen_meta.
- T0.2 PASS: RF / XGB / SVM reproduse la 4 zecimale din revision/; predictii si probabilitati salvate in results/; acord SVM argmax(proba) vs predict() = 0.9115.
- T0.3 PASS: praguri de etichetare parametrizate, identitate verificata (ALL PASS, 10,201 perechi + 29 configuratii, test_t3_labels.py, 2026-09-14).
- T0.4 PASS: persistence baseline F1 macro 0.3039 [0.2612, 0.3501], F1 Active 0.0000, MCC -0.0566 [-0.1632, 0.0597], accuracy 0.5625; majority reproduce 0.2811 / 0.7292. Egalitatea balanced_accuracy == f1_macro verificata si explicata (nu e bug; vezi nota si _check_t04_coincidence.py).

Concluzie-cheie pentru response letter: CI-ul persistence pe F1 macro [0.261, 0.350] NU se suprapune cu CI-ul RF [0.373, 0.610]; persistence are F1 Active = 0; iar persistence este o functie determinista a readiness_score_lag1 si sleep_score_lag1, doua dintre cele 19 features ale modelului, deci RF are strict mai multa informatie decat baseline-ul.
