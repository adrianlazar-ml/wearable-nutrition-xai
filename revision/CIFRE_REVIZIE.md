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
| T1.1 | RF balanced accuracy [CI 95%] | 0.5227 [0.3851, 0.6634] | revision/scripts/extended_metrics.py | 42 | 2026-10-18 | 1000 resamples, matrice partajata, 0 sarite |
| T1.1 | RF Cohen's kappa (simplu) [CI 95%] | 0.1415 [-0.0050, 0.2924] | extended_metrics.py | 42 | 2026-10-18 | QWK canonic 0.2551 ramane |
| T1.1 | RF G-mean [CI 95%] | 0.4923 [0.2960, 0.6228] | extended_metrics.py | 42 | 2026-10-18 | media geometrica a recall-urilor Rest/Normal/Active |
| T1.1 | RF macro ROC-AUC OvR [CI 95%] | 0.5581 [0.4199, 0.6921] | extended_metrics.py | 42 | 2026-10-18 | CI include 0.5; de raportat ca atare |
| T1.1 | RF macro PR-AUC [CI 95%] | 0.4983 [0.3716, 0.6359] | extended_metrics.py | 42 | 2026-10-18 | peste prevalenta medie 0.3333 |
| T1.1 | XGB balanced accuracy [CI 95%] | 0.4500 [0.3439, 0.5763] | extended_metrics.py | 42 | 2026-10-18 | |
| T1.1 | XGB Cohen's kappa (simplu) [CI 95%] | 0.1459 [0.0065, 0.2905] | extended_metrics.py | 42 | 2026-10-18 | |
| T1.1 | XGB G-mean [CI 95%] | 0.3759 [0.0000, 0.5156] | extended_metrics.py | 42 | 2026-10-18 | CI atinge 0: in unele resample-uri recall Active = 0 |
| T1.1 | XGB macro ROC-AUC OvR [CI 95%] | 0.5892 [0.4610, 0.7022] | extended_metrics.py | 42 | 2026-10-18 | cel mai mare ROC-AUC dintre cele 3; CI include 0.5 |
| T1.1 | XGB macro PR-AUC [CI 95%] | 0.4598 [0.3616, 0.6082] | extended_metrics.py | 42 | 2026-10-18 | |
| T1.1 | SVM balanced accuracy [CI 95%] | 0.4342 [0.3288, 0.5683] | extended_metrics.py | 42 | 2026-10-18 | |
| T1.1 | SVM Cohen's kappa (simplu) [CI 95%] | 0.0955 [-0.0395, 0.2310] | extended_metrics.py | 42 | 2026-10-18 | |
| T1.1 | SVM G-mean [CI 95%] | 0.3824 [0.0000, 0.5355] | extended_metrics.py | 42 | 2026-10-18 | CI atinge 0 |
| T1.1 | SVM macro ROC-AUC OvR [CI 95%] | 0.5595 [0.4493, 0.6733] | extended_metrics.py | 42 | 2026-10-18 | probabilitati Platt; CI include 0.5 |
| T1.1 | SVM macro PR-AUC [CI 95%] | 0.4603 [0.3565, 0.6064] | extended_metrics.py | 42 | 2026-10-18 | |
| T1.1 | Persistence balanced accuracy [CI 95%] | 0.3039 [0.2602, 0.3520] | extended_metrics.py | 42 | 2026-10-18 | == F1 macro la punct (vezi nota T0.4); CI difera |
| T1.1 | Persistence Cohen's kappa (simplu) [CI 95%] | -0.0566 [-0.1626, 0.0593] | extended_metrics.py | 42 | 2026-10-18 | == MCC la punct (vezi nota T0.4) |
| T1.1 | Persistence G-mean [CI 95%] | 0.0000 [0.0000, 0.0000] | extended_metrics.py | 42 | 2026-10-18 | recall Active = 0 in toate resample-urile |
| T1.1 | Persistence / Majority ROC-AUC, PR-AUC | n/a | extended_metrics.py | 42 | 2026-10-18 | hard labels only |
| T1.1 | Majority balanced accuracy [CI 95%] | 0.3333 [0.3333, 0.3333] | extended_metrics.py | 42 | 2026-10-18 | = 1/3 exact |
| T1.1 | Majority Cohen's kappa (simplu) [CI 95%] | 0.0000 [0.0000, 0.0000] | extended_metrics.py | 42 | 2026-10-18 | |
| T1.1 | Majority G-mean [CI 95%] | 0.0000 [0.0000, 0.0000] | extended_metrics.py | 42 | 2026-10-18 | |
| T1.1 | RF F1 / acc / MCC / QWK (sanity, CI pe matricea partajata) | 0.5063 [0.3784, 0.6130] / 0.6406 [0.5729, 0.7083] / 0.1416 [-0.0050, 0.2929] / 0.2551 [0.1025, 0.3966] | extended_metrics.py | 42 | 2026-10-18 | puncte identice cu CIFRE_CANONICE; CI difera usor de cel canonic [0.373, 0.610] / [-0.0045, 0.2843] (alte resample-uri) |
| T1.1 | XGB F1 / acc / MCC / QWK (sanity) | 0.4760 [0.3406, 0.5933] / 0.6875 [0.6198, 0.7552] / 0.1503 [0.0067, 0.3042] / 0.1805 [0.0192, 0.3376] | extended_metrics.py | 42 | 2026-10-18 | puncte identice cu CIFRE_CANONICE |
| T1.1 | SVM F1 / acc / MCC / QWK (sanity) | 0.4347 [0.3279, 0.5395] / 0.6302 [0.5573, 0.6979] / 0.0956 [-0.0398, 0.2316] / 0.1709 [0.0263, 0.3146] | extended_metrics.py | 42 | 2026-10-18 | puncte identice cu CIFRE_CANONICE |
| T1.1 | Persistence F1 / acc / MCC / QWK (sanity) | 0.3039 [0.2612, 0.3501] / 0.5625 [0.4948, 0.6354] / -0.0566 [-0.1632, 0.0597] / 0.0055 [-0.1175, 0.1225] | extended_metrics.py | 42 | 2026-10-18 | CI F1 si MCC identice cu T0.4 (acelasi flux RNG) |
| T1.1 | Majority F1 / acc / MCC / QWK (sanity) | 0.2811 [0.2667, 0.2946] / 0.7292 [0.6667, 0.7917] / 0.0000 [0.0000, 0.0000] / 0.0000 [0.0000, 0.0000] | extended_metrics.py | 42 | 2026-10-18 | |
| T1.1 | Abatere maxima suma probabilitati (RF, XGB, SVM) | 1.066e-07 | extended_metrics.py | 42 | 2026-10-18 | prag 1e-6; PASS |
| T1.1 | Resample-uri sarite per metrica | 0 pentru toate cele 9 metrici | extended_metrics.py | 42 | 2026-10-18 | niciun resample fara vreo clasa in y_true (1000 x 192, seed 42) |
| T1.2 | McNemar RF-XGB (n01 / n10) | 15 / 24; chi2_cc 1.6410, p_cc 0.2002, p_exact 0.1996, p_holm 0.4004 | revision/scripts/statistical_tests.py | 42 | 2026-10-18 | n00 45, n11 108; discordante 39 >= 25, varianta cc raportata; nesemnificativ la 0.05 |
| T1.2 | McNemar RF-SVM (n01 / n10) | 22 / 20; chi2_cc 0.0238, p_cc 0.8774, p_exact 0.8776, p_holm 0.8774 | statistical_tests.py | 42 | 2026-10-18 | n00 49, n11 101; discordante 42; nesemnificativ |
| T1.2 | McNemar XGB-SVM (n01 / n10) | 27 / 16; chi2_cc 2.3256, p_cc 0.1273, p_exact 0.1263, p_holm 0.3818 | statistical_tests.py | 42 | 2026-10-18 | n00 44, n11 105; discordante 43; nesemnificativ (XGB are acuratetea cea mai mare, 0.6875) |
| T1.2 | Paired bootstrap RF-XGB, F1 macro | diff 0.0303 [-0.0796, 0.1561]; proportie resample-uri cu diff > 0 = 0.684 | statistical_tests.py | 42 | 2026-10-18 | CI include 0; matricea partajata T1.1 |
| T1.2 | Paired bootstrap RF-XGB, balanced accuracy | diff 0.0727 [-0.0322, 0.1998]; proportie diff > 0 = 0.891 | statistical_tests.py | 42 | 2026-10-18 | CI include 0 |
| T1.2 | Paired bootstrap RF-SVM, F1 macro | diff 0.0716 [-0.0337, 0.1775]; proportie diff > 0 = 0.902 | statistical_tests.py | 42 | 2026-10-18 | CI include 0 |
| T1.2 | Paired bootstrap RF-SVM, balanced accuracy | diff 0.0885 [-0.0232, 0.2159]; proportie diff > 0 = 0.929 | statistical_tests.py | 42 | 2026-10-18 | CI include 0 (cel mai aproape de a-l exclude) |
| T1.2 | Paired bootstrap XGB-SVM, F1 macro | diff 0.0413 [-0.1021, 0.1647]; proportie diff > 0 = 0.730 | statistical_tests.py | 42 | 2026-10-18 | CI include 0 |
| T1.2 | Paired bootstrap XGB-SVM, balanced accuracy | diff 0.0158 [-0.1195, 0.1508]; proportie diff > 0 = 0.606 | statistical_tests.py | 42 | 2026-10-18 | CI include 0 |
| T1.2 | Cochran's Q (RF, XGB, SVM, corect/incorect) | Q 3.3226, df 2, p 0.1899 | statistical_tests.py | 42 | 2026-10-18 | 192 zile x 3 modele; nesemnificativ; fara post-hoc suplimentar (perechile McNemar sunt post-hoc-ul) |
| T1.2 | McNemar RF-persistence (n01 / n10), comparatie cu baseline-ul | 50 / 35; chi2_cc 2.3059, p_cc 0.1289, p_exact 0.1284 | statistical_tests.py | 42 | 2026-10-18 | in afara familiei Holm (neajustat); n00 34, n11 73; NESEMNIFICATIV la 0.05 pe corect/incorect per zi, desi CI-urile marginale F1 macro RF / persistence nu se suprapun (T0.4); vezi nota |
| T1.2 | Versiune statsmodels | 0.15.0 | pip; pin in revision/requirements.txt | | 2026-10-18 | instalat la T1.2; python / numpy / pandas / sklearn neschimbate (3.11.9 / 2.4.3 / 2.3.3 / 1.8.0) |
| T1.2 | Resample-uri sarite (paired bootstrap) | 0 pentru toate cele 6 perechi x metrici | statistical_tests.py | 42 | 2026-10-18 | 1000 folosite din 1000; 0 warning-uri sklearn/statsmodels in bucla |

## Artefacte

Modele si preprocesoare salvate la T0.2 (`revision/scripts/save_test_predictions.py`, rulat 2026-09-14; inventar `_check_t12_inputs.py`, 2026-10-18). NU sunt folosite de T1.1 si T1.2 (care citesc doar csv-urile si matricea npy); necesare incepand cu T1.3 (calibrare) si T1.4. Toate sunt urmarite de git din commit-ul d3e1047 (Faza 0); total 2,779,243 bytes (2.78 MB), sub pragul de 20 MB.

| Cale | Bytes | sha256 (primele 16) | Data modificarii | Script producator | Status git |
|---|---|---|---|---|---|
| revision/results/models/rf_revision.joblib | 1,810,593 | b8113662207ab548 | 2026-09-14 22:46:54 | revision/scripts/save_test_predictions.py | urmarit |
| revision/results/models/xgb_revision.joblib | 751,753 | 268760acaf1055c5 | 2026-09-14 22:46:54 | revision/scripts/save_test_predictions.py | urmarit |
| revision/results/models/svm_revision.joblib | 213,939 | 811a86c96f08a586 | 2026-09-14 22:46:54 | revision/scripts/save_test_predictions.py | urmarit |
| revision/results/models/imputer_revision.joblib | 1,295 | 9b5e16ba6cf9a70a | 2026-09-14 22:46:54 | revision/scripts/save_test_predictions.py | urmarit |
| revision/results/models/scaler_revision.joblib | 1,663 | 981b7001eaf30521 | 2026-09-14 22:46:54 | revision/scripts/save_test_predictions.py | urmarit |

## Note metodologice

- T0.4: persistence = eticheta zilei precedente din seria de 958 zile (randul anterior, nu neaparat ziua calendaristica anterioara; scriptul raporteaza cate randuri din test au gap > 1 zi). Prima zi de test foloseste ultima zi de train.
- T0.4: pe acest dataset persistence e o functie determinista a doua dintre cele 19 features (readiness_score_lag1, sleep_score_lag1); RF are strict mai multa informatie decat baseline-ul. De folosit in response letter.

- T0.2: `predict()` la SVC(probability=True) foloseste functia de decizie, iar `predict_proba` foloseste Platt scaling (CV intern); pot diferi pe cateva zile. Faza 1 foloseste `y_pred` (= predict, ca in scriptul canonic) pentru metrici si McNemar, si probabilitatile doar pentru calibrare.

- T0.4: balanced_accuracy == f1_macro pentru persistence (0.30389610...) NU e bug. Verificat in `revision/scripts/_check_t04_coincidence.py` (rulat 2026-10-14): numarul de predictii per clasa (44/140/8) este exact egal cu suportul per clasa, deci precision = recall per clasa, deci F1 per clasa = recall si macro F1 = macro recall = balanced accuracy. Structural: predictiile persistence sunt etichetele de test shiftate cu o zi, iar ultima eticheta din train (Normal) coincide cu ultima din test (Normal), deci multiset-ul predictiilor = multiset-ul adevarului. Acelasi fapt explica si kappa == MCC. Scriptul persistence_baseline.py e corect.

- T1.1: toate metricile sunt calculate din predictiile salvate la T0.2 (`results/test_predictions_{rf,xgb,svm}.csv`), fara reantrenare. Inventarul intrarilor (`_check_t11_inputs.py`, 2026-10-18): 192 randuri x 6 coloane in fiecare fisier, etichete int 0/1/2 (Rest/Normal/Active), coloane p_rest, p_normal, p_active in ordinea claselor [0, 1, 2], date sortata crescator si identica in cele trei fisiere, y_true identic element cu element, suport 44/140/8. `persistence_baseline.csv` e doar rezumat; predictiile persistence au fost reconstruite din y_true cu pred[0] = ultima eticheta din train citita din `models/dataset_snapshot_20260331.csv` (= 1, Normal) si sunt identice cu `test_predictions_persistence.csv` salvat la T0.4. Metricile canonice recalculate din csv (F1, acc, MCC, QWK) coincid la 4 zecimale pentru RF, XGB, SVM.

- T1.1, G-mean multiclass: G = (recall_Rest x recall_Normal x recall_Active)^(1/3), cu recall_c = TP_c / suport_c si clasele fixate explicit (labels = [0, 1, 2]). Implementare manuala in `extended_metrics.gmean()`, testata in `revision/scripts/test_extended_metrics.py` (6 teste, ALL PASS, 2026-10-18: predictie perfecta = 1, clasa neprezisa = 0, recall-uri cunoscute 0.5/1.0/0.25 -> 0.5, egalitate cu media geometrica a `recall_score`, nan cand o clasa lipseste din y_true, independenta de ordinea etichetelor). Pentru persistence si majority G-mean = 0 pentru ca recall Active = 0 (Active nu e prezis niciodata corect).

- T1.1, ROC-AUC si PR-AUC n/a pentru persistence si majority: cele doua baseline-uri produc doar etichete hard (one-hot), nu scoruri ordonabile; o curba ROC/PR pe one-hot are un singur punct operational, iar AUC-ul rezultat nu e comparabil cu cel al modelelor cu probabilitati. Pentru RF, XGB, SVM probabilitatile vin din p_rest, p_normal, p_active (verificat: ordinea coloanelor = ordinea claselor la `label_binarize(classes=[0, 1, 2])`; abaterea maxima a sumei pe rand fata de 1 este 1.066e-07). Etichetele hard pentru balanced accuracy, kappa, G-mean vin din y_pred (SVM: predict(), nu argmax(proba), conform notei T0.2).

- T1.1, bootstrap: o singura matrice de indici 1000 x 192, `np.random.default_rng(42).integers(0, 192, size=(1000, 192))`, salvata in `results/bootstrap_indices_1000x192_seed42.npy` si folosita pentru toate sistemele si toate metricile (T1.2 o va folosi pentru paired bootstrap). CI = percentile 2.5 / 97.5. Niciun resample nu are vreo clasa absenta din y_true, deci 0 resample-uri sarite pentru fiecare dintre cele 9 metrici si 0 warning-uri sklearn in bucla. Matricea coincide cu fluxul RNG din `persistence_baseline.py` (1000 apeluri succesive `integers(0, 192, 192)` cu acelasi seed), de aceea CI-urile persistence pentru F1 si MCC sunt identice cu T0.4. CI-urile RF pe F1 [0.3784, 0.6130] si MCC [-0.0050, 0.2929] difera usor de cele canonice [0.373, 0.610] / [-0.0045, 0.2843], care au fost obtinute cu alte resample-uri; punctele sunt identice. In manuscris se citeaza CI-urile canonice pentru cifrele canonice si CI-urile T1.1 pentru metricile noi.

- T1.1, plauzibilitate (Task 5, raportat): persistence balanced accuracy == F1 macro (0.303896) si kappa == MCC (-0.056604) la punct, conform notei T0.4; CI-urile lor difera insa ([0.2602, 0.3520] vs [0.2612, 0.3501]) pentru ca egalitatea multiset-urilor nu se pastreaza in resample-uri. Majority: balanced accuracy = 1/3, kappa = 0, G-mean = 0, exact. RF: balanced accuracy 0.5227 intre F1 macro 0.5063 si 1.0. ROC-AUC peste 0.5 si PR-AUC peste prevalenta medie (1/3) pentru toate cele trei modele la punct; dar CI-ul ROC-AUC include 0.5 pentru toate trei (RF [0.4199, 0.6921], XGB [0.4610, 0.7022], SVM [0.4493, 0.6733]), iar CI-ul G-mean atinge 0 pentru XGB si SVM. Se raporteaza ca atare, consistent cu regula 4.

- T1.1, ierarhie: pe balanced accuracy si G-mean RF ramane primul (0.5227 / 0.4923), XGB si SVM sunt apropiate (0.4500 / 0.3759 si 0.4342 / 0.3824); pe ROC-AUC XGB e primul (0.5892), pe PR-AUC RF (0.4983). Kappa simplu este foarte apropiat de MCC pentru toate sistemele (diferenta maxima 0.0044, la XGB: 0.1459 vs 0.1503).

- T1.2, McNemar: pentru perechea A-B, corect = `y_pred == y_true` pe fiecare dintre cele 192 zile; n01 = A corect si B gresit, n10 = A gresit si B corect, n11 = ambele corecte, n00 = ambele gresite; tabelul dat la statsmodels este [[n11, n01], [n10, n00]], iar statistica depinde doar de n01 si n10. McNemar testeaza acuratetea per zi (corect/incorect), nu F1 macro: e insensibil la clasa pe care se greseste, deci nu "vede" diferentele pe clasa Active. Etichetele hard vin din `y_pred` (SVM: `predict()`, conform notei T0.2); probabilitatile nu intra in T1.2.

- T1.2, varianta exacta: chi-patratul cu corectie de continuitate (`mcnemar(exact=False, correction=True)`) e o aproximare asimptotica; cu putine discordante (regula uzuala: n01 + n10 < 25) se recomanda testul binomial exact (`exact=True`). Aici discordantele sunt 39 / 42 / 43 (85 pentru RF-persistence), deci varianta cc e cea raportata, iar cea exacta e listata alaturi; difera abia la a 3-a zecimala si duce la aceeasi concluzie. Coloana `recommended` din `t12_mcnemar.csv` aplica regula automat.

- T1.2, "proportie resample-uri cu diff > 0" NU e p-value: e fractia din distributia bootstrap a diferentei metric_A - metric_B care cade peste 0, o descriere a distributiei, nu o probabilitate sub ipoteza nula (nu exista o ipoteza nula simulata). Inferenta se face pe CI-ul percentile al diferentei: daca include 0, diferenta nu e distinsa de 0 la 95%. Toate cele 6 CI-uri (3 perechi x F1 macro / balanced accuracy) includ 0.

- T1.2, familii de teste: ajustarea Holm (`multipletests(method="holm")`) se aplica pe familia de 3 perechi de modele (RF-XGB, RF-SVM, XGB-SVM), separat pentru p_cc si p_exact. RF-persistence raspunde la o alta intrebare (model vs baseline, cerinta Reviewer 2), deci e raportat in afara familiei, neajustat. Cochran's Q e testul global pe cele 3 modele simultan (Q 3.3226, df 2, p 0.1899); perechile McNemar sunt post-hoc-ul lui, nu se face alt post-hoc.

- T1.2, paired bootstrap: aceeasi matrice de indici ca T1.1 (`bootstrap_indices_1000x192_seed42.npy`, verificata byte-identic cu `default_rng(42).integers(0, 192, size=(1000, 192))`), acelasi vector de indici aplicat la y_true, y_pred_A si y_pred_B in fiecare resample, deci diferenta e pe aceleasi zile. Sanity check marginal PASS: cele 12 CI-uri marginale (RF / XGB / SVM x F1 macro / balanced accuracy) coincid la 4 zecimale cu T1.1, iar cele 6 diferente la punct coincid cu diferentele cifrelor canonice / T1.1 (de ex. RF-XGB F1 0.5063 - 0.4760 = 0.0303). 0 resample-uri sarite, 0 warning-uri in bucla.

- T1.2, rezultat neasteptat (raportat, nu ajustat): McNemar RF vs persistence NU e semnificativ la 0.05 (p_cc 0.1289; 50 zile RF corect / persistence gresit vs 35 invers), desi la T0.4 CI-urile marginale F1 macro ale RF [0.3784, 0.6130] si persistence [0.2612, 0.3501] nu se suprapun. Nu e o contradictie: McNemar compara acuratetea per zi (RF 0.6406 vs persistence 0.5625), pe care persistence o obtine prin zilele Normal consecutive, iar F1 macro penalizeaza puternic F1 Active = 0 al persistence. In response letter se formuleaza exact asa: avantajul RF fata de persistence e pe clasele minoritare (F1 macro, balanced accuracy, G-mean), nu pe acuratetea bruta; comparatia formala RF-persistence pe F1 macro (paired bootstrap, aceeasi procedura ca 2b) nu a fost ceruta la T1.2 si ramane de adaugat daca e nevoie pentru T4.2.

## Status Faza 1

In lucru. Deschisa: 2026-10-18.

- T1.1 PASS (2026-10-18, commit e54883d): inventar intrari PASS (`_check_t11_inputs.py`), metrici canonice din csv PASS la 4 zecimale, `extended_metrics.py` rulat de doua ori consecutiv cu iesiri byte-identice (sha256 identic pe csv, md si npy), 26.5 s / 18.0 s, 0 erori; tabel md == csv (`_check_t11_outputs.py`); diff pe CIFRE_CANONICE.md gol. Iesiri: `results/t11_extended_metrics.csv`, `results/t11_extended_metrics_table.md`, `results/bootstrap_indices_1000x192_seed42.npy`.
- T1.2 PASS (2026-10-18): inventar intrari PASS (`_check_t12_inputs.py`: 4 csv x 192 randuri, `date` si `y_true` identice, suport 44/140/8, matrice npy (1000, 192) int64 byte-identica cu `default_rng(42)`); `statistical_tests.py` cu 29 sanity check-uri interne PASS (sanity 2b: 12 CI marginale == T1.1 la 4 zecimale, 6 diff_point == diferente canonice; p-value-uri in [0, 1]; n00+n01+n10+n11 = 192 pe fiecare pereche); teste unitare 7/7 PASS (`test_statistical_tests.py`, pytest 9.1.1, 2.7 s); rulat de doua ori consecutiv cu iesiri byte-identice (sha256 identic pe 5 fisiere csv/md, json identic fara `run_date`), 10.0 s / 10.0 s, 0 warning-uri; LF pe toate iesirile si pe CIFRE_REVIZIE.md (`_check_t12_outputs.py`); diff gol pe CIFRE_CANONICE.md si pe rezultatele T0.x / T1.1; statsmodels 0.15.0 instalat si pinuit in `revision/requirements.txt`. Iesiri: `results/t12_mcnemar.csv`, `results/t12_mcnemar_table.md`, `results/t12_paired_bootstrap.csv`, `results/t12_paired_bootstrap_table.md`, `results/t12_cochran_q.csv`, `results/t12_run_meta.json`.
- T1.3, T1.4, T1.5: nu au inceput.

## Status Faza 0

Inchisa: 2026-10-14, commit d3e1047.

- T0.1 PASS: rerun sanity_check_frozen.py reproduce exact cifrele canonice RF (F1 0.5063, acc 0.6406, MCC 0.1416), hash dataset match; versiuni mediu identice cu frozen_meta.
- T0.2 PASS: RF / XGB / SVM reproduse la 4 zecimale din revision/; predictii si probabilitati salvate in results/; acord SVM argmax(proba) vs predict() = 0.9115.
- T0.3 PASS: praguri de etichetare parametrizate, identitate verificata (ALL PASS, 10,201 perechi + 29 configuratii, test_t3_labels.py, 2026-09-14).
- T0.4 PASS: persistence baseline F1 macro 0.3039 [0.2612, 0.3501], F1 Active 0.0000, MCC -0.0566 [-0.1632, 0.0597], accuracy 0.5625; majority reproduce 0.2811 / 0.7292. Egalitatea balanced_accuracy == f1_macro verificata si explicata (nu e bug; vezi nota si _check_t04_coincidence.py).

Concluzie-cheie pentru response letter: CI-ul persistence pe F1 macro [0.261, 0.350] NU se suprapune cu CI-ul RF [0.373, 0.610]; persistence are F1 Active = 0; iar persistence este o functie determinista a readiness_score_lag1 si sleep_score_lag1, doua dintre cele 19 features ale modelului, deci RF are strict mai multa informatie decat baseline-ul.
