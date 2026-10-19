# F5 - raport de format (T5.3b, T5.4, T5.5), Revision 1 KBS

Data: 2026-10-20. Fisiere verificate: revision/manuscript/KBS_Manuscript_R1.docx, KBS_Manuscript_R1_marked.docx, KBS_Supplementary_R1.docx, KBS_Response_R1.docx, KBS_Highlights_R1.docx, F4_manuscript_changes.md, F4_response_letter.md. Toate docx-urile sunt regenerate de `revision/scripts/assemble_manuscript.py` din BASE_MANUSCRIPT_v8.docx + sursele F4 (o rerulare fara modificari a fost verificata ca identica la nivel de text, stiluri, evidentieri si comentarii cu versiunea comisa; de aceea modificarile de text din F5 s-au facut in surse si prin rerulare, nu in docx direct). Nicio cifra din manuscris nu s-a modificat (T5.5).

## T5.3b - ortografie americana in textul nou

Standard: engleza americana (v8 verificat 20 oct: Personalized, behavior, characterized, labeling, emphasize, analyzed, standardized, favor, prioritize). Metoda: (1) grep pe `-is(e|ed|es|ing|ation)\b` si `-our\b` plus lista explicita de perechi UK -> US, pe toate cele 7 fisiere, cu listarea tuturor potrivirilor INAINTE de orice inlocuire (scripturile de scanare in scratchpad-ul sesiunii; listarea completa a inlocuirilor este afisata de `revision/scripts/f5_us_spelling.py --dry-run`); (2) inlocuire in sursele F4 cu `f5_us_spelling.py` (liniile de citat ale reviewerilor, care incep cu ">", sunt sarite explicit), in trei siruri hard-codate din `assemble_manuscript.py` (doua pentru fisierul Supplementary, una pentru propozitia din Related Work preluata din baza v8) si rerularea asamblarii; (3) re-scanare.

### Potriviri UK gasite inainte de inlocuire (toate fisierele)

| Fisier | Forme UK in corpul textului (numar) | Total |
|---|---|---|
| KBS_Manuscript_R1.docx | labelling 4 (3.1, 3.10, 4.7 x2), labelled 2 (4.7 text, legenda Table 10), personalised 2 (contributii, Discussion), favourable 2 (4.8, Discussion), analysed 1 (3.2), standardised 1 (3.4), characterise 1 (3.6), characterised 1 (Conclusions), hypothesise 1 (Discussion), personalisation 1 (Conclusions), prioritise 1 (Conclusions), neighbouring 1 (Discussion), artefact 1 (4.7), modelling 1 (Related Work, text de baza v8: "survival modelling") | 20 |
| KBS_Manuscript_R1_marked.docx | identic cu manuscrisul (copia marcata are acelasi corp de text) | 20 |
| KBS_Supplementary_R1.docx | labelling 1 (introducerea S3), labelled 1 (legenda Table S3) | 2 |
| KBS_Response_R1.docx | labelling 2, personalised 2, generalisation 1 (titlul Comment 1), optimisation 1 (titlul Comment 4), favourable 1, hypothesise 1, personalisation 1 | 9 |
| F4_manuscript_changes.md | aceleasi 19 forme ca in manuscris mai putin "survival modelling" (care vine din baza v8) | 19 |
| F4_response_letter.md | aceleasi 9 forme ca in KBS_Response_R1.docx | 9 |

Potriviri ale grep-ului generic excluse ca false positives (nu sunt forme britanice): Pairwise / pairwise (6 in manuscris, 1 in Supplementary, 1 in Response), promising (1), characteristics (manuscris 1, Response 1), analyses (substantiv; manuscris 3, Response 2, F4 md 3), optimistic (manuscris 3, Response 1, F4 md 3), precise / otherwise / raise / noise / hour / four (din lista de excludere, fara potriviri relevante), numele de variabile sleep_deep_hours si sleep_rem_hours (celule din Table 1), cuvintele romanesti "scrise" si "submise" din notele meta ale F4_manuscript_changes.md, "acknowledgements" (2 note meta in romana din F4_manuscript_changes.md si numele de sectiune folosit de Elsevier; neschimbat).

### Inlocuiri efectuate

| Fisier | Inlocuiri UK -> US | Cum |
|---|---|---|
| F4_manuscript_changes.md | 19 | f5_us_spelling.py |
| F4_response_letter.md | 9 (0 sarite in citate: citatele reviewerilor nu contin forme UK) | f5_us_spelling.py |
| assemble_manuscript.py (siruri) | 3: "labelling rule" si "labelled days" (Supplementary S3), "survival modelling in aged care" -> "survival modeling in aged care" (Related Work, text de baza v8) | editare manuala a sirurilor + regula de inlocuire in text de baza |
| KBS_Manuscript_R1.docx | 20 (19 din F4 + 1 din baza) | rerulare asamblare |
| KBS_Manuscript_R1_marked.docx | 20 (propozitia din Related Work apare acum evidentiata; paragrafe marcate 97 -> 98) | rerulare asamblare |
| KBS_Supplementary_R1.docx | 2 | rerulare asamblare |
| KBS_Response_R1.docx | 9 | rerulare asamblare |
| KBS_Highlights_R1.docx | 0 (nicio forma UK) | - |

Perechi aplicate: characterise/characterized (+ -ed), personalisation/personalization, personalised/personalized, modelling/modeling, labelling/labeling, labelled/labeled, analysed/analyzed, standardised/standardized, hypothesise/hypothesize, prioritise/prioritize, favourable/favorable, neighbouring/neighboring, artefact/artifact (manuscrisul foloseste deja "counting artifact"), generalisation/generalization, optimisation/optimization. Celelalte perechi din lista (behaviour, centre, colour, summarise, emphasise, optimise etc.) nu au avut potriviri.

### Exceptii pastrate (cu locatia)

- Titlul referintei Susnjak & Griffin (2025), "An explainable AI approach to survival modelling in residential aged care", in lista References din KBS_Manuscript_R1.docx si KBS_Manuscript_R1_marked.docx (1 aparitie in fiecare): ramane "modelling", asa cum este publicat.
- Citatele reviewerilor din KBS_Response_R1.docx (12 paragrafe italice indentate; in F4_response_letter.md liniile cu ">"): verificate, 0 forme UK, nimic de modificat.
- Nume de fisiere si variabile: sleep_deep_hours, sleep_rem_hours (Table 1), neschimbate.

### Rezultat dupa inlocuire

Re-scanare cu lista de perechi pe toate cele 7 fisiere: 0 forme UK in corpul textului; singura potrivire ramasa este titlul referintei Susnjak & Griffin (exceptia de mai sus). PASS.

## T5.4 - verificari de format (doar raport, nimic modificat)

### a) Cross-referinte fara numar de sectiune (KBS p. 25)

Cautat in KBS_Manuscript_R1.docx (fara lista de referinte): "the text" 0, "as mentioned" 0, "see text" 0, "aforementioned" 0, "above" 10, "below" 9. Din cele 19 aparitii de above / below, 12 sunt comparatii numerice (de ex. "7.8 points above the ring alone", "below the majority baseline", "Platt slopes ... below one", "remained below every tree ensemble", "0.009 below the selected one", "scored below its default"), nu cross-referinte. Cele 7 cross-referinte pozitionale, cu paragraful:

| # | Sectiune | Text |
|---|---|---|
| 1 | 1. Introduction, paragraful 5 | "This paper addresses the gap identified above by investigating whether ..." |
| 2 | 3.4 Classification Models, paragraful 3 | "In addition to the three classifiers above, four further models were evaluated ..." |
| 3 | 3.11 Temporal Robustness | "... baselines, bootstrap intervals and paired differences were computed as above." |
| 4 | 4. Results, fraza introductiva | "All results below use the JSON data source adopted as primary." |
| 5 | 4.6 Ablation by Device and by Feature Group, ultimul paragraf | "... and the two results above are consistent." |
| 6 | 4.10 Data Quality: XML versus JSON | "... differs from the 19-feature configuration reported above." |
| 7 | 5. Discussion, Practical implications | "(Temporal robustness, above)" |

Rezolvat la T6.1 (2026-10-20). Inlocuirile s-au facut in surse (F4_manuscript_changes.md pentru #2, #3, #5, #7; tuplurile de inlocuire in textul de baza v8 din assemble_manuscript.py pentru #1, #4, #6) si docx-urile au fost regenerate. Tintele au fost verificate in docx-ul asamblat: 3.5 defineste baselines (persistence, majority), 3.6 bootstrap-ul si diferentele paired, Table 7 = ablatia per dispozitiv, Table 8 = leave-one-group-out, 4.2 = rezultatele pe 19 features (Table 4), 4.8 = Temporal Robustness.

| # | Sectiune | Inainte | Dupa | Tinta |
|---|---|---|---|---|
| 1 | 1. Introduction, paragraful 5 | "the gap identified above" | "that gap" | referinta in aceeasi sectiune, fara numar: s-a revenit la formularea submisa (v8) prin eliminarea tuplului T4.1-D |
| 2 | 3.4 Classification Models, paragraful 3 | "the three classifiers above" | "Random Forest, XGBoost and the SVM" | referinta in aceeasi sectiune: numele celor trei clasificatori din paragraful 1 al 3.4, in loc de o auto-referinta "Section 3.4" |
| 3 | 3.11 Temporal Robustness | "computed as above" | "computed as in Sections 3.5 and 3.6" | 3.5 (baselines), 3.6 (bootstrap, paired differences) |
| 4 | 4. Results, fraza introductiva | "All results below use" | "All results in Section 4 use" | Section 4 |
| 5 | 4.6 Ablation by Device and by Feature Group | "the two results above are consistent" | "the two results (Table 7 and Table 8) are consistent" | Table 7, Table 8 (scris in F4 ca "Table 6 and Table 6b", renumerotat de script) |
| 6 | 4.10 Data Quality: XML versus JSON | "reported above" | "reported in Section 4.2" | Section 4.2 (Table 4) |
| 7 | 5. Discussion, Limitations (1) | "(Temporal robustness, above)" | "(Section 4.8)" | Section 4.8 |

Dupa T6.1: 0 cross-referinte pozitionale; raman cele 12 comparatii numerice above / below. In F4_manuscript_changes.md a fost corectat la fel si blocul draft neutilizat "Limitation (1), rescrisa" ("the temporal evaluation above" -> "the temporal evaluation of Section 4.8"); blocul nu intra in docx.

### b) Supplementary (KBS p. 22)

Citari in KBS_Manuscript_R1.docx: 7 in total; Supplementary Table S1 1 (4.3), Supplementary Table S2 1 (4.5), Supplementary Table S3 2 (4.7), Supplementary Figure S1 1 (4.7), Supplementary Figure S2 1 (4.11), Supplementary Figure S3 1 (4.12). Fiecare element din fisierul Supplementary este citat cel putin o data (verificat si de `_check_t41_outputs.py`, 6 verificari PASS). Sintagma "Supplementary Material" nu apare in corpul manuscrisului (0); o fraza de tip "Supplementary material associated with this article can be found in the online version" este adaugata de editura la productie, nu de autori.

KBS_Supplementary_R1.docx: 61 blocuri (21 paragrafe, 6 tabele). Are titlu ("Supplementary Material" + titlul manuscrisului) si o fraza introductiva care enumera continutul; trei sectiuni cu heading (S1 Statistical comparison between classifiers, S2 Calibration of the random forest, S3 Agreement of the labels with ring signals and lag-1 indicators) si sectiunea "Supplementary figures". Tabele: 6, fiecare cu caption descriptiv ("Table S1. (a) Pairwise McNemar tests ...", "Table S1. (b) Paired bootstrap differences ...", "Table S2. (a) Calibration metrics ...", "Table S2. (b) Per-class calibration ...", "Table S2. (c) Safety layer at the 0.50 confidence threshold ...", "Table S3. Ring signals (level B) and lag-1 activity indicators ..."). Figuri: 3 imagini incorporate (word/media/image4-6.png), fiecare cu legenda descriptiva (Figure S1, S2, S3; textul in revision/figures/FIGURES_MANIFEST.md). PASS la cerinta "titlu si caption descriptiv per tabel / figura".

### c) Acknowledgements (KBS p. 26)

Sectiune Acknowledgements: 0 aparitii ("Acknowledg" 0 in tot textul). Ordinea sectiunilor finale: 6. Conclusions and Future Work -> Declaration of generative AI and AI-assisted technologies in the manuscript preparation process -> Data availability -> Appendix A. Hyperparameter search -> References. Nu exista text si nici placeholder. Daca se adauga, KBS cere o sectiune separata la sfarsitul articolului, inainte de referinte (deci intre Data availability si Appendix A sau imediat inaintea References, dupa anexa). Intrebare pentru Adrian: exista finantare / persoane de mentionat?

### d) Data availability, Declaration of competing interest, CRediT

- Data availability: PREZENTA (heading "Data availability", 1 paragraf). Text exact: "The raw physiological data cannot be shared publicly because they constitute sensitive personal health data from a single identifiable participant. The source code and the analysis scripts will be deposited in a public repository with a persistent identifier upon acceptance, and an anonymized access route can be provided to reviewers on request during the review process."
- Declaration of generative AI: PREZENTA (text exact: "During the preparation of this work the authors used Claude (Anthropic) in order to support language editing and the structuring of the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.").
- Declaration of competing interest: ABSENTA (0 aparitii "competing interest", "Conflict of interest"); fara placeholder. Elsevier o cere obligatoriu (de regula ca fisier separat generat in Editorial Manager, dar si ca sectiune in manuscris).
- CRediT authorship contribution statement: ABSENTA (0 aparitii); fara placeholder.
- Funding: ABSENTA (0 aparitii "Funding" / "funding"); fara placeholder.
- Pagina de titlu: manuscrisul incepe direct cu titlul si Abstract, fara autori, afilieri si autor corespondent (baza este v8 "Anonymized"). De clarificat daca KBS primeste pagina de titlu separat in Editorial Manager sau in manuscris.

### e) Comentarii Word si field codes

| Fisier | Comentarii | Text comentarii | fldSimple | instrText | Marcaje Zotero / CSL |
|---|---|---|---|---|---|
| KBS_Manuscript_R1.docx | 6 | toate: "verify in Zotero (provisional APA 7 entry added at assembly, T4.1)", autor "T4.1 assembly", pe cele 6 referinte noi | 0 | 0 | 0 |
| KBS_Manuscript_R1_marked.docx | 6 | identic | 0 | 0 | 0 |
| KBS_Supplementary_R1.docx | 0 | - | 0 | 0 | 0 |
| KBS_Response_R1.docx | 0 | - | 0 | 0 | 0 |
| KBS_Highlights_R1.docx | 0 | - | 0 | 0 | 0 |

Referintele sunt text simplu (fara campuri Zotero); cele 6 comentarii se scot dupa importul in Zotero (task separat, conform regulii). Copia marcata foloseste evidentiere galbena, nu track changes (188 run-uri evidentiate la T4.1; 98 paragrafe si 408 celule marcate dupa T5.3b).

## T5.5 - verificare finala a cifrelor

`_check_t41_numbers.py` rerulat dupa T5.3 si T5.3b (2026-10-20):

| Sursa | Inainte (commit 7ce81b1 / baza T4.1) | Dupa T5.3 + T5.3b | Rezultat |
|---|---|---|---|
| KBS_Manuscript_R1.docx | 1509 numere; FOUND 1061; NEGASIT 0; NEGASIT explicate (numere din baza, in afara fisierelor de cifre) 10; SKIPPED 438 | 1509; 1061; 0; 10; 438 | identic, PASS |
| KBS_Supplementary_R1.docx | (nu exista rulare anterioara) | 372 numere; FOUND 308; NEGASIT 0; SKIPPED 64 | 0 cifre neexplicate, PASS (t41_number_check_S1.md, nou) |
| F4_manuscript_changes.md | 1175 numere; FOUND 937; NEGASIT 0 | 1175; 937; 0 | identic, PASS |

Formulari interzise (TEXT_DECISIONS.md) in manuscris: best performing 0, best overall 0, stable over time 0, optimal threshold 0, three years 0; "significant" 1 (cu test numit in aceeasi fraza), "optimal" 1 (documentatia producatorului, scoruri >= 85), neschimbate fata de T4.1. `_check_t41_outputs.py`: 67 verificari, 0 FAIL dupa fiecare task (regula git a checker-ului extinsa pentru livrabilele F5: revision/F5_*, revision/references_new*, revision/figures/; caption-ul anexei verificat acum ca "Table A.1").

## Mediu

bibtexparser 2.1.0 instalat pentru T5.1 (pachet pur Python, nu schimba nicio versiune din mediul inghetat: Python 3.11.9, scikit-learn 1.8.0, numpy 2.4.3, pandas 2.3.3, matplotlib 3.10.8).

## T6.5 - Figuri fara script in repo (Figure 6 si Figure 7), doar raport

Provenienta comuna (verificata 2026-10-20): ambele fisiere vin din `C:\Users\Adrian\Downloads\assa\` (Figure2_shap_bar_rf.pdf/.png, Figure3_confusion_matrices.pdf/.png, data fisierelor 2026-06-21 18:19), setul de figuri al manuscrisului submis la ESWA si apoi la KBS (copiat in revision/manuscript/figures/ la T4.1 si in revision/figures/ la T5.2). Producer PDF: Matplotlib 3.11.0 (mediul inghetat al reviziei are 3.10.8), font DejaVu Serif (TrueType, embedded), PNG la 1000 dpi. In acelasi folder exista scriptul care le-a generat: `fix_eswa_figures.py` ("Regenerare figuri pentru submisia ESWA", autor Adrian Lazar, 2026), care NU este in repo. Scriptul reutilizeaza `cap5_compare_algorithms_json_ext.load_and_prepare()` din radacina repo-ului (pipeline-ul inghetat), reantreneaza RF / XGB / SVM (split 766 / 192, imputer median, SMOTE k = 3, seed 42, hiperparametrii canonici), scrie in `outputs/figures/eswa/` (folder absent acum din repo) cu `pdf.fonttype 42`, fara titlu in imagine, nume de feature-uri fara sufixul _lag1 (de ex. aw_hr_mean).

| Figura | Caption (din KBS_Manuscript_R1.docx) | Sursa cifrelor aratate | Cost estimat de reconstructie |
|---|---|---|---|
| Figure 6 (SHAP bar, RF) | "Global feature importance for the Random Forest model on the 19-feature configuration, computed as the mean absolute SHAP value across the temporal test set. The mean 24-hour heart rate from the Apple Watch (aw_hr_mean) is the dominant predictor, with an importance more than twice that of the second feature. Feature names prefixed with aw_ originate from the Apple Watch; the remaining features originate from the Oura Ring. The order of features from the third rank downward is not stable across retrained models (Section 4.11); only the first two ranks should be interpreted." | Cifrele exista in repo: `revision/results/t14_shap_reference.csv` (19 feature-uri, mean_abs_shap, rank; RF canonic rf_revision.joblib, TreeExplainer, medie a valorilor absolute pe 192 zile x 3 clase, aceeasi procedura ca `run_shap()` din scriptul canonic), produs de `shap_stability.py` si verificat de `_check_t14_inputs.py`; rand in CIFRE_REVIZIE.md: T1.4 "Ranking SHAP de referinta" (aw_hr_mean 0.0611; hrv_balance 0.0273; aw_active_calories 0.0205; readiness_score 0.0192; activity_score 0.0191), consistent cu caption-ul (raport 2.2 intre locurile 1 si 2). CIFRE_CANONICE.md sectiunile 1-16 nu contin un tabel cu valorile SHAP (doar lista de feature-uri si, in sectiunea Revision 1, randurile T1.4). | MIC: script existent (in afara repo-ului) si date existente. Varianta A, fara reantrenare: o functie de plot de circa 20 de linii in `export_figures_kbs.py` care citeste t14_shap_reference.csv (barh sortat, pdf.fonttype 42, 300 dpi), verificabila valoare cu valoare fata de csv. Varianta B: copierea `fix_eswa_figures.py` in revision/scripts si rularea pe pipeline-ul inghetat (reantreneaza 3 modele, circa 1 minut; necesita matplotlib 3.10.8, nu 3.11). De decis: alinierea numelor la conventia noilor figuri (prefix oura_ pentru feature-urile inelului, TEXT_DECISIONS punctul 3) ar schimba etichetele fata de figura submisa. |
| Figure 7 (matrici de confuzie RF / XGB / SVM) | "Confusion matrices on the temporal test set (192 days: 44 Rest, 140 Normal, 8 Active) for the three classifiers: (a) Random Forest, (b) XGBoost, and (c) SVM with an RBF kernel. Rows are true labels and columns are predicted labels. All three models systematically misclassify the minority Active class; Random Forest correctly identifies 4 of the 8 Active days." | Cifrele exista in repo: `revision/results/test_predictions_{rf,xgb,svm}.csv` (T0.2, `save_test_predictions.py`, predictiile canonice pe cele 192 zile). Matricile calculate din ele la 2026-10-20 (randuri = adevarat Rest / Normal / Active, coloane = prezis): RF [14, 30, 0; 29, 105, 6; 0, 4, 4]; XGB [11, 32, 1; 20, 119, 1; 0, 6, 2]; SVM [13, 31, 0; 28, 106, 6; 1, 5, 2]; consistent cu caption-ul (RF 4 din 8 Active). Matricea RF apare si in `t31_threshold_grid.csv` (celula R60_A85, coloanele cm_*) si `t33_expanding_window.csv` (fold 1). Nu exista un rand explicit in CIFRE_REVIZIE.md sau in CIFRE_CANONICE.md cu cele trei matrici complete. | MIC: date existente, fara reantrenare: o functie de plot (sklearn ConfusionMatrixDisplay, 1 x 3 panouri (a) (b) (c), cmap Blues, ca in `fix_eswa_figures.py`) pe cele trei csv, in `export_figures_kbs.py`. Daca se reconstruieste, cele trei matrici intra ca rand nou in CIFRE_REVIZIE.md (script, seed 42, data) inainte de folosire, conform regulii 2 din plan. |

Concluzie T6.5: niciuna dintre cele doua figuri nu are script in repo, dar ambele au scriptul original identificat in Downloads/assa si cifrele complete in revision/results/, deci reconstructia este de cost mic (plot pe csv existente, fara reantrenare) si verificabila numeric. Nicio reconstructie nu s-a facut in acest task; decizia (reconstructie cu nume aliniate sau pastrarea fisierelor submise) ramane la Adrian.
