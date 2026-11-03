# F10_kbs_conformity.md - conformitate KBS si v8 (T10.7, 1 nov 2026; DOAR RAPORT, nimic aplicat)

Fisiere analizate: `revision/manuscript/BASE_MANUSCRIPT_v8.docx` (referinta de conventie), `KBS_Manuscript_R1.docx`, `KBS_Supplementary_R1.docx`, `KBS_Response_R1.docx`, `KBS_Highlights_R1.docx`, toate la commit-ul T10.6 (0732616). Textul a fost extras cu python-docx (paragrafe, celule de tabel); numaratorile sunt pe textul intreg (proza + tabele) daca nu se spune altfel. Script de evidenta: scratchpad `t107.py` (nu e in repo; cifrele de mai jos sunt reproductibile cu grep pe textul extras).

## (1) Tipografie (v8 = referinta)

| Element | v8 | R1 manuscris | R1 Supplementary | R1 Response | Observatie |
|---|---|---|---|---|---|
| ">=" (ASCII) | 17 | 17 | 0 | 0 | R1 pastreaza conventia v8 (aceleasi 17: 3.2 si tabelul pragurilor, v8 Table in 4.5 = R1 Table 7) |
| "≥" (U+2265) | 0 | 0 | 0 | 0 | conform |
| "<=" / "≤" | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | conform |
| "+/-" (ASCII) | 4 | 11 | 0 | 0 | conventia v8 (ASCII) pastrata in cele 7 aparitii noi (Table 4 randul "CV macro F1", Table A.1 coloanele de cautare / CV) |
| "±" (U+00B1) | 0 | 0 | 0 | 0 | conform |
| hyphen-minus inaintea unei cifre, in tabele | 2 | 51 | 27 | 0 | conventia v8 (hyphen-minus) pastrata; cifrele negative noi (Cliff's delta, MCC, intervale) folosesc toate hyphen-minus |
| U+2212 in tabele | 0 | 0 | 0 | 0 | conform |
| U+2212 in proza | 1 | 1 | 0 | 0 | aceeasi aparitie v8: "[−1, 1]" in fraza despre MCC (3.6), paragraf v8 neatins; nu e o diferenta R1 |
| "24-hour" | 6 | 8 | 0 | 3 | proza: conventia v8 "24-hour" pastrata in textul nou |
| "24h" | 1 | 1 | 0 | 0 | Table 1, randul 12 "Mean 24h heart rate (bpm)", identic cu v8 |
| "24-h" | 0 | 1 | 0 | 0 | **DIFERA de v8**: Table 10 (indicatori HealthKit pe clase), randul "Mean 24-h heart rate (bpm)"; v8 are "24h" in tabel (Table 1) si "24-hour" in proza; forma "24-h" nu apare in v8 |
| "24 h" | 0 | 0 | 0 | 0 | conform |
| em dash (U+2014) | 0 | 0 | 0 | 0 | conform (verificat si de check-urile permanente) |
| en dash (U+2013) | 0 | 0 | 0 | 0 | conform |

Unde R1 difera de conventia v8: un singur loc, "Mean 24-h heart rate (bpm)" in Table 10 (v8: "24h" in Table 1, "24-hour" in proza). Nimic aplicat.

## (2) Ordinea primei citari

Tabele, manuscris (prima citare in text, legendele si captions excluse): 1, A.1, 2, 3, 4, 5, S1, 6, S2, 7, 8, 9, 10, S3, 11, 12, 13, 14.
- Tables 1-14: crescatoare, fiecare citata cel putin o data (check permanent "Tables first cited in numeric order" PASS).
- Table A.1 (anexa): prima citare in 3.5 (paragraful despre hyperparameter search), inainte de Table 2; fiind tabel de anexa, numerotarea A.1 nu intra in secventa 1-14 (observatie, nu abatere).
- Captions in ordinea aparitiei: Table 1, 2, Figure 1, Table 3, 4, 5, 6, Figure 2, Table 7, 8, Figure 3, Table 9, 10, 11, Figure 4, Table 12, 13, Figure 5, 6, 7, Table 14, Table A.1.

Figuri, manuscris: 1, 2, 3, S1, 4, 5, 6, S3, 7 (plus Figure S2, citata prima data in legenda Figure 5, "seeds mode in Supplementary Figure S2", deci intre S1 si S3). Figures 1-7 crescatoare, fiecare citata (check permanent PASS).

Supplementary: Tables S1, S2, S3 citate in manuscris in aceasta ordine (4.3, 4.5, 4.7); Figures S1 (4.7), S2 (legenda Figure 5), S3 (4.12) in aceasta ordine. In fisierul Supplementary, captions in ordinea: Table S1 (a), Table S1 (b), Table S2 (a), (b), (c), Table S3, Figure S1, S2, S3 (sub-tabelele (a)-(c) poarta acelasi numar S, conform listei SUPPLEMENTARY din F4).

Concluzie (2): ordinea e crescatoare pe toate cele patru serii; fiecare element e citat cel putin o data.

## (3) Citari in text vs lista de referinte

Lista de referinte: 40 de intrari (6 noi in revizie, cu comentariu Word "verify in Zotero"). Check permanent "Every reference entry is cited in the text" PASS (0 necitate); check-ul T4.2 pe response letter PASS (9 citari asteptate, 4 referinte eliminate fata de v8: De La Torre, Ferreira, Gu & Qian, Liang, listate in General changes).

Verificare independenta (euristica (Autor, an) pe textul manuscrisului): 44 de chei distincte in text, 40 de intrari. Aparentele orfane sunt toate false pozitive ale euristicii, verificate manual:
- "cited, no entry": "February 2025", "July 2024", "June 2023", "March 2026", "September 2025" (date din 3.1 / 3.5, nu citari); "Griffin 2025" si "Rodriguez-Villegas 2024" (al doilea autor din "Susnjak & Griffin, 2025" si "Abdulsadig & Rodriguez-Villegas, 2024", ambele cu intrare); "Shapley 1953" (citare "Shapley, 1953/2016", intrarea e Shapley (2016)).
- "entry, not cited": Abdulsadig & Rodriguez-Villegas (2024) si Susnjak & Griffin (2025) citate cu "&"; Shaffer & Ginsberg (2017) citata in Table 1 (celula "Mean nocturnal HRV (RMSSD, ms; Shaffer & Ginsberg, 2017)"), nu in proza; Shapley (2016) citata ca "Shapley, 1953/2016".

Orfani in ambele sensuri: 0.

## (4) Numerotarea limitarilor

Manuscris, Discussion, paragraful "Limitations. Nine limitations bound the interpretation of these results.": markerii (1)-(9) apar o singura data fiecare, in ordine: (1) single-subject design si evaluarea temporala; (2) incertitudine statistica; (3) sistem open-loop, fara validare nutritionala / clinica; (4) clasele definite prin praguri pe scoruri compozite; (5) probabilitatile RF discrimineaza slab clasa Active; (6) doar capul ranking-ului SHAP e stabil; (7) doua generatii de dispozitive, ordinea standardizare / oversampling; (8) coeficientul caloric unic 25 kcal/kg/zi; (9) trei feature-uri Watch includ esantioane de la inel.

Trimiteri "Limitation N":
- Manuscris: una, in 3.12, "(Limitation 3, Section 5)", pentru offsetul caloric prescriptiv; consistenta cu (3). Forma: fara paranteze in jurul cifrei.
- Response letter: Limitation (1) x2 (Comment 10, R1.1 / Changes), Limitation (3) x1 (R2.a, Changes), Limitation (4) x1 (R1.5 Changes), Limitation (5) x1 (Comment 9 Changes), Limitation (6) x1 (Comment 6 Changes), Limitation (9) x3 (R1.5 Response si Changes, General changes bullet 8 ca "new Limitation 9"). Toate consistente cu continutul (1)-(9).
- Observatie de forma (neaplicat): manuscrisul scrie "Limitation 3", scrisoarea "Limitation (N)" si o data "new Limitation 9"; cifrele sunt consistente, forma nu e uniforma.

## (5) Keywords si Abstract

- Keywords v8: 7 (Personalized nutrition; Wearable devices; Explainable AI; SHAP; Physiological state classification; Random Forest; mHealth). R1 la T10.7: 7 (Random Forest inlocuit cu Decision support systems, decizie 19 oct). R1 la T11b.1 (3 nov 2026, decizia coautorului): 7 (Personalized nutrition; Wearable devices; Multimodal physiological data; Knowledge-based decision support; Explainable artificial intelligence (XAI); Decision support systems; SHAP: cele 5 din metadatele Editorial Manager ale submisiei originale, in ordinea EM, plus Decision support systems si SHAP; KEYWORDS_MODE = 'em7'; regenerat cu `_check_t11b_inventory.py --after`). Limita KBS 1-7 (F4_manuscript_changes.md, FORMAT KBS): toate conforme; nimic de semnalat (se semnaleaza doar peste 7).
- Abstract R1: 250 de cuvinte (split pe spatii; limita 250, check permanent PASS); 0 citari (nicio paranteza cu an), 0 trimiteri la sectiuni, 0 trimiteri la tabele / figuri.

## (6) Highlights

5 bullets (limita 3-5), lungimi cu spatii: 75; 83; 78; 85; 82 (limita 85 inclusiv; bullet-ul 4 e exact la limita, decizie T9.1). Fisier separat cu "Highlights" in nume; titlul v8 pe linia a doua. Neschimbate la T10 (document.xml byte-identic inainte / dupa T10.4; textul extras identic cu 9f44018).

## Rezumat

- Difera de v8: o singura forma tipografica, "24-h" in Table 10 (v8: "24h" in tabel / "24-hour" in proza). Restul conventiilor (">=", "+/-", hyphen-minus in tabele, "24-hour" in proza, fara em dash) sunt pastrate.
- Ordinea citarilor: crescatoare pe tabele, figuri, S-tabele, S-figuri; Table A.1 citata devreme (3.5), ca tabel de anexa.
- Orfani: 0 in ambele sensuri (40 de intrari, toate citate; nicio citare fara intrare).
- Limitari: (1)-(9) o data fiecare, in ordine; toate trimiterile consistente; forma "Limitation 3" (manuscris) vs "Limitation (N)" (scrisoare) neuniforma.
- Keywords: 7 / 7 (limita 1-7, conform; lista em7 de la T11b.1, 3 nov 2026); Abstract 250 de cuvinte, fara citari si fara trimiteri.
- Highlights: 5 bullets, 75 / 83 / 78 / 85 / 82 caractere.
