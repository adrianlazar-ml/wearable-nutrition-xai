# T10_summary.md - final pass Revision 1 (KBS), branch r1-final-pass

Data: 1 nov 2026. Baza: master 9f44018 (T9.1). Decizii Buiu (WhatsApp, vineri 30 oct 2026): Q1 RF canonic ramane modelul raportat, titlul ramane cel submis, validarea nutritionala declarata obiectiv viitor (Reviewer 2): DA; Q8 GOAL_FIX (3.12 code-faithful: goal-ul e camp de profil si nu modifica tinta): CONFIRMAT; Q9 T3.2b intra in Supplementary S3 ca O SINGURA fraza, cu rezerve.

Regula de lucru: docx-urile nu se editeaza direct; sursele sunt F4_manuscript_changes.md, F4_response_letter.md si sirurile din assemble_manuscript.py (Supplementary); reasamblare dupa fiecare modificare. Branch r1-final-pass, un commit per task, NU s-a facut merge in master. CIFRE_CANONICE.md: sha256 a80e9d6c5986c98b... inainte si dupa (identic). Nicio cifra noua in afara celor 10 din T10.2 (toate din randurile T3.2b ale registrului, 2026-10-26).

## Tabel task / commit / rezultat

| Task | Commit | Rezultat |
|---|---|---|
| T10.0 Inventar | b22de8a | PASS: 21 siruri OLD, fiecare gasit exact o data in locul asteptat; 0 aparitii in BASE_MANUSCRIPT_v8; 0 NEGASIT; aw_resp_rate confirmat (917 / 952 = 96.3% zile cu o inregistrare Oura, 2.2% din inregistrari) |
| T10.1 Respiratory rate | bd6f9af | PASS: 0 aparitii ale celor 4 formulari retrase in cele 3 docx; numerele identice pe toate cele 7 paragrafe schimbate (3.10, 4.7, S3, R1.5, General changes, + copia marcata); Highlights identice |
| T10.2 T3.2b in S3 | 49027a9 | PASS: fraza de 74 de cuvinte inserata exact (hyphen-minus ASCII, 0 caractere non-ASCII); cele 10 numere FOUND in CIFRE_REVIZIE (S1: 387 / 321 / 0 NEGASIT); _check_t32b_outputs ALL PASS; 7 formulari "doar registru" din CIFRE_REVIZIE devenite "Supplementary S3, o fraza (decizie Buiu, 30 oct 2026); nu intra in manuscrisul principal" (4 din cele trei formulari listate in brief + 3 "doar registru" din randul T3.2b, Artefacte si Status Faza 3); R1.5 Changes completat; singura exceptie de paritate: paragraful S3 (10 numere noi, declarate) |
| T10.3 Response letter | 2078ed8 | PASS: _check_t42_response ALL PASS; style_check Response 5 / 5, medie 27.0 cuvinte / fraza (<= 27, la limita), 0 termeni interzisi; "concern deserves" / "concern implies" adaugate in STYLE_GUIDE sectiunea 2 si in style_check.py (FORBIDDEN), dovada: scrisoarea de dinainte de T10.3 da FAIL 'concern deserves': 1, 'concern implies': 1; R1.8 NEW verificat fata de 4.2 si Table 4 (coincide) |
| T10.4 Prudenta | fd0a2a3 | PASS: Abstract 250 cuvinte (<= 250, fara taierea lui "alone"); check 74 (provenance wording) PASS; Highlights byte-identice (word/document.xml sha256 80771c7e41702d78 inainte si dupa); numere identice pe cele 4 paragrafe schimbate (4.4 ramane); (a) aplicat, fara dovada (vezi mai jos) |
| T10.5 Eticheta "Internal" | a738321 | PASS: in cele 5 docx 0 'Internal' in subsoluri / antete, 0 proprietati de clasificare in docProps (partea custom.xml si docMetadata/LabelInfo.xml nu mai sunt scrise); textul extras in rest identic (singura diferenta: cele doua texte 'Internal' din footer1 si footer3); check nou 75 in _check_t41_outputs.py |
| T10.6 Documente de decizie | 0732616 | PASS: TEXT_DECISIONS sectiunea T10, STYLE_GUIDE sectiunea 5 (3.12 GOAL_FIX confirmat), AUDIT_DECISIONS (GOAL_FIX si P5 confirmate, pasul 1 facut partial), F4 header (versiunea submisa confirmata 20 oct, RF canonic confirmat 30 oct, Limitations (9), Abstract 250), lista SUPPLEMENTARY cu fraza T3.2b; reasamblarea din surse reproduce cele 5 docx fara diff pe textul extras (toate partile word/*.xml) |
| T10.7 Conformitate KBS / v8 | 9041cf3 | PASS: raport scris cu cifre in revision/F10_kbs_conformity.md, nimic aplicat |
| T10.8 Verificare finala | (acest commit) | PASS: toate check-urile de mai jos; NEGASIT 0; diferentele de text fata de 9f44018 explicate integral de T10.1-T10.5 |

## T10.0 Inventar (fara modificari)

HEAD contine T9.1: Highlight 4 "Labels agree with rule-independent indicators: Active days are recovery, not exertion" este in F4_manuscript_changes.md (linia 307) si in KBS_Highlights_R1.docx (bullet 4, 85 caractere).

Confirmare aw_resp_rate (nicio cifra noua): `revision/audit/_check_f4_empirical_output.txt` linia 238: "aw_resp_rate <- respiratory_rate: days 952, days with >=1 Oura record 917 (0.963), days with NO non-Oura record 12, Oura share of records 0.022"; linia 44: "respiratory_rate: ... ALL: 0.022; pure-Oura records 837"; linia 35: "ALL lab: 0.963 (917/952)". CIFRE_REVIZIE.md, randul AUDIT "Zile etichetate (din 952 cu fisier) cu cel putin o inregistrare Oura" (linia 530), coloana Nota: "aw_resp_rate 1 inregistrare Oura pe zi (2.2% din inregistrari)". Deci: o inregistrare Oura pe zi pe 96.3% din zilele etichetate (917 / 952), 2.2% din inregistrari. Confirmat; s-a continuat. (Observatie: randul din registru poarta "1 pe zi" si "2.2%", iar 96.3% apare in fisierul de audit, nu in registru; T10.1 nu introduce nicio cifra, deci nu s-a adaugat rand.)

Sursa textului Supplementary: sirurile din `revision/scripts/assemble_manuscript.py`, functia `make_supplementary()` (paragraful S3); nu exista md separat.

Inventarul sirurilor OLD (grep exact; "docx" = textul extras din paragrafe, tabele, antete si subsoluri cu python-docx; v8 = BASE_MANUSCRIPT_v8.docx):

| Sir OLD | Fisier(e) | Aparitii | Stare |
|---|---|---|---|
| T10.1 3.10 "Validation rests on the agreement analysis, and only on the indicators whose sensor is not shared with the ring scores." | F4_manuscript_changes.md (3.10); Manuscript_R1, _marked | 1 / 1 / 1 | modificat |
| T10.1 3.10 "HRV, SpO2 and respiratory rate are also independent of the ring's sensors, whereas" | F4_manuscript_changes.md (3.10); Manuscript_R1, _marked | 1 / 1 / 1 | modificat |
| T10.1 4.7 "Among the indicators whose sensor is not shared with the ring scores, respiratory rate follows" | F4_manuscript_changes.md (4.7); Manuscript_R1, _marked | 1 / 1 / 1 | modificat |
| T10.1 4.7 "including the respiratory rate measured by the watch alone, and describe" | F4_manuscript_changes.md (4.7); Manuscript_R1, _marked; F4_response_letter.md si Response_R1 (R1.5 Response reia fraza manuscrisului; acoperit de OLD R1.5b, acelasi NEW) | 1 / 1 / 1; 1 / 1 | modificat |
| T10.1 S3 "so only HRV, SpO2 and respiratory rate are also independent of the ring's sensors." | assemble_manuscript.py (make_supplementary); Supplementary_R1 | 1 / 1 | modificat |
| T10.1 R1.5 "so only HRV, SpO2 and respiratory rate are independent of the ring's sensors as well." | F4_response_letter.md (R1.5 Response, linia 49); Response_R1 | 1 / 1 | modificat |
| T10.1 R1.5 "including the respiratory rate measured by the watch alone," | F4_response_letter.md linia 49; Response_R1 (si ca prefix al OLD 4.7 de mai sus in manuscris) | 1 / 1 | modificat |
| T10.1 General changes "is restricted to HRV, SpO2 and respiratory rate." | F4_response_letter.md linia 128 (bullet provenance); Response_R1 | 1 / 1 | modificat |
| T10.2 R1.5 "Supplementary Table S3 and Figure S1" | F4_response_letter.md linia 51 (Changes.); Response_R1 | 1 / 1 | modificat (insertie dupa) |
| T10.3 "Two corrections to the original submission are disclosed below" | F4_response_letter.md linia 9; Response_R1 | 1 / 1 | modificat |
| T10.3 "and we report both orderings, as the reviewer's concern implies we should." | F4_response_letter.md linia 41 (R1.4); Response_R1 | 1 / 1 | modificat |
| T10.3 "so tuning did not change their standing; the reviewer's concern that defaults disadvantage them is addressed by the result." | F4_response_letter.md linia 41 (R1.4); Response_R1 | 1 / 1 | modificat |
| T10.3 "We also report, as the reviewer's concern deserves, that on three" | F4_response_letter.md linia 107 (R2.b); Response_R1 | 1 / 1 | modificat |
| T10.3 R1.8 "Random Forest has the highest point estimate on balanced accuracy, G-mean and PR-AUC, XGBoost on ROC-AUC and accuracy." | F4_response_letter.md linia 73; Response_R1 | 1 / 1 | modificat |
| T10.4 (a) "Two results are robust under retraining and across label thresholds. The first is" | F4_manuscript_changes.md (Principal findings); Manuscript_R1, _marked | 1 / 1 / 1 | modificat (fara dovada, vezi T10.4) |
| T10.4 (b) "and mean 24-hour heart rate is the stable head of the SHAP ranking." | F4_manuscript_changes.md (CONCLUSIONS); Manuscript_R1, _marked | 1 / 1 / 1 | modificat |
| T10.4 (c) "(a 4.4-fold gain in F1 for the minority Active class)" | F4_manuscript_changes.md (Contributiile, bullet 1); Manuscript_R1, _marked | 1 / 1 / 1 | modificat |
| T10.4 (c) "(a 4.4-fold gain in Active-class F1)" | F4_manuscript_changes.md (CONCLUSIONS); Manuscript_R1, _marked | 1 / 1 / 1 | modificat |
| T10.4 (d) "Combining both data sources raises Active-class F1 4.4-fold," | F4_manuscript_changes.md (ABSTRACT); Manuscript_R1, _marked | 1 / 1 / 1 | modificat |
| T10.4 (d) "and an expanding-window temporal evaluation." | F4_manuscript_changes.md (ABSTRACT); Manuscript_R1, _marked; F4_response_letter.md linia 9 si Response_R1 (fraza introductiva a scrisorii, NU abstractul; lasata neschimbata, T10.4(d) vizeaza doar Abstractul) | 1 / 1 / 1; 1 / 1 | modificat (Abstract) |
| T10.4 (d) "for single-subject wearable data, reported in full." | F4_manuscript_changes.md (ABSTRACT); Manuscript_R1, _marked | 1 / 1 / 1 | modificat |

Toate: 0 aparitii in BASE_MANUSCRIPT_v8.docx (niciun OLD nu e text v8). Niciun NEGASIT. Highlights_R1: 0 aparitii ale oricarui OLD.

Verificarea T10.3 R1.8 fata de manuscris (4.2 si Table 4, KBS_Manuscript_R1.docx): 4.2 spune "Random Forest has the highest point estimate on macro F1, balanced accuracy, G-mean, PR-AUC and QWK; XGBoost on accuracy, MCC, Cohen's kappa and ROC-AUC"; Table 4: RF macro F1 0.506, balanced accuracy 0.523, G-mean 0.492, PR-AUC 0.498, QWK 0.255 (maxime pe coloana); XGBoost accuracy 68.8%, MCC 0.150, kappa 0.146, ROC-AUC 0.589 (maxime). NEW R1.8 coincide.

Baseline la HEAD 9f44018 (inainte de orice modificare): `_check_t41_outputs.py` 74 / 74 PASS; `_check_t8_parity.py --manuscript --response --supp` 19 / 26 (cele 7 FAIL sunt cele cunoscute din audit-apply si T9.1: textul nu mai e egal cu f8/*.txt din 25 oct; checker-ul compara cu textele F8). Paritatea numerelor pentru T10 s-a verificat per paragraf schimbat cu un script dedicat (textul extras la 9f44018 fata de textul reasamblat; rezultatele in tabelul de mai sus si in sectiunea T10.8).

## T10.2: fraza S3 si numarul de cuvinte

Fraza inserata (exact, dupa fraza NEW din T10.1 in paragraful introductiv S3): "On the 320 labeled days with at least one HealthKit heart-rate sample not written by the ring (median one such sample per day; a sensitivity check on a non-random subset, not a validation on an independent sensor), the Active versus Rest contrast persists on those samples alone (Cliff's delta -0.703 [-0.885, -0.472], against -0.835 [-0.983, -0.602] on all samples of the same days), whereas the Active versus Normal contrast does not (-0.135 [-0.373, +0.112])."

Numar de cuvinte: 74 (split pe spatii), peste tinta de sub 70. style_check.py in modul "other" (Supplementary) nu are prag de lungime a frazei (doar 'random forest' = 0 si termeni interzisi = 0; 2 / 2 PASS), deci nu a fost caz de STOP; fraza NU a fost impartita (textul e cel din brief, neschimbat). Toate semnele minus sunt hyphen-minus ASCII; 0 caractere non-ASCII in fraza.

## T10.4 (a): dovada de rerulare a ablatiei pe dispozitiv

Cautare in CIFRE_REVIZIE.md (Oura-only / Watch-only / ring-only / both / ablatie pe dispozitive): singurele randuri sunt T1.5 "Identitate T1.5.0b: ablatia canonica pe dispozitive prin fit_eval, Oura-only (11) / Watch-only (8): F1 macro, F1 Active | 0.428134 / 0.285714; 0.324330 / 0.100000 | group_ablation.py | 42 | 2026-10-19" (linia 101) si nota T1.5 "procedura in spatiu redus identica cu ablatia canonica (dovedita de T1.5.0b)" (linia 401): o singura rulare de identitate pe seed 42, pe matricea canonica. T1.4 (100 de paduri pe bootstrap), T3.1 (grila de praguri) si T3.3 (expanding window) ruleaza doar modelul complet cu 19 feature-uri; T1.5 LOGO / KOGI sunt grupuri fiziologice, nu dispozitive. Rezultat: **fara dovada**; s-a aplicat "Two results stand out. The first is".

## T10.4 (d): Abstract

250 de cuvinte, regula de numarare: `len(abstract.split())` (split pe spatii), aceeasi cu check-ul permanent "Abstract under 250 words" din `_check_t41_outputs.py` (linia 117) si cu F4 (249 la T7b.1). D1 (+5 cuvinte) compensata prin "and an expanding-window evaluation." (-1) si "for single-subject wearable data." (-3): 249 + 5 - 4 = 250 <= 250, deci "alone" a ramas (nu a fost nevoie de taiere). "SHAP (Shapley additive explanations)" si "not clinical readiness" pastrate (verificat in docx). Observatie: `_check_t8_parity.py` are propriul prag istoric "Abstract <= 249 words" (valoarea R1 din 25 oct) si da acum FAIL cu 250; nu a fost slabit, se raporteaza (limita KBS e 250, check-ul permanent PASS).

## T10.5: ce era in BASE_MANUSCRIPT_v8.docx

Eticheta nu e paragraf in corp; este marcajul de continut al unei etichete de sensibilitate Microsoft Purview: (i) un text box ancorat in subsol (w:drawing cu wp:docPr descr="Internal" si aclsh:classification classificationOutcomeType="ftr") in word/footer1.xml, footer2.xml si footer3.xml, textul "Internal" de doua ori per subsol (mc:Choice + mc:Fallback in acelasi w:r) = cele "doua paragrafe Internal" vazute la extragerea textului cu un tool care citeste text box-urile; (ii) docProps/custom.xml cu trei proprietati ClassificationContentMarkingFooterShapeIds / ...FooterFontProps / ...FooterText = "Internal" (nu chei MSIP_Label_*); (iii) docMetadata/LabelInfo.xml (clbl:labelList, label id {7e753c2c-...}, method Standard). v8: 10 aparitii "Internal" in subsoluri, 3 proprietati, LabelInfo prezent; identic in BASE_MANUSCRIPT.docx (v2) prin aceeasi origine; toate cele 5 docx generate le mosteneau (toate pornesc din docx.Document(BASE)).

Solutie (assemble_manuscript.py, `strip_classification_marking(doc)`, apelata pe manuscris dupa incarcarea bazei si in `empty_doc_from_base()` pentru Highlights, Supplementary si Response; copia marcata porneste din manuscrisul deja curatat): run-ul care contine desenul e eliminat din fiecare subsol (2 run-uri per docx: footer1 si footer3; footer2 nu are run cu aclsh:classification in textul w:t), iar relatiile de pachet custom-properties si classificationlabels sunt eliminate, deci cele doua parti nu mai sunt scrise (nici in [Content_Types].xml). Fisierele de baza nu se modifica (sha256 verificat de script si de check-urile 58-60). Rezultat: in cele 5 docx 0 'Internal' in oricare parte XML, 0 proprietati de clasificare, 0 LabelInfo; textul w:t al tuturor partilor word/*.xml identic cu cel dinainte, cu exceptia celor doua texte 'Internal' din footer1 / footer3. Check permanent nou: 75 "T10.5: 0 sensitivity-label marking in the 5 docx".

## T10.7 pe scurt (raportul complet: revision/F10_kbs_conformity.md)

- Difera de v8: o singura forma tipografica, "Mean 24-h heart rate (bpm)" in Table 10 (v8: "24h" in Table 1, "24-hour" in proza). ">=" (17 / 17), "+/-" (4 / 11, ASCII), hyphen-minus in tabele (2 / 51, 0 U+2212), "24-hour" in proza, 0 em dash: conventia v8 pastrata.
- Ordinea primei citari: Tables 1-14 crescatoare (A.1 citata in 3.5, tabel de anexa), Figures 1-7 crescatoare, Tables S1-S3 si Figures S1-S3 crescatoare (S2 in legenda Figure 5); fiecare element citat cel putin o data.
- Orfani: 0 in ambele sensuri (40 de intrari, toate citate; aparentele orfane ale euristicii sunt date si al doilea autor, verificate manual).
- Limitari: (1)-(9) o data fiecare, in ordine; trimiterile din manuscris (Limitation 3 in 3.12) si din scrisoare ((1), (3), (4), (5), (6), (9)) consistente; forma "Limitation 3" vs "Limitation (N)" neuniforma (neaplicat).
- Keywords: 7 in v8 si 7 in R1 (limita KBS 1-7; nimic de semnalat). Abstract: 250 de cuvinte, fara citari, fara trimiteri la sectiuni / tabele / figuri.
- Highlights: 5 bullets, 75 / 83 / 78 / 85 / 82 caractere.

## T10.8 Verificare finala (dupa reasamblarea celor 5 docx din surse)

| Check | Rezultat |
|---|---|
| _check_t41_outputs.py | 75 / 75 PASS (74 + check-ul nou T10.5) |
| _check_t41_numbers.py R1 (manuscris) | 1581 numere, FOUND 1099, NEGASIT 0 (+ 10 explicate, neschimbat fata de 9f44018) |
| _check_t41_numbers.py S1 (Supplementary) | 387 numere, FOUND 321, NEGASIT 0 (377 / 311 la 9f44018; +10 = cele 10 cifre T3.2b) |
| _check_table_rounding.py | 402 celule, 0 nepotriviri |
| _check_t32b_outputs.py | ALL PASS (rulat, apoi results/t32b_run_meta.json restaurat din git: checker-ul rescrie run_date / elapsed_seconds) |
| _check_t42_response.py | ALL PASS |
| style_check.py manuscris / Supplementary / Response | 8 / 8 (medie 26.4 cuvinte / fraza) ; 2 / 2 ; 5 / 5 (medie 27.0, prag <= 27, 0 termeni interzisi inclusiv cei doi noi) |
| _check_t91_wording.py revision/manuscript | PASS, 0 hits (check 74 PASS) |
| _check_t8_parity.py --manuscript --response --supp | 18 / 26 (19 / 26 la 9f44018). Cele 7 FAIL preexistente (text != f8/*.txt din 25 oct, 3.12 GOAL_FIX, numere audit in scrisoare / Supplementary, capitalizare Highlights) raman; FAIL nou: "Abstract <= 249 words" (250; prag istoric al checker-ului T8, nu limita KBS; neslabit, raportat). "Supplementary: number set unchanged" include acum si cele 10 cifre T3.2b (exceptia declarata). Paritatea per paragraf schimbat (script dedicat, 9f44018 vs final): manuscris 6 paragrafe, toate IDENTICAL; Response 7 paragrafe, 5 IDENTICAL, R1.5 Changes +"3" din eticheta "Section S3", R1.8 +"1" din "macro F1" (etichete, nu cifre); Supplementary 1 paragraf cu cele 10 cifre noi |
| pytest revision/scripts/ -q | 111 passed in 64.82 s, exit 0 |
| Comentarii Word | 9 in manuscris (6 "verify in Zotero" + 3 "T6.2 draft"), 1 in Response ("T7 draft"), 0 in rest: neschimbat |
| CIFRE_CANONICE.md | sha256 a80e9d6c5986c98b... identic inainte / dupa |

Diferentele de text fata de 9f44018 (textul extras din paragrafe si tabele; toate partile word/*.xml comparate):
- KBS_Manuscript_R1.docx si copia marcata, 6 paragrafe: Abstract (T10.4 d), contributia 1 (T10.4 c), 3.10 (T10.1), 4.7 paragraful "Agreement with label-independent indicators" (T10.1), Principal findings (T10.4 a), Conclusions (T10.4 b, c). Tabele, legende, Highlights, title page, declaratii, Table 1, comentarii: neschimbate.
- KBS_Response_R1.docx, 7 paragrafe: introducerea (T10.3), R1.4 Response (T10.3, doua formulari), R1.5 Response (T10.1), R1.5 Changes (T10.2), R1.8 Response (T10.3), R2.b Response (T10.3), General changes bullet-ul de provenienta (T10.1).
- KBS_Supplementary_R1.docx, 1 paragraf: introducerea S3 (T10.1 + fraza T10.2).
- KBS_Highlights_R1.docx: 0 paragrafe schimbate.
- Toate cele 5 docx: subsolurile footer1 / footer3 fara textul 'Internal', fara docProps/custom.xml si docMetadata/LabelInfo.xml (T10.5).
Nicio alta diferenta.

Neatinse (regulile din brief): title page, CRediT / competing interest / Funding / GenAI / Data availability, comentariile Word, Highlights, Table 1, paragrafele identice cu v8 in afara celor listate; fara em dash; engleza americana; LF.

Ramas deschis (nu face parte din T10): Q2-Q7 pentru Buiu; forma "Limitation 3" vs "Limitation (N)" si "24-h" din Table 10 (raportate in F10, neaplicate); pragul istoric 249 din _check_t8_parity.py; merge-ul r1-final-pass in master (decizie Adrian).

## T10.9 (1 nov 2026): trei corectii din review-ul T10_summary / F10 (commit-uri a07b4a4, e9e7dcc, 663729a + summary)

| Subtask | Commit | Rezultat |
|---|---|---|
| T10.9a Abstract 249 | a07b4a4 | PASS: OLD gasit o data (F4 ABSTRACT + manuscris + copia marcata; 0 in Highlights si Response); Abstract 249 cuvinte (split pe spatii); set de numere identic (958, 33 prezente, nimic altceva schimbat); _check_t8_parity 19 / 26 (18 / 26 la T10.8, regula "Abstract <= 249" trece, restul neschimbat); check 74 PASS |
| T10.9b "24-h" -> "24-hour" | e9e7dcc | PASS: inventar "24-h" inainte: o singura aparitie in cele 3 docx, eticheta de rand din Table 10 (F4 4.7, text de revizie, nu v8); dupa: 0 in manuscris, Supplementary, Response; _check_table_rounding 402 / 0 dupa redenumirea cheii de rand "Mean 24-h heart rate" -> "Mean 24-hour heart rate" in _check_table_rounding.py (cheia leaga eticheta de aw_hr_mean; aceleasi 7 celule verificate, nu e o slabire; fara redenumire scriptul se oprea cu "STOP: row 'Mean 24-h heart rate' matched 0 rows") |
| T10.9c trimiteri la limitari | 663729a | PASS: 0 trimiteri cu paranteze in afara enumerarii; lista N inainte == dupa |
| T10.9d verificare finala | (acest commit) | PASS: toate check-urile de mai jos |

Inventar T10.9c (manuscris + Response, enumerarea din Discussion exclusa):
- Manuscris: 0 trimiteri cu paranteze inainte si dupa; singura trimitere e "Limitation 3, Section 5" (3.12), neschimbata; enumerarea (1)-(9) din paragraful "Limitations" intacta.
- Response, inainte (8): "Limitation (1) rewritten" (R1.1 Changes); "Table 1 and Limitation (9)." (R1.5 Response); "Limitation (4); new Limitation (9)" (R1.5 Changes); "Figure 6 legend; Limitation (6)." (Comment 6 Changes); "Figure S3; Limitation (5)." (Comment 9 Changes); "Temporal robustness"; Limitation (1); Abstract" (Comment 10 Changes); "Changes. Limitation (3) rewritten with figures" (R2.a Changes). Dupa: aceleasi 8 locuri cu "Limitation N"; lista N inainte [1, 9, 4, 9, 6, 5, 1, 3] (+ "new Limitation 9" din General changes, deja fara paranteze) == dupa. Neatinse: "### Limitation (1), rescrisa" si "Limitations (9)" din titlurile de bloc ale F4_manuscript_changes.md (adnotari editoriale, nu text de manuscris).
- TEXT_DECISIONS.md: linia T10.9c adaugata la finalul sectiunii T10.

Verificare finala T10.9d (dupa reasamblarea celor 5 docx): _check_t41_outputs 75 / 75; t41 R1 1581 / 1099 / 0 si S1 387 / 321 / 0 (neschimbate); _check_table_rounding 402 / 0; _check_t32b_outputs ALL PASS (run_meta restaurat); _check_t42_response ALL PASS; style_check 8 / 8 (26.4), 2 / 2, 5 / 5 (27.0); _check_t91_wording PASS 0 hits; _check_t8_parity 19 / 26 (+1 fata de T10.8: Abstract <= 249); pytest 111 passed; comentarii Word 9 + 1; CIFRE_CANONICE.md sha256 a80e9d6c5986c98b... identic.

Diferente de text fata de ec85606 (toate partile word/*.xml comparate; subsolurile neschimbate): manuscris si copia marcata 2 paragrafe (Abstract, T10.9a; randul "Mean 24-hour heart rate (bpm)" din Table 10, T10.9b); Response 7 paragrafe (cele 8 trimiteri la limitari, T10.9c; R1.5 Changes contine doua); Supplementary si Highlights 0. Numerele identice pe fiecare paragraf schimbat. Nicio alta diferenta. Adnotarea din F4 (titlul blocului ABSTRACT) actualizata la 249 in commit-ul de summary.
