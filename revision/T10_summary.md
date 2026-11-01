# T10_summary.md - final pass Revision 1 (KBS), branch r1-final-pass

Data: 1 nov 2026. Baza: master 9f44018 (T9.1). Decizii Buiu (WhatsApp, 30 oct 2026): Q1 RF canonic ramane modelul raportat, titlul ramane cel submis, validarea nutritionala declarata obiectiv viitor (Reviewer 2): DA; Q8 GOAL_FIX (3.12 code-faithful): CONFIRMAT; Q9 T3.2b intra in Supplementary S3 ca o singura fraza, cu rezerve.

Regula de lucru: docx-urile nu se editeaza direct; sursele sunt F4_manuscript_changes.md, F4_response_letter.md si sirurile din assemble_manuscript.py (Supplementary); reasamblare dupa fiecare modificare.

## Tabel task / commit / rezultat

| Task | Commit | Rezultat |
|---|---|---|
| T10.0 Inventar | (acest commit) | PASS: toate cele 21 de siruri OLD gasite exact o data in locul asteptat; 0 aparitii in BASE_MANUSCRIPT_v8; 0 NEGASIT |

## T10.0 Inventar (fara modificari)

HEAD contine T9.1: Highlight 4 "Labels agree with rule-independent indicators: Active days are recovery, not exertion" este in F4_manuscript_changes.md (linia 307) si in KBS_Highlights_R1.docx (bullet 4, 85 caractere).

Confirmare aw_resp_rate (nicio cifra noua): `revision/audit/_check_f4_empirical_output.txt` linia 238: "aw_resp_rate <- respiratory_rate: days 952, days with >=1 Oura record 917 (0.963), days with NO non-Oura record 12, Oura share of records 0.022"; linia 44: "respiratory_rate: ... ALL: 0.022; pure-Oura records 837"; linia 35: "ALL lab: 0.963 (917/952)". CIFRE_REVIZIE.md, randul AUDIT "Zile etichetate (din 952 cu fisier) cu cel putin o inregistrare Oura" (linia 530), coloana Nota: "aw_resp_rate 1 inregistrare Oura pe zi (2.2% din inregistrari)". Deci: o inregistrare Oura pe zi pe 96.3% din zilele etichetate (917 / 952), 2.2% din inregistrari. Confirmat; se continua. (Observatie: randul din registru poarta "1 pe zi" si "2.2%", iar 96.3% apare in fisierul de audit, nu in registru; T10.1 nu introduce nicio cifra, deci nu se adauga rand.)

Sursa textului Supplementary: sirurile din `revision/scripts/assemble_manuscript.py`, functia `make_supplementary()` (paragraful S3 la linia 1320-1328); nu exista md separat.

Inventarul sirurilor OLD (grep exact; "docx" = textul extras din paragrafe, tabele, antete si subsoluri cu python-docx; v8 = BASE_MANUSCRIPT_v8.docx):

| Sir OLD | Fisier(e) | Aparitii | Stare |
|---|---|---|---|
| T10.1 3.10 "Validation rests on the agreement analysis, and only on the indicators whose sensor is not shared with the ring scores." | F4_manuscript_changes.md (3.10); Manuscript_R1, _marked | 1 / 1 / 1 | de modificat |
| T10.1 3.10 "HRV, SpO2 and respiratory rate are also independent of the ring's sensors, whereas" | F4_manuscript_changes.md (3.10); Manuscript_R1, _marked | 1 / 1 / 1 | de modificat |
| T10.1 4.7 "Among the indicators whose sensor is not shared with the ring scores, respiratory rate follows" | F4_manuscript_changes.md (4.7); Manuscript_R1, _marked | 1 / 1 / 1 | de modificat |
| T10.1 4.7 "including the respiratory rate measured by the watch alone, and describe" | F4_manuscript_changes.md (4.7); Manuscript_R1, _marked; F4_response_letter.md si Response_R1 (R1.5 Response reia fraza manuscrisului; acoperit de OLD R1.5b, acelasi NEW) | 1 / 1 / 1; 1 / 1 | de modificat |
| T10.1 S3 "so only HRV, SpO2 and respiratory rate are also independent of the ring's sensors." | assemble_manuscript.py linia 1322; Supplementary_R1 | 1 / 1 | de modificat |
| T10.1 R1.5 "so only HRV, SpO2 and respiratory rate are independent of the ring's sensors as well." | F4_response_letter.md (R1.5 Response, linia 49); Response_R1 | 1 / 1 | de modificat |
| T10.1 R1.5 "including the respiratory rate measured by the watch alone," | F4_response_letter.md linia 49; Response_R1 (si ca prefix al OLD 4.7 de mai sus in manuscris) | 1 / 1 | de modificat |
| T10.1 General changes "is restricted to HRV, SpO2 and respiratory rate." | F4_response_letter.md linia 128 (bullet provenance); Response_R1 | 1 / 1 | de modificat |
| T10.2 R1.5 "Supplementary Table S3 and Figure S1" | F4_response_letter.md linia 51 (Changes.); Response_R1 | 1 / 1 | de modificat (insertie dupa) |
| T10.3 "Two corrections to the original submission are disclosed below" | F4_response_letter.md linia 9; Response_R1 | 1 / 1 | de modificat |
| T10.3 "and we report both orderings, as the reviewer's concern implies we should." | F4_response_letter.md linia 41 (R1.4); Response_R1 | 1 / 1 | de modificat |
| T10.3 "so tuning did not change their standing; the reviewer's concern that defaults disadvantage them is addressed by the result." | F4_response_letter.md linia 41 (R1.4); Response_R1 | 1 / 1 | de modificat |
| T10.3 "We also report, as the reviewer's concern deserves, that on three" | F4_response_letter.md linia 107 (R2.b); Response_R1 | 1 / 1 | de modificat |
| T10.3 R1.8 "Random Forest has the highest point estimate on balanced accuracy, G-mean and PR-AUC, XGBoost on ROC-AUC and accuracy." | F4_response_letter.md linia 73; Response_R1 | 1 / 1 | de modificat |
| T10.4 (a) "Two results are robust under retraining and across label thresholds. The first is" | F4_manuscript_changes.md (Principal findings, linia 251); Manuscript_R1, _marked | 1 / 1 / 1 | de modificat (fara dovada, vezi T10.4) |
| T10.4 (b) "and mean 24-hour heart rate is the stable head of the SHAP ranking." | F4_manuscript_changes.md (CONCLUSIONS); Manuscript_R1, _marked | 1 / 1 / 1 | de modificat |
| T10.4 (c) "(a 4.4-fold gain in F1 for the minority Active class)" | F4_manuscript_changes.md (Contributiile, bullet 1); Manuscript_R1, _marked | 1 / 1 / 1 | de modificat |
| T10.4 (c) "(a 4.4-fold gain in Active-class F1)" | F4_manuscript_changes.md (CONCLUSIONS); Manuscript_R1, _marked | 1 / 1 / 1 | de modificat |
| T10.4 (d) "Combining both data sources raises Active-class F1 4.4-fold," | F4_manuscript_changes.md (ABSTRACT); Manuscript_R1, _marked | 1 / 1 / 1 | de modificat |
| T10.4 (d) "and an expanding-window temporal evaluation." | F4_manuscript_changes.md (ABSTRACT); Manuscript_R1, _marked; F4_response_letter.md linia 9 si Response_R1 (fraza introductiva a scrisorii, NU abstractul; ramane neschimbata, T10.4(d) vizeaza doar Abstractul) | 1 / 1 / 1; 1 / 1 | de modificat (Abstract) |
| T10.4 (d) "for single-subject wearable data, reported in full." | F4_manuscript_changes.md (ABSTRACT); Manuscript_R1, _marked | 1 / 1 / 1 | de modificat |

Toate: 0 aparitii in BASE_MANUSCRIPT_v8.docx (niciun OLD nu e text v8). Niciun NEGASIT. Highlights_R1: 0 aparitii ale oricarui OLD.

Verificarea T10.3 R1.8 fata de manuscris (4.2 si Table 4, KBS_Manuscript_R1.docx): 4.2 spune "Random Forest has the highest point estimate on macro F1, balanced accuracy, G-mean, PR-AUC and QWK; XGBoost on accuracy, MCC, Cohen's kappa and ROC-AUC"; Table 4: RF macro F1 0.506, balanced accuracy 0.523, G-mean 0.492, PR-AUC 0.498, QWK 0.255 (maxime pe coloana); XGBoost accuracy 68.8%, MCC 0.150, kappa 0.146, ROC-AUC 0.589 (maxime). NEW R1.8 coincide; se continua.

Baseline la HEAD (inainte de orice modificare): `_check_t41_outputs.py` 74 / 74 PASS; `_check_t8_parity.py --manuscript --response --supp` 19 / 26 (cele 7 FAIL sunt cele cunoscute din audit-apply si T9.1: textul nu mai e egal cu f8/*.txt; checker-ul compara cu textele F8 din 25 oct). Paritatea numerelor pentru T10 se verifica per paragraf schimbat cu un script dedicat (textul extras la HEAD, salvat inainte de T10.1, fata de textul reasamblat).

Eticheta "Internal" (T10.5, constatare la inventar): nu e paragraf in corp; este un text box ancorat in subsol (w:drawing cu wp:docPr descr="Internal", aclsh:classification classificationOutcomeType="ftr") in word/footer1-3.xml, de doua ori per subsol (mc:Choice + mc:Fallback), plus trei proprietati in docProps/custom.xml: ClassificationContentMarkingFooterShapeIds, ClassificationContentMarkingFooterFontProps, ClassificationContentMarkingFooterText = "Internal" (nu chei MSIP_Label_*). Prezent identic in BASE_MANUSCRIPT_v8.docx si mostenit de toate cele 5 docx generate (toate pornesc din docx.Document(BASE)).
