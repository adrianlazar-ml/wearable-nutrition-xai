# F5 - checklist de submisie KBS (Revision 1)

Creat: 2026-10-20, din lista de task-uri F5 primita de la Adrian (textul separat al checklist-ului nu era in repo; fisierul reproduce cerintele asa cum au fost formulate in sesiunea din 20 oct 2026). Referinta: Guide for Authors, Knowledge-Based Systems (KBS_journal.pdf, pp. 20-37). Nicio cifra din manuscris nu se modifica in F5.

Stare la deschidere: F0-F4 inchise (CIFRE_REVIZIE.md, assembly_plan.md), docx-urile F4 asamblate pe baza v8 (confirmata 20 oct ca versiunea submisa): KBS_Manuscript_R1.docx, KBS_Manuscript_R1_marked.docx, KBS_Supplementary_R1.docx, KBS_Response_R1.docx, KBS_Highlights_R1.docx. Standard de limba: engleza americana (v8 verificat 20 oct).

| Task | Cerinta (KBS) | Livrabil | Criteriu PASS | Status |
|---|---|---|---|---|
| T4.1-F | Referinte complete, proceedings nu arXiv (p. 32) | revision/references_new_verified.md | 6/6 cu toate campurile si DOI sau URL stabil; 0 arXiv; campurile negasite marcate NEGASIT | |
| T5.1 | Referinte importabile in Zotero | revision/references_new.bib | parsare fara erori, 6 intrari, chei unice | |
| T5.2 | Figuri ca fisiere separate, vector + 300 dpi (p. 20) | revision/figures/Figure_N.{pdf,png}, FIGURES_MANIFEST.md | PDF + PNG per figura; PNG >= 2244 px (pagina) sau >= 1063 px (coloana); fonturi embedded (pdf.fonttype 42); continut neschimbat | |
| T5.3 | Numerotare anexa "Table A.1" (p. 28) | docx + md actualizate | 0 "Table A1" ramase; N inainte == N dupa | |
| T5.3b | Ortografie US in textul nou | revision/F5_format_report.md (lista matches, inlocuiri) | 0 forme UK in corp; exceptii listate (titluri de referinte, citate reviewer, nume de fisiere / variabile) | |
| T5.4 | Verificari de format, doar raport | revision/F5_format_report.md | 5 puncte cu cifre: cross-referinte, Supplementary, Acknowledgements, declaratii, comentarii / field codes | |
| T5.5 | Verificare finala a cifrelor | t41_number_check_R1.md rerulat | acelasi rezultat ca inainte de T5.3 / T5.3b (0 cifre neexplicate) | |

Reguli: o singura cifra nu se schimba; nu se inventeaza date bibliografice; comentariile "verify in Zotero" raman pana la importul in Zotero; inlocuirile se fac pe liste de matches afisate inainte; un commit per task cu prefixul task-ului.

Ramase dupa F5 (din plan, sectiunea 5 si 8): review Buiu (minimum o saptamana), regenerarea referintelor din Zotero (APA 7), repo public cu DOI (anonimizare profil, demo_end_to_end_v2.py, fisierele nestaged), submisie cu cel putin 5 zile inainte de 1 decembrie 2026.
