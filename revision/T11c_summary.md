# T11c_summary.md - scoaterea celor 6 comentarii "verify in Zotero" (branch t11c-zotero-comments)

Data: 3 nov 2026. Baza: master 8fad285 (T11b). Branch t11c-zotero-comments, un commit per task, merge fast-forward in master la final. Motiv: BASE_MANUSCRIPT_v8.docx nu contine campuri Zotero (verificat in sesiune pe word/document.xml: 0 ZOTERO_ITEM, 0 ZOTERO_BIBL, 0 CSL_CITATION, 0 fldChar, 0 fldSimple; 34 de intrari de referinta ca text static), deci lista de referinte este text static si cele 6 intrari noi (forma APA 7 verificata manual la F5, 6 / 6, references_new_verified.md) nu au nevoie de un import in Zotero inainte de submisie; comentariile erau placeholder-e pentru acel import (importul ramane optional, pentru consistenta de format). Nicio cifra noua; textul livrabilelor neschimbat (doar comentariile Word scoase); CIFRE_CANONICE.md sha256 a80e9d6c5986c98b... identic; CIFRE_REVIZIE.md neschimbat (git diff fata de 680cc65 gol).

## Tabel task / commit / rezultat

| Task | Commit | Rezultat |
|---|---|---|
| T11c.1 Comentarii | be81eda | PASS: `doc.add_comment(... REF_COMMENT ...)` nu se mai apeleaza (textul celor 6 intrari neschimbat); comentarii Word 0 / 0 / 0 in manuscris, copia marcata, Response (si 0 in Supplementary, Highlights, cover letter); paragrafele listei de referinte identice inainte / dupa (40 in manuscris, 69 in copia marcata, comparate ca text, baseline salvat inainte de editare); asteptarile din `_check_t41_outputs.py` (doua verificari) si `_check_t8_parity.py` trecute de la 6 / 0 la 0 / 0; textele din Deviations (assembly_plan.md) actualizate |
| T11c.2 Copia marcata, keywords | (fara editare; inregistrat in a199a07) | PASS: paragraful 11 "Keywords: Personalized nutrition; Wearable devices; Multimodal physiological data; Knowledge-based decision support; Explainable artificial intelligence (XAI); Decision support systems; SHAP" apare ca insertie (ambele run-uri, "Keywords: " si lista, evidentiate); la "Removed from the submitted version" (27 de intrari: 18 "Paragraph:" / "Cell:" + 9 "From the paragraph beginning ..."; paragraful 204): 'Paragraph: "Keywords: Personalized nutrition; Wearable devices; Explainable AI; SHAP; Physiological state classification; Random Forest; mHealth"' |
| T11c.3 Documentatie | a199a07 | PASS: TEXT_DECISIONS.md, sectiunea T11b, bullet T11c (fara campuri Zotero in v8, lista statica, APA 7 verificat manual la F5, import optional); F5_submission_checklist.md: punctul deschis 2 (import Zotero) INCHIS cu acest motiv, regula "comentariile raman pana la import" si linia "Ramase dupa F5" marcate inchise |
| T11c.4 Verificare finala | (acest commit) | PASS: toate check-urile la valorile T11b, cu exceptia comentariilor (0 / 0 / 0); t41_outputs_check.md regenerat dupa commit-ul T11c.3 si comis aici |

## Valorile check-urilor (T11c.4, dupa reasamblarea celor 6 docx din surse)

| Check | Asteptat (T11b) | Obtinut (T11c) |
|---|---|---|
| `_check_t41_outputs.py` (dupa commit-ul T11c.3) | 76 / 76 | 76 checks, 0 FAIL (comentarii 0 / 0 / 0; "0 'verify in Zotero'") |
| `_check_t41_numbers.py` R1 / S1 | 1,581 / 1,099 / 0 ; 387 / 321 / 0 | identic (raportul R1 neschimbat fata de T11b) |
| `_check_table_rounding.py` | 402 / 0 | 402 celule, 0 nepotriviri |
| `style_check.py` manuscris / Supplementary / Response / Highlights / cover letter | 8 / 8 (26.4), 3 / 3, 5 / 5 (27.0), 3 / 3, 3 / 3 | identic |
| `_check_t8_parity.py` | 19 / 26, aceleasi 7 FAIL istorice | 19 / 26, aceleasi 7 FAIL istorice; asteptarea comentariilor 6 -> 0 |
| `_check_t32b_outputs.py` / `_check_t42_response.py` / `_check_t91_wording.py` / `_check_t115_cover_letter.py` / `_check_t11b_inventory.py --after` | ALL PASS | ALL PASS (t32b_run_meta.json restaurat din index dupa rulare) |
| pytest (revision/scripts) | 113 | 113 passed |
| CIFRE_CANONICE.md sha256 / CIFRE_REVIZIE.md | a80e9d6c5986c98b... / neschimbat | identic / neschimbat |
| LF, em dash | i/lf, 0 | i/lf pe fisierele editate, 0 em dash |

## Puncte deschise (dupa T11c)

Cele din T11b_summary.md fara importul in Zotero: T12b (clona publica, DOI Zenodo, fraza a doua din Data availability), raspunsul editorial office (tipul de item pentru manuscrisul curat), corectarea numelui in EM (coautor), campurile EM la upload (abstract 249 de cuvinte, 7 keywords em7, "[date of submission]", formularul de competing interests, figurile separate daca Guide for Authors le cere), textul verbatim al mesajelor WhatsApp din 3 nov (primul autor). Pachetul la coautor: 9-10 nov 2026.
