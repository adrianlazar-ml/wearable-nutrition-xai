# T8: uniformizare stilistică a livrabilelor R1 cu stilul manuscrisului v8 (25 oct 2026)

## Context

Repo disertatie_v2, folder revision/. Stare: f6eefc0 (T7b), F5 închisă tehnic, 73/73 verificări, 9 comentarii Word (6 Zotero, 3 coautor). Baza manuscrisului e BASE_MANUSCRIPT_v8.docx (v8 submis, confirmat de Buiu).

Problema: textul adăugat în revizie (R1) are alt stil decât v8. Măsurat pe proza din docx (fără tabele, legende, referințe): v8 23.5 cuvinte/frază, 9% fraze peste 40 de cuvinte, „Random Forest" cu majuscule, „we" rar; textul nou R1 33 cuvinte/frază, 32% fraze peste 40, 11 fraze peste 60 (max 104), „random forest" cu minuscule de 15 ori, „we" de 3 ori mai des, „not distinguished from zero" de 9 ori. Paragrafele păstrate din v8 sunt la standard; cusătura e exact între v8 și textul nou.

Decizia (Adrian, 25 oct 2026): tot textul din toate documentele KBS se aliniază la stilul v8. Rescrierea s-a făcut deja în proiectul de decizie și e validată paragraf cu paragraf. T8 APLICĂ textul validat, nu îl rescrie. Sursele sunt cele patru fișiere din revision/f8/:

- f8_front_methods_new.txt: 18 paragrafe (Abstract; Introduction paragraful de poziționare și 4 contribuții; 3.2; 3.4 p3; 3.5 p2 și p3; 3.6 p3; 3.7; 3.8 p2; 3.9; 3.10; 3.11; Conclusions p1; Future work).
- f8_results_new.txt: 11 paragrafe (4.2 p1; 4.3 p2; 4.4 p1 și p2; 4.5; 4.6 p2; 4.7 ambele; 4.8 p1; 4.11; 4.12).
- f8_discussion_new.txt: 6 paragrafe (Principal findings; Temporal robustness; Choice of classifier; Generalization from a single subject; Limitations; Practical implications).
- f8_response_new.txt: paragrafele Response și Changes ale response letter-ului, în ordinea R1.1-R1.10, R2 Weakness 1-3, precedate de paragraful introductiv.
- style_compare.py: scriptul de măsurare folosit la decizie; se generalizează în T8.1.

Reguli de corespondență: fiecare paragraf nou înlocuiește paragraful R1 care începe cu aceleași prime 3-6 cuvinte (sau cu același titlu bold: „Principal findings.", „Threshold sensitivity.", „Agreement with label-independent indicators." etc.). Paragrafele se înlocuiesc integral, unu la unu. Dacă un paragraf nou nu are corespondent unic, STOP și raportează; nu ghici.

Ce NU se atinge: 3.12 (conține fraza GOAL_FIX „adjusted by the declared goal", decizie Buiu în așteptare); orice paragraf identic cu v8 (inclusiv „We emphasize" din 3.2, care rămâne); tabelele și cifrele lor; referințele; cele 9 comentarii Word; Appendix A; declarațiile.

## Reguli generale

1. Zero cifre noi, zero cifre mutate. Pentru fiecare paragraf înlocuit, setul de numere (regex `[-+]?\d[\d,]*\.?\d*`) din vechi și din nou trebuie să fie identic, cu două excepții documentate: Generalization (nou are în plus 2024 și 2025, anii citărilor Zignoli și Kim, ambele deja în References) și nimic altceva. Orice altă diferență = FAIL.
2. Sursa de adevăr pentru text rămâne F4_manuscript_changes.md și F4_response_letter.md: se modifică ÎNTÂI acolo, apoi se regenerează docx-urile prin pipeline-ul de asamblare existent (assembly_plan.md). Dacă pipeline-ul nu permite regenerare completă, se editează docx-ul cu python-docx păstrând stilurile paragrafelor, bold-urile din titlurile inline și cele 9 comentarii; se raportează explicit care cale s-a folosit.
3. Engleză americană (v8 e US); LF peste tot; Git Bash, nu PowerShell. Fără em dash nicăieri.
4. Un commit per task, mesaj prefixat „T8.x:". Înainte de orice înlocuire: listează potrivirile (grep) și confirmă unicitatea; nu face find-replace orb.
5. Nu rescrie și nu „îmbunătățește" textul din f8_*. Dacă găsești o eroare în el, raportează și oprește-te la paragraful respectiv.

## Task-uri

### T8.1 Ghid de stil și script de verificare

Creează revision/STYLE_GUIDE.md cu profilul v8, derivat din măsurători: fraze de 20-25 de cuvinte în medie, rareori peste 40, niciodată peste 60 (excepție: enumerări de definiții, ex. cele șase grupuri din 3.9); subiect impersonal sau numele obiectului, „we" doar pentru decizii ale autorilor; „Random Forest" cu majuscule în tot manuscrisul, Supplementary și Highlights; „separated from zero" / „includes zero" în loc de „distinguished from zero"; 3 zecimale în textul nou, 4 doar unde v8 avea deja (Table 3, 5, 7, 12); „percentage points" sau „pp" pentru diferențe exprimate în puncte; fără formule retorice; interdicțiile din TEXT_DECISIONS rămân („best performing", „significant" ca etichetă, „stable over time", „optimal thresholds"). Response letter: aceleași reguli de lungime și vocabular, „we" permis.

Generalizează style_compare.py în revision/scripts/style_check.py: primește un docx, extrage proza (exclude tabele, legende Table/Figure, References, declarații, Appendix), raportează pe întreg documentul și separat pe paragrafele diferite de v8 (similaritate difflib < 0.9 față de BASE_MANUSCRIPT_v8.docx): cuvinte/frază (medie, mediană, p90, max), % fraze > 40, număr fraze > 60, „we"/1k, „Random Forest" vs „random forest", „distinguish", termeni interziși, numere cu 4 vs 3 zecimale. Exit code 1 dacă pragurile de la T8.6 nu sunt îndeplinite.

PASS: STYLE_GUIDE.md există; style_check.py rulează pe KBS_Manuscript_R1.docx curent (înainte de T8.2) și reproduce ordinul de mărime al cifrelor din Context (text nou ~33 cuv./frază, ~15 „random forest" minuscule). Commit „T8.1: style guide and style_check.py".

### T8.2 Manuscris: înlocuirea celor 35 de paragrafe

Surse: f8_front_methods_new.txt (18), f8_results_new.txt (11), f8_discussion_new.txt (6). Ancorele paragrafelor vechi (primele cuvinte în R1):

Front/Methods/Conclusions: „Personalized nutrition depends"; „The system studied here"; „A machine learning pipeline that"; „An end-to-end explainable"; „An evaluation methodology for"; „The results of that evaluation"; „Each day is assigned"; „In addition to Random Forest"; „Two baselines are reported"; „Hyperparameters were searched"; „Because macro F1, accuracy"; „Calibration of the random forest"; „The stability of the SHAP"; „Complementing the per-device"; „The pseudo-labels were examined"; „To assess whether the single"; „This paper presented and evaluated"; „Future work should prioritize".

Results: „Table 4 reports"; „Pairwise comparisons"; „Table 6 reports"; „Hyperparameter search (Appendix"; „The raw random forest"; „Table 8 and Figure 3"; „Threshold sensitivity."; „Agreement with label"; „Table 11 and Figure 4"; „The SHAP analysis was replicated"; „The confidence threshold bounds".

Discussion: „Principal findings."; „Temporal robustness."; „Choice of classifier."; „Generalization from a single subject."; „Limitations."; „Practical implications.".

Titlurile inline bold (ex. „**Principal findings. **", „**Threshold sensitivity. **") se păstrează ca formatare; textul nou le conține ca text simplu la început de paragraf, deci la aplicare titlul rămâne bold și restul paragrafului e înlocuit.

După înlocuire, trei tăieri de „therefore" (introduse de spargerea frazelor):
- 4.4: „the default-parameter comparison therefore does not establish an ordering" → „the default-parameter comparison does not establish an ordering"
- 4.5: „Calibration is therefore reported as a result and as a limitation" → „Calibration is reported as a result and as a limitation"
- 4.6: „The model therefore does not depend on the composite scores" → „The model does not depend on the composite scores"

Legende, „random forest" → „Random Forest" (numai în legende noi sau modificate în revizie; v8 are deja majuscule): Table 4 (caption: „of the random forest are in Table 5"), Table 8, Table 11, Figure 2, Figure 3, Figure 4, Figure 5. Listează potrivirile înainte; corpul tabelelor nu se atinge.

PASS: 35 de paragrafe înlocuite, fiecare cu potrivire unică; paritate numerică per paragraf (regula 1) cu singura excepție documentată; 0 apariții „random forest" (minuscule) în proză și legende; 0 „distinguish"; „we" în proză ≤ 6 (v8 are 4; „We emphasize" și „to the best of our knowledge" rămân); Abstract ≤ 250 de cuvinte la numărătoarea Word (python-docx: ≤ 249 la split pe spații, identic cu R1); 3.12 byte-identic cu înainte; cele 9 comentarii Word prezente; marked.docx regenerat față de v8. Commit „T8.2: manuscript text aligned with v8 style (35 paragraphs)".

### T8.3 Supplementary și Highlights

Supplementary: „random forest" → „Random Forest" în titlul „S2. Calibration of the random forest", legenda Table S2(a), legendele Figure S1, S2, S3 (6 apariții așteptate; listează înainte). Nimic altceva.
Highlights: „Random forest exceeds" → „Random Forest exceeds" (83 de caractere, neschimbat).

PASS: 6 + 1 înlocuiri, 0 „random forest" minuscule în ambele; cele 5 highlights ≤ 85 caractere; Supplementary 372 / 308 / 0 la verificarea cifrelor. Commit „T8.3: Supplementary and Highlights capitalization".

### T8.4 Response letter

Înlocuiește paragraful introductiv („We thank the editor...") și fiecare pereche Response / Changes cu textul din f8_response_new.txt, în ordinea R1.1-R1.10, R2 Weakness 1-3 (paragraful „In the course of this analysis..." din R1.9 e al doilea paragraf Response al acelui comentariu). Antetele de comentarii, citatele din review (italice) și comentariul Word de pe linia Authors rămân.

Adaugă în „General changes not prompted by a specific comment" bullet-ul: „The wording of the text added in revision was aligned with the style of the submitted manuscript (sentence length, impersonal voice, capitalization of model names); no figure, claim or qualification was changed." Celelalte intrări pendinte (declarații, cross-referințe, Table A.1, US) NU se adaugă acum; rămân pentru după răspunsul lui Buiu.

Actualizează F4_response_letter.md ca sursă.

PASS: 1 + 13 perechi înlocuite; paritate numerică pe întreg documentul (setul de numere din R1 ⊆ setul din nou și invers; raportează diferențele, așteptat zero); 0 „distinguish"; 0 „significant" (fraza din R1.3 e acum „rather than by a significance label"); 0 „random forest" minuscule; 40/40 trimiteri la secțiuni, tabele și figuri rezolvă în manuscrisul T8.2 (reutilizează verificarea T4.2); comentariul Word pe Authors prezent. Commit „T8.4: response letter aligned with v8 style".

### T8.5 Documente de proces

- TEXT_DECISIONS.md: secțiune „F8 (25 oct 2026), stil": convențiile din STYLE_GUIDE.md; „separated from zero" / „includes zero" înlocuiesc „distinguished from zero"; „Random Forest" cu majuscule peste tot; 3.12 exclus până la GOAL_FIX; R1.1 primește fraza despre participant și citarea Zignoli/Kim (decizie Adrian 25 oct); R1.3 „significance label".
- assembly_plan.md: notă că paragrafele listate la T8.2 provin din revision/f8/ și înlocuiesc versiunile F4/T7.
- CIFRE_REVIZIE.md și CIFRE_CANONICE.md: NU se modifică (nicio cifră nouă). Confirmă printr-un git diff gol pe ele.
- KBS_Revision_Plan.md: adaugă T8 la calendar ca închis, cu commit-urile.

PASS: fișierele actualizate, diff-ul pe CIFRE_* gol. Commit „T8.5: process docs for T8".

### T8.6 Verificare finală

Rulează și raportează, cu valorile obținute:
1. _check_t41 pe manuscris: 1516 / 1061 / 0 neexplicate (identic cu F5); pe Supplementary: 372 / 308 / 0; _check_t41_outputs 69/69.
2. style_check.py pe KBS_Manuscript_R1.docx: pe paragrafele diferite de v8, medie ≤ 27 cuvinte/frază, p90 ≤ 45, fraze > 60 ≤ 1 (enumerarea din 3.9), „random forest" minuscule = 0, „distinguish" = 0, termeni interziși = 0; „we" în proza manuscrisului ≤ 6. Pe Response: medie ≤ 27, fraze > 60 ≤ 1.
3. grep pe toate cele 5 docx: „best performing", „significant", „stable over time", „optimal thresholds", „honestly", „distinguish", em dash „—": toate 0.
4. Afirmații-cheie prezente verbatim în manuscris (grep): „the final block is the most favorable of the four"; „not as a time-invariant property of the approach"; „The reported setting (60/85/75) gives the highest macro F1 among the comparable settings, with T2 within 0.001"; „describe recovery rather than exertion"; „these offsets are prescriptive rules, not measured effects"; „the results support technical feasibility, not clinical readiness"; „Only the head of the ranking is robust"; „is not separated from persistence"; „Multi-subject validation remains the first item of future work"; „These figures correct those in the original submission (80% coverage, 76 to 69 kcal)".
5. Spelling US (reutilizează check-ul permanent din T6): 0 forme UK.
6. Cele 9 comentarii Word în manuscris și 1 în response letter; cross-referințele către Table A.1 și secțiuni intacte.
7. Fiecare docx se deschide și se convertește în PDF cu LibreOffice fără eroare.
8. Proprietatea Word „Internal" (2 apariții la final) NEMODIFICATĂ: doar confirmă prezența, rămâne în F5 checklist.

PASS: toate cele 8 verzi. Dacă una pică, nu corecta textul din f8_*; raportează paragraful și valoarea.

### T8.7 Fișiere pentru KB

Listează fișierele de reîncărcat în KB (docx-urile modificate, TEXT_DECISIONS.md, assembly_plan.md, F4_*.md, STYLE_GUIDE.md, KBS_Revision_Plan.md) cu sha256, pentru verificarea post-upload.

## Rezumat final + commit

La final: lista commit-urilor T8.1-T8.5 cu hash; tabelul cu cifrele style_check înainte și după (medie, p90, max, % > 40, „we", „random forest"); rezultatul celor 8 verificări din T8.6 cu valori; orice paragraf la care te-ai oprit și de ce; lista T8.7. Nu reformula și nu rezuma conținutul textelor din f8_*.
