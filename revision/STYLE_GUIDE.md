# STYLE_GUIDE.md - profilul stilistic al manuscrisului KBS (Revision 1)

T8.1, 25 oct 2026. Referinta este manuscrisul submis, `revision/manuscript/BASE_MANUSCRIPT_v8.docx` (v8, confirmat de Buiu). Tot textul nou sau modificat in revizie (manuscris, Supplementary, Highlights, response letter) se aliniaza la acest profil. Masurat cu `revision/scripts/style_check.py` (generalizarea lui `revision/f8/style_compare.py`, scriptul folosit la decizia din 25 oct); cifrele de referinta sunt in sectiunea 4.

## 1. Lungimea frazelor

- Media: 20-25 de cuvinte pe fraza (v8: 24.6, mediana 22.5).
- Rareori peste 40 de cuvinte (v8: 10% din fraze); niciodata peste 60.
- Exceptie admisa: enumerarile de definitii, de exemplu cele sase grupuri de feature-uri din 3.9 (o singura fraza peste 60 de cuvinte in textul nou al manuscrisului).
- O fraza lunga cu doua puncte sau cu punct si virgula se sparge in doua fraze; conectorul rezultat ("therefore", "in addition") se pastreaza numai daca rationamentul il cere (la T8.2 s-au taiat trei "therefore" introduse de spargere).

## 2. Voce si subiect

- Subiect impersonal sau numele obiectului: "The evaluation adds...", "Calibration of the Random Forest probabilities was assessed...", "The configuration fixed before this revision is kept...".
- "we" numai pentru decizii ale autorilor ("We emphasize that these labels are operational pseudo-labels", "to the best of our knowledge"); in proza manuscrisului cel mult 6 aparitii (v8 are 4).
- Response letter: aceleasi reguli de lungime si vocabular; "we" este permis (autorii raspund in nume propriu).
- Fara formule retorice ("the picture is that", "we report it honestly", "as the reviewer's concern deserves" se inlocuiesc cu enuntul direct). Termeni interzisi in response letter, verificati de style_check.py (T10.3, 1 nov 2026): "concern deserves", "concern implies".

## 3. Vocabular si notatie

- "Random Forest" cu majuscule, ca nume al modelului, in tot manuscrisul, in Supplementary, in Highlights si in response letter; "forest" / "the forest" singur ramane cu minuscule ("the raw forest", "100 forests retrained"). Pluralul este "Random Forest models" ("100 Random Forest models retrained", legendele Figure 5 si Figure S2; T8.8b), nu "Random Forests".
- Intervale care contin zero: "separated from zero" / "is not separated from zero" / "the interval includes zero"; NU "distinguished from zero", "distinguishable from zero", "excludes zero" (cu exceptia descrierii tehnice "intervals excluding the raw value" din 4.5, pastrata din F4).
- Retelele vs persistence: "not separated from persistence after tuning" (formularea T7 "not distinguished from persistence" devine "not separated from persistence"; regula T7 ramane: fara "do not exceed persistence" despre retele fara "after tuning").
- Zecimale: 3 in textul nou; 4 numai unde v8 avea deja 4 (Table 3, 5, 7, 12 si tabelele Supplementary S1, S2, care reproduc sursele).
- Diferente in puncte procentuale: "percentage points" sau "pp" ("7.8 percentage points", "2.41 points" ramane in textul v8 pastrat).
- Interdictiile din TEXT_DECISIONS raman: "best performing", "best overall", "best model", "significant" / "significantly" ca eticheta (fara test raportat; "significance label" in response letter R1.3 este descrierea a ceea ce NU se face), "stable over time", "optimal thresholds", "honestly", "three years". Fara em dash nicaieri (nici in response letter).
- Engleza americana (v8 este US): verificarea permanenta din `_check_t41_outputs.py` (perechile din `f5_us_spelling.py`).
- Numele claselor Rest / Normal / Active cu majuscula; "macro F1", "F1 Active", "balanced accuracy", "paired bootstrap difference" ca in v8.
- Provenance wording (T9.1, 26 oct 2026): in Abstract, Highlights, Introduction, Conclusions si response letter NU apar "two devices are complementary", "both devices are complementary", "device-independent", "watch indicators", "ring and watch are complementary"; comparatia dispozitivelor se numeste "two data paths" (in Abstract "both data sources"), iar indicatorii de validare a etichetelor "rule-independent indicators" / "indicators that do not enter the labeling rule". Methods, Results si Discussion nu intra in verificare (formularile hardware din 3.1, 3.3, 3.9, 4.6 si Discussion raman). Verificare permanenta in `_check_t41_outputs.py` (check "Provenance wording (T9.1)").

## 4. Cifre de referinta (style_check.py, proza fara tabele, legende, note, referinte, declaratii, anexa)

| Text | Fraze | Cuvinte/fraza (medie / mediana / p90 / max) | % fraze > 40 | Fraze > 60 | "we" | "random forest" minuscule | "distinguish" |
|---|---|---|---|---|---|---|---|
| v8 (BASE_MANUSCRIPT_v8.docx, 40 paragrafe) | 152 | 24.6 / 22.5 / 40 / 79 | 9.9% | 2 | 4 | 0 | 1 |
| R1 inainte de T8.2, paragrafe identice cu v8 (24) | 86 | 23.3 / 21.5 / 39 / 56 | 9.3% | 0 | 2 | 0 | 0 |
| R1 inainte de T8.2, paragrafe diferite de v8 (36) | 178 | 34.5 / 33.0 / 58 / 104 | 32.0% | 13 | 18 | 17 | 14 |
| Response letter inainte de T8.4 (20 paragrafe) | 83 | 37.7 / 37 / 58 / 89 | 39.8% | 8 | 43 | 11 | 6 |

Praguri (T8.6; `style_check.py` iese cu cod 1 daca nu sunt indeplinite): pe paragrafele diferite de v8, medie <= 27, p90 <= 45, fraze > 60 <= 1, "random forest" minuscule = 0, "distinguish" = 0, termeni interzisi = 0; "we" in proza manuscrisului <= 6. Response letter: medie <= 27, fraze > 60 <= 1, "random forest" = 0, "distinguish" = 0, termeni interzisi = 0. Masurare (T8.8c, 25 oct 2026): numai pe paragrafele Response (paragraful introductiv, paragrafele "Response." si al doilea paragraf Response din R1.9); paragrafele "Changes." (liste de localizari, o singura "fraza" fiecare) si bullet-urile din "General changes" sunt enumerari si nu intra in profil. Pana la T8.8c se masurau toate paragrafele autorilor (27.7 cuvinte/fraza dupa T8.4, fata de 26.9 pe paragrafele Response). Supplementary si Highlights (`--mode other`, T10.9e, 1 nov 2026): fraze > 60 cuvinte = 0 pe paragrafele de proza adaugate in revizie; legendele de tabel si de figura si notele italice de sub tabele sunt excluse (aceeasi regula ca listele "Changes" pe Response), plus "random forest" minuscule = 0 si termeni interzisi = 0. Cover letter (T11.5, 3 nov 2026; `KBS_Cover_Letter_R1.docx` din `F4_cover_letter_R1.md`): `--mode other` (termeni interzisi 0, fraze > 60 cuvinte 0, "random forest" minuscule 0) plus lista suplimentara din `revision/scripts/_check_t115_cover_letter.py`: 0 "deduplicated", "deployable", "complete system", "honest", "candor", "best", "significant", "optimal", "stable over time"; 0 em dash; ortografie US; sub 350 de cuvinte; fara cifre de rezultate (singurele numere admise: numarul manuscrisului, codul postal, data); "[date of submission]" singurul placeholder.

## 5. Ce nu se atinge la alinierea stilistica

- Paragrafele identice cu v8 (inclusiv "We emphasize..." din 3.2 si "to the best of our knowledge" din contributii).
- 3.12: GOAL_FIX aplicat, confirmat de Buiu 30 oct 2026 (obiectivul declarat este camp de profil si nu modifica tinta; fraza v8 "adjusted by the declared goal" a disparut la apply-audit).
- Tabelele si cifrele lor, referintele, cele 9 comentarii Word ale manuscrisului si comentariul de pe linia Authors din response letter, Appendix A, declaratiile.
- Nicio cifra noua si nicio cifra mutata: pentru fiecare paragraf inlocuit, setul de numere din vechi si din nou este identic (verificat de `revision/scripts/_check_t8_parity.py`).
