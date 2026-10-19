"""T4.3: append the section "Revision 1 (KBS, 2026)" to CIFRE_CANONICE.md (root) without touching the existing text.

The file uses CRLF line endings and Romanian decimal commas in sections 1-16; the appended section keeps CRLF and uses the
decimal point of CIFRE_REVIZIE.md (stated in the section). Run once from the repo root; it refuses to run twice.
Verification: the original bytes must be a prefix of the new file (asserted here and checked with cmp in the T4.3 log).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "CIFRE_CANONICE.md"
MARK = "## Revision 1 (KBS, 2026)"

SECTION = """

---

## Revision 1 (KBS, 2026)

Sectiune adaugata la T4.3 (2026-10-19), dupa planul `revision/KBS_Revision_Plan.md`. Sectiunile 1-16 de mai sus NU au fost modificate (byte-identice; verificat prin cmp la adaugare, `revision/scripts/_append_t43_canonice.py`). Toate cifrele de mai jos sunt copiate din `revision/CIFRE_REVIZIE.md` (randul indicat), cu punct zecimal ca in acel fisier; ele sunt aditive si NU schimba cifrele canonice ale modelului raportat (RF 100 / 10 / 5, seed 42: F1 macro 0.5063, accuracy 0.6406, MCC 0.1416).

### (a) Errata la sectiunea 7 (Confidence threshold)

Tabelul din sectiunea 7 NU provine din modelul canonic (RF 19 features, JSON). A fost produs de `confidence_threshold.main()` din pipeline-ul XML cu 16 features (commit f7cc4ca, 2026-05-26); evidenta: `revision/scripts/_check_t13_safety_layer.py` (exit 1 intentionat, 2026-10-19) si randurile T1.3 din CIFRE_REVIZIE.md. Auditul de provenienta din 19 oct 2026 arata ca celelalte 15 sectiuni sunt pe modelul canonic.

| Marime (prag 0.50) | Sectiunea 7 (gresit, pipeline XML 16f) | Corect (RF canonic, test 192 zile) | Sursa |
|---|---|---|---|
| Coverage | 80,2% (38 suprascrise) | 65.6% (126 / 192), 66 suprascrise | CIFRE_REVIZIE T1.3, `safety_layer_calibrated.py` |
| Eroare calorica medie la 0.50 | 68,8 kcal | 58.3 kcal | CIFRE_REVIZIE T1.3 |
| Eroare calorica medie la prag 0 (fara threshold) | 76,0 kcal | 71.9 kcal | CIFRE_REVIZIE T1.3 |
| Eroare calorica maxima | 200 kcal | 200 kcal (se mentine: nicio confuzie Active-Rest) | CIFRE_REVIZIE T1.3 |
| F1 macro / accuracy / F1 Active dupa fallback; stari emise R/N/A | 0,522 / 65,6% / nu era raportat | 0.5067 / 0.7083 / 0.4000; 28 / 162 / 2 | CIFRE_REVIZIE T1.3 |

Manuscrisul revizuit (sectiunea 4.12) foloseste cifrele corecte; corectia este dezvaluita in response letter (Reviewer 1, comentariul 9). Pragul 0.50 nu a fost reales. Eroarea calorica medie nu se raporteaza singura: sistemul "mereu Normal" are 54.2 kcal tocmai pentru ca nu emite nicio recomandare Rest / Active (CIFRE_REVIZIE T1.3).

### (b) Cifre noi care intra in manuscrisul revizuit (un rand per cifra)

| Cifra | Valoare | Rand CIFRE_REVIZIE | Loc in manuscris |
|---|---|---|---|
| Persistence F1 macro [CI 95%] | 0.3039 [0.2612, 0.3501] | T0.4 | Table 4, Table 6, 4.2, 4.3 |
| Persistence F1 Rest / F1 Normal / F1 Active | 0.205 / 0.707 / 0.0000 | T0.4 | Table 4 |
| Persistence accuracy / MCC [CI 95%] / QWK | 0.5625 / -0.0566 [-0.1632, 0.0597] / 0.0055 | T0.4, T1.1 | Table 4 |
| Persistence balanced accuracy [CI] / kappa [CI] / G-mean | 0.3039 [0.2602, 0.3520] / -0.0566 [-0.1626, 0.0593] / 0.0000 | T1.1 | Table 4 |
| Majority balanced accuracy / kappa / G-mean | 0.3333 / 0.0000 / 0.0000 | T1.1 | Table 4 |
| RF balanced accuracy [CI 95%] | 0.5227 [0.3851, 0.6634] | T1.1 | Table 4, 4.2 |
| RF Cohen's kappa [CI 95%] | 0.1415 [-0.0050, 0.2924] | T1.1 | Table 4 |
| RF G-mean [CI 95%] | 0.4923 [0.2960, 0.6228] | T1.1 | Table 4 |
| RF macro ROC-AUC OvR [CI 95%] | 0.5581 [0.4199, 0.6921] (include 0.5) | T1.1 | Table 4, 4.2 |
| RF macro PR-AUC [CI 95%] | 0.4983 [0.3716, 0.6359] | T1.1 | Table 4 |
| XGB balanced acc / kappa / G-mean / ROC-AUC / PR-AUC [CI] | 0.4500 [0.3439, 0.5763] / 0.1459 [0.0065, 0.2905] / 0.3759 [0.0000, 0.5156] / 0.5892 [0.4610, 0.7022] / 0.4598 [0.3616, 0.6082] | T1.1 | Table 4 |
| SVM balanced acc / kappa / G-mean / ROC-AUC / PR-AUC [CI] | 0.4342 [0.3288, 0.5683] / 0.0955 [-0.0395, 0.2310] / 0.3824 [0.0000, 0.5355] / 0.5595 [0.4493, 0.6733] / 0.4603 [0.3565, 0.6064] | T1.1 | Table 4 |
| RF F1 macro, CI pe matricea partajata (seed 42, 1000 resamples) | 0.5063 [0.3784, 0.6130] (CI canonic [0.373, 0.610] ramane in Table 5) | T1.1 | Table 6, Table 9, Table 11 |
| McNemar RF-XGB / RF-SVM / XGB-SVM, p Holm | 0.4004 / 0.8774 / 0.3818 | T1.2 | 4.3, Supplementary Table S1 |
| Cochran's Q (RF, XGB, SVM) | Q 3.3226, df 2, p 0.1899 | T1.2 | 4.3, Supplementary Table S1 |
| Paired bootstrap F1 macro RF-XGB / RF-SVM / XGB-SVM | 0.0303 [-0.0796, 0.1561] / 0.0716 [-0.0337, 0.1775] / 0.0413 [-0.1021, 0.1647] | T1.2 | 4.3, Table 6 |
| Paired bootstrap RF minus persistence, F1 macro / balanced accuracy | 0.2024 [0.0642, 0.3173] / 0.2188 [0.0728, 0.3630] | T1.2b | Abstract, 4.3, Table 6, 5 |
| McNemar RF-persistence (neajustat) | p_cc 0.1289 | T1.2 | 4.3 |
| ECE top-label RF test, raw / Platt / isotonic [CI] | 0.1377 [0.0815, 0.2074] / 0.0361 [0.0123, 0.0996] / 0.0405 [0.0189, 0.1077] | T1.3 | 4.5, Supplementary Table S2 |
| Brier multiclass [0, 2] raw / Platt / isotonic | 0.5091 / 0.4124 / 0.4095 | T1.3 | 4.5 |
| Log loss raw / Platt / isotonic | 0.8522 / 0.6969 / 0.7111 | T1.3 | 4.5 |
| Pante Platt (Rest / Normal / Active) | 0.3970 / 0.2210 / 0.2095 | T1.3 | 4.5 |
| p_active maxim pe test, raw / Platt / isotonic | 0.6312 / 0.1029 / 0.1896 | T1.3 | 4.5 |
| Argmax calibrat: F1 macro / F1 Active (toate 192 zile Normal) | 0.2811 / 0.0000 (= majority) | T1.3 | 4.5, Limitation (5) |
| Safety layer @0.50 raw (vezi errata) | 65.6% / 66 / 58.3 kcal (71.9 la prag 0) / 200; 0.5067 / 0.7083 / 0.4000; 28 / 162 / 2 | T1.3 | 4.12 |
| aw_hr_mean pe rangul 1 SHAP, bootstrap / seeds | 88 / 100 (top-3: 99 / 100) / 100 / 100 | T1.4 | Abstract, 4.11, 5 |
| Jaccard top-5 mediu, bootstrap / seeds | 0.4904 / 0.5276 | T1.4 | 4.11 |
| Kendall tau fata de referinta, bootstrap / seeds | 0.3909 [0.1813, 0.5912] / 0.7506 [0.6257, 0.8541] | T1.4 | 4.11 |
| Frecventa in top-5 (seeds): aw_active_calories / aw_sleep_deep_hours | 0.29 / 0.86 | T1.4 | 4.11 |
| Distanta intre locurile 3-7 (mean abs SHAP) | 0.00003 - 0.0005 | T1.4 (nota) | 4.11 |
| F1 macro al celor 100 modele pe seeds (dispersie seed-to-seed, NON-canonic) | 0.5142 [0.4682, 0.5677] | T1.4 | 4.6, 4.7 (referinta pentru "zgomot") |
| LOGO G1 composite scores: F1 macro / F1 Active; delta [CI] | 0.5148 / 0.5333; -0.0085 [-0.0893, 0.0640] | T1.5 | Table 8, 4.6 |
| LOGO G2 heart rate | 0.4467 / 0.4211; +0.0596 [-0.0150, 0.1274]; F1 Rest 0.1579 | T1.5 | Table 8, 4.6 |
| LOGO G3 HRV | 0.4730 / 0.4444; +0.0333 [-0.0415, 0.1091] | T1.5 | Table 8 |
| LOGO G4 sleep architecture | 0.5141 / 0.4706; -0.0077 [-0.0795, 0.0547] | T1.5 | Table 8 |
| LOGO G5 activity / energy (singurul CI care exclude 0) | 0.3780 / 0.1250; +0.1284 [0.0164, 0.2388] | T1.5 | Table 8, 4.6, 5 |
| LOGO G6 respiration / thermal / SpO2 | 0.5150 / 0.5000; -0.0087 [-0.0862, 0.0527] | T1.5 | Table 8 |
| KOGI: cel mai bun grup singur (heart rate) / G4 / G6 Active | 0.3589 / 0.2751 (sub majority) / 0 Active corecte | T1.5 | 4.6, Figure 3 |
| T2.1 defaults, F1 macro: CatBoost / XGB / SVM / FT-T / LightGBM / TabNet | 0.4980 / 0.4760 / 0.4347 / 0.4115 / 0.4089 / 0.2911 | T2.1 | Table 6, 4.4 |
| T2.1 defaults, F1 Active: CatBoost / FT-T / LightGBM / TabNet | 0.4615 / 0.1739 / 0.2222 / 0.0000 | T2.1 | Table 6 |
| T2.1 defaults, MCC: CatBoost / FT-T / LightGBM / TabNet | 0.1296 / 0.1218 / 0.0782 / -0.0204 | T2.1 | Table 6 |
| T2.1 delta RF minus model, F1 macro [CI]: CatBoost / FT-T / LightGBM / TabNet | +0.0083 [-0.0841, +0.1131] / +0.0948 [-0.0085, +0.1912] / +0.0974 [-0.0627, +0.2335] / +0.2152 [+0.0772, +0.3345] | T2.1 | Table 6, 4.4 |
| T2.1 McNemar TabNet vs RF | p 0.0006, Holm 0.0025 | T2.1 | 4.4 |
| T2.1 CatBoost pe 5 seed-uri (NON-canonic) | 0.4980 - 0.5388, peste RF pe 4 din 5 | T2.1 | 4.4 |
| T2.1 retele pe 5 seed-uri, maxim: FT-T / TabNet | 0.4194 / 0.3599 | T2.1 | 4.4 |
| T2.1 LightGBM default: zile prezise Normal | 165 / 192 | T2.1 | 4.4 |
| T2.2 CV: castigatori si scor (RF / XGB / SVM / LightGBM / CatBoost / TabNet / FT-T) | 0.3680 / 0.3925 / 0.4078 / 0.4013 / 0.3834 / 0.4162 / 0.4348; referinta RF rang 3 din 21 (0.3595) | T2.2 | Table A1, 4.4 |
| T2.2 tuned pe test, F1 macro [CI]: RF / XGB / SVM / LightGBM / CatBoost / TabNet / FT-T | 0.5683 [0.4172, 0.6761] / 0.5043 [0.3262, 0.6176] / 0.4386 [0.3463, 0.5307] / 0.5739 [0.4165, 0.6804] / 0.5368 [0.3938, 0.6443] / 0.3412 [0.2773, 0.4093] / 0.3698 [0.2987, 0.4522] | T2.2 | Table A1, 4.4, 5 |
| T2.2 tuned minus RF canonic, F1 macro [CI]: RF / LightGBM / CatBoost | +0.0619 [+0.0028, +0.1398] / +0.0676 [+0.0001, +0.1483] / +0.0305 [-0.0325, +0.1002] | T2.2 | Table A1, 4.4 |
| T2.2 tuned minus RF tuned: LightGBM / CatBoost | +0.0056 [-0.0305, +0.0450] / -0.0314 [-0.0810, +0.0100] | T2.2 | 4.4 |
| T2.2 retele tuned minus persistence: TabNet / FT-T | +0.0373 [-0.0486, +0.1204] / +0.0659 [-0.0208, +0.1655] (includ 0) | T2.2 | 4.4, 5 |
| T2.2 McNemar tuned vs RF canonic, Holm | niciuna < 0.05 (minim 0.0682 TabNet) | T2.2 | 4.4 |
| T2.2 dispersie pe seed-uri (NON-canonic): RF / LightGBM / CatBoost tuned | 0.5085 - 0.5816 / 0.4976 - 0.5739 / 0.5220 - 0.5642 | T2.2 | Table A1, 4.4 |
| T3.1 celule comparabile (6): interval F1 macro / F1 Active; RF > persistence si > majority | 0.3718 - 0.5063 / 0.2222 - 0.6154; 6 / 6 | T3.1 | Table 9, 4.7 |
| T3.1 R55_A85 (T2): F1 macro [CI]; persistence [CI]; delta | 0.5053 [0.3574, 0.6084]; 0.3211 [0.2771, 0.3679]; +0.1842 [+0.0281, +0.2919] | T3.1 | Table 9 |
| T3.1 R65_A85 (T5) | 0.4557 [0.3536, 0.5527]; 0.3089 [0.2581, 0.3592]; +0.1468 [+0.0288, +0.2656] | T3.1 | Table 9 |
| T3.1 R55_A80 / R65_A80: delta include 0 | +0.0514 [-0.0288, +0.1347] / +0.0804 [-0.0142, +0.1738]; F1 Active 0.2500 / 0.2222 | T3.1 | 4.7 |
| T3.1 R60_A80 | F1 macro 0.4129, F1 Active 0.2581, delta +0.1127 [+0.0207, +0.2053] | T3.1 | 4.7 (grila completa in Figure S1) |
| T3.1 celule A_r = 90 (T4 / T6, 1 Active in test, neinterpretate) | 0.6870 [0.6427, 0.7347] / 0.6822 [0.6364, 0.7237]; persistence 0.3199 / 0.3288; delta +0.3671 / +0.3534; 660 / 1000 resamples | T3.1 | Table 9 (asterisc) |
| T3.2 nivel A, aw_hr_mean: mediane R / N / A; eps2; Cliff A-R [CI] | 91.95 / 86.02 / 81.32 bpm; 0.1926; -0.840 [-0.912, -0.749] | T3.2 | Table 10, 4.7, 5 |
| T3.2 aw_resp_rate | 16.42 / 15.61 / 14.85; 0.1159; -0.732 [-0.817, -0.640] | T3.2 | Table 10 |
| T3.2 aw_hrv_mean | 22.82 / 24.87 / 23.85; 0.0191; +0.129 [-0.063, +0.308] (include 0) | T3.2 | Table 10 |
| T3.2 aw_spo2 | 94.36 / 94.70 / 94.81; 0.0060; +0.113 [-0.055, +0.306] | T3.2 | Table 10 |
| T3.2 aw_active_calories (DISCORDANT) | 883.6 / 809.4 / 579.3 kcal; 0.0201; -0.420 [-0.574, -0.246] | T3.2 | Table 10, 4.7, 5 |
| T3.2 aw_steps (DISCORDANT) | 8,057 / 7,079 / 4,499; 0.0132; -0.319 [-0.479, -0.129] | T3.2 | Table 10 |
| T3.2 nivel B: rhr_lowest / hrv_mean / temp_deviation / steps_oura, Cliff A-R | -0.879 / +0.603 / -0.270 / -0.260 | T3.2 | 4.7, Supplementary Table S3 |
| T3.2 lag-1: aw_active_calories / aw_steps / steps_oura, Cliff A-R | -0.267 / -0.196 / -0.202 | T3.2 | 4.7, Supplementary Table S3 |
| T3.3 fold 4 (test Dec 2023 - Jul 2024, train 190, 9 Active): F1 macro [CI] / F1 Active / persistence [CI] / delta [CI] | 0.3605 [0.3095, 0.4069] / 0.0000 / 0.3161 [0.2735, 0.3627] / +0.0444 [-0.0317, +0.1168] | T3.3 | Table 11, 4.8, 5 |
| T3.3 fold 3 (test Jul 2024 - Feb 2025, train 382, 14 Active) | 0.3771 [0.3127, 0.4456] / 0.0741 [0.0000, 0.2222] / 0.4120 [0.3253, 0.5136] / -0.0349 [-0.1600, +0.0841] (RF sub persistence) | T3.3 | Table 11, 4.8, 5 |
| T3.3 fold 2 (test Mar 2025 - Sep 2025, train 574, 22 Active) | 0.3124 [0.2561, 0.3749] / 0.1026 [0.0000, 0.2424] / 0.3070 [0.2559, 0.3682] / +0.0054 [-0.0768, +0.0930] | T3.3 | Table 11, 4.8, 5 |
| T3.3 fold 1 (canonic) | 0.5063 [0.3784, 0.6130] / 0.4444 [0.1176, 0.7000] / 0.3039 [0.2612, 0.3501] / +0.2024 [+0.0642, +0.3173] | T3.3 | Table 11 |
| T3.3 primul feature SHAP pe fold-urile 4 / 3 / 2 / 1 | aw_hr_mean / oura_readiness_score / oura_readiness_score / aw_hr_mean | T3.3 | Table 11, 4.8, 5 |
| T3.3 pooled (768 zile; 166 / 565 / 37): F1 macro [CI] / F1 Active [CI] | 0.3932 [0.3523, 0.4349] / 0.1386 [0.0522, 0.2316] (NON-canonic, nu inlocuieste 0.5063) | T3.3 | Table 11, 4.8, 5 |
| Durata dataset-ului | 10 iunie 2023 - 30 martie 2026, aproximativ 33 de luni (1,009 zile colectate / 958 utile) | KBS_Revision_Plan sectiunea 8.7 | 3.1, Abstract |

### (c) Decizia privind modelul raportat

RF canonic (100 arbori, depth 10, min leaf 5, seed 42; F1 macro 0.5063) ramane modelul raportat al sistemului (decizie Adrian, 19 oct 2026; de confirmat cu coautorul la draft-ul din 15 nov). Motive (CIFRE_REVIZIE T2.2): pe CV configuratia canonica este pe rangul 3 din 21 pentru RF, la 0.009 sub cea selectata; castigatorii T2.2 (LightGBM tuned 0.5739, RF tuned 0.5683, CatBoost tuned 0.5368) depasesc RF canonic la punct, dar CI-urile paired au limita inferioara la 0 (+0.0001 / +0.0028), niciun McNemar nu este semnificativ dupa Holm, iar cele trei ansambluri tuned nu sunt distinse intre ele; a alege modelul dupa rezultatul pe test ar contrazice protocolul de evaluare. Rezultatele tuned intra in manuscris ca analiza de sensibilitate (4.4, Appendix A, Table A1, Discussion "Choice of classifier"). Toate analizele de explicabilitate, calibrare si robustete temporala se refera la RF canonic.
"""


def main() -> None:
    orig = TARGET.read_bytes()
    if MARK.encode() in orig:
        raise SystemExit("already appended; nothing done")
    if not orig.endswith(b"\r\n"):
        raise SystemExit("STOP: unexpected line ending at the end of CIFRE_CANONICE.md")
    new = orig + SECTION.replace("\n", "\r\n").encode("utf-8")
    TARGET.write_bytes(new)
    assert TARGET.read_bytes()[: len(orig)] == orig
    print(f"appended {len(new) - len(orig)} bytes; existing {len(orig)} bytes byte-identical")


if __name__ == "__main__":
    main()
