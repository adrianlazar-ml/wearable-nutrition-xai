"""T11.5 (2026-11-03): checks on the Revision 1 cover letter (revision/manuscript/KBS_Cover_Letter_R1.docx, built by
assemble_manuscript.make_cover_letter from revision/F4_cover_letter_R1.md).

1. Under 350 words (split on spaces); the only numbers are the manuscript number (26, 14863), the street number (313) and
   the postal code (060042); '[date of submission]' is the only placeholder.
2. The two formulations that must coincide with the response letter: 'immediately after multi-subject validation'
   (Reviewer 2, Weakness 1, literal) and 'reported model is unchanged' (no literal counterpart in the letter; coincides
   with R1.4 'The tuned results are reported as a sensitivity analysis' against 'the reported Random Forest', and with
   manuscript 4.4 'The configuration fixed before this revision is therefore kept as the system's model'; decision Q1 of
   the co-author, 30 Oct 2026). The sentences found are printed.
3. 0 occurrences of the forbidden list of the brief, 0 em dash, 0 British spellings (f5_us_spelling pairs).
4. The docx text equals the markdown source with the parameters resolved (paragraph by paragraph).

Usage (repo root): python revision/scripts/_check_t115_cover_letter.py
Exit 1 on any FAIL.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble_manuscript import COVER_MD, OUT_COVER, LETTER, F4, cover_letter_paragraphs, FIRST_AUTHOR, SECOND_AUTHOR  # noqa: E402
from f5_us_spelling import PATTERN as UK_PATTERN  # noqa: E402

FORBIDDEN = ["deduplicated", "deployable", "complete system", "honest", "candor", "best", "significant", "optimal", "stable over time"]
ALLOWED_NUMBERS = {"26", "14863", "313", "060042"}
fails = 0


def rec(name: str, ok: bool, detail: str = "") -> None:
    global fails
    fails += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def main() -> None:
    d = docx.Document(str(OUT_COVER))
    pars = [p.text for p in d.paragraphs if p.text.strip()]
    text = "\n".join(pars)
    n_words = len(text.split())
    rec("Cover letter under 350 words (split on spaces)", n_words < 350, f"{n_words} words, {len(pars)} paragraphs")
    nums = set(re.findall(r"\d+", text))
    rec("Only the manuscript number, street number and postal code carry digits (0 result figures)", nums <= ALLOWED_NUMBERS, f"numbers: {sorted(nums)}")
    ph = sorted(set(re.findall(r"\[[^\]]+\]", text)))
    rec("'[date of submission]' is the only placeholder", ph == ["[date of submission]"], str(ph))
    rec(f"Authors named as '{FIRST_AUTHOR}' and '{SECOND_AUTHOR}'; 0 'Adrian A. Lazar'",
        f"{FIRST_AUTHOR} and {SECOND_AUTHOR}" in text and "Adrian A. Lazar" not in text,
        f"'{FIRST_AUTHOR}' {text.count(FIRST_AUTHOR)}, '{SECOND_AUTHOR}' {text.count(SECOND_AUTHOR)}")

    letter = LETTER.read_text(encoding="utf-8")
    manu = F4.read_text(encoding="utf-8")
    s1 = "immediately after multi-subject validation"
    hits1 = [s.strip() for s in re.split(r"(?<=\.)\s", letter) if s1 in s]
    rec(f"'{s1}' coincides with the response letter (Reviewer 2, Weakness 1)", s1 in text and len(hits1) == 1, hits1[0][:200] if hits1 else "not in the letter")
    s2 = "reported model is unchanged"
    hits2 = [s.strip() for s in re.split(r"(?<=\.)\s", letter) if "reported as a sensitivity analysis" in s and "tuned" in s]
    hits2m = [s.strip() for s in re.split(r"(?<=\.)\s", manu) if "fixed before this revision is therefore kept as the system's model" in s]
    rec(f"'{s2}' coincides with the response letter (R1.4, tuned results as sensitivity analysis) and manuscript 4.4 (configuration kept)",
        s2 in text and len(hits2) >= 1 and len(hits2m) == 1,
        ("letter: " + hits2[0][:160] + " | manuscript: " + hits2m[0][:160]) if hits2 and hits2m else "no matching sentence")

    low = text.lower()
    forb = {w: len(re.findall(r"\b" + re.escape(w), low)) for w in FORBIDDEN}
    rec("0 forbidden formulations (deduplicated, deployable, complete system, honest, candor, best, significant, optimal, stable over time)",
        not any(forb.values()), str({k: v for k, v in forb.items() if v}) or "0 hits")
    rec("0 em dash (U+2014)", "—" not in text, f"{text.count(chr(0x2014))} found")
    uk = [m.group(0) for m in UK_PATTERN.finditer(text)]
    rec("0 British spellings (f5_us_spelling pairs)", not uk, "; ".join(uk) if uk else "0 hits")
    long_sents = [s for p in pars for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", p) if len(s.split()) > 60]
    rec("0 sentences over 60 words", not long_sents, f"{len(long_sents)}")

    md_pars = cover_letter_paragraphs(COVER_MD.read_text(encoding="utf-8"))
    rec("docx text == markdown source with the parameters resolved (paragraph by paragraph)", pars == md_pars,
        f"{len(pars)} docx paragraphs, {len(md_pars)} md paragraphs" + ("" if pars == md_pars else "; first difference: " +
        next((f"{a[:80]!r} != {b[:80]!r}" for a, b in zip(pars, md_pars) if a != b), "length")))
    rec("0 Word comments in the cover letter", len(list(d.comments)) == 0, f"{len(list(d.comments))}")
    print(f"{'ALL PASS' if not fails else f'{fails} FAIL'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
