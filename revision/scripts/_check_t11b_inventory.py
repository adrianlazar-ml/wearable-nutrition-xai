"""T11b.0 (2026-11-03): inventory of the strings touched by the co-author decisions of 3 Nov 2026, BEFORE any edit.

Searches the sources (assemble_manuscript.py, F4_manuscript_changes.md, F4_response_letter.md, F4_cover_letter_R1.md)
line by line and the 6 docx (python-docx: body paragraphs with their index, table cells, Word comments) for each target
string, prints every hit (path, paragraph index / line, exact text) and judges the expected locations: PASS when every
target is found exactly as many times as expected where expected (Acknowledgements: 0 everywhere), FAIL otherwise
(exit 1). Hits outside the expected locations are reported, not judged. The Data availability paragraph is printed
verbatim (T11b.4). Run again after the edits (--after) to check the post-edit expectations.

Usage (repo root): python revision/scripts/_check_t11b_inventory.py [--after]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "revision" / "manuscript"
SOURCES = {
    "assemble_manuscript.py": ROOT / "revision" / "scripts" / "assemble_manuscript.py",
    "F4_manuscript_changes.md": MAN / "F4_manuscript_changes.md",
    "F4_response_letter.md": MAN / "F4_response_letter.md",
    "F4_cover_letter_R1.md": ROOT / "revision" / "F4_cover_letter_R1.md",
}
DOCX = {
    "manuscript": MAN / "KBS_Manuscript_R1.docx",
    "marked": MAN / "KBS_Manuscript_R1_marked.docx",
    "response": MAN / "KBS_Response_R1.docx",
    "supplementary": MAN / "KBS_Supplementary_R1.docx",
    "highlights": MAN / "KBS_Highlights_R1.docx",
    "cover": MAN / "KBS_Cover_Letter_R1.docx",
}
KW_OLD = "Keywords: Personalized nutrition; Wearable devices; Explainable AI; Decision support systems; SHAP; Physiological state classification; mHealth"
KW_NEW = ("Keywords: Personalized nutrition; Wearable devices; Multimodal physiological data; Knowledge-based decision support; "
          "Explainable artificial intelligence (XAI); Decision support systems; SHAP")
FUNDING = "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors."

# (label, string, expected counts {location: n}); locations not listed are reported only
BEFORE = [
    ("keywords line (docx)", KW_OLD, {"manuscript": 1, "marked": 1}),
    ("keywords line (F4 source)", "Keywords (1-7, KBS): Personalized nutrition; Wearable devices; Explainable AI; Decision support systems; SHAP; Physiological state classification; mHealth",
     {"F4_manuscript_changes.md": 1}),
    ("General changes bullet on keywords", 'One keyword was changed ("Random Forest" to "Decision support systems").',
     {"F4_response_letter.md": 1, "response": 1}),
    ("cover letter signatory", "Catalin Buiu, on behalf of", {"F4_cover_letter_R1.md": 1, "cover": 1}),
    ("cover letter e-mail", "***REMOVED-EMAIL***", {"F4_cover_letter_R1.md": 1, "cover": 1}),
    ("CRediT comment", "confirm roles with coauthor", {"manuscript#comments": 1, "marked#comments": 1}),
    ("Funding comment", "confirm no funding with coauthor", {"manuscript#comments": 1, "marked#comments": 1}),
    ("Funding paragraph", FUNDING, {"F4_manuscript_changes.md": 1, "manuscript": 1, "marked": 1}),
    ("Acknowledgements", "Acknowledgements", {k: 0 for k in DOCX}),
    ("Data availability heading", "Data availability", {"manuscript": 1, "marked": 1}),
]
AFTER = [
    ("keywords line (docx, em7)", KW_NEW, {"manuscript": 1, "marked": 1}),
    ("old keywords line (docx)", KW_OLD, {k: 0 for k in DOCX}),
    ("keywords line (F4 source, em7)", "Keywords (1-7, KBS): Personalized nutrition; Wearable devices; Multimodal physiological data; Knowledge-based decision support; Explainable artificial intelligence (XAI); Decision support systems; SHAP",
     {"F4_manuscript_changes.md": 1}),
    ("General changes bullet on keywords (new)", "Keywords aligned with the Editorial Manager record of the original submission, completed with Decision support systems and SHAP (7)",
     {"F4_response_letter.md": 1, "response": 1}),
    ("General changes bullet on keywords (old)", 'One keyword was changed ("Random Forest" to "Decision support systems").', {"F4_response_letter.md": 0, "response": 0}),
    ("cover letter signatory (old)", "Catalin Buiu, on behalf of", {"F4_cover_letter_R1.md": 0, "cover": 0}),
    ("cover letter signatory (new)", "on behalf of the authors Adrian Lazar and Catalin Buiu", {"F4_cover_letter_R1.md": 1, "cover": 1}),
    ("cover letter e-mail (old)", "***REMOVED-EMAIL***", {"F4_cover_letter_R1.md": 0, "cover": 0}),
    ("cover letter e-mail (new)", "***REMOVED-EMAIL***", {"F4_cover_letter_R1.md": 1, "cover": 1}),
    ("CRediT comment", "confirm roles with coauthor", {"manuscript#comments": 0, "marked#comments": 0}),
    ("Funding comment", "confirm no funding with coauthor", {"manuscript#comments": 0, "marked#comments": 0}),
    ("Funding paragraph", FUNDING, {"F4_manuscript_changes.md": 1, "manuscript": 1, "marked": 1}),
    ("Acknowledgements", "Acknowledgements", {k: 0 for k in DOCX}),
    ("Data availability heading", "Data availability", {"manuscript": 1, "marked": 1}),
]


def docx_units(path: Path):
    """(location suffix, index, text) for body paragraphs, table cells and comments."""
    d = docx.Document(str(path))
    units = []
    for i, p in enumerate(d.paragraphs):
        units.append(("", i, p.text))
    for ti, t in enumerate(d.tables):
        for ri, row in enumerate(t.rows):
            for ci, cell in enumerate(row.cells):
                units.append(("#tables", f"t{ti}r{ri}c{ci}", cell.text))
    for ci, c in enumerate(d.comments):
        units.append(("#comments", ci, c.text))
    return units, d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", action="store_true", help="check the post-edit expectations (T11b.1-T11b.4)")
    args = ap.parse_args()
    targets = AFTER if args.after else BEFORE
    src_lines = {k: p.read_text(encoding="utf-8").splitlines() for k, p in SOURCES.items()}
    units = {}
    docs = {}
    for k, p in DOCX.items():
        units[k], docs[k] = docx_units(p)
    fails = 0
    rows = []
    for label, s, expected in targets:
        hits = {}
        for k, lines in src_lines.items():
            for ln, line in enumerate(lines, 1):
                if s in line:
                    hits.setdefault(k, []).append((ln, line.strip()))
        for k, us in units.items():
            for suffix, idx, text in us:
                if s in text:
                    hits.setdefault(k + suffix, []).append((idx, text.strip()))
        counts = {loc: len(v) for loc, v in hits.items()}
        ok = all(counts.get(loc, 0) == n for loc, n in expected.items())
        fails += not ok
        rows.append((label, s, expected, counts, ok))
        print(f"\n[{'PASS' if ok else 'FAIL'}] {label}: expected {expected}; found {counts or '{}'}")
        for loc, lst in hits.items():
            judged = loc in expected
            for idx, text in lst:
                print(f"    {loc}{'' if judged else ' (not judged)'} @ {idx}: {text[:300]}")
    # Data availability paragraph, verbatim (T11b.4)
    print("\n--- Data availability paragraph (manuscript), verbatim ---")
    pars = docs["manuscript"].paragraphs
    for i, p in enumerate(pars):
        if p.text.strip() == "Data availability":
            print(f"    manuscript @ {i}: {p.text}")
            print(f"    manuscript @ {i + 1}: {pars[i + 1].text}")
    # section order after Conclusions (T11b.3): the unnumbered headings are 'Normal' paragraphs holding exactly the title
    print("\n--- section order after Conclusions (manuscript and marked copy) ---")
    expected_order = ["6. Conclusions and Future Work", "CRediT authorship contribution statement", "Declaration of competing interest",
                      "Funding", "Declaration of generative AI and AI-assisted technologies in the manuscript preparation", "Data availability",
                      "Appendix A. Hyperparameter search", "References"]
    for k in ("manuscript", "marked"):
        ps = docs[k].paragraphs
        found = [(i, p.text.strip()) for i, p in enumerate(ps) if p.text.strip() in expected_order or p.text.strip().startswith("Declaration of generative AI")]
        order = [t if t in expected_order else "Declaration of generative AI and AI-assisted technologies in the manuscript preparation" for _, t in found]
        ok = order == expected_order
        fails += not ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {k}: " + " > ".join(f"{t[:28]}@{i}" for i, t in found))
    print("\n--- Word comments per docx ---")
    for k, d in docs.items():
        cs = list(d.comments)
        print(f"    {k}: {len(cs)}" + (" (" + "; ".join(c.text[:40] for c in cs) + ")" if cs else ""))
    print("\n--- summary table ---")
    print("| target | expected | found | result |\n|---|---|---|---|")
    for label, s, expected, counts, ok in rows:
        print(f"| {label} | {expected} | {counts or '{}'} | {'PASS' if ok else 'FAIL'} |")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAIL'} ({'after' if args.after else 'before'} edits)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
