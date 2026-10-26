"""T9.1 T0 inventory (2026-10-26): where do the provenance / complementarity formulations appear in the three
deliverables (manuscript, highlights, response letter), by section. Read-only; prints one line per occurrence.

Usage (from the repo root): python revision/scripts/_check_t91_inventory.py [path/to/dir-with-docx]
The optional argument points to another folder holding the three docx (the T0 copies), default revision/manuscript/.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
MAN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "revision" / "manuscript"
PATTERNS = ["two devices", "both devices", "device-independent", "watch indicators", "ring and watch",
            "watch mean heart rate", "96% to 100%", "96 % to 100 %"]
RX = re.compile("|".join(re.escape(p) for p in PATTERNS), re.I)


def body(d):
    for child in d.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            yield "p", Paragraph(child, d)
        elif tag == "tbl":
            yield "t", Table(child, d)


def manuscript_sections(d):
    """Yield (section label, text) for every paragraph / table row of the manuscript."""
    section = "Front matter"
    contrib = 0
    disc_par = 0
    for kind, el in body(d):
        if kind == "t":
            for row in el.rows:
                yield section + " (table)", " | ".join(c.text for c in row.cells)
            continue
        t = el.text.strip()
        st = el.style.name if el.style is not None else ""
        if not t:
            continue
        if st.startswith("Heading") or t in ("Abstract", "References", "Highlights"):
            section = t if len(t) < 60 else t[:57] + "..."
            contrib = 0
            disc_par = 0
            continue
        if section.startswith("1.") and st == "List Paragraph":
            contrib += 1
            yield f"Introduction contribution {contrib}", t
            continue
        if section.startswith("5.") or section.startswith("Discussion"):
            disc_par += 1
            yield f"{section} paragraph {disc_par}", t
            continue
        yield section, t


def response_sections(d):
    section = "Header"
    for kind, el in body(d):
        if kind == "t":
            continue
        t = el.text.strip()
        if not t:
            continue
        m = re.match(r"(Comment|Weakness) (\d+)\.", t)
        if m:
            section = ("R1." if m.group(1) == "Comment" else "R2.") + m.group(2)
            continue
        if t.startswith("General changes"):
            section = "General changes"
            continue
        yield section, t


def highlight_sections(d):
    n = 0
    for p in d.paragraphs:
        if p.style.name == "List Paragraph" and p.text.strip():
            n += 1
            yield f"Highlight {n}", p.text.strip()


def main() -> None:
    files = [("manuscript", MAN / "KBS_Manuscript_R1.docx", manuscript_sections),
             ("highlights", MAN / "KBS_Highlights_R1.docx", highlight_sections),
             ("response", MAN / "KBS_Response_R1.docx", response_sections)]
    n = 0
    print("| # | File | Section | Pattern | Exact string (context) |")
    print("|---|---|---|---|---|")
    for name, path, fn in files:
        d = docx.Document(str(path))
        for section, text in fn(d):
            for m in RX.finditer(text):
                n += 1
                s, e = max(0, m.start() - 45), min(len(text), m.end() + 45)
                ctx = ("..." if s > 0 else "") + text[s:e].replace("|", "/") + ("..." if e < len(text) else "")
                print(f"| {n} | {name} | {section} | {m.group(0)} | {ctx} |")
    print(f"\n{n} occurrences")


if __name__ == "__main__":
    main()
