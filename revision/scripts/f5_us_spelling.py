"""T5.3b (F5): British -> American spelling in the NEW revision text (F4 markdown sources).

The manuscript base (v8) is in American English; the text written in F4 mixed in British forms. This script
replaces the forms listed in PAIRS in revision/manuscript/F4_manuscript_changes.md and
revision/manuscript/F4_response_letter.md, prints every match with its context BEFORE replacing (the list goes
into revision/F5_format_report.md), and skips:
  - reviewer quotes in the response letter (markdown lines starting with ">"),
  - reference titles (none of the F4 lines are reference entries; the base-manuscript reference titles are not
    touched by this script at all because they live in BASE_MANUSCRIPT_v8.docx),
  - file names and variable names (no PAIRS form occurs in one; checked by the listing).
The docx files are then regenerated with assemble_manuscript.py (which also carries the two British strings
hard-coded for the Supplementary file and the 'survival modelling' base sentence; see T5.3b in the report).

Usage (repo root):  python revision/scripts/f5_us_spelling.py [--dry-run]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = [ROOT / "revision" / "manuscript" / "F4_manuscript_changes.md",
         ROOT / "revision" / "manuscript" / "F4_response_letter.md"]

# lower-case UK -> US; capitalised forms are handled by case preservation
PAIRS = {
    "characterise": "characterize", "characterised": "characterized", "characterises": "characterizes",
    "characterising": "characterizing", "characterisation": "characterization",
    "personalisation": "personalization", "personalised": "personalized", "personalise": "personalize",
    "modelling": "modeling", "modelled": "modeled",
    "labelling": "labeling", "labelled": "labeled",
    "behaviour": "behavior", "behaviours": "behaviors", "behavioural": "behavioral",
    "optimisation": "optimization", "optimise": "optimize", "optimised": "optimized", "optimising": "optimizing",
    "standardise": "standardize", "standardised": "standardized", "standardisation": "standardization",
    "analyse": "analyze", "analysed": "analyzed", "analysing": "analyzing",
    "summarise": "summarize", "summarised": "summarized", "summarising": "summarizing",
    "emphasise": "emphasize", "emphasised": "emphasized", "emphasising": "emphasizing",
    "hypothesise": "hypothesize", "hypothesised": "hypothesized",
    "prioritise": "prioritize", "prioritised": "prioritized",
    "generalisation": "generalization", "generalise": "generalize", "generalised": "generalized",
    "normalisation": "normalization", "normalised": "normalized", "visualisation": "visualization",
    "minimise": "minimize", "minimised": "minimized", "maximise": "maximize", "maximised": "maximized",
    "utilise": "utilize", "utilised": "utilized", "recognise": "recognize", "recognised": "recognized",
    "organise": "organize", "organised": "organized", "categorise": "categorize", "categorised": "categorized",
    "stabilise": "stabilize", "stabilised": "stabilized", "realise": "realize", "realised": "realized",
    "centre": "center", "centres": "centers", "colour": "color", "colours": "colors",
    "favour": "favor", "favourable": "favorable", "favoured": "favored",
    "neighbouring": "neighboring", "neighbour": "neighbor", "neighbours": "neighbors",
    "artefact": "artifact", "artefacts": "artifacts",
    "programme": "program", "whilst": "while", "amongst": "among", "grey": "gray",
}
PATTERN = re.compile(r"\b(" + "|".join(sorted(map(re.escape, PAIRS), key=len, reverse=True)) + r")\b", re.I)


def us(word: str) -> str:
    rep = PAIRS[word.lower()]
    if word.isupper():
        return rep.upper()
    if word[0].isupper():
        return rep[0].upper() + rep[1:]
    return rep


def main(dry_run: bool) -> None:
    total = 0
    for f in FILES:
        lines = f.read_text(encoding="utf-8").split("\n")
        out, n_file, skipped = [], 0, 0
        print(f"===== {f.relative_to(ROOT)}")
        for i, line in enumerate(lines, 1):
            if line.lstrip().startswith(">"):                   # reviewer quote: never changed
                for m in PATTERN.finditer(line):
                    skipped += 1
                    print(f"  [line {i}] SKIP (reviewer quote) {m.group(0)!r}")
                out.append(line)
                continue
            def sub(m):
                nonlocal n_file
                n_file += 1
                s, e = m.start(), m.end()
                print(f"  [line {i}] {m.group(0)!r} -> {us(m.group(0))!r}: ...{line[max(0, s - 45):e + 45]}...")
                return us(m.group(0))
            out.append(PATTERN.sub(sub, line))
        print(f"  -> {n_file} replacements, {skipped} skipped in reviewer quotes")
        total += n_file
        if not dry_run and n_file:
            f.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"TOTAL {total} replacements" + (" (dry run, nothing written)" if dry_run else ""))


if __name__ == "__main__":
    main("--dry-run" in sys.argv[1:])
