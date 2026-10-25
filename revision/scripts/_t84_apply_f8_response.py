"""T8.4 (2026-10-25): apply the validated F8 response-letter paragraphs (revision/f8/f8_response_new.txt) to the source
revision/manuscript/F4_response_letter.md. One-off, kept as evidence; stops at the first paragraph without a unique
counterpart and changes nothing in that case.

f8_response_new.txt holds 28 paragraphs in letter order: the introductory paragraph, then for each of the 13 reviewer
comments (R1.1-R1.10, R2 Weakness 1-3) one 'Response.' paragraph and one 'Changes.' paragraph, with the second Response
paragraph of R1.9 ('In the course of this analysis...') between its Response and Changes. The F8 text carries the FINAL
numbering of the assembled manuscript; F4_response_letter.md is resolved by renumber() at assembly, so the final labels
are mapped back to the F4 labels and the round trip renumber(F4) == F8 is asserted for every paragraph.
Also adds, under 'General changes not prompted by a specific comment', the bullet on the style alignment (T8 brief).

Usage (repo root): python revision/scripts/_t84_apply_f8_response.py [--dry-run]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "scripts"))
from assemble_manuscript import renumber  # noqa: E402
from _t82_apply_f8 import inv, NUM_RE  # noqa: E402  (same inverse mapping as the manuscript)

LETTER = ROOT / "revision" / "manuscript" / "F4_response_letter.md"
F8 = ROOT / "revision" / "f8" / "f8_response_new.txt"
NEW_BULLET = ("- The wording of the text added in revision was aligned with the style of the submitted manuscript (sentence length, "
              "impersonal voice, capitalization of model names); no figure, claim or qualification was changed.")
STARE_NOTE = (" T8.4 (25 oct 2026): paragraful introductiv si toate perechile Response / Changes inlocuite cu textul validat din "
              "revision/f8/f8_response_new.txt (stil v8; etichetele finale mapate inapoi la etichetele F4 de revision/scripts/_t84_apply_f8_response.py); "
              "antetele, citatele din review si linia Authors neschimbate; bullet nou in 'General changes' despre alinierea stilistica.")


def inv_letter(text: str) -> str:
    text = text.replace("Tables 4, 6, 9 and 11", "Tables 4, A, 7 and 8")  # special case of renumber()
    return inv(text)


def main(dry_run: bool) -> None:
    f8 = [p.strip() for p in F8.read_text(encoding="utf-8").split("\n") if p.strip()]
    if len(f8) != 28:
        raise SystemExit(f"STOP: f8_response_new.txt has {len(f8)} paragraphs, expected 28")
    lines = LETTER.read_text(encoding="utf-8").split("\n")
    # authors' paragraphs of the source, in order: intro, then Response. / Changes. / 'In the course...' lines
    idx = [i for i, l in enumerate(lines) if l.startswith(("We thank the editor", "Response. ", "Changes. ", "In the course of this analysis"))]
    if len(idx) != 28:
        raise SystemExit(f"STOP: F4_response_letter.md has {len(idx)} authors' paragraphs, expected 28: {idx}")
    kinds_f4 = [lines[i].split(" ")[0] for i in idx]
    kinds_f8 = [p.split(" ")[0] for p in f8]
    if kinds_f4 != kinds_f8:
        raise SystemExit(f"STOP: paragraph kinds differ between F4 and F8:\n F4 {kinds_f4}\n F8 {kinds_f8}")
    # headings between paragraphs: each F8 pair must sit under the same '### Comment' heading as the F4 pair it replaces
    print("| # | heading above | kind | F4 line | words old -> new | numbers old \\ new | numbers new \\ old |")
    print("|---|---|---|---|---|---|---|")
    new_lines = list(lines)
    for n, (i, p) in enumerate(zip(idx, f8), 1):
        heading = next((lines[k] for k in range(i, -1, -1) if lines[k].startswith("### ") or lines[k].startswith("## ")), "")
        old = lines[i]
        new = inv_letter(p)
        rt = renumber(new)
        if rt != p:
            j = next((x for x in range(min(len(rt), len(p))) if rt[x] != p[x]), min(len(rt), len(p)))
            raise SystemExit(f"STOP: round trip differs for paragraph {n} ({heading}) at char {j}:\n  got: {rt[j-40:j+60]}\n  exp: {p[j-40:j+60]}")
        n_old, n_new = {t.rstrip('.,') for t in NUM_RE.findall(old)}, {t.rstrip('.,') for t in NUM_RE.findall(new)}
        print(f"| {n} | {heading[4:60]} | {new.split(' ')[0]} | {i+1} | {len(old.split())} -> {len(new.split())} | {sorted(n_old - n_new)} | {sorted(n_new - n_old)} |")
        new_lines[i] = new
    # General changes bullet (after the last existing bullet of that section)
    gi = next(i for i, l in enumerate(new_lines) if l.startswith("## General changes not prompted"))
    last_bullet = max(i for i in range(gi, len(new_lines)) if new_lines[i].startswith("- "))
    if any(l.startswith("- The wording of the text added in revision") for l in new_lines):
        raise SystemExit("STOP: the style-alignment bullet is already present")
    new_lines.insert(last_bullet + 1, NEW_BULLET)
    print(f"| - | General changes | bullet | {last_bullet + 2} | new bullet ({len(NEW_BULLET.split())} words) | | |")
    si = next(i for i, l in enumerate(new_lines) if l.startswith("Stare:"))
    new_lines[si] = new_lines[si].rstrip() + STARE_NOTE
    if dry_run:
        print("dry run: nothing written")
        return
    LETTER.write_text("\n".join(new_lines), encoding="utf-8", newline="\n")
    print(f"written: {LETTER}")


if __name__ == "__main__":
    main("--dry-run" in sys.argv[1:])
