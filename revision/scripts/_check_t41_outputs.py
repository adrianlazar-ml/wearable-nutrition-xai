"""T4.1 structural checks on the assembled files (Pass criteria of Task 3 of the T4.1-A brief).

Usage (from the repo root): python revision/scripts/_check_t41_outputs.py
Writes revision/manuscript/t41_outputs_check.md and exits 1 if any check fails.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "revision" / "manuscript"
BASE_SHA = "6cfe1763e7dca2b4a54d9d5cad02d07089fdf695a317bc8d68d721b5c36017f6"  # ESWA_Manuscript_Anonymized_v2.docx
DOCS = {"manuscript": MAN / "KBS_Manuscript_R1.docx", "highlights": MAN / "KBS_Highlights_R1.docx",
        "supplementary": MAN / "KBS_Supplementary_R1.docx", "response": MAN / "KBS_Response_R1.docx",
        "marked": MAN / "KBS_Manuscript_R1_marked.docx"}
PLACEHOLDER = re.compile(r"\[t\d+\]|\[Section 4\.x\]|\b3\.x\b|\[Figure tNN\]|\[Table X\]|\[title as submitted\]|\[t11\]")
RESULTS: list[tuple[str, bool, str]] = []


def rec(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def body(d):
    out = []
    for child in d.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            out.append(("p", Paragraph(child, d)))
        elif tag == "tbl":
            out.append(("t", Table(child, d)))
    return out


def all_text(d) -> str:
    parts = []
    for kind, el in body(d):
        if kind == "p":
            parts.append(el.text)
        else:
            for row in el.rows:
                parts.append(" | ".join(c.text for c in row.cells))
    for c in d.comments:
        parts.append(c.text)
    return "\n".join(parts)


def main() -> None:
    docs = {}
    for k, p in DOCS.items():
        try:
            docs[k] = docx.Document(str(p))
            rec(f"{p.name} opens with python-docx", True, f"{len(docs[k].paragraphs)} paragraphs, {len(docs[k].tables)} tables")
        except Exception as e:  # noqa: BLE001
            rec(f"{p.name} opens with python-docx", False, str(e))
    if len(docs) < 5:
        finish(); return
    # marked copy: same body text as the manuscript (before the appended "Removed" section), highlights present
    mk_pars = [p.text for p in docs["marked"].paragraphs]
    cut = mk_pars.index("Removed from the submitted version") if "Removed from the submitted version" in mk_pars else len(mk_pars)
    rec("Marked copy: body text identical to KBS_Manuscript_R1.docx", mk_pars[:cut] == [p.text for p in docs["manuscript"].paragraphs])
    n_hl = sum(1 for p in docs["marked"].paragraphs for r in p.runs if r.font.highlight_color is not None)
    rec("Marked copy: highlighted runs present and a 'Removed from the submitted version' section exists", n_hl > 0 and cut < len(mk_pars), f"{n_hl} highlighted runs, {len(mk_pars) - cut - 1} removed items")

    man = docs["manuscript"]
    items = body(man)
    pars = [el for k, el in items if k == "p"]

    # abstract word count
    idx = next(i for i, p in enumerate(pars) if p.text.strip() == "Abstract")
    abstract = pars[idx + 1].text
    n_words = len(abstract.split())
    rec("Abstract under 250 words", n_words <= 250, f"{n_words} words")

    # highlights
    hl = [p.text for p in docs["highlights"].paragraphs if p.style.name == "List Paragraph" and p.text.strip()]
    rec("Highlights: 3 to 5 bullets", 3 <= len(hl) <= 5, f"{len(hl)} bullets")
    rec("Highlights: each at most 85 characters", all(len(h) <= 85 for h in hl), "; ".join(str(len(h)) for h in hl))

    # em dash and placeholders in every docx
    for k, d in docs.items():
        t = all_text(d)
        rec(f"{DOCS[k].name}: 0 em dash (U+2014)", "—" not in t, f"{t.count(chr(0x2014))} found")
        ph = PLACEHOLDER.findall(t)
        rec(f"{DOCS[k].name}: 0 placeholders", not ph, ", ".join(ph[:10]))
        if k in ("manuscript", "marked"):
            body_t = t.split("Removed from the submitted version")[0] if k == "marked" else t
            rec(f"{DOCS[k].name}: 0 'canonical' in the text", len(re.findall(r"canonical", body_t, re.I)) == 0, f"{len(re.findall(r'canonical', body_t, re.I))} found")
            rec(f"{DOCS[k].name}: 0 paragraphs containing 'Internal'", "Internal" not in t)

    # tables: captions in order, cited in text, first citation in order
    text_pars = [p.text for p in pars]
    full = "\n".join(text_pars)
    caps = [re.match(r"Table (A?\d+)\.", t).group(1) for t in text_pars if re.match(r"Table (A?\d+)\. ", t)]
    main_caps = [c for c in caps if not c.startswith("A")]
    rec("Table captions numbered consecutively 1..N in order of appearance", main_caps == [str(i) for i in range(1, len(main_caps) + 1)], " ".join(main_caps))
    rec("Appendix table caption A1 present", "A1" in caps, " ".join(caps))
    body_wo_caps = "\n".join(t for t in text_pars if not re.match(r"Table (A?\d+)\. ", t))
    first_cite = {}
    for m in re.finditer(r"\bTables? ((?:A?\d+)(?:,? (?:and )?A?\d+)*)", body_wo_caps):
        for n in re.findall(r"A?\d+", m.group(1)):
            first_cite.setdefault(n, m.start())
    missing = [c for c in caps if c not in first_cite]
    rec("Every table cited in the text at least once", not missing, "missing: " + ", ".join(missing) if missing else "")
    order = [c for c in sorted(main_caps, key=lambda c: first_cite.get(c, 1e9))]
    rec("Tables first cited in numeric order", order == main_caps, " ".join(order))
    n_tables = sum(1 for k, _ in items if k == "t")
    rec("Number of Word tables equals number of captions", n_tables == len(caps), f"{n_tables} tables, {len(caps)} captions")

    # figures: legends 1..7 in order, cited before/at legend in order
    legs = [re.match(r"Figure (\d+)\.", t).group(1) for t in text_pars if re.match(r"Figure (\d+)\. ", t)]
    rec("Figure legends numbered consecutively 1..N in order of appearance", legs == [str(i) for i in range(1, len(legs) + 1)], " ".join(legs))
    body_wo_legs = "\n".join(t for t in text_pars if not re.match(r"Figure (\d+)\. ", t))
    fcite = {}
    for m in re.finditer(r"\bFigures? (\d+(?:,? (?:and )?\d+)*)", body_wo_legs):
        for n in re.findall(r"\d+", m.group(1)):
            fcite.setdefault(n, m.start())
    missing = [c for c in legs if c not in fcite]
    rec("Every figure cited in the text at least once", not missing, "missing: " + ", ".join(missing) if missing else "")
    order = sorted(legs, key=lambda c: fcite.get(c, 1e9))
    rec("Figures first cited in numeric order", order == legs, " ".join(order))
    rec("Figure files present for Figures 1..7 (pdf + png)", all((MAN / "figures").glob(f"Figure{i}_*.{ext}") and any((MAN / "figures").glob(f"Figure{i}_*.{ext}")) for i in range(1, 8) for ext in ("pdf", "png")))

    # supplementary cross-references
    sup_text = all_text(docs["supplementary"])
    for lab in ("Table S1", "Table S2", "Table S3", "Figure S1", "Figure S2", "Figure S3"):
        rec(f"Main text cites Supplementary {lab} and the Supplementary file has it", f"Supplementary {lab}" in full and re.search(rf"{lab}\. ", sup_text) is not None)

    # section cross-references resolve to existing headings
    headings = {re.match(r"(\d+(?:\.\d+)?)\.", p.text).group(1) for p in pars if p.style.name.startswith("Heading") and re.match(r"\d+(\.\d+)?\.", p.text)}
    refs = set()
    for m in re.finditer(r"\bSections? (\d+(?:\.\d+)?(?:(?:,| and| to) \d+(?:\.\d+)?)*)", full):
        refs.update(re.findall(r"\d+(?:\.\d+)?", m.group(1)))
    bad = sorted(r for r in refs if r not in headings)
    rec("All 'Section x.y' cross-references point to existing headings", not bad, "unknown: " + ", ".join(bad) if bad else f"{len(refs)} distinct references")
    exp_h2 = [f"3.{i}" for i in range(1, 14)] + [f"4.{i}" for i in range(1, 13)]
    got_h2 = [re.match(r"(\d+\.\d+)\.", p.text).group(1) for p in pars if p.style.name == "Heading 2" and re.match(r"\d+\.\d+\.", p.text)]
    rec("Subsection headings are 3.1-3.13 and 4.1-4.12 in order", got_h2 == exp_h2, " ".join(got_h2))

    # table style: no vertical borders, no shading
    for k, d in docs.items():
        bad = 0
        for t in d.tables:
            x = t._tbl.xml
            if re.search(r'<w:shd [^>]*w:fill="(?!auto)', x):
                bad += 1
            tb = re.search(r"<w:tblBorders>.*?</w:tblBorders>", x, re.S)
            if not tb or not all(f'<w:{e} w:val="nil"' in tb.group(0) for e in ("left", "right", "insideV")):
                bad += 1
        rec(f"{DOCS[k].name}: tables without vertical lines and shading", bad == 0, f"{len(d.tables)} tables, {bad} non-conforming")

    # every reference entry cited in the text (surname followed by the year within 80 characters)
    ref_started, refs, body_txt = False, [], []
    for kind, el in items:
        if kind == "p":
            if el.text.strip() == "References":
                ref_started = True; continue
            (refs if ref_started else body_txt).append(el.text)
        elif kind == "t" and not ref_started:
            body_txt.extend(" ".join(c.text for c in r.cells) for r in el.rows)
    btxt = "\n".join(body_txt)
    uncited = []
    for rt in refs:
        if not rt.strip():
            continue
        surname = rt.split(",")[0].strip()
        m = re.search(r"\((\d{4})", rt)
        if not m or not re.search(re.escape(surname) + r"[^)]{0,80}?" + m.group(1), btxt):
            uncited.append(rt[:40])
    rec("Every reference entry is cited in the text", not uncited, f"{len([r for r in refs if r.strip()])} entries; uncited: " + "; ".join(uncited))

    # comments on new references
    cm = [c for c in man.comments if "verify in Zotero" in c.text]
    rec("6 Word comments 'verify in Zotero' on the new references", len(cm) == 6, f"{len(cm)} comments")

    # response letter structure
    rtext = [p.text for p in docs["response"].paragraphs]
    rec("Response letter: 10 Reviewer 1 comments and 3 Reviewer 2 weaknesses", sum(t.startswith("Comment ") for t in rtext) == 10 and sum(t.startswith("Weakness ") for t in rtext) == 3)

    # base unchanged
    sha = hashlib.sha256((MAN / "BASE_MANUSCRIPT.docx").read_bytes()).hexdigest()
    rec("BASE_MANUSCRIPT.docx unchanged (sha256 of the submitted v2)", sha == BASE_SHA, sha[:16])

    # git: tracked files in revision/ outside manuscript/ and scripts/ unchanged (requirements.txt pin excepted)
    st = subprocess.run(["git", "status", "--porcelain", "--", "revision/", "CIFRE_CANONICE.md"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    others = [l for l in st if not l[3:].startswith(("revision/manuscript/", "revision/scripts/"))
              and l[3:] not in ("revision/requirements.txt", "revision/CIFRE_REVIZIE.md")]
    rec("Git: no changes in tracked revision/ files outside manuscript/, scripts/ (requirements.txt pin and the CIFRE_REVIZIE T0.4 row excepted)", not others, "; ".join(others))
    for f in ("CIFRE_CANONICE.md", "revision/TEXT_DECISIONS.md"):
        r = subprocess.run(["git", "diff", "--quiet", "--", f], cwd=ROOT)
        rec(f"Git diff empty on {f}", r.returncode == 0)
    # CIFRE_REVIZIE.md: either unchanged or changed only by the T0.4 persistence per-class row (T4.1-B)
    d = subprocess.run(["git", "diff", "--", "revision/CIFRE_REVIZIE.md"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    changed = [l for l in d.splitlines() if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---"))]
    ok = not changed or (len(changed) == 1 and changed[0].startswith("+| T0.4 | Persistence F1 Rest / F1 Normal"))
    rec("Git diff on revision/CIFRE_REVIZIE.md empty or limited to the T0.4 persistence per-class row", ok, f"{len(changed)} changed lines")
    cif = (ROOT / "revision" / "CIFRE_REVIZIE.md").read_text(encoding="utf-8")
    rec("CIFRE_REVIZIE.md has the T0.4 row for persistence F1 Rest / F1 Normal (0.205 / 0.707)", "| T0.4 | Persistence F1 Rest / F1 Normal | 0.205 / 0.707 |" in cif)
    rec("LF line endings: CIFRE_REVIZIE.md", b"\r\n" not in (ROOT / "revision" / "CIFRE_REVIZIE.md").read_bytes())

    # LF on produced text files
    for f in list(MAN.glob("*.md")) + [ROOT / "revision" / "scripts" / n for n in ("assemble_manuscript.py", "_check_t41_numbers.py", "_check_t41_outputs.py")]:
        rec(f"LF line endings: {f.name}", b"\r\n" not in f.read_bytes())

    finish()


def finish() -> None:
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    L = ["# T4.1 outputs check", "", f"{len(RESULTS)} checks, {n_fail} FAIL.", "", "| Check | Result | Detail |", "|---|---|---|"]
    for name, ok, detail in RESULTS:
        L.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    (MAN / "t41_outputs_check.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(RESULTS)} checks, {n_fail} FAIL -> revision/manuscript/t41_outputs_check.md")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
