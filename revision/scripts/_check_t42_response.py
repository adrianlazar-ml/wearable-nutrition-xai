"""T4.2-A check: the response letter docx has no em dash and no placeholder, and every "Section N.N", "Table N" and
"Figure N" it cites exists as a heading / caption / legend in KBS_Manuscript_R1.docx (Supplementary S-items in
KBS_Supplementary_R1.docx). Also checks that the citation list and the removed-reference list in the letter match
assembly_plan.md.

Usage (from the repo root): python revision/scripts/_check_t42_response.py
Exit 1 on any failure.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "revision" / "manuscript"
RESP = MAN / "KBS_Response_R1.docx"
MANU = MAN / "KBS_Manuscript_R1.docx"
SUPP = MAN / "KBS_Supplementary_R1.docx"
PLAN = MAN / "assembly_plan.md"

EXPECTED_CITATIONS = ["Breiman, 2001", "Chen & Guestrin, 2016", "Vapnik, 1995", "Chawla et al., 2002", "Lundberg & Lee, 2017",
                      "Lundberg et al., 2020", "Shapley, 1953/2016", "Zheng et al., 2024", "Thapa et al., 2026"]
EXPECTED_REMOVED = ["De La Torre, K.", "Ferreira, R. E. P.", "Gu, L., & Qian, C.", "Liang, Y.-T."]


def doc_text(path: Path) -> str:
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    parts += [" | ".join(c.text for c in r.cells) for t in d.tables for r in t.rows]
    parts += [c.text for c in d.comments]
    return "\n".join(parts)


def main() -> None:
    fails = 0

    def rec(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    resp = doc_text(RESP)
    manu_d = docx.Document(str(MANU))
    manu_pars = [p.text for p in manu_d.paragraphs]
    manu = "\n".join(manu_pars)
    supp = doc_text(SUPP)

    rec("Response: 0 em dash", "—" not in resp, f"{resp.count(chr(0x2014))} found")
    ph = re.findall(r"\b3\.x\b|\[Table|\[Figure|\[t\d*|\[Section|\[title as submitted\]", resp)
    rec("Response: 0 placeholders", not ph, ", ".join(ph))

    headings = {m.group(1) for t in manu_pars for m in [re.match(r"(\d+(?:\.\d+)?)\. ", t)] if m}
    captions = {m.group(1) for t in manu_pars for m in [re.match(r"Table (A?\d+)\. ", t)] if m}
    legends = {m.group(1) for t in manu_pars for m in [re.match(r"Figure (\d+)\. ", t)] if m}
    supp_items = set(re.findall(r"\b(Table S\d|Figure S\d)\. ", supp))

    sec_refs = set()
    for m in re.finditer(r"\bSections? (\d+(?:\.\d+)?(?:(?:,| and| to) \d+(?:\.\d+)?)*)", resp):
        sec_refs.update(re.findall(r"\d+(?:\.\d+)?", m.group(1)))
    tab_refs = set()
    for m in re.finditer(r"\bTables? ((?:A?\d+)(?:,? (?:and )?A?\d+)*)", resp):
        tab_refs.update(re.findall(r"A?\d+", m.group(1)))
    fig_refs = set()
    for m in re.finditer(r"\bFigures? (\d+(?:,? (?:and )?\d+)*)", resp):
        fig_refs.update(re.findall(r"\d+", m.group(1)))
    supp_refs = set(re.findall(r"Supplementary (Table S\d|Figure S\d)", resp)) | set(re.findall(r"\b(Figure S\d|Table S\d)\b", resp))

    bad_sec = sorted(r for r in sec_refs if r not in headings)
    bad_tab = sorted(r for r in tab_refs if r not in captions)
    bad_fig = sorted(r for r in fig_refs if r not in legends)
    bad_sup = sorted(r for r in supp_refs if r not in supp_items)
    rec(f"Response: all {len(sec_refs)} distinct Section references exist as manuscript headings", not bad_sec, "missing: " + ", ".join(bad_sec))
    rec(f"Response: all {len(tab_refs)} distinct Table references exist as manuscript captions", not bad_tab, "missing: " + ", ".join(bad_tab))
    rec(f"Response: all {len(fig_refs)} distinct Figure references exist as manuscript legends", not bad_fig, "missing: " + ", ".join(bad_fig))
    rec(f"Response: all {len(supp_refs)} distinct Supplementary references exist in the Supplementary file", not bad_sup, "missing: " + ", ".join(bad_sup))

    # citation list and removed references vs assembly_plan.md
    plan = PLAN.read_text(encoding="utf-8")
    dev = [l for l in plan.splitlines() if l.startswith("- base = v2 text with v8 citations")]
    plan_cites = re.search(r"in-text citations (.*?)\. The v8 title", dev[0]).group(1) if dev else ""
    n_in_plan = len(re.findall(r", \d{4}(?:/\d{4})?(?:,|$)", plan_cites + ","))
    ok = all(c in plan_cites for c in EXPECTED_CITATIONS) and n_in_plan == len(EXPECTED_CITATIONS)
    rec("Plan v8 citation list equals the expected nine", ok, f"{n_in_plan} citations in the plan line")
    letter_bullet = [l for l in resp.splitlines() if l.startswith("In-text citations were added")]
    ok = bool(letter_bullet) and all(c.replace(" & ", " and ") in letter_bullet[0] for c in EXPECTED_CITATIONS) and "four entries" in letter_bullet[0]
    rec("Letter bullet lists the same nine citations (with 'and') and the four removed entries", ok)
    removed_line = [l for l in plan.splitlines() if l.startswith("- References not cited anywhere")]
    ok = bool(removed_line) and all(r.split(",")[0] in removed_line[0] for r in EXPECTED_REMOVED) and "Kept although uncited: none" in removed_line[0]
    rec("Plan removed-reference list is De La Torre, Ferreira, Gu and Qian, Liang and nothing kept uncited", ok)
    rec("Letter Comment 4: 'Section 3.5 extended with the search procedure' present, 'New Section 3.5' absent",
        "Section 3.5 extended with the search procedure" in resp and "New Section 3.5" not in resp)

    print(f"{'ALL PASS' if not fails else str(fails) + ' FAIL'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
