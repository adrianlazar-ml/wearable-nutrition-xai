"""T8 (2026-10-25) parity check: the F8 style alignment changed wording only.

For the manuscript (T8.2): each of the 35 replaced paragraphs is located in the docx BEFORE T8 (taken from git, default
revision f6eefc0 = T7b) and in the current docx by its anchor (case-insensitive prefix, unique on both sides); the current
text must equal the F8 paragraph (revision/f8/*.txt; the three 'therefore' cuts of the brief applied) and the set of
numbers (regex [-+]?\\d[\\d,]*\\.?\\d* of the brief, trailing '.' / ',' stripped) must be identical, except the documented
exception (Generalization: citation years 2024 and 2025 of Zignoli et al. and Kim et al.). Section / Table / Figure labels
are compared separately and reported, not failed (Generalization adds the cross-reference 'Section 2'). Also checks:
0 lower-case 'random forest' and 0 'distinguish' in paragraphs (prose, captions, legends), 3.12 byte-identical (paragraph
XML between the 3.12 and 3.13 headings), 8 Word comments (9 until T11.2), Abstract <= 249 words (split on spaces), marked copy changed.
For the response letter (T8.4, --response): the 28 authors' paragraphs (intro, Response, Changes) in order must equal
revision/f8/f8_response_new.txt and the whole-document number set must be identical to the one before T8.
For Supplementary and Highlights (T8.3, --supp): 0 lower-case 'random forest', highlights <= 85 characters, number sets
identical to before.

Usage (repo root):
  python revision/scripts/_check_t8_parity.py [--before-rev f6eefc0] [--manuscript] [--response] [--supp]
Exit 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import docx
from docx.text.paragraph import Paragraph
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "revision" / "manuscript"
F8 = ROOT / "revision" / "f8"
NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
LABEL_RE = re.compile(r"\b(Section|Sections|Table|Tables|Figure|Figures|Limitation|Limitations|Comment|comment|Weakness)\s+(?:\(?[A-Z]?\d+(?:\.\d+)?\)?(?:,\s*|\s+and\s+|\s+to\s+)?)+")
ANCHORS = ["Personalized nutrition depends", "The system studied here", "A machine learning pipeline that", "An end-to-end explainable",
           "An evaluation methodology for", "The results of that evaluation", "Each day is assigned", "In addition to Random Forest",
           "Two baselines are reported", "Hyperparameters were searched", "Because macro F1, accuracy", "Calibration of the random forest",
           "The stability of the SHAP", "Complementing the per-device", "The pseudo-labels were examined", "To assess whether the single",
           "This paper presented and evaluated", "Future work should prioritize",
           "Table 4 reports", "Pairwise comparisons", "Table 6 reports", "Hyperparameter search (Appendix", "The raw random forest",
           "Table 8 and Figure 3", "Threshold sensitivity.", "Agreement with label", "Table 11 and Figure 4", "The SHAP analysis was replicated",
           "The confidence threshold bounds",
           "Principal findings.", "Temporal robustness.", "Choice of classifier.", "Generalization from a single subject.", "Limitations.",
           "Practical implications."]
THEREFORE_CUTS = [
    ("the default-parameter comparison therefore does not establish an ordering", "the default-parameter comparison does not establish an ordering"),
    ("Calibration is therefore reported as a result and as a limitation", "Calibration is reported as a result and as a limitation"),
    ("The model therefore does not depend on the composite scores", "The model does not depend on the composite scores"),
]
ALLOWED_EXTRA = {"Generalization from a single subject.": {"2024", "2025"}}
RESULTS: list[tuple[str, bool, str]] = []


def rec(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def numbers(text: str) -> set[str]:
    return {t.rstrip(".,") for t in NUM_RE.findall(text) if t.rstrip(".,")}


def labels(text: str) -> set[str]:
    return {m.group(0).strip() for m in LABEL_RE.finditer(text)}


def body_numbers(text: str) -> set[str]:
    """Numbers that are not part of a Section / Table / Figure / Limitation / comment label."""
    stripped = LABEL_RE.sub(" ", text)
    return numbers(stripped)


def git_docx(rev: str, relpath: str, outdir: Path) -> Path:
    out = outdir / Path(relpath).name
    data = subprocess.run(["git", "show", f"{rev}:{relpath}"], cwd=ROOT, check=True, capture_output=True).stdout
    out.write_bytes(data)
    return out


def find_unique(d, anchor: str) -> Paragraph:
    hits = [p for p in d.paragraphs if p.text.lower().startswith(anchor.lower())]
    if len(hits) != 1:
        raise SystemExit(f"STOP: anchor {anchor!r} matched {len(hits)} paragraphs")
    return hits[0]


def section_xml(d, start: str, end: str) -> bytes:
    body = d.element.body
    out, on = [], False
    for child in body.iterchildren():
        tag = child.tag.split("}")[1]
        text = Paragraph(child, d).text if tag == "p" else ""
        if tag == "p" and text.startswith(start):
            on = True
        elif tag == "p" and text.startswith(end):
            break
        if on:
            out.append(etree.tostring(child, method="c14n"))
    return b"\n".join(out)


def f8_paragraphs(name: str) -> list[str]:
    return [p.strip() for p in (F8 / name).read_text(encoding="utf-8").split("\n") if p.strip()]


def check_manuscript(before: Path, after: Path) -> None:
    f8 = f8_paragraphs("f8_front_methods_new.txt") + f8_paragraphs("f8_results_new.txt") + f8_paragraphs("f8_discussion_new.txt")
    if len(f8) != 35:
        raise SystemExit(f"STOP: expected 35 F8 paragraphs, found {len(f8)}")
    old, new = docx.Document(str(before)), docx.Document(str(after))
    print("| # | anchor | words before -> after | text == F8 | numbers before \\ after | numbers after \\ before | labels after \\ before |")
    print("|---|---|---|---|---|---|---|")
    all_ok = True
    for i, (anchor, expect) in enumerate(zip(ANCHORS, f8), 1):
        for o, n in THEREFORE_CUTS:
            expect = expect.replace(o, n)
        po, pn = find_unique(old, anchor), find_unique(new, anchor)
        same_text = pn.text == expect
        n_old, n_new = body_numbers(po.text), body_numbers(pn.text)
        extra, missing = n_new - n_old, n_old - n_new
        allowed = ALLOWED_EXTRA.get(anchor, set())
        ok = same_text and not missing and extra <= allowed
        all_ok &= ok
        lab_extra = labels(pn.text) - labels(po.text)
        print(f"| {i} | {anchor} | {len(po.text.split())} -> {len(pn.text.split())} | {'yes' if same_text else 'NO'} | "
              f"{sorted(missing)} | {sorted(extra)}{' (allowed)' if extra and extra <= allowed else ''} | {sorted(lab_extra)} |")
        if not same_text:
            j = next((x for x in range(min(len(pn.text), len(expect))) if pn.text[x] != expect[x]), min(len(pn.text), len(expect)))
            print(f"    first difference at char {j}: docx '{pn.text[j-30:j+50]}' vs F8 '{expect[j-30:j+50]}'")
    rec("35 paragraphs: text equals the F8 text and the number set is unchanged (documented exception only)", all_ok)
    # lead formatting kept (bold run-in)
    leads_ok = True
    for anchor in ("Principal findings.", "Temporal robustness.", "Choice of classifier.", "Generalization from a single subject.",
                   "Limitations.", "Practical implications.", "Threshold sensitivity.", "Agreement with label"):
        p = find_unique(new, anchor)
        r0 = p.runs[0]
        leads_ok &= bool(r0.bold) and r0.text.strip().endswith(".") and len(p.runs) >= 2 and not p.runs[1].bold
    rec("bold run-in leads kept on the 8 lead paragraphs (first run bold, rest not bold)", leads_ok)
    txt = "\n".join(p.text for p in new.paragraphs)
    rf = [p.text[:70] for p in new.paragraphs if re.search(r"random forest", p.text)]
    rec("0 lower-case 'random forest' in paragraphs (prose, captions, legends)", not rf, "; ".join(rf))
    cells = sum(len(re.findall(r"random forest", c.text)) for t in new.tables for r in t.rows for c in r.cells)
    rec("0 lower-case 'random forest' in table cells", cells == 0, str(cells))
    dist = [p.text[:70] for p in new.paragraphs if "distinguish" in p.text]
    rec("0 'distinguish' in paragraphs", not dist, "; ".join(dist))
    we = len(re.findall(r"\b[Ww]e\b", "\n".join(p.text for p in new.paragraphs if not p.text.startswith(("Table", "Figure")))))
    rec("'we' in the manuscript paragraphs <= 6", we <= 6, str(we))
    rec("3.12 paragraphs byte-identical (c14n XML between the 3.12 and 3.13 headings)",
        section_xml(old, "3.12.", "3.13.") == section_xml(new, "3.12.", "3.13.") and len(section_xml(new, "3.12.", "3.13.")) > 0)
    # T11.2 (2026-11-03): the T6.2 anonymization comment on the CRediT heading was removed (single-blind review), 9 -> 8
    rec("8 Word comments in the manuscript (9 before T11.2)", len(list(new.comments)) == 8, str(len(list(new.comments))))
    idx = next(i for i, p in enumerate(new.paragraphs) if p.text.strip() == "Abstract")
    n_abs = len(new.paragraphs[idx + 1].text.split())
    rec("Abstract <= 249 words (split on spaces, as R1)", n_abs <= 249, f"{n_abs} words")
    marked_before = subprocess.run(["git", "show", f"{ARGS.before_rev}:revision/manuscript/KBS_Manuscript_R1_marked.docx"], cwd=ROOT,
                                   check=True, capture_output=True).stdout
    marked_now = (MAN / "KBS_Manuscript_R1_marked.docx").read_bytes()
    mtext = "\n".join(p.text for p in docx.Document(str(MAN / "KBS_Manuscript_R1_marked.docx")).paragraphs)
    rec("marked copy regenerated (bytes differ from before; contains the F8 Abstract)", marked_before != marked_now and "The results support technical feasibility, not clinical readiness." in mtext)


def response_paragraphs(d) -> list[str]:
    out, started = [], False
    for p in d.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        if t.startswith("Manuscript: "):
            started = True
            continue
        if not started:
            continue
        if p.style is not None and p.style.name.startswith("Heading"):
            continue
        runs = [r for r in p.runs if r.text.strip()]
        if runs and all(r.italic for r in runs):
            continue  # reviewer quote
        if t.startswith("General changes") or p.paragraph_format.left_indent:
            continue
        out.append(t)
    return out


def check_response(before: Path, after: Path) -> None:
    f8 = f8_paragraphs("f8_response_new.txt")
    old, new = docx.Document(str(before)), docx.Document(str(after))
    po, pn = response_paragraphs(old), response_paragraphs(new)
    # the bullets of 'General changes' are body paragraphs too: keep only the 28 authors' paragraphs (intro + Response/Changes)
    pn_main = [t for t in pn if t.startswith(("We thank the editor", "Response. ", "Changes. ", "In the course of this analysis"))]
    po_main = [t for t in po if t.startswith(("We thank the editor", "Response. ", "Changes. ", "In the course of this analysis"))]
    rec("response letter: 28 authors' paragraphs (intro + 14 Response + 13 Changes) in the docx", len(pn_main) == 28 and len(f8) == 28, f"{len(pn_main)} docx, {len(f8)} F8")
    diffs = [(i, a[:60]) for i, (a, b) in enumerate(zip(pn_main, f8), 1) if a != b]
    rec("response letter: every authors' paragraph equals the F8 text, in order", not diffs, "; ".join(f"#{i} {t}" for i, t in diffs[:5]))
    for i, (a, b) in enumerate(zip(pn_main, f8), 1):
        if a != b:
            j = next((x for x in range(min(len(a), len(b))) if a[x] != b[x]), min(len(a), len(b)))
            print(f"    #{i} first difference at char {j}: docx '{a[j-30:j+50]}' vs F8 '{b[j-30:j+50]}'")
    n_old = numbers("\n".join(po))
    n_new = numbers("\n".join(pn))
    # documented exception (same as the manuscript's Generalization paragraph): R1.1 'Changes' now cites Zignoli et al. (2024)
    # and Kim et al. (2025), both in the References of the manuscript (decision of the first author, 25 Oct 2026)
    rec("response letter: number set unchanged (R1 subset of new; new minus R1 within the documented exception {2024, 2025})",
        not (n_old - n_new) and (n_new - n_old) <= {"2024", "2025"},
        f"before \\ after {sorted(n_old - n_new)}; after \\ before {sorted(n_new - n_old)}")
    txt = "\n".join(p.text for p in new.paragraphs)
    rec("response letter: 0 'distinguish'", "distinguish" not in txt, str(txt.count("distinguish")))
    rec("response letter: 0 'significant' / 'significantly'", not re.search(r"\bsignificant(ly)?\b", txt), "")
    rec("response letter: 0 lower-case 'random forest'", "random forest" not in txt, str(txt.count("random forest")))
    # T11.2 (2026-11-03): the 'T7 draft' comment on the Authors line was removed (single-blind review), 1 -> 0
    rec("response letter: 0 Word comments (the Authors-line comment removed at T11.2)", len(list(new.comments)) == 0, str(len(list(new.comments))))
    bullets = [p.text for p in new.paragraphs if p.text.startswith("The wording of the text added in revision was aligned")]
    rec("response letter: 'General changes' bullet on the style alignment present once", len(bullets) == 1, str(len(bullets)))


def check_supp(before_s: Path, after_s: Path, before_h: Path, after_h: Path) -> None:
    for name, b, a in (("Supplementary", before_s, after_s), ("Highlights", before_h, after_h)):
        old, new = docx.Document(str(b)), docx.Document(str(a))
        t_old = "\n".join([p.text for p in old.paragraphs] + [c.text for t in old.tables for r in t.rows for c in r.cells])
        t_new = "\n".join([p.text for p in new.paragraphs] + [c.text for t in new.tables for r in t.rows for c in r.cells])
        rec(f"{name}: 0 lower-case 'random forest'", "random forest" not in t_new, str(t_new.count("random forest")))
        rec(f"{name}: number set unchanged", numbers(t_old) == numbers(t_new), f"{sorted(numbers(t_old) ^ numbers(t_new))}")
        # T8.8b (2026-10-25): the plural reads '100 Random Forest models' (was '100 random forests' before T8, 'Random Forests' at T8.3)
        norm = lambda s: re.sub(r"[Rr]andom [Ff]orests?(?: models)?", "RF", s)
        rec(f"{name}: text differs from before only by 'random forest' -> 'Random Forest' ('random forests' -> 'Random Forest models')",
            norm(t_old) == norm(t_new))
    hl = [p.text for p in docx.Document(str(after_h)).paragraphs if p.style is not None and p.text.strip() and p.text[0].isupper()
          and not p.text.startswith(("Highlights", "Explainable Artificial"))]
    rec("Highlights: 5 bullets, each <= 85 characters", len(hl) == 5 and all(len(h) <= 85 for h in hl), "; ".join(f"{len(h)}" for h in hl))
    rec("Highlights: 'Random Forest exceeds persistence and majority baselines on the final 192-day block' (83 characters)",
        any(h == "Random Forest exceeds persistence and majority baselines on the final 192-day block" and len(h) == 83 for h in hl))


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-rev", default="f6eefc0", help="git revision holding the docx before T8 (default T7b)")
    ap.add_argument("--manuscript", action="store_true")
    ap.add_argument("--response", action="store_true")
    ap.add_argument("--supp", action="store_true")
    ARGS = ap.parse_args()
    if not (ARGS.manuscript or ARGS.response or ARGS.supp):
        ARGS.manuscript = ARGS.response = ARGS.supp = True
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        if ARGS.manuscript:
            print(f"## manuscript (before: {ARGS.before_rev})")
            check_manuscript(git_docx(ARGS.before_rev, "revision/manuscript/KBS_Manuscript_R1.docx", tdp), MAN / "KBS_Manuscript_R1.docx")
        if ARGS.response:
            print(f"## response letter (before: {ARGS.before_rev})")
            check_response(git_docx(ARGS.before_rev, "revision/manuscript/KBS_Response_R1.docx", tdp), MAN / "KBS_Response_R1.docx")
        if ARGS.supp:
            print(f"## supplementary and highlights (before: {ARGS.before_rev})")
            check_supp(git_docx(ARGS.before_rev, "revision/manuscript/KBS_Supplementary_R1.docx", tdp), MAN / "KBS_Supplementary_R1.docx",
                       git_docx(ARGS.before_rev, "revision/manuscript/KBS_Highlights_R1.docx", tdp), MAN / "KBS_Highlights_R1.docx")
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks PASS")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
