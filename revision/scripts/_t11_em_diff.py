"""T11.1 (2026-11-03): programmatic comparison of the Editorial Manager PDF of the original submission
(revision/manuscript/submitted/KNOSYS-D-26-14863.pdf, received from the co-author on 2 November 2026) with the
working base of the revision, revision/manuscript/BASE_MANUSCRIPT_v8.docx.

The PDF text is extracted with pdftotext (xpdf 4.00, the only extractor available in the environment; PyMuPDF is
not installed). Two modes are used: -raw (content-stream order, keeps the hyphen of a compound broken at a line end,
used for the diff and the anchors) and the default mode (no -layout; kept for the record, but xpdf's default mode
drops the hyphen when it joins 'lag-' + '1' into 'lag1', so it cannot serve an exact comparison). The text is cut in
four blocks on the markers of the brief: 'Powered by Editorial Manager' (end of the metadata page), 'on behalf of the
authors' (end of the cover letter), the manuscript title followed by the authors line (start of the manuscript),
'Declaration of interests' (start of the form). Block 3 (the manuscript) is compared sentence by sentence with the
v8 docx text (paragraphs and tables in body order, python-docx); each difference is classified as
(a) TITLE PAGE, (b) RENDERING (equations of 3.6, table rows, legends, page furniture) or (c) OTHER, which must be 0.
Five anchors are compared exactly. The report is revision/audit/em_submission_diff.md.

Usage (repo root): python revision/scripts/_t11_em_diff.py
Exit 1 if a class (c) difference remains, an anchor differs or a marker is not found.
"""
from __future__ import annotations

import difflib
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "revision" / "manuscript" / "submitted" / "KNOSYS-D-26-14863.pdf"
V8 = ROOT / "revision" / "manuscript" / "BASE_MANUSCRIPT_v8.docx"
OUT = ROOT / "revision" / "audit" / "em_submission_diff.md"
TITLE = ("Explainable Artificial Intelligence for Personalized Nutrition: A Multimodal Framework Integrating "
         "Smart Ring and Smartwatch Data")
# sentences of the 3.6 equation paragraph (OMML objects in the docx, invisible to python-docx; rendered symbols in the PDF)
EQUATION_MARKERS = ("is defined as", "harmonic mean of precision", "quadratic weights", "observed and E the expected",
                    "MCC generalizes", "Matthews correlation")
# formulations of the original cover letter that the revision retracted
RETRACTED = [("deduplicated data source", "AUDIT_DECISIONS #12 (the JSON export is the export application's merged aggregate, "
                                          "not a deduplicated source; 3.3, 4.10 and the Figure 1 legend rewritten)"),
             ("complete, deployable system", "R1.7 / response letter and Discussion: the contribution is a feasibility study, "
                                             "not a deployable system (the wording 'deployable' is not used in Revision 1)")]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pdftotext(mode: str) -> str:
    args = ["pdftotext"] + (["-raw"] if mode == "raw" else []) + ["-enc", "UTF-8", str(PDF), "-"]
    r = subprocess.run(args, capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"STOP: pdftotext failed: {r.stderr.decode('utf-8', 'replace')}")
    return r.stdout.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def pdftotext_version() -> str:
    r = subprocess.run(["pdftotext", "-v"], capture_output=True)
    return (r.stderr or r.stdout).decode("utf-8", "replace").splitlines()[0].strip()


# ----------------------------------------------------------------------------------------------
# segmentation
# ----------------------------------------------------------------------------------------------
def segment(lines: list[str]) -> dict:
    def find(pred, start=0, what=""):
        for i in range(start, len(lines)):
            if pred(i):
                return i
        raise SystemExit(f"STOP: marker not found: {what}")

    i1 = find(lambda i: "Powered by Editorial Manager" in lines[i], 0, "Powered by Editorial Manager")
    i2 = find(lambda i: "on behalf of the authors" in lines[i], i1 + 1, "on behalf of the authors")

    def title_then_authors(i):
        if not lines[i].startswith("Explainable Artificial Intelligence for Personalized Nutrition"):
            return False
        return any("Lazar" in lines[j] and "Buiu" in lines[j] for j in range(i + 1, min(i + 4, len(lines))))

    i3 = find(title_then_authors, i2 + 1, "manuscript title followed by the authors line")
    i4 = find(lambda i: lines[i].startswith("Declaration of interests") or lines[i].startswith("Declaration of competing interests"),
              i3 + 1, "Declaration of interests")
    return {"i1": i1, "i2": i2, "i3": i3, "i4": i4,
            "block1": lines[: i1 + 1], "block2": lines[i1 + 1: i2 + 1], "wrapper": lines[i2 + 1: i3],
            "block3": lines[i3: i4], "block4": lines[i4:]}


# ----------------------------------------------------------------------------------------------
# normalization
# ----------------------------------------------------------------------------------------------
QUOTES = {"’": "'", "‘": "'", "“": '"', "”": '"', "−": "-", "–": "-", "—": "-",
          " ": " ", "•": " ", "ﬁ": "fi", "ﬂ": "fl"}


def norm(s: str, glue_hyphen: bool = True) -> str:
    s = s.replace("\x0c", "\n").replace("\x02", "")
    s = "".join(ch for ch in s if ord(ch) >= 32 or ch in "\n\t")
    for k, v in QUOTES.items():
        s = s.replace(k, v)
    if glue_hyphen:
        s = re.sub(r"(\w)-\n(?=\w)", r"\1-", s)  # compound broken at a line end: keep the hyphen, drop the break
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in text.split(". ")]
    return [p + "." if i < len(parts) - 1 else p for i, p in enumerate(parts) if p]


def sentences_pos(text: str) -> list[tuple[str, int]]:
    """Sentences split on '. ' with the character offset of each sentence start in `text`."""
    out, pos = [], 0
    for m in re.finditer(r"(.*?\. )|(.+$)", text, re.S):
        s = m.group(0)
        if s.strip():
            out.append((s.strip(), m.start()))
    return out


# ----------------------------------------------------------------------------------------------
# v8 text with origin tags
# ----------------------------------------------------------------------------------------------
def v8_items() -> list[tuple[str, str, bool]]:
    """(origin, normalized text, has_omml) for every body paragraph (non-empty) and table of the v8 docx."""
    d = docx.Document(str(V8))
    out = []
    n_tab = 0
    for k, child in enumerate(d.element.body.iterchildren()):
        tag = child.tag.split("}")[1]
        if tag == "p":
            p = Paragraph(child, d)
            t = norm(p.text, glue_hyphen=False)
            if t:
                out.append((f"paragraph {k} '{t[:40]}'", t, "<m:oMath" in child.xml))
        elif tag == "tbl":
            n_tab += 1
            rows = [" ".join(norm(c.text, glue_hyphen=False) for c in r.cells) for r in Table(child, d).rows]
            out.append((f"table {n_tab} '{rows[0][:40]}'", " ".join(rows), False))
    return out


def main() -> None:
    raw = pdftotext("raw")
    default = pdftotext("default")
    lines_raw = raw.replace("\x0c", "\n").split("\n")
    seg = segment(lines_raw)
    seg_default = segment(default.replace("\x0c", "\n").split("\n"))
    # title page: lines from the title to the line before 'Abstract' (class (a), removed from block 3 before the diff;
    # the title line itself, which v8 also has, stays in the compared text)
    b3 = seg["block3"]
    i_abs = next(i for i, s in enumerate(b3) if s.strip() == "Abstract")
    i_auth = next(i for i in range(0, 4) if "Lazar" in b3[i] and "Buiu" in b3[i])
    title_text = norm("\n".join(b3[:i_auth]))
    if title_text != TITLE:
        raise SystemExit(f"STOP: title in the PDF differs from the v8 title: {title_text!r}")
    WRAPPER = ("Manuscript", "Click here to view linked References")
    title_page_lines = [s for s in b3[i_auth:i_abs] if s.strip() and not s.strip().startswith(WRAPPER)]
    wrapper_in_b3 = [s for s in b3[i_auth:i_abs] if s.strip().startswith(WRAPPER)]
    block3_text = norm(title_text + "\n" + "\n".join(b3[i_abs:]))

    # ---- v8 sentences with origin (split on the whole joined text, origin by character offset) ---
    v8 = v8_items()
    v8_full, bounds, off = "", [], 0
    for origin, text, has_math in v8:
        bounds.append((off, off + len(text), origin, has_math, origin.startswith("table")))
        v8_full += text + " "
        off += len(text) + 1
    v8_sents, v8_origin = [], []
    for s, p in sentences_pos(v8_full):
        covering = [b for b in bounds if b[0] <= p < b[1]] or [bounds[-1]]
        b = covering[0]
        v8_sents.append(s)
        v8_origin.append((b[2], b[3], b[4]))
    pdf_sents = [s for s, _ in sentences_pos(block3_text)]

    sm = difflib.SequenceMatcher(None, pdf_sents, v8_sents, autojunk=False)
    diffs = [("a", "title page (between the title and 'Abstract', absent from the anonymized v8)", line, "",
              "title page line: " + ("authors" if "Buiu" in line else "corresponding-author mark" if line.startswith("*") else
                                     "e-mail / telephone" if "@" in line else "affiliation")) for line in title_page_lines]
    diffs += [("b", "Editorial Manager wrapper, first manuscript page", line, "",
               "page furniture added by Editorial Manager ('Manuscript' header, 'Click here to view linked References' link)")
              for line in wrapper_in_b3]
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        ptxt = " ".join(pdf_sents[i1:i2])
        vtxt = " ".join(v8_sents[j1:j2])
        origins = sorted({v8_origin[j][0] for j in range(j1, j2)})
        loc = "; ".join(origins) if origins else f"PDF sentences {i1 + 1}-{i2} (no v8 counterpart)"
        if any(v8_origin[j][1] for j in range(j1, j2)) or any(m in ptxt or m in vtxt for m in EQUATION_MARKERS):
            cls, reason = "b", "equations of 3.6: OMML objects (invisible to python-docx) rendered as symbols in the PDF"
        elif any(v8_origin[j][2] for j in range(j1, j2)):
            # table: same cell sequence when the pdf text equals the v8 text without spaces, otherwise reflowed cells
            same = re.sub(r"\s", "", ptxt) == re.sub(r"\s", "", vtxt)
            origin = next(v8_origin[j][0] for j in range(j1, j2) if v8_origin[j][2])
            header = re.search(r"'(.*)'", origin).group(1)  # first 40 characters of the header row
            repeated = re.sub(r"\s", "", ptxt).replace(re.sub(r"\s", "", header), "", 1) == re.sub(r"\s", "", vtxt).replace(re.sub(r"\s", "", header), "", 1) \
                and ptxt.count(header[:20]) == vtxt.count(header[:20]) + 1
            cls, reason = "b", "table rendered cell by cell" + (" (identical character sequence)" if same else
                                                               " (header row repeated at a page break)" if repeated else
                                                               " (cells reflowed by the extractor)")
        elif j1 == j2 and all(re.fullmatch(r"\d+\.?", s) for s in pdf_sents[i1:i2]):
            cls, reason = "b", "page number"
        elif re.sub(r"\s", "", ptxt) == re.sub(r"\s", "", vtxt):
            cls, reason = "b", "identical character sequence, different sentence split (line break or tab in the docx)"
        else:
            cls, reason = "c", "OTHER"
        diffs.append((cls, loc, ptxt, vtxt, reason))

    # ---- anchors ---------------------------------------------------------------------------------
    anchors = []
    v8_pars = [t for o, t, m in v8 if o.startswith("paragraph")]
    # 1 Abstract
    v8_abs = v8_pars[2]
    i_kw = next(i for i, s in enumerate(b3) if s.startswith("Keywords:"))
    pdf_abs = norm("\n".join(b3[i_abs + 1: i_kw]))
    anchors.append(("Abstract (v8 paragraph, word by word)", pdf_abs == v8_abs, pdf_abs, v8_abs))
    # 2 Keywords line of the body
    v8_kw = v8_pars[3]
    i_intro = next(i for i in range(i_kw + 1, len(b3)) if b3[i].startswith("1. Introduction"))
    pdf_kw = norm("\n".join(b3[i_kw:i_intro]))
    anchors.append(("Keywords line of the body", pdf_kw == v8_kw, pdf_kw, v8_kw))
    # 3 first sentence of 3.1
    v8_31 = next(t for t in v8_pars if t.startswith("The dataset comprises physiological data"))
    s_v8 = sentences(v8_31)[0]
    i = block3_text.index("The dataset comprises physiological data")
    s_pdf = sentences(block3_text[i:])[0]
    anchors.append(("First sentence of 3.1", s_pdf == s_v8, s_pdf, s_v8))
    # 4 last sentence of Conclusions
    v8_concl = next(t for t in v8_pars if t.startswith("Future work should prioritize"))
    s_v8 = sentences(v8_concl)[-1]
    i = block3_text.index("Future work should prioritize")
    j = block3_text.index("Declaration of generative AI", i)
    s_pdf = sentences(block3_text[i:j])[-1]
    anchors.append(("Last sentence of Conclusions", s_pdf == s_v8, s_pdf, s_v8))
    # 5 first and last reference entries
    i_ref = max(k for k, t in enumerate(v8_pars) if t == "References")
    first_ref, last_ref = v8_pars[i_ref + 1], v8_pars[-1]
    i = block3_text.rindex(" References ")
    refs_pdf = block3_text[i + len(" References "):]
    ok = refs_pdf.startswith(first_ref) and refs_pdf.endswith(last_ref)
    anchors.append(("First and last entry of References (first entry opens the list, last entry closes block 3)", ok,
                    refs_pdf[: len(first_ref)] + " ... " + refs_pdf[-len(last_ref):], first_ref + " ... " + last_ref))

    # ---- keywords: EM metadata vs body ----------------------------------------------------------
    b1 = norm("\n".join(seg["block1"]))
    m = re.search(r"Personalized nutrition; Wearable devices; Multimodal[^\n]*?\(XAI\)", b1)
    em_kw = m.group(0) if m else "NOT FOUND"
    body_kw = pdf_kw[len("Keywords:"):].strip()
    em_list = [k.strip() for k in em_kw.split(";")]
    body_list = [k.strip() for k in body_kw.split(";")]

    # ---- report ----------------------------------------------------------------------------------
    n = {c: sum(1 for d in diffs if d[0] == c) for c in "abc"}
    L = ["# em_submission_diff.md - T11.1: Editorial Manager PDF of the original submission vs BASE_MANUSCRIPT_v8.docx", ""]
    L.append(f"Generated by `revision/scripts/_t11_em_diff.py` on 2026-11-03. PDF: `revision/manuscript/submitted/KNOSYS-D-26-14863.pdf`, "
             f"sha256 `{sha256(PDF)}`, received from the co-author on 2 November 2026 (Editorial Manager, manuscript KNOSYS-D-26-14863). "
             f"Base: `revision/manuscript/BASE_MANUSCRIPT_v8.docx`, sha256 `{sha256(V8)}`. Extractor: {pdftotext_version()} "
             f"(xpdf; PyMuPDF not installed). The diff and the anchors use `pdftotext -raw -enc UTF-8` (no -layout; the content-stream "
             f"order keeps the hyphen of a compound broken at a line end, e.g. 'health-' + 'data'); the default mode of xpdf 4.00 joins such "
             f"words without the hyphen ('healthdata', 'lag1', 'Knowledgebased') and was used only to confirm the segmentation "
             f"(same four cuts: lines {seg_default['i1'] + 1}, {seg_default['i2'] + 1}, {seg_default['i3'] + 1}, {seg_default['i4'] + 1}).")
    L.append("")
    L.append("Normalization applied to both texts: control characters removed (the PDF text carries form feeds between pages and no "
             "U+0002), 'word-' + line break + continuation glued with the hyphen kept, whitespace collapsed, curly quotes and apostrophes "
             "unified to ASCII, U+2212 / U+2013 / U+2014 unified to '-', bullet characters dropped, non-breaking spaces to spaces. "
             "v8 text: body paragraphs and tables in document order (python-docx; OMML equations are not part of the paragraph text). "
             "Sentence split on '. '.")
    L.append("")
    L.append("## 1. Segmentation (pdftotext -raw)")
    L.append("")
    L.append("| Cut | Marker | Line | First 60 characters of the line |")
    L.append("|---|---|---|---|")
    for name, key, marker in (("end of block 1 (EM metadata page)", "i1", "Powered by Editorial Manager"),
                              ("end of block 2 (cover letter)", "i2", "on behalf of the authors"),
                              ("start of block 3 (manuscript)", "i3", "title followed by the authors line"),
                              ("start of block 4 (form)", "i4", "Declaration of interests")):
        L.append(f"| {name} | {marker} | {seg[key] + 1} | {lines_raw[seg[key]][:60]} |")
    L.append("")
    wrap = [s for s in seg["wrapper"] if s.strip()]
    L.append("Lines between block 2 and block 3 in the raw order: " + ("; ".join(repr(s) for s in wrap) if wrap else
             "none (the Editorial Manager wrapper of the first manuscript page, 'Manuscript' and 'Click here to view linked "
             "References', comes after the title page in the content stream and is listed in sections 2 and 3)") + ".")
    L.append(f"Block sizes (non-empty lines): 1 = {sum(1 for s in seg['block1'] if s.strip())}, 2 = {sum(1 for s in seg['block2'] if s.strip())}, "
             f"3 = {sum(1 for s in seg['block3'] if s.strip())}, 4 = {sum(1 for s in seg['block4'] if s.strip())}. "
             f"Block 3 ends with the last reference entry; the 'Declaration of competing interests' / 'Declaration of interests' lines that "
             f"follow belong to the form (block 4), not to the manuscript (v8 has no such section).")
    L.append("")
    L.append("## 2. Title page of the submitted manuscript (block 3, literal, before 'Abstract')")
    L.append("")
    for s in b3[:i_auth] + title_page_lines:
        L.append(f"    {s}")
    L.append("")
    L.append(f"The title (three lines in the raw extraction) equals the v8 title. Editorial Manager furniture on the same page, not part of "
             f"the title page: " + "; ".join(repr(s.strip()) for s in wrapper_in_b3) + ".")
    L.append("")
    L.append("The title page carries 'Adrian A. Lazar'. This is the form of the Editorial Manager record (metadata page: First Author "
             "'Adrian A. Lazar'); the 'A.' is the father's initial of the Romanian civil records, not a middle name, and in a publication "
             "it would read as a different author name. The revision deliverables use 'Adrian Lazar' (the form of the dissertation and of "
             "the original cover letter, block 2); the Editorial Manager record is to be corrected by the corresponding author "
             "(TEXT_DECISIONS.md section T11, decision of the first author of 3 November 2026). This file is a historical record and is "
             "reproduced as is. The title page also omits the faculty line and the postal code that the cover letter carries "
             "(AFFILIATION_LINES of T11.2 take the four-line form of the cover letter, provisional until the co-author confirms).")
    L.append("")
    L.append("## 3. Sentence-level diff of block 3 against the v8 text")
    L.append("")
    L.append(f"PDF sentences: {len(pdf_sents)}; v8 sentences: {len(v8_sents)}; equal opcodes cover "
             f"{sum(i2 - i1 for t, i1, i2, j1, j2 in sm.get_opcodes() if t == 'equal')} PDF sentences. "
             f"Differences: (a) TITLE PAGE {n['a']}, (b) RENDERING {n['b']}, (c) OTHER {n['c']}.")
    L.append("")
    L.append("| # | Class | Location (v8) | Reason | PDF text (first 160 characters) | v8 text (first 160 characters) |")
    L.append("|---|---|---|---|---|---|")
    for k, (cls, loc, p, v, reason) in enumerate(diffs, 1):
        cell = lambda s: s[:160].replace("|", "/") + ("..." if len(s) > 160 else "")
        L.append(f"| {k} | ({cls}) | {loc.replace('|', '/')} | {reason} | {cell(p)} | {cell(v)} |")
    L.append("")
    L.append("## 4. Anchors (exact comparison after normalization)")
    L.append("")
    L.append("| # | Anchor | Result | Text |")
    L.append("|---|---|---|---|")
    for k, (name, ok, p, v) in enumerate(anchors, 1):
        shown = p if ok else f"PDF: {p} /// v8: {v}"
        L.append(f"| {k} | {name} | {'IDENTICAL' if ok else 'DIFFERENT'} | {shown.replace('|', '/')} |")
    L.append("")
    L.append("## 5. Keywords: Editorial Manager metadata (block 1) versus manuscript body (block 3)")
    L.append("")
    L.append(f"- EM metadata ({len(em_list)}): {em_kw}")
    L.append(f"- Manuscript body ({len(body_list)}): {body_kw}")
    only_em = [k for k in em_list if k not in body_list]
    only_body = [k for k in body_list if k not in em_list]
    common = [k for k in em_list if k in body_list]
    L.append(f"- Common ({len(common)}): {'; '.join(common)}. Only in EM ({len(only_em)}): {'; '.join(only_em)}. "
             f"Only in the body ({len(only_body)}): {'; '.join(only_body)}.")
    L.append("- Revision 1 body (F4_manuscript_changes.md, T4.1): Personalized nutrition; Wearable devices; Explainable AI; Decision support "
             "systems; SHAP; Physiological state classification; mHealth (7; 'Random Forest' of the submitted body replaced by 'Decision support "
             "systems'). Which list the revised submission carries is the co-author's decision (question sent 3 November 2026); "
             "KEYWORDS_MODE in assemble_manuscript.py selects between 'manuscript' (current) and 'em' (T11.4).")
    L.append("")
    L.append("## 6. Cover letter of the original submission (block 2, historical quotation)")
    L.append("")
    L.append("Reproduced as extracted (July 2, 2026), for the record. Two of its formulations were retracted in the revision: "
             + "; ".join(f"'{s}' ({why})" for s, why in RETRACTED) + ". The cover letter of Revision 1 (T11.5) does not reuse them. "
             "The line 'Cover letter' inside the quotation is the Editorial Manager page header of the second page (raw content-stream order).")
    L.append("")
    cover = norm("\n".join(seg["block2"]), glue_hyphen=True)
    for s, _ in RETRACTED:
        if s not in cover:
            raise SystemExit(f"STOP: retracted formulation '{s}' not found in the cover letter")
    for line in seg["block2"]:
        if line.strip():
            L.append(f"> {line.strip()}")
    L.append("")
    L.append("## 7. Result")
    L.append("")
    ok_anchors = all(a[1] for a in anchors)
    L.append(f"- Class (c) differences: {n['c']} (required 0). Class (b): {n['b']}, each listed in section 3 with its location. Class (a): {n['a']}.")
    L.append(f"- Anchors identical: {sum(a[1] for a in anchors)}/5.")
    L.append(f"- Block 3 of the PDF is the v8 text plus the title page; the manuscript submitted to Knowledge-Based Systems is BASE_MANUSCRIPT_v8.docx "
             f"(confirms the co-author's statement of 20 October 2026). Review is single-blind: the submitted main document carries the authors.")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"diff: (a) {n['a']}, (b) {n['b']}, (c) {n['c']}; anchors {sum(a[1] for a in anchors)}/5 -> {OUT.relative_to(ROOT)}")
    for cls, loc, p, v, reason in diffs:
        if cls == "c":
            print(f"  (c) {loc}\n      PDF: {p[:300]}\n      v8 : {v[:300]}")
    for name, ok, p, v in anchors:
        if not ok:
            print(f"  ANCHOR DIFFERENT: {name}\n      PDF: {p[:300]}\n      v8 : {v[:300]}")
    sys.exit(0 if n["c"] == 0 and ok_anchors else 1)


if __name__ == "__main__":
    main()
