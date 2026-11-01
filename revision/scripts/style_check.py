"""T8.1 (2026-10-25): style profile of a KBS Revision 1 docx against the submitted manuscript (v8).

Generalizes revision/f8/style_compare.py (the measurement script of the F8 decision of 25 Oct 2026) to a docx input.
It extracts the prose of the document (body paragraphs; tables, Table/Figure captions and legends, notes, headings,
keywords, References, declarations and Appendix A are excluded), classifies every prose paragraph as identical to or
different from the v8 base (difflib SequenceMatcher ratio >= 0.9 against the best v8 paragraph = "same as v8") and
reports, for the whole document and for the paragraphs different from v8:
  words per sentence (mean, median, p90, max), % sentences > 40 words, sentences > 60 words, "we" per 1,000 words,
  "Random Forest" vs "random forest", "distinguish", forbidden terms (TEXT_DECISIONS / STYLE_GUIDE), 4- vs 3-decimal
  numbers, "percentage points" / "pp".
Thresholds (T8.6 of the T8 brief) are applied with --mode and the exit code is 1 when one of them is not met.

Usage (repo root):
  python revision/scripts/style_check.py revision/manuscript/KBS_Manuscript_R1.docx --mode manuscript
  python revision/scripts/style_check.py revision/manuscript/KBS_Response_R1.docx --mode response
  python revision/scripts/style_check.py revision/manuscript/KBS_Supplementary_R1.docx --mode other
Options: --v8 PATH (default revision/manuscript/BASE_MANUSCRIPT_v8.docx), --json PATH (machine-readable report),
         --no-fail (report only, exit 0), --longest N (print the N longest sentences of the different-from-v8 group).

Thresholds by mode (different-from-v8 paragraphs unless stated):
  manuscript: mean <= 27 words/sentence, p90 <= 45, sentences > 60 <= 1, 'random forest' (lower case) = 0,
              'distinguish' = 0, forbidden terms = 0; 'we' in the whole prose <= 6 occurrences.
  response:   mean <= 27, sentences > 60 <= 1, 'random forest' = 0, 'distinguish' = 0, forbidden terms = 0
              (the letter has no v8 counterpart, so every paragraph counts as different; 'we' is allowed).
              T8.8c (2026-10-25): measured on the Response paragraphs only (introductory paragraph, 'Response.' paragraphs
              and the second Response paragraph of R1.9); the 'Changes.' paragraphs (location lists, one 'sentence' each)
              and the 'General changes' bullets are enumerations and are excluded. Before T8.8c all authors' paragraphs
              were measured (27.7 words/sentence after T8.4 against 26.9 on the Response paragraphs).
  other:      'random forest' = 0, forbidden terms = 0, sentences > 60 words = 0 (T10.9e; Supplementary, Highlights; captions,
              legends and italic notes excluded).
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics as st
import sys
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
V8_DEFAULT = ROOT / "revision" / "manuscript" / "BASE_MANUSCRIPT_v8.docx"

MIN_WORDS = 25  # as in style_compare.py: shorter paragraphs (headings, notes, bullets) are not prose for the profile
# unnumbered headings after which the body is no longer manuscript prose (declarations, appendix, references)
STOP_HEADINGS = ("CRediT authorship contribution statement", "Declaration of competing interest", "Funding",
                 "Declaration of generative AI", "Data availability", "Appendix A", "References",
                 "Removed from the submitted version", "Supplementary figures")
CAPTION_RE = re.compile(r"^(Table|Figure) (S?\d+|A\.1|A1)\b")
NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.\s")
LEAD_RE = re.compile(r"^(?:[A-Z][^.]{2,70}\.|Response\.|Changes\.)\s$")
FORBIDDEN = ["best performing", "best overall", r"\bsignificant(ly)?\b", "stable over time", "optimal thresholds",
             "honestly", "—",  # em dash
             "concern deserves", "concern implies"]  # T10.3 (2026-11-01): rhetorical formulas, response letter (STYLE_GUIDE section 2)
ABBR = r"(?:et al|vs|Fig|e\.g|i\.e|approx|St|No|cf)\."


# ----------------------------------------------------------------------------------------------
def body_paragraphs(d):
    """Body paragraphs in document order (table cells excluded)."""
    out = []
    for child in d.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            out.append(Paragraph(child, d))
    return out


def strip_lead(p: Paragraph) -> str:
    """Paragraph text without a bold run-in lead ('Principal findings. ', 'Response. ')."""
    text = p.text
    runs = p.runs
    if runs and runs[0].bold and LEAD_RE.match(runs[0].text or ""):
        return text[len(runs[0].text):]
    m = re.match(r"^(Response|Changes)\. ", text)
    if m:
        return text[m.end():]
    return text


def is_quote(p: Paragraph) -> bool:
    runs = [r for r in p.runs if r.text.strip()]
    return bool(runs) and all(r.italic for r in runs)


def prose_paragraphs(path: Path, mode: str) -> list[str]:
    d = docx.Document(str(path))
    out, skipping = [], False
    for p in body_paragraphs(d):
        text = p.text.strip()
        if not text:
            continue
        if any(text.startswith(h) for h in STOP_HEADINGS):
            skipping = True
            continue
        if NUMBERED_HEADING_RE.match(text) and len(text.split()) <= 12:
            skipping = False  # a numbered heading restarts the prose (none follows the stop headings in practice)
            continue
        if skipping:
            continue
        if p.style is not None and p.style.name.startswith("Heading"):
            continue
        if CAPTION_RE.match(text) or text.startswith("Keywords") or text.startswith("* "):
            continue
        if text.startswith("Manuscript: ") or text.startswith("Highlights") or text.startswith("Supplementary Material"):
            continue
        if mode == "response" and is_quote(p):
            continue  # reviewer quotes (italic, indented) are not the authors' text
        if mode == "response" and (text.startswith("Changes. ") or (p.style is not None and p.style.name == "List Paragraph")):
            continue  # T8.8c: 'Changes.' location lists and the 'General changes' bullets are enumerations, not prose
        if is_quote(p):
            continue  # italic notes under tables
        body = strip_lead(p).strip()
        if len(body.split()) < MIN_WORDS:
            continue
        out.append(body)
    return out


def sentences(p: str) -> list[str]:
    t = re.sub(ABBR, lambda m: m.group(0).replace(".", "@"), p)
    t = re.sub(r"(\d)\.(\d)", r"\1@\2", t)
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", t) if x.strip()]
    return [x.replace("@", ".") for x in s]


def classify(r1: list[str], v8: list[str]) -> tuple[list[str], list[str], list[float]]:
    """Split R1 paragraphs into (same as v8, different from v8); ratio >= 0.9 against the best v8 paragraph = same."""
    same, diff, ratios = [], [], []
    v8set = set(v8)
    for p in r1:
        if p in v8set:
            same.append(p); ratios.append(1.0); continue
        best = 0.0
        for q in v8:
            sm = difflib.SequenceMatcher(None, p, q, autojunk=False)
            if sm.real_quick_ratio() < 0.9 or sm.quick_ratio() < 0.9:
                continue  # upper bounds below the threshold: the real ratio cannot reach 0.9
            best = max(best, sm.ratio())
            if best >= 0.9:
                break
        ratios.append(best)
        (same if best >= 0.9 else diff).append(p)
    return same, diff, ratios


def profile(paras: list[str]) -> dict:
    sents = [s for p in paras for s in sentences(p)]
    L = [len(s.split()) for s in sents]
    txt = " ".join(paras)
    words = sum(L)
    c = lambda pat: len(re.findall(pat, txt))
    if not L:
        return {"paragraphs": 0, "sentences": 0, "words": 0}
    Ls = sorted(L)
    return {
        "paragraphs": len(paras), "sentences": len(sents), "words": words,
        "mean": round(st.mean(L), 1), "median": st.median(L), "p90": Ls[min(len(Ls) - 1, int(0.9 * len(Ls)))], "max": max(L),
        "pct_gt40": round(100 * sum(l > 40 for l in L) / len(L), 1), "n_gt40": sum(l > 40 for l in L), "n_gt60": sum(l > 60 for l in L),
        "we": c(r"\b[Ww]e\b"), "we_per_1k": round(1000 * c(r"\b[Ww]e\b") / words, 1),
        "Random Forest": c(r"Random Forest"), "random forest": c(r"random forest"),
        "distinguish": c(r"distinguish"),
        "forbidden": {pat: c(pat) for pat in FORBIDDEN if c(pat)},
        "forbidden_total": sum(c(pat) for pat in FORBIDDEN),
        "dec4": c(r"\b\d\.\d{4}\b"), "dec3": c(r"\b\d\.\d{3}\b"),
        "percentage points": c("percentage points"), "pp": c(r"\d pp\b"),
        "therefore": c(r"\btherefore\b"), "semicolons": c(";"),
        "longest": sorted(sents, key=lambda s: -len(s.split()))[:5],
    }


def fmt(name: str, pr: dict) -> str:
    if pr["paragraphs"] == 0:
        return f"== {name}: 0 paragraphs"
    return (f"== {name}: {pr['paragraphs']} paragraphs, {pr['sentences']} sentences, {pr['words']} words\n"
            f"   words/sentence mean {pr['mean']}, median {pr['median']}, p90 {pr['p90']}, max {pr['max']}; "
            f"> 40 words {pr['n_gt40']} ({pr['pct_gt40']}%), > 60 words {pr['n_gt60']}\n"
            f"   'we' {pr['we']} ({pr['we_per_1k']}/1k words); 'Random Forest' {pr['Random Forest']} vs 'random forest' {pr['random forest']}; "
            f"'distinguish' {pr['distinguish']}; forbidden {pr['forbidden_total']} {pr['forbidden'] or ''}\n"
            f"   4-decimal {pr['dec4']} vs 3-decimal {pr['dec3']}; 'percentage points' {pr['percentage points']}, 'pp' {pr['pp']}; "
            f"'therefore' {pr['therefore']}; semicolons {pr['semicolons']}")


def thresholds(mode: str, all_pr: dict, diff_pr: dict) -> list[tuple[str, bool, str]]:
    T = []
    if mode == "manuscript":
        T.append(("different-from-v8: mean words/sentence <= 27", diff_pr["mean"] <= 27, str(diff_pr["mean"])))
        T.append(("different-from-v8: p90 <= 45", diff_pr["p90"] <= 45, str(diff_pr["p90"])))
        T.append(("different-from-v8: sentences > 60 words <= 1", diff_pr["n_gt60"] <= 1, str(diff_pr["n_gt60"])))
        T.append(("different-from-v8: 'random forest' (lower case) = 0", diff_pr["random forest"] == 0, str(diff_pr["random forest"])))
        T.append(("different-from-v8: 'distinguish' = 0", diff_pr["distinguish"] == 0, str(diff_pr["distinguish"])))
        T.append(("different-from-v8: forbidden terms = 0", diff_pr["forbidden_total"] == 0, str(diff_pr["forbidden"])))
        T.append(("whole prose: 'we' <= 6", all_pr["we"] <= 6, str(all_pr["we"])))
        T.append(("whole prose: 'random forest' (lower case) = 0", all_pr["random forest"] == 0, str(all_pr["random forest"])))
    elif mode == "response":
        T.append(("response prose: mean words/sentence <= 27", all_pr["mean"] <= 27, str(all_pr["mean"])))
        T.append(("response prose: sentences > 60 words <= 1", all_pr["n_gt60"] <= 1, str(all_pr["n_gt60"])))
        T.append(("response prose: 'random forest' (lower case) = 0", all_pr["random forest"] == 0, str(all_pr["random forest"])))
        T.append(("response prose: 'distinguish' = 0", all_pr["distinguish"] == 0, str(all_pr["distinguish"])))
        T.append(("response prose: forbidden terms = 0", all_pr["forbidden_total"] == 0, str(all_pr["forbidden"])))
    else:
        T.append(("prose: 'random forest' (lower case) = 0", all_pr.get("random forest", 0) == 0, str(all_pr.get("random forest", 0))))
        T.append(("prose: forbidden terms = 0", all_pr.get("forbidden_total", 0) == 0, str(all_pr.get("forbidden", {}))))
        # T10.9e (2026-11-01): revision prose of the Supplementary / Highlights; captions, legends and italic table notes are
        # excluded by prose_paragraphs (same rule as the 'Changes' lists of the response letter, STYLE_GUIDE section 4)
        T.append(("prose: sentences > 60 words = 0", all_pr.get("n_gt60", 0) == 0, str(all_pr.get("n_gt60", 0))))
    return T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("docx", type=Path)
    ap.add_argument("--v8", type=Path, default=V8_DEFAULT)
    ap.add_argument("--mode", choices=["manuscript", "response", "other"], default="manuscript")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--no-fail", action="store_true")
    ap.add_argument("--longest", type=int, default=3)
    a = ap.parse_args()

    r1 = prose_paragraphs(a.docx, a.mode)
    if a.mode == "manuscript":
        v8 = prose_paragraphs(a.v8, "manuscript")
        same, diff, _ = classify(r1, v8)
        v8_pr = profile(v8)
    else:
        same, diff, v8_pr = [], list(r1), None
    all_pr, same_pr, diff_pr = profile(r1), profile(same), profile(diff)

    print(f"# style_check: {a.docx.name} (mode {a.mode}; v8 reference {a.v8.name if a.mode == 'manuscript' else 'n/a'})")
    if v8_pr:
        print(fmt("v8 prose (reference)", v8_pr))
    print(fmt(f"{a.docx.name}: all prose", all_pr))
    if a.mode == "manuscript":
        print(fmt("  same as v8 (ratio >= 0.9)", same_pr))
        print(fmt("  different from v8 (ratio < 0.9)", diff_pr))
    for s in (diff_pr if a.mode == "manuscript" else all_pr).get("longest", [])[: a.longest]:
        print(f"   LONGEST ({len(s.split())} words): {s[:160]}{'...' if len(s) > 160 else ''}")
    T = thresholds(a.mode, all_pr, diff_pr if a.mode == "manuscript" else all_pr)
    n_fail = 0
    print("## thresholds")
    for name, ok, val in T:
        n_fail += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {val}")
    if a.json:
        out = {"docx": str(a.docx), "mode": a.mode, "all": all_pr, "same_as_v8": same_pr, "different_from_v8": diff_pr,
               "v8": v8_pr, "thresholds": [{"name": n, "pass": ok, "value": v} for n, ok, v in T]}
        for k in ("all", "same_as_v8", "different_from_v8", "v8"):
            if out[k]:
                out[k] = {kk: vv for kk, vv in out[k].items() if kk != "longest"}
        a.json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"json written to {a.json}")
    print(f"{len(T) - n_fail}/{len(T)} thresholds met")
    if n_fail and not a.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
