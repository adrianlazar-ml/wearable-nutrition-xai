"""T4.1 check: every number in the revised text must have a source (CIFRE_REVIZIE.md, CIFRE_CANONICE.md or
revision/results/*.csv|*.md) at the matching rounding, plus a grep for the forbidden formulations of TEXT_DECISIONS.md.

Usage (from the repo root):
  python revision/scripts/_check_t41_numbers.py --source revision/manuscript/F4_manuscript_changes.md --out revision/manuscript/t41_number_check_F4.md
  python revision/scripts/_check_t41_numbers.py --source revision/manuscript/KBS_Manuscript_R1.docx --out revision/manuscript/t41_number_check_R1.md --phrases

Rules: a text value with d decimals matches a source value v when |v - t| <= 0.5 * 10^-d (i.e. v rounds to t), or when the
exact token appears in a source file; percentages are compared with v and 100 * v; integers must appear as a token (with or
without thousands separator). Section / table / figure numbers, years, dates, CI levels and integers below 20 are skipped
and listed as such. Numbers that are NOT FOUND (NEGASIT) are reported with the three closest source values; nothing is
corrected automatically.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / "revision" / "CIFRE_REVIZIE.md", ROOT / "CIFRE_CANONICE.md"]
RESULTS = ROOT / "revision" / "results"

NUM_RE = re.compile(r"(?<![\w.])([-+−]?)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+(?:e-?\d+)?|\d+e-?\d+|\d+)(%?)")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

FORBIDDEN = ["best performing", "best overall", "stable over time", "optimal threshold", "three years"]
EXTRA_WATCH = ["best model", "significantly", "optimal", "validated labels", "robust to temporal", "generalises across periods"]
# Numbers that come from the unchanged base manuscript (not from the revision) and have no row in the cifre files.
EXPLAINED = {
    "8,734": "base manuscript (3.3, 4.10, contribution list): XML vs JSON step MAE from the dissertation data-quality analysis; not in CIFRE_*; unchanged",
    "15,634": "base manuscript Table 13 / 4.10: mean daily steps XML; dissertation data-quality analysis; unchanged",
    "6,902": "base manuscript Table 13 / 4.10: mean daily steps JSON; dissertation data-quality analysis; unchanged",
    "63.9%": "base manuscript Table 13: XML accuracy, 16 features, aligned subset; dissertation; unchanged",
    "61.8%": "base manuscript Table 13: JSON accuracy, 16 features, aligned subset; dissertation; unchanged",
    "2.41": "base manuscript 4.1: SVM delta 19f minus 16f in percentage points; CIFRE_CANONICE section 16 lists it as -2,41 (sign convention); unchanged",
    "63,153": "base manuscript Table 14: N of Mason et al. (2022), literature value; unchanged",
    "953": "v8 wording of 4.10 (T4.1-C): aligned XML/JSON subset of the pre-freeze dataset, 16 features; dissertation data-quality analysis",
}
TEST_MARKERS = ("p =", "p-value", "p-values", "McNemar", "Holm", "Cochran", "interval", "bootstrap", "test", "excluding zero", "exclude zero")


# ----------------------------------------------------------------------------------------------
def parse_number(tok: str, romanian: bool = False) -> list[float]:
    """Return the float interpretations of a numeric token (several when the format is ambiguous)."""
    out = []
    t = tok
    if romanian:
        # CIFRE_CANONICE: decimal comma ("0,5063", "64,1"), dot thousands ("1.009")
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
            out.append(float(t.replace(".", "")))
        if "," in t and not re.fullmatch(r"\d{1,3}(,\d{3})+", t):
            try:
                out.append(float(t.replace(".", "").replace(",", ".")))
            except ValueError:
                pass
        try:
            out.append(float(t.replace(",", "")))
        except ValueError:
            pass
        return out
    try:
        out.append(float(t.replace(",", "")))
    except ValueError:
        pass
    return out


def load_sources() -> tuple[set[float], list[tuple[str, int, str]], dict[float, tuple[str, int]]]:
    values: set[float] = set()
    where: dict[float, tuple[str, int]] = {}
    lines: list[tuple[str, int, str]] = []
    files = list(SOURCES) + sorted(RESULTS.glob("*.csv")) + sorted(RESULTS.glob("*.md"))
    for f in files:
        romanian = f.name == "CIFRE_CANONICE.md"
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            lines.append((rel, i, line))
            for m in re.finditer(r"[-+−]?\d[\d,.]*(?:e-?\d+)?", line):
                tok = m.group(0).rstrip(".,")
                sign = -1.0 if tok[0] in "-−" else 1.0
                body = tok.lstrip("+-−")
                for v in parse_number(body, romanian):
                    for val in (sign * v,):
                        if val not in where:
                            where[val] = (rel, i)
                        values.add(val)
    return values, lines, where


def decimals_of(tok: str) -> int:
    if "e" in tok:
        return 6
    return len(tok.split(".")[1]) if "." in tok else 0


def context(text: str, start: int, end: int, width: int = 45) -> str:
    return text[max(0, start - width): end + width].replace("\n", " ").replace("|", "/")


def extract_text_md(path: Path) -> str:
    """F4 blocks only: skip the header, headings, Romanian meta sections."""
    skip_sections = ("FORMAT KBS", "SUPPLEMENTARY", "REFERINTE NOI", "De sters", "Ordinea finala", "HIGHLIGHTS")
    out = []
    key = None
    started = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## METHODS"):
            started = True
        if not started:
            continue
        if line.startswith("## ") or line.startswith("### "):
            key = line.lstrip("#").strip()
            continue
        if key and key.startswith(skip_sections):
            continue
        if line.startswith("#") or line.strip() == "---":
            continue
        if key and key.startswith("Contributiile") and line.startswith("Ultimul"):
            continue
        out.append(line)
    return "\n".join(out)


TITLE_PAGE_STYLE = "TitlePage"  # T11.2 (2026-11-03): the title-page paragraphs of assemble_manuscript.py (street number,
TITLE_PAGE_SKIPPED = {"paragraphs": 0, "numbers": 0}  # postal code, telephone) are not result figures and are skipped


def extract_text_docx(path: Path) -> str:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    d = docx.Document(str(path))
    out = []
    for child in d.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            par = Paragraph(child, d)
            t = par.text
            if t.strip() == "References":
                break
            if par.style is not None and par.style.name == TITLE_PAGE_STYLE:
                TITLE_PAGE_SKIPPED["paragraphs"] += 1
                TITLE_PAGE_SKIPPED["numbers"] += len(NUM_RE.findall(t))
                continue
            out.append(t)
        elif tag == "tbl":
            for row in Table(child, d).rows:
                out.append(" | ".join(c.text for c in row.cells))
    return "\n".join(out)


def skip_reason(text: str, m: re.Match, tok: str, pct: str) -> str | None:
    pre = text[max(0, m.start() - 30): m.start()]
    post = text[m.end(): m.end() + 25]
    if re.search(r"(Sections?|Tables?|Figures?|Appendix|Appendix A,|Comment|Weakness|Limitation|Limitations?|fold|Fold|rows?|S|T|G|v|version|Python|Supplementary Table S|Supplementary Figure S)\s*$", pre):
        return "label (section / table / figure / set / fold)"
    if re.search(r"(Sections?|Tables?|Figures?) \d+(\.\d+)? (to|and) $", pre):
        return "label (section / table / figure range)"
    if re.search(r"\(\d+\)\s*$", text[max(0, m.start() - 1): m.end() + 1]) and "." not in tok and int(tok.replace(",", "")) <= 10:
        return "enumeration (n)"
    if re.fullmatch(r"(19|20)\d\d", tok):
        return "year"
    if re.fullmatch(r"\d+\.\d+", tok) and (m.start() == 0 or text[m.start() - 1] == "\n") and re.match(r"\. [A-Z]", post):
        return "heading number"
    if re.match(r"\s+(" + "|".join(MONTHS) + r")\b", post):
        return "date"
    if re.search(r"\b(" + "|".join(MONTHS) + r")\s*$", pre) and "." not in tok:
        return "date"
    if tok in ("95", "2.5", "97.5") and (pct or "percentile" in post or "level" in post or "CI" in post or "interval" in post):
        return "CI level / percentile"
    if "." not in tok and "e" not in tok and int(tok.replace(",", "")) < 20:
        return "small integer (count)"
    if re.search(r"\bC\s*=\s*$", pre) or re.search(r"\bgamma\s*=\s*$", pre):
        return None
    return None


def check(text: str, values: set[float], where: dict, lines: list) -> list[dict]:
    rows = []
    sorted_vals = sorted(values)
    src_text = "\n".join(l for _, _, l in lines)
    for m in NUM_RE.finditer(text):
        sign, tok, pct = m.group(1), m.group(2), m.group(3)
        reason = skip_reason(text, m, tok, pct)
        ctx = context(text, m.start(), m.end())
        if reason:
            rows.append({"number": sign + tok + pct, "status": "SKIPPED", "detail": reason, "context": ctx})
            continue
        if "e" in tok and "." not in tok.split("e")[0] and "e-" not in tok:
            rows.append({"number": tok, "status": "SKIPPED", "detail": "not a number", "context": ctx})
            continue
        neg = sign in ("-", "−")
        try:
            val = float(tok.replace(",", "")) * (-1 if neg else 1)
        except ValueError:
            rows.append({"number": tok, "status": "SKIPPED", "detail": "unparsable", "context": ctx})
            continue
        d = decimals_of(tok)
        found = None
        if "." in tok or "e" in tok:
            tol = 0.5 * 10 ** (-d) + 1e-12
            cands = [tol, tol / 100.0] if pct else [tol]
            for v in sorted_vals:
                if abs(v - val) <= cands[0] or (pct and abs(v * 100 - val) <= tol):
                    found = v; break
            if found is None and pct:
                for v in sorted_vals:
                    if abs(v - val / 100.0) <= tol / 100.0:
                        found = v; break
        else:
            # integer: exact token (with or without thousands separator) in the sources, or equal value
            plain = tok.replace(",", "")
            if re.search(r"(?<![\d.,])" + re.escape(tok) + r"(?![\d])", src_text) or \
               re.search(r"(?<![\d.,])" + re.escape(plain) + r"(?![\d.,]\d)", src_text):
                found = val
            elif val in values:
                found = val
            elif abs(val) >= 100:
                # integer written from a decimal source (e.g. median 600.7 kcal -> 601)
                for v in sorted_vals:
                    if abs(v - val) <= 0.5 + 1e-12 and v != val:
                        found = v; break
        if found is None and re.search(r"(?<![\d.])" + re.escape(tok) + r"(?![\d])", src_text):
            found = val  # exact textual token (e.g. 4.4-fold, 0.00003)
        if found is not None:
            loc = where.get(found) or where.get(val) or ("token match", 0)
            note = "" if found == val else f" (source {found:g}, rounded)"
            rows.append({"number": sign + tok + pct, "status": "FOUND", "detail": f"{loc[0]}:{loc[1]}" + (note or f" (source {found:g})"), "context": ctx})
        elif (sign + tok + pct) in EXPLAINED:
            rows.append({"number": sign + tok + pct, "status": "NEGASIT (explained)", "detail": EXPLAINED[sign + tok + pct], "context": ctx})
        else:
            near = sorted(sorted_vals, key=lambda v: abs(v - val))[:3]
            rows.append({"number": sign + tok + pct, "status": "NEGASIT", "detail": "closest source values: " + ", ".join(f"{v:g}" for v in near), "context": ctx})
    return rows


def phrase_check(text: str) -> list[str]:
    out = []
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    for ph in FORBIDDEN:
        hits = [s for s in sentences if ph.lower() in s.lower()]
        out.append(f"- `{ph}`: {len(hits)} occurrence(s)")
        for s in hits:
            out.append(f"    - {s.strip()[:300]}")
    hits = [s for s in sentences if re.search(r"\bsignificant\b", s, re.I)]
    out.append(f"- `significant` (word): {len(hits)} occurrence(s); each listed with whether a test / interval is named in the same sentence")
    for s in hits:
        has = any(k.lower() in s.lower() for k in TEST_MARKERS)
        out.append(f"    - [{'test named' if has else 'NO TEST NAMED, check manually'}] {s.strip()[:300]}")
    out.append("- additional watch list (not forbidden, for manual review):")
    for ph in EXTRA_WATCH:
        hits = [s for s in sentences if re.search(r"\b" + re.escape(ph) + r"\b", s, re.I)]
        out.append(f"    - `{ph}`: {len(hits)}" + ("" if not hits else " -> " + " || ".join(h.strip()[:160] for h in hits[:6])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phrases", action="store_true", help="also grep the forbidden formulations (TEXT_DECISIONS)")
    args = ap.parse_args()
    src = Path(args.source)
    text = extract_text_docx(src) if src.suffix == ".docx" else extract_text_md(src)
    values, lines, where = load_sources()
    rows = check(text, values, where, lines)
    n_found = sum(r["status"] == "FOUND" for r in rows)
    n_neg = sum(r["status"] == "NEGASIT" for r in rows)
    n_expl = sum(r["status"] == "NEGASIT (explained)" for r in rows)
    n_skip = sum(r["status"] == "SKIPPED" for r in rows)
    L = [f"# T4.1 number check: {src.name}", "",
         f"Source files: CIFRE_REVIZIE.md, CIFRE_CANONICE.md, {len(list(RESULTS.glob('*.csv')))} csv and {len(list(RESULTS.glob('*.md')))} md files in revision/results/. "
         f"Numbers extracted: {len(rows)}; FOUND {n_found}; NEGASIT {n_neg}; NEGASIT (explained, base-manuscript numbers outside the cifre files) {n_expl}; "
         f"SKIPPED {n_skip} (labels, heading numbers, years, dates, CI levels, integers below 20). A FOUND match at d decimals means some source value rounds to the text value; "
         "with about 10,000 source values, short decimals (two decimals or fewer) can match by coincidence, so the source location is given for every number in the last table. "
         f"Title-page paragraphs skipped (T11.2, style {TITLE_PAGE_STYLE}): {TITLE_PAGE_SKIPPED['paragraphs']} paragraphs, {TITLE_PAGE_SKIPPED['numbers']} numbers "
         "(street number, postal code, telephone; not result figures).", ""]
    L.append("## NEGASIT (not found at the matching rounding; to be explained or corrected by hand, never automatically)\n")
    L.append("| # | Number | Status | Context | Closest source values / explanation |\n|---|---|---|---|---|")
    k = 0
    for r in rows:
        if r["status"].startswith("NEGASIT"):
            k += 1
            L.append(f"| {k} | {r['number']} | {r['status']} | ...{r['context']}... | {r['detail']} |")
    if k == 0:
        L.append("| - | none | | | |")
    if args.phrases:
        L.append("\n## Forbidden formulations (TEXT_DECISIONS.md) and watch list\n")
        L.extend(phrase_check(text))
    L.append("\n## All numbers\n")
    L.append("| # | Number | Status | Source / reason | Context |\n|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | {r['number']} | {r['status']} | {r['detail']} | ...{r['context']}... |")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"{src.name}: {len(rows)} numbers, FOUND {n_found}, NEGASIT {n_neg}, SKIPPED {n_skip}; title page skipped: "
          f"{TITLE_PAGE_SKIPPED['paragraphs']} paragraphs, {TITLE_PAGE_SKIPPED['numbers']} numbers -> {args.out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
