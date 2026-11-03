"""T12 (2026-11-03): sensitive-string grep over the tracked tree of a repository (HEAD), for the public-release dry-run.
Report only. Each hit is classified: (a) under a path of the exclusion list (disappears at filtering), (b) in code or
documents that stay (to be decided), (c) false positive by rule (the word used in another sense). The docx files of
revision/manuscript/ are searched on their extracted text (paragraphs, tables, headers, footers, comments).

Usage: python revision/scripts/_t12_grep.py [--repo PATH] [--out FILE.md] [--label private|clone]
Exit 0 always.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

EXCL = ["data/", "apple_watch_api/raw_json/", "_referinte_upb/", "disertatie_audit.zip", "audit_csv_output.txt",
        "sanity_check_output.txt", "revision/manuscript/submitted/"]
PATTERNS = {
    "e-mail address": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "0040 (phone prefix)": r"0040",
    "tel:": r"tel:",
    "C:\\Users": r"C:\\\\Users",
    "/Users/": r"/Users/",
    "token": r"[Tt]oken",
    "api_key / apikey": r"api_?key",
    "Bearer": r"Bearer",
    "password": r"[Pp]assword",
    "secret": r"[Ss]ecret",
    "client_id": r"client_id",
    ".env": r"\.env\b",
    "Oura + account/invoice/subscription": r"[Oo]ura.*(account|invoice|subscription)|(account|invoice|subscription).*[Oo]ura",
}
# rule-based false positives (c): the word in a non-credential sense
FALSE_POS = {
    "token": [r"tokeniz", r"\btokens?\b.*(word|text|sentence|split|regex|numeric|number|whitespace|exact)", r"(word|text|exact|numeric)[- ]tokens?", r"_tokens\(", r"tokens? = ", r"token match",
              r"^tokenizers==", r"numeric token", r"token \(with", r"\btok\b"],
    "0040 (phone prefix)": [r"[\d.]0040", r"0040\d*[.,]\d"],  # a decimal value such as 0.0040 or 20040.5, not the +40 prefix
    "secret": [r"secretly"],
    "password": [],
    "Bearer": [],
    "client_id": [],
    ".env": [r"environment", r"\.env\w"],
    "e-mail address": [r"noreply@anthropic\.com", r"@\w+\.(org|edu)\b.*(doi|journal)"],
}
DOCX_DIR = "revision/manuscript"


def docx_text(path: Path) -> str:
    import docx
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    parts += [c.text for t in d.tables for r in t.rows for c in r.cells]
    for s in d.sections:
        for hf in (s.header, s.footer, s.first_page_header, s.first_page_footer, s.even_page_header, s.even_page_footer):
            parts += [p.text for p in hf.paragraphs]
    parts += [c.text for c in d.comments]
    with zipfile.ZipFile(path) as z:  # text boxes and drawings are invisible to python-docx: scan the raw XML parts too
        for n in z.namelist():
            if n.startswith("word/") and n.endswith(".xml"):
                parts += re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", z.read(n).decode("utf-8", "replace"))
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="private")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    files = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True, encoding="utf-8").stdout.split("\n")
    files = [f for f in files if f]
    rows = []  # (pattern, file, line, text, class, why)
    for name, pat in PATTERNS.items():
        r = subprocess.run(["git", "grep", "-n", "-I", "-E", pat, "HEAD", "--"], cwd=repo, capture_output=True)
        for line in r.stdout.decode("utf-8", "replace").splitlines():
            m = re.match(r"HEAD:([^:]+):(\d+):(.*)", line)
            if not m:
                continue
            f, ln, txt = m.group(1), int(m.group(2)), m.group(3).strip()
            if any(f.startswith(e) for e in EXCL):
                cls, why = "a", "under an excluded path"
            elif any(re.search(fp, txt, re.I) for fp in FALSE_POS.get(name, [])):
                cls, why = "c", "rule: word in a non-credential sense"
            else:
                cls, why = "b", "stays after filtering; to be decided"
            rows.append((name, f, ln, txt[:160], cls, why))
    # docx of revision/manuscript (extracted text), e-mail / phone only
    docx_rows = []
    for f in files:
        if f.startswith(DOCX_DIR + "/") and f.endswith(".docx"):
            try:
                t = docx_text(repo / f)
            except Exception as e:  # noqa: BLE001
                docx_rows.append((f, f"could not read: {e}", "?"))
                continue
            for name in ("e-mail address", "0040 (phone prefix)", "tel:"):
                hits = sorted(set(re.findall(PATTERNS[name], t)))
                docx_rows.append((f, name, "; ".join(hits) if hits else "0"))
    n = {c: sum(1 for r in rows if r[4] == c) for c in "abc"}
    L = [f"# T12 grep ({a.label}): {repo}", "",
         f"Tracked files: {len(files)}. Hits: {len(rows)} ((a) excluded path {n['a']}, (b) stays {n['b']}, (c) false positive {n['c']}). "
         "Binary files are skipped by git grep -I; the docx files are searched on their extracted text (table below).", "",
         "| Pattern | File | Line | Class | Text |", "|---|---|---|---|---|"]
    for name, f, ln, txt, cls, why in sorted(rows, key=lambda r: (r[4], r[0], r[1], r[2])):
        L.append(f"| {name} | {f} | {ln} | ({cls}) | {txt.replace('|', '/')} |")
    L += ["", "## docx of revision/manuscript (extracted text incl. headers, footers, comments, raw w:t runs)", "",
          "| File | Pattern | Hits |", "|---|---|---|"]
    for f, name, hits in docx_rows:
        L.append(f"| {f} | {name} | {hits} |")
    out = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8", newline="\n")
    print(f"{a.label}: {len(rows)} hits: (a) {n['a']}, (b) {n['b']}, (c) {n['c']}; docx rows {len(docx_rows)}" + (f" -> {a.out}" if a.out else ""))
    for name, f, ln, txt, cls, why in sorted(rows, key=lambda r: (r[4], r[0], r[1], r[2])):
        if cls != "a":
            print(f"  ({cls}) [{name}] {f}:{ln}: {txt[:110]}")
    for f, name, hits in docx_rows:
        if hits != "0":
            print(f"  docx {f} [{name}]: {hits}")


if __name__ == "__main__":
    main()
