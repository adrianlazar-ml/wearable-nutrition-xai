"""T12b.3 (2026-11-03): sensitive-string grep over the WHOLE HISTORY of a repository (git log -p, all refs), for the
filtered public clone. Report only; exit 1 when a hit is classified as real.

Patterns (the T12b brief): e-mail regex, 0040 (Romanian telephone prefix), tel:, OURA_, client_secret, Bearer, token=,
C:\\Users, invoice. Every hit is classified by rule:
  (real) a sensitive value: an e-mail address that is not a no-reply / documentation address, a telephone number
         (0040 followed by nine digits, or tel: followed by a number), a credential value (OURA_* / client_secret /
         token= assigned a literal of 8+ characters, Bearer followed by a literal of 16+ characters);
  (b)    the word in a credential or local-path sense but without a value: the mechanism in oura_auth.py / oura_collect.py
         and in the process documents (os.environ, 'the token was revoked'), the local Windows paths that carry the author's
         own user name, the names of removed files (Oura subscription documents) in the inventories;
  (c)    false positive by rule: decimal / hexadecimal / numeric values that contain 0040, the words token / tokens /
         tokenizer in the text or model sense, the pattern definitions in the grep scripts, tel: without a number,
         no-reply and example.org addresses (RFC 2606), the GitHub no-reply author address.
PASS of T12b.3 = 0 (real) hits; the (b) hits are listed in full for the author's decision, the (c) hits are counted.
Commit author / committer e-mails and commit messages are scanned separately (git log --format).

Usage: python revision/scripts/_t12b_grep_history.py --repo PATH [--out FILE.md] [--label clone]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

PATTERNS = {
    "e-mail address": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "0040": r"0040",
    "tel:": r"tel:",
    "OURA_": r"OURA_",
    "client_secret": r"client_secret",
    "Bearer": r"Bearer",
    "token=": r"token=",
    "C:\\Users": r"C:\\+Users",
    "invoice": r"[Ii]nvoice",
}
REAL = {
    "e-mail address": lambda m, line: not re.search(r"noreply|no-reply|@example\.(org|com|net)\b", m.group(0), re.I),
    "0040": lambda m, line: bool(re.search(r"(?<![\w.])0040[ \d]{9,}(?!\d)", line)) and not re.search(r"[\d.]0040|0040\d*[.,]\d", line),
    "tel:": lambda m, line: bool(re.search(r"tel:\s*[+\d]", line)),
    "OURA_": lambda m, line: bool(re.search(r"OURA_\w+\s*=\s*[\"']?[A-Za-z0-9\-_.:/]{8,}", line)) and "os.environ" not in line and "getenv" not in line,
    "client_secret": lambda m, line: bool(re.search(r"client_secret[\"']?\s*[=:]\s*[\"']?[A-Za-z0-9\-_.]{8,}", line)) and "os.environ" not in line and "getenv" not in line,
    "Bearer": lambda m, line: bool(re.search(r"Bearer\s+[A-Za-z0-9\-_.]{16,}", line)),
    "token=": lambda m, line: bool(re.search(r"token=[\"']?[A-Za-z0-9\-_.]{16,}", line)),
    "C:\\Users": lambda m, line: False,
    "invoice": lambda m, line: False,
}
FALSE_POS = {  # (c) rules, applied to the line when the hit is not real
    "0040": [r"[\d.]0040", r"0040\d*[.,]\d", r"[0-9a-f]{6,}0040|0040[0-9a-f]{6,}", r"0\.0040"],
    "tel:": [r"tel:(?!\s*[+\d])"],
    "e-mail address": [r"noreply|no-reply|@example\.(org|com|net)\b"],
    "token=": [r"tokeniz", r"_token=", r"token=\s*$", r"token=None"],
    "OURA_": [],
    "client_secret": [],
    "Bearer": [],
    "C:\\Users": [],
    "invoice": [],
}
B_WHY = {
    "OURA_": "credential mechanism, no value (environment variable name)",
    "client_secret": "credential mechanism, no value",
    "Bearer": "HTTP header built from a variable, no value",
    "token=": "mechanism / documentation, no value",
    "C:\\Users": "local Windows path with the author's own user name (dry-run: harmless, cosmetic)",
    "invoice": "name of a removed file in an inventory or report; the files themselves are not in the history",
    "tel:": "the word without a number",
    "e-mail address": "address without a person (documentation)",
    "0040": "numeric value",
}


def run(args: list[str], repo: Path) -> bytes:
    return subprocess.run(args, cwd=repo, capture_output=True, check=True).stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="clone")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    head = run(["git", "rev-parse", "--short", "HEAD"], repo).decode().strip()
    ncommits = int(run(["git", "rev-list", "--all", "--count"], repo).decode().strip())
    log = run(["git", "log", "-p", "--all", "--format=COMMIT:%H", "--no-color"], repo).decode("utf-8", "replace")
    rows = []  # (pattern, class, commit, file, text, why)
    commit = file = "?"
    for raw in log.split("\n"):
        if raw.startswith("COMMIT:"):
            commit, file = raw[7:14], "?"
            continue
        if raw.startswith("diff --git "):
            m = re.match(r"diff --git a/(.*) b/(.*)$", raw)
            file = m.group(2) if m else raw[11:]
            continue
        if raw[:1] in "+-" and not raw.startswith("+++") and not raw.startswith("---"):
            line = raw[1:]
        elif raw.startswith("Binary files") or raw.startswith("index ") or raw.startswith("@@"):
            continue
        else:
            continue  # context lines are the same text as a + line of an earlier commit
        for name, pat in PATTERNS.items():
            for m in re.finditer(pat, line):
                if REAL[name](m, line):
                    rows.append((name, "real", commit, file, line.strip()[:140], "sensitive value"))
                elif any(re.search(fp, line, re.I) for fp in FALSE_POS[name]):
                    rows.append((name, "c", commit, file, line.strip()[:140], "rule"))
                else:
                    rows.append((name, "b", commit, file, line.strip()[:140], B_WHY[name]))
    # commit metadata
    authors = sorted(set(run(["git", "log", "--all", "--format=%ae%n%ce"], repo).decode().split()))
    msgs = run(["git", "log", "--all", "--format=COMMIT:%H%n%B"], repo).decode("utf-8", "replace")
    msg_rows = []
    c = "?"
    for line in msgs.split("\n"):
        if line.startswith("COMMIT:"):
            c = line[7:14]
            continue
        for name, pat in PATTERNS.items():
            for m in re.finditer(pat, line):
                if REAL[name](m, line):
                    msg_rows.append((name, "real", c, line.strip()[:140]))
                elif any(re.search(fp, line, re.I) for fp in FALSE_POS[name]):
                    msg_rows.append((name, "c", c, line.strip()[:140]))
                else:
                    msg_rows.append((name, "b", c, line.strip()[:140]))
    real_authors = [x for x in authors if not re.search(r"noreply|no-reply", x)]
    n_real = sum(1 for r in rows if r[1] == "real") + sum(1 for r in msg_rows if r[1] == "real") + len(real_authors)
    # report
    L = [f"# T12b.3 history grep ({a.label}): {repo}", "",
         f"HEAD {head}, {ncommits} commits on all refs; `git log -p --all` scanned line by line (added and removed lines; binary blobs "
         f"appear only as 'Binary files differ'). Classes: (real) sensitive value, (b) word in a credential or local-path sense without a "
         f"value, (c) false positive by rule (see the script docstring). PASS = 0 (real).", "",
         "| Pattern | real | (b) | (c) |", "|---|---|---|---|"]
    for name in PATTERNS:
        cnt = Counter(r[1] for r in rows if r[0] == name)
        L.append(f"| {name} | {cnt['real']} | {cnt['b']} | {cnt['c']} |")
    cnt = Counter(r[1] for r in rows)
    L.append(f"| total | {cnt['real']} | {cnt['b']} | {cnt['c']} |")
    L += ["", f"Commit author / committer addresses: {', '.join(authors)} (real: {len(real_authors)}).",
          f"Commit messages: {sum(1 for r in msg_rows if r[1] == 'real')} real, {sum(1 for r in msg_rows if r[1] == 'b')} (b), "
          f"{sum(1 for r in msg_rows if r[1] == 'c')} (c).", "",
          f"## (real) hits: {cnt['real']}", "", "| Pattern | Commit | File | Text |", "|---|---|---|---|"]
    for name, cls, c, f, t, why in rows:
        if cls == "real":
            L.append(f"| {name} | {c} | {f} | {t.replace('|', '/')} |")
    for name, cls, c, t in msg_rows:
        if cls == "real":
            L.append(f"| {name} | {c} | (commit message) | {t.replace('|', '/')} |")
    L += ["", f"## (b) hits: {cnt['b']} (file, pattern, number of lines over the history; one example line)", "",
          "| File | Pattern | Lines | Why | Example |", "|---|---|---|---|---|"]
    groups: dict[tuple[str, str], list] = {}
    for name, cls, c, f, t, why in rows:
        if cls == "b":
            groups.setdefault((f, name), []).append((c, t, why))
    for (f, name), items in sorted(groups.items()):
        L.append(f"| {f} | {name} | {len(items)} | {items[0][2]} | {items[0][0]} `{items[0][1].replace('|', '/').replace('`', chr(39))[:100]}` |")
    for name, cls, c, t in msg_rows:
        if cls == "b":
            L.append(f"| (commit message) | {name} | 1 | {B_WHY[name]} | {c} `{t.replace('|', '/')[:100]}` |")
    L += ["", f"## (c) hits: {cnt['c']} (by file and pattern, counts only)", "", "| File | Pattern | Lines |", "|---|---|---|"]
    cg = Counter((f, name) for name, cls, c, f, t, why in rows if cls == "c")
    for (f, name), n in sorted(cg.items()):
        L.append(f"| {f} | {name} | {n} |")
    out = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8", newline="\n")
    print(f"{a.label}: HEAD {head}, {ncommits} commits; hits real {cnt['real']} (+{sum(1 for r in msg_rows if r[1] == 'real')} in messages, "
          f"+{len(real_authors)} author addresses), (b) {cnt['b']}, (c) {cnt['c']}" + (f" -> {a.out}" if a.out else ""))
    for name, cls, c, f, t, why in rows:
        if cls == "real":
            print(f"  REAL [{name}] {c} {f}: {t[:110]}")
    for name, cls, c, t in msg_rows:
        if cls == "real":
            print(f"  REAL [{name}] {c} (message): {t[:110]}")
    print("PASS" if n_real == 0 else f"FAIL: {n_real} real hits")
    sys.exit(0 if n_real == 0 else 1)


if __name__ == "__main__":
    main()
