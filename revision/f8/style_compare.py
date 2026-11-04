"""Compare writing style: v8 body paragraphs vs R1 paragraphs (kept vs new)."""
import re, sys, difflib, statistics as st
from collections import Counter

v8 = open(sys.argv[1]).read().split("\n\n")
r1 = open(sys.argv[2]).read().split("\n\n")

def is_prose(p):
    if len(p.split()) < 25: return False
    if re.match(r"^(Table|Figure|\d+\.|Keywords|Abstract|References)", p): return False
    if p.count("|") > 2: return False
    if re.search(r"\(\d{4}\)\. ", p) and "https://" in p: return False
    return True

v8p = [p.strip() for p in v8 if is_prose(p.strip())]
r1p = [p.strip() for p in r1 if is_prose(p.strip())]

# classify r1 paragraphs
def best(p):
    return max(difflib.SequenceMatcher(None, p, q, autojunk=False).ratio() for q in v8p)
kept, modified, new = [], [], []
for p in r1p:
    r = best(p)
    (kept if r > 0.9 else modified if r > 0.5 else new).append(p)

ABBR = r"(?:et al|vs|Fig|e\.g|i\.e|approx|St)\."
def sentences(p):
    t = re.sub(ABBR, lambda m: m.group(0).replace(".", "@"), p)
    t = re.sub(r"(\d)\.(\d)", r"\1@\2", t)
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"])", t) if x.strip()]
    return [x.replace("@", ".") for x in s]

def stats(name, paras):
    sents = [s for p in paras for s in sentences(p)]
    L = [len(s.split()) for s in sents]
    words = sum(L)
    txt = " ".join(paras)
    c = lambda pat: len(re.findall(pat, txt))
    print(f"\n== {name}: {len(paras)} paragraphs, {len(sents)} sentences, {words} words")
    print(f"  words/sentence mean {st.mean(L):.1f}, median {st.median(L)}, p90 {sorted(L)[int(0.9*len(L))]}, max {max(L)}")
    print(f"  sentences >40 words: {sum(l>40 for l in L)} ({100*sum(l>40 for l in L)/len(L):.0f}%), >60: {sum(l>60 for l in L)}")
    per1k = lambda n: f"{1000*n/words:.1f}/1k words"
    semi, colon, paren = c(';'), c(r':\s'), c(r'\(')
    print(f"  semicolons {semi} ({per1k(semi)}); colons {colon} ({per1k(colon)}); parentheses {paren} ({per1k(paren)})")
    ci, we, therefore = c(r'\[-?\+?\d'), c(r'\b[Ww]e\b'), c(r'\btherefore\b')
    print(f"  brackets [CI] {ci}; 'we ' {we} ({per1k(we)}); 'therefore' {therefore}")
    print(f"  'distinguish' {c(r'distinguish')}; 'not distinguished' {c(r'not distinguish')}; 'includes zero' {c(r'include[sd]? zero')}; 'excludes zero/excluding zero' {c(r'exclud(es|ing) zero')}")
    print(f"  'Random Forest' {c(r'Random Forest')} vs 'random forest' {c(r'random forest')}")
    best_w = c(r'\bbest\b')
    print(f"  'significant' {c(r'significan')}; 'best' {best_w}; 'robust' {c(r'robust')}; 'honest' {c(r'honest')}; 'deliberate' {c(r'deliberate')}")
    dec4, dec3, pp = c(r'\b0\.\d{4}\b'), c(r'\b0\.\d{3}\b'), c(r'\d pp\b')
    print(f"  4-decimal numbers {dec4}; 3-decimal {dec3}; 'percentage points' {c('percentage points')}; ' pp' {pp}; 'points above/below' {c(r'points (above|below)')}")
    passive = c(r"\b(was|were|is|are|be|been)\s+\w+ed\b")
    print(f"  passive-ish (aux + -ed) {passive} ({per1k(passive)})")
    top = sorted(sents, key=lambda s: -len(s.split()))[:3]
    for s in top: print(f"  LONGEST ({len(s.split())}w): {s[:140]}...")

stats("v8 (ESWA, all prose)", v8p)
stats("R1 kept from v8 (ratio>0.9)", kept)
stats("R1 modified (0.5-0.9)", modified)
stats("R1 new (ratio<0.5)", new)
stats("R1 all prose", r1p)
