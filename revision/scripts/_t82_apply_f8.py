"""T8.2 (2026-10-25): apply the validated F8 paragraphs (revision/f8/*.txt, style aligned with v8) to the manuscript text
source revision/manuscript/F4_manuscript_changes.md. One-off, kept as evidence; it stops at the first paragraph whose F4
counterpart is not unique and changes nothing in that case.

The f8 files carry the FINAL numbering of the assembled manuscript (Table 6, Figure 4, Appendix A, Table A.1, ...), the
in-text citations that assemble_manuscript.py inserts at assembly (Ke / Prokhorenkova in 3.4, Tomczak in 3.10) and the
bold run-in leads of the Discussion paragraphs. F4_manuscript_changes.md is the source BEFORE assembly, so each f8
paragraph is mapped back to the F4 convention (F4 table / figure labels, no assembly-time citations, no leads) and the
round trip F4 -> renumber() -> f8 is asserted. The three 'therefore' cuts of the T8 brief (4.4, 4.5, 4.6) are applied
here, in the source. Captions / legends listed in the brief get 'random forest' -> 'Random Forest'.

Usage (repo root): python revision/scripts/_t82_apply_f8.py [--dry-run]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "scripts"))
from assemble_manuscript import renumber  # noqa: E402

F4 = ROOT / "revision" / "manuscript" / "F4_manuscript_changes.md"
F8 = ROOT / "revision" / "f8"
NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")

# final label -> F4 label (inverse of TABLE_MAP / FIGURE_MAP in assemble_manuscript.py)
INV_TABLE = {"6": "A", "7": "6", "8": "6b", "9": "7", "10": "7b", "11": "8", "13": "9", "14": "10"}
INV_FIGURE = {"2": "[t13]", "3": "[t15]", "4": "[t33]", "5": "[t14]", "6": "2", "7": "3"}

# text added by assemble_manuscript.py at assembly (must not be in the F4 source)
CIT_34_F8 = "LightGBM (Ke et al., 2017) and CatBoost (Prokhorenkova et al., 2018), both gradient-boosted trees,"
CIT_34_F4 = "LightGBM and CatBoost, both gradient-boosted trees,"
CIT_310_F8 = "Kruskal-Wallis tests and epsilon-squared (Tomczak & Tomczak, 2014),"
CIT_310_F4 = "Kruskal-Wallis tests and epsilon-squared,"
FIG3_46_F8 = "Table 8 and Figure 3 extend the device ablation"
FIG3_46_F4 = "Table 6b extends the device ablation"

THEREFORE_CUTS = [
    ("4.4", "the default-parameter comparison therefore does not establish an ordering", "the default-parameter comparison does not establish an ordering"),
    ("4.5", "Calibration is therefore reported as a result and as a limitation", "Calibration is reported as a result and as a limitation"),
    ("4.6", "The model therefore does not depend on the composite scores", "The model does not depend on the composite scores"),
]

# base (v8) part of the 3.2 paragraph: the F4 block holds only the appended text
BASE_32 = ("Each day is assigned to one of three nutritionally relevant classes using thresholds on the ring readiness and sleep "
           "scores: Rest (readiness < 60 or sleep < 60), Active (readiness >= 85 and sleep >= 75), and Normal (all other days). "
           "These thresholds are calibrated on the manufacturer's documentation, which defines scores >= 85 as optimal and scores "
           "< 70 as warranting attention. The resulting distribution is Rest 202 days (21.1%), Normal 710 days (74.1%), and Active "
           "46 days (4.8%). We emphasize that these labels are operational pseudo-labels derived from composite scores, not "
           "externally validated clinical states; this distinction frames the interpretation of all downstream results. ")


def inv(text: str) -> str:
    """Final numbering -> F4 labels (tables, figures, appendix)."""
    text = text.replace("Appendix A, Table A.1", "Appendix Table B1")
    text = re.sub(r"\bTable (\d+)(?![0-9A-Za-z])", lambda m: f"Table {INV_TABLE.get(m.group(1), m.group(1))}", text)
    text = re.sub(r"\bFigure (\d+)(?![0-9A-Za-z])", lambda m: f"Figure {INV_FIGURE.get(m.group(1), m.group(1))}", text)
    return text


def strip_lead(text: str, lead: str) -> str:
    if not text.startswith(lead):
        raise SystemExit(f"STOP: expected lead {lead!r} in: {text[:60]}")
    return text[len(lead):]


def load(name: str, n: int) -> list[str]:
    paras = [p.strip() for p in (F8 / name).read_text(encoding="utf-8").split("\n") if p.strip()]
    if len(paras) != n:
        raise SystemExit(f"STOP: {name} has {len(paras)} paragraphs, expected {n}")
    return paras


def main(dry_run: bool) -> None:
    front = load("f8_front_methods_new.txt", 18)
    res = load("f8_results_new.txt", 11)
    disc = load("f8_discussion_new.txt", 6)
    lines = F4.read_text(encoding="utf-8").split("\n")

    # (label, F4 line prefix (current), f8 text, transform f8 -> F4 source, expected final text after the therefore cuts)
    def identity(t):
        return t

    def t_32(t):
        if not t.startswith(BASE_32):
            raise SystemExit("STOP: f8 3.2 paragraph does not start with the v8 base text")
        return t[len(BASE_32):]

    def t_34(t):
        if t.count(CIT_34_F8) != 1 or t.count("(Section 3.5)") != 1:
            raise SystemExit("STOP: f8 3.4 paragraph: expected citations / 'Section 3.5' not found exactly once")
        return t.replace(CIT_34_F8, CIT_34_F4).replace("(Section 3.5)", "(Section 3.x)")

    def t_310(t):
        if t.count(CIT_310_F8) != 1:
            raise SystemExit("STOP: f8 3.10 paragraph: Tomczak citation not found exactly once")
        return t.replace(CIT_310_F8, CIT_310_F4)

    def t_44(t):
        return cut("4.4", inv(t))

    def t_45(t):
        return cut("4.5", inv(t))

    def t_46(t):
        t = inv(t)
        if t.count(inv(FIG3_46_F8)) != 1:
            raise SystemExit("STOP: f8 4.6 paragraph does not start with the expected Table 8 / Figure 3 sentence")
        return cut("4.6", t.replace(inv(FIG3_46_F8), FIG3_46_F4))

    def cut(sec, t):
        old, new = [(o, n) for s, o, n in THEREFORE_CUTS if s == sec][0]
        if t.count(old) != 1:
            raise SystemExit(f"STOP: 'therefore' cut for {sec} not found exactly once")
        return t.replace(old, new)

    def lead(lead_text):
        return lambda t: inv(strip_lead(t, lead_text))

    plan = [
        ("Abstract", "Personalized nutrition depends on individual", front[0], identity),
        ("Introduction / positioning", "The system studied here is a knowledge-based", front[1], identity),
        ("Contribution 1", "- A machine learning pipeline that, to the best", front[2], lambda t: "- " + t),
        ("Contribution 2", "- An end-to-end explainable prototype", front[3], lambda t: "- " + t),
        ("Contribution 3", "- An evaluation methodology for single-subject", front[4], lambda t: "- " + t),
        ("Contribution 4", "- The results of that evaluation reported in full", front[5], lambda t: "- " + t),
        ("3.2 (appended block)", "The class names are prescriptive, not descriptive:", front[6], t_32),
        ("3.4", "In addition to Random Forest, XGBoost and the SVM", front[7], t_34),
        ("3.5 baselines", "Two baselines are reported alongside", front[8], identity),
        ("3.5 hyperparameter search (3.x)", "Hyperparameters were searched on the training period", front[9], inv),
        ("3.6", "Because macro F1, accuracy, MCC and QWK", front[10], identity),
        ("3.7", "Calibration of the random forest's probabilities", front[11], identity),
        ("3.8", "The stability of the SHAP ranking", front[12], identity),
        ("3.9", "Complementing the per-device ablation", front[13], identity),
        ("3.10", "The pseudo-labels were examined in two ways", front[14], t_310),
        ("3.11", "To assess whether the single temporal split", front[15], identity),
        ("Conclusions p1", "This paper presented and evaluated an explainable", front[16], identity),
        ("Conclusions p2 (future work)", "Future work should prioritize multi-subject", front[17], identity),
        ("4.2", "Table 4 reports the metric set", res[0], inv),
        ("4.3", "Pairwise comparisons confirm what", res[1], identity),
        ("4.4 p1", "Table A reports the seven classifiers", res[2], t_44),
        ("4.4 p2", "Hyperparameter search (Appendix Table B1)", res[3], inv),
        ("4.5", "The raw random forest is over-confident", res[4], t_45),
        ("4.6", "Table 6b extends the device ablation", res[5], t_46),
        ("4.7 threshold sensitivity", "Threshold sensitivity. Table 7 extends", res[6], inv),
        ("4.7 agreement", "Agreement with label-independent indicators. Table 7b compares", res[7], inv),
        ("4.8", "Table 8 and Figure [t33] report", res[8], inv),
        ("4.11", "The SHAP analysis was replicated across data sources", res[9], inv),
        ("4.12", "The confidence threshold bounds the maximum caloric error", res[10], inv),
        ("Discussion / Principal findings", "Within a single-subject design, passively collected", disc[0], lead("Principal findings. ")),
        ("Discussion / Temporal robustness", "The figures reported in Sections 4.2 and 4.3", disc[1], lead("Temporal robustness. ")),
        ("Discussion / Choice of classifier", "With seven classifiers, two baselines and a reportable search", disc[2], lead("Choice of classifier. ")),
        ("Discussion / Generalization", "The single-subject design is a deliberate choice", disc[3], lead("Generalization from a single subject. ")),
        ("Discussion / Limitations", "Eight limitations bound the interpretation", disc[4], lead("Limitations. ")),
        ("Discussion / Practical implications", "Despite these limitations, the system demonstrates", disc[5], lead("Practical implications. ")),
    ]
    captions = [
        ("Table 4 caption", "Table 4. Metrics on 19 features", "of the random forest are in Table 5", "of the Random Forest are in Table 5"),
        ("Table 6b caption (final Table 8)", "Table 6b. Leave-one-group-out ablation", "(random forest, reported hyperparameters)", "(Random Forest, reported hyperparameters)"),
        ("Table 8 caption (final Table 11)", "Table 8. Expanding-window evaluation", "(random forest, reported hyperparameters;", "(Random Forest, reported hyperparameters;"),
        ("Figure [t13] legend (final Figure 2)", "Figure [t13]. (a) Reliability diagram", "Reliability diagram of the random forest", "Reliability diagram of the Random Forest"),
        ("Figure [t14] legend (final Figure 5)", "Figure [t14]. Distribution of SHAP rank", "across 100 random forests retrained", "across 100 Random Forests retrained"),
    ]

    print("| # | block | F4 line | words old -> new | numbers old \\ new | numbers new \\ old | changed |")
    print("|---|---|---|---|---|---|---|")
    new_lines = list(lines)
    for i, (label, prefix, f8text, tf) in enumerate(plan, 1):
        hits = [k for k, l in enumerate(lines) if l.startswith(prefix)]
        if len(hits) != 1:
            raise SystemExit(f"STOP: F4 prefix {prefix!r} matched {len(hits)} lines: {hits}; nothing written")
        k = hits[0]
        old = lines[k]
        new = tf(f8text)
        # round trip: the assembler must reproduce the f8 text (modulo assembly-time insertions and the therefore cuts)
        rt = renumber(new.lstrip("- "))
        expect = f8text
        for sec, o, n in THEREFORE_CUTS:
            if label.startswith(sec):
                expect = expect.replace(o, n)
        if label == "3.2 (appended block)":
            expect = expect[len(BASE_32):]
        if label == "3.4":
            rt = rt.replace(CIT_34_F4, CIT_34_F8)
        if label == "3.10":
            rt = rt.replace(CIT_310_F4, CIT_310_F8)
        if label == "4.6":
            rt = rt.replace("Table 8 extends the device ablation", FIG3_46_F8, 1)
        if label.startswith("Discussion"):
            rt = f8text.split(". ")[0] + ". " + rt
        if label.startswith("Contribution"):
            rt = rt
        if rt != expect:
            # show the first difference
            j = next((x for x in range(min(len(rt), len(expect))) if rt[x] != expect[x]), min(len(rt), len(expect)))
            raise SystemExit(f"STOP: round trip differs for {label} at char {j}:\n  got: {rt[j-40:j+60]}\n  exp: {expect[j-40:j+60]}")
        n_old, n_new = set(NUM_RE.findall(old)), set(NUM_RE.findall(new))
        print(f"| {i} | {label} | {k+1} | {len(old.split())} -> {len(new.split())} | {sorted(n_old - n_new)} | {sorted(n_new - n_old)} | {'yes' if old != new else 'no (identical)'} |")
        new_lines[k] = new
    for label, prefix, old_s, new_s in captions:
        hits = [k for k, l in enumerate(lines) if l.startswith(prefix)]
        if len(hits) != 1 or new_lines[hits[0]].count(old_s) != 1:
            raise SystemExit(f"STOP: caption {label}: prefix matched {len(hits)} lines or phrase not unique; nothing written")
        k = hits[0]
        new_lines[k] = new_lines[k].replace(old_s, new_s)
        print(f"| - | {label} | {k+1} | '{old_s}' -> '{new_s}' | | | yes |")
    # header notes
    for k, l in enumerate(new_lines):
        if l.startswith("### ABSTRACT (inlocuieste integral; 249 cuvinte dupa T7b.1"):
            n_abs = len(front[0].split())
            new_lines[k] = l.rstrip(")") + f"; T8.2 (25 oct 2026): text F8 aliniat la stilul v8, {n_abs} cuvinte)"
        if l.startswith("T7 (22 oct 2026, review intern):"):
            new_lines.insert(k + 2, "T8.2 (25 oct 2026, stil): cele 35 de paragrafe listate in revision/f8/T8_prompt_llm_assistant.md inlocuite cu textul validat din "
                                    "revision/f8/f8_front_methods_new.txt, f8_results_new.txt, f8_discussion_new.txt (mapate inapoi la etichetele F4 de "
                                    "revision/scripts/_t82_apply_f8.py; trei 'therefore' taiate in 4.4, 4.5, 4.6; 'random forest' -> 'Random Forest' in legendele "
                                    "Table 4, Table 6b, Table 8, Figure [t13], Figure [t14]); 3.12 si paragrafele identice cu v8 neatinse; nicio cifra noua.")
            new_lines.insert(k + 3, "")
            break
    if dry_run:
        print("dry run: nothing written")
        return
    F4.write_text("\n".join(new_lines), encoding="utf-8", newline="\n")
    print(f"written: {F4}")


if __name__ == "__main__":
    main("--dry-run" in sys.argv[1:])
