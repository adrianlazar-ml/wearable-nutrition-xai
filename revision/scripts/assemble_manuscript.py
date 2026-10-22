"""T4.1-A: assemble the revised KBS manuscript (Revision 1) from the submitted base docx
and the F4 text blocks, plus Highlights, Supplementary, Response letter and the figure folder.

Inputs (never modified):
  revision/manuscript/BASE_MANUSCRIPT.docx      copy of ESWA_Manuscript_Anonymized_v2.docx (submitted to KBS)
  revision/manuscript/F4_manuscript_changes.md  final text blocks (verbatim; only placeholders are resolved)
  revision/manuscript/F4_response_letter.md     response letter text
  revision/results/*.md, *.csv                  Supplementary tables
  revision/results/figures/*.png|pdf            revision figures
  <legacy-figures-dir>/Figure*_*.png|pdf        Figure 1-3 of the submitted manuscript (outside the repo)

Outputs (revision/manuscript/):
  assembly_plan.md, KBS_Manuscript_R1.docx, KBS_Highlights_R1.docx, KBS_Supplementary_R1.docx,
  KBS_Response_R1.docx, figures/

Run from the repo root:  python revision/scripts/assemble_manuscript.py [--legacy-figures-dir DIR]
Only dependency beyond the frozen environment: python-docx 1.2.0 (revision/requirements.txt).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "revision" / "manuscript"
RES = ROOT / "revision" / "results"
FIGS = RES / "figures"
BASE = MAN / "BASE_MANUSCRIPT_v8.docx"  # T4.1-E: v8 is the version submitted to KBS (title confirmed in Editorial Manager)
BASE_V2 = MAN / "BASE_MANUSCRIPT.docx"  # v2, kept for the record (base of T4.1-A to T4.2-A)
F4 = MAN / "F4_manuscript_changes.md"
LETTER = MAN / "F4_response_letter.md"
OUT_MAN = MAN / "KBS_Manuscript_R1.docx"
OUT_HIGH = MAN / "KBS_Highlights_R1.docx"
OUT_SUPP = MAN / "KBS_Supplementary_R1.docx"
OUT_RESP = MAN / "KBS_Response_R1.docx"
OUT_PLAN = MAN / "assembly_plan.md"
OUT_MARKED = MAN / "KBS_Manuscript_R1_marked.docx"
# T4.1-B: base references not cited in the v2 text that are removed (the four that v8 had already dropped)
UNCITED_REFS_TO_REMOVE = ("De La Torre, K.", "Ferreira, R. E. P.", "Gu, L., & Qian, C.", "Liang, Y.-T.")
# T4.1-C: paragraphs whose wording is taken verbatim from ESWA_Manuscript_Anonymized_v8.docx (in-text citations of the
# method references, Figure 1 legend, 3.9 first sentence, 4.7 aligned-subset sentence). (base prefix, v8 prefix)
V8_WORDING = [
    ("AI systems for nutritional recommendation. AI applications", "AI systems for nutritional recommendation. AI applications"),
    ("Wearables in personalized health. Consumer wearables", "Wearables in personalized health. Consumer wearables"),
    ("Three supervised algorithms with distinct inductive biases", "Three supervised algorithms with distinct inductive biases"),
    ("Data were split temporally: the first 80% of days", "Data were split temporally: the first 80% of days"),
    ("SHAP (SHapley Additive exPlanations) was integrated", "SHAP (SHapley Additive exPlanations; Lundberg & Lee, 2017)"),
    ("The system follows a five-stage flow", "The system follows a staged flow"),
    ("Figure 1. Overview of the five-stage system architecture", "Figure 1. End-to-end flow of the proposed system"),
    ("Table 9 compares models trained on the XML and JSON", "Table 9 compares models trained on the XML and JSON"),
]
# T4.1-D item 13: GenAI declaration text supplied by the first author on 2026-10-19 (names the tool, as the v8 declaration did)
GENAI_DECLARATION = ("During the preparation of this work the authors used Claude (Anthropic) in order to support language editing and "
                     "the structuring of the manuscript. After using this tool, the authors reviewed and edited the content as needed "
                     "and take full responsibility for the content of the published article.")
V8_EXPECTED_CITATIONS = ("Breiman, 2001", "Chen & Guestrin, 2016", "Vapnik, 1995", "Chawla et al., 2002", "Lundberg & Lee, 2017",
                         "Shapley, 1953/2016", "Lundberg et al., 2020", "Zheng et al., 2024", "Thapa et al., 2026")
OUT_FIGDIR = MAN / "figures"

RUN_DATE = "2026-10-19"
TITLE = ("Explainable Artificial Intelligence for Personalized Nutrition: "
         "A Multimodal Framework Integrating Smart Ring and Smartwatch Data")  # v8 title, as submitted

# ----------------------------------------------------------------------------------------------
# Final numbering (order of appearance in the assembled manuscript)
# ----------------------------------------------------------------------------------------------
# F4 label -> final table number
TABLE_MAP = {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "A": "6", "6": "7", "6b": "8",
             "7": "9", "7b": "10", "8": "11", "9": "13", "10": "14",
             "B1": "A.1"}  # T5.3 (KBS p. 28): the appendix table is "Table A.1"
# base table label -> final (base numbering differs from F4 for Table 8/9: base Table 8 = imbalance)
BASE_TABLE_MAP = {"6": "7", "8": "12", "9": "13", "10": "14"}
# F4 figure label -> final figure number
FIGURE_MAP = {"1": "1", "t13": "2", "t15": "3", "t33": "4", "t14": "5", "2": "6", "3": "7"}
SECTION_MAP_TEXT = {"3.x": "3.5"}
# base heading text -> final heading text (sections 3 and 4)
HEADING_MAP = {
    "3.5. Class-Imbalance Handling and Data Splitting": "3.5. Data Splitting, Class Imbalance, Baselines and Hyperparameter Search",
    "3.7. Explainability": "3.8. Explainability",
    "3.8. Nutritional Recommendation Module": "3.12. Nutritional Recommendation Module",
    "3.9. System Architecture and Implementation": "3.13. System Architecture and Implementation",
    "4.3. Statistical Uncertainty": "4.3. Statistical Uncertainty and Model Comparison",
    "4.4. Device-Complementarity Ablation": "4.6. Ablation by Device and by Feature Group",
    "4.5. Labeling-Threshold Sensitivity": "4.7. Label Validation",
    "4.6. Class-Imbalance Strategies": "4.9. Class-Imbalance Strategies",
    "4.7. Data Quality: XML versus JSON": "4.10. Data Quality: XML versus JSON",
    "4.8. SHAP Robustness": "4.11. SHAP Robustness",
    "4.9. Safety Layer": "4.12. Safety Layer",
}
NEW_HEADINGS = {
    "3.7": "3.7. Calibration",
    "3.9": "3.9. Feature-Group Ablation",
    "3.10": "3.10. Label Validation",
    "3.11": "3.11. Temporal Robustness",
    "4.4": "4.4. Additional Classifiers and Hyperparameter Search",
    "4.5": "4.5. Calibration",
    "4.8": "4.8. Temporal Robustness",
}
# cross-references inside retained base paragraphs: (paragraph prefix, old, new)
BASE_XREFS = [
    ("XML versus JSON data source. A key design decision", "Section 4.7", "Section 4.10"),
    ("Data were split temporally: the first 80% of days", "(Section 4.6)", "(Section 4.9)"),
    ("SHAP (SHapley Additive exPlanations", "(Section 4.8)", "(Section 4.11)"),
    ("The system follows a ", "(Section 4.9)", "(Section 4.12)"),
    ("An ablation study trained the Random Forest model", "(Table 6)", "(Table 7)"),
    ("Table 6. Per-device ablation", "Table 6.", "Table 7."),
    ("Five strategies were compared (Table 8)", "(Table 8)", "(Table 12)"),
    ("Table 8. Comparison of class-imbalance strategies", "Table 8.", "Table 12."),
    ("Table 9 compares models trained on the XML and JSON", "Table 9 compares", "Table 13 compares"),
    ("Table 9. Data-source impact", "Table 9.", "Table 13."),
    ("Figure 2. Global feature importance for the Random Forest", "Figure 2.", "Figure 6."),
    ("Figure 3. Confusion matrices on the temporal test set", "Figure 3.", "Figure 7."),
    ("Comparison with the state of the art. Table 10 positions", "Table 10 positions", "Table 14 positions"),
    ("Table 10. Positioning against representative", "Table 10.", "Table 14."),
    # T4.1-B: wording in the retained base paragraph of 4.3
    ("Because the Active class is represented by only 8 test samples", "for the best model", "for the reported model"),
    # T4.1-D: editorial fixes in retained base text (items 5, 6, 7, 8, 14 of the read-through list)
    ("Ring data were retrieved through a cloud REST API",
     "After feature engineering and labeling (Sections 3.2 and 3.3), the unified dataset contained 958 usable days described by nineteen lag-1 features (eleven from the ring, eight from the watch), summarized in Table 1.",
     "The nineteen lag-1 features (eleven from the ring, eight from the watch) are summarized in Table 1."),
    ("Table 2. Macronutrient distribution", "by predicted activity class", "by predicted physiological state"),
    ("Table 7. Per-device ablation", "(Random Forest, T3 labels)", "(Random Forest, reported labels)"),  # after the renumbering above
    # T5.3b (F5): American spelling in the retained base text of Related Work (the title of the Susnjak & Griffin
    # reference keeps 'survival modelling', as published)
    ("Explainability. Explainability is increasingly regarded", "survival modelling in aged care", "survival modeling in aged care"),
    ("SHAP (SHapley Additive exPlanations", "all three algorithms", "the three original algorithms"),
    # T6.1 (F5, KBS p. 25): positional cross-references in retained base text replaced by section numbers;
    # the T4.1-D tuple 'that gap' -> 'the gap identified above' was removed (the submitted wording 'that gap' stands).
    ("All results below use the JSON data source", "All results below use", "All results in Section 4 use"),
    ("Table 13 compares models trained on the XML and JSON", "reported above", "reported in Section 4.2"),  # after the Table 9 -> 13 renumbering above
    # T7.2 (2026-10-22, internal review): the caloric offset per state is defined in Methods 3.12. Verified in the
    # application code: nutrition_recommender.py lines 93-96 (Rest: base_cal - 200), 131 (Normal: base_cal), 163-164
    # (Active: base_cal + 200) with base_cal = round(25 * weight_kg); confidence_threshold.py line 58
    # CALORIC_MAP = {0: -200, 1: 0, 2: 200} (caloric error of Section 4.12). The 200 kcal figure is in CIFRE_CANONICE.md
    # (safety layer, maximum caloric error).
    ("The predicted physiological state is passed to a deterministic, rule-based module",
     "adjusted by the declared goal (maintenance, deficit, or surplus).",
     "adjusted by the declared goal (maintenance, deficit, or surplus). The predicted state then shifts this target by a fixed "
     "offset of -200 kcal for Rest, 0 for Normal and +200 kcal for Active; these offsets are prescriptive rules, not measured "
     "effects (Limitation 3, Section 5), and define the caloric error reported in Section 4.12."),
]
# [t11] cells: persistence per-class F1 from results/persistence_baseline.csv (3 decimals)
T11_SOURCE = RES / "persistence_baseline.csv"

# New references (provisional APA 7 text, F4 section REFERINTE NOI; each gets a Word comment)
# T4.1-F (2026-10-20): fields verified on the publishers' pages, see revision/references_new_verified.md and
# revision/references_new.bib (NeurIPS / AAAI papers cited as proceedings, not arXiv; NeurIPS volumes carry no DOI).
NEW_REFERENCES = [
    "Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning. Proceedings of the "
    "AAAI Conference on Artificial Intelligence, 35(8), 6679-6687. https://doi.org/10.1609/aaai.v35i8.16826",
    "Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for "
    "tabular data. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P. S. Liang, & J. Wortman Vaughan (Eds.), Advances "
    "in Neural Information Processing Systems 34 (pp. 18932-18943). Curran Associates. "
    "https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html",
    "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly "
    "efficient gradient boosting decision tree. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, "
    "S. Vishwanathan, & R. Garnett (Eds.), Advances in Neural Information Processing Systems 30 (pp. 3146-3154). "
    "Curran Associates. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
    "Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting "
    "with categorical features. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, & R. Garnett "
    "(Eds.), Advances in Neural Information Processing Systems 31 (pp. 6638-6648). Curran Associates. "
    "https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html",
    "Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. Frontiers in "
    "Public Health, 5, Article 258. https://doi.org/10.3389/fpubh.2017.00258",
    "Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. An overview of some "
    "recommended measures of effect size. Trends in Sport Sciences, 21(1), 19-25. "
    "https://tss.awf.poznan.pl/The-need-to-report-effect-size-estimates-revisited-An-overview-of-some-recommended,188960,0,2.html",
]
REF_COMMENT = "verify in Zotero (provisional APA 7 entry added at assembly, T4.1)"

# Figure legends written at assembly for figures without a legend in F4 (flagged in assembly_plan.md)
LEGEND_FIG3 = (
    "Figure 3. Feature-group ablation of the random forest (reported hyperparameters, 192-day test set). "
    "(a) Leave-one-group-out: paired bootstrap difference (full model minus ablated model) in macro F1 (circles) "
    "and F1 Active (squares) with 95% intervals on the shared resampling matrix; a positive value means that "
    "removing the group lowers the score. (b) Keep-one-group-in: macro F1 of a forest trained on each group alone, "
    "with F1 Active in parentheses; the dashed line is the full 19-feature model (0.506) and the dotted line the "
    "majority baseline (0.281). Values are in Table 8.")
LEGEND_FIG4 = (
    "Figure 4. Expanding-window evaluation of the random forest (reported hyperparameters) on four disjoint "
    "192-day test blocks, each preceded by all earlier days as training data (190, 382, 574 and 766 days). "
    "Circles: macro F1 with 95% bootstrap interval; squares: F1 Active with interval; diamonds: persistence "
    "baseline macro F1 with interval; crosses: majority baseline macro F1. The rightmost block is the reported "
    "test set. Values are in Table 11.")
LEGEND_S1 = (
    "Figure S1. Labeling-threshold sensitivity grid: (a) macro F1 and (b) F1 Active of the random forest on the "
    "192-day test block for a Rest threshold in {55, 60, 65} and an Active readiness threshold in {80, 85, 90}, "
    "with the Active sleep threshold fixed at 75. The outlined cell is the reported setting (60/85/75). Hatched "
    "cells marked with an asterisk have a single Active test day and are not interpreted. Each cell gives the "
    "number of Active test days. Values for the comparable settings are in Table 9.")
LEGEND_S2 = (
    "Figure S2. Distribution of SHAP rank per feature across 100 random forests trained on the full 766-day "
    "training set with random seeds 0 to 99 (boxes: median and interquartile range over the 100 runs; diamonds: "
    "rank in the reported model, seed 42; features ordered by their rank in the reported model). Companion to "
    "Figure 5 of the main text, which shows the bootstrap mode.")
LEGEND_S3 = (
    "Figure S3. Safety-layer coverage (percentage of the 192 test days that receive the model's prediction rather "
    "than the Normal fallback) as a function of the confidence threshold, for raw, Platt-calibrated and "
    "isotonic-calibrated random forest probabilities. The dotted line marks the 0.50 threshold used by the system. "
    "Under calibrated probabilities the argmax is Normal on every test day at every threshold, so the coverage of "
    "the calibrated variants is not informative.")

PLAN_ROWS: list[dict] = []
DEVIATIONS: list[str] = []


def plan(block: str, target: str, op: str, final: str) -> None:
    PLAN_ROWS.append({"block": block, "target": target[:60], "op": op, "final": final})


# ----------------------------------------------------------------------------------------------
# F4 markdown parsing
# ----------------------------------------------------------------------------------------------
def parse_md_blocks(path: Path) -> dict[str, list[tuple[str, object]]]:
    """Return {heading: [('p', text) | ('table', rows) | ('bullet', text)]} for ## and ### headings."""
    blocks: dict[str, list] = {}
    key = None
    table_rows: list[list[str]] = []
    para_lines: list[str] = []

    def flush_para():
        nonlocal para_lines
        if para_lines and key is not None:
            blocks[key].append(("p", " ".join(s.strip() for s in para_lines)))
        para_lines = []

    def flush_table():
        nonlocal table_rows
        if table_rows and key is not None:
            blocks[key].append(("table", table_rows))
        table_rows = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("## ") or line.startswith("### "):
            flush_para(); flush_table()
            key = line.lstrip("#").strip()
            blocks[key] = []
            continue
        if key is None or line.startswith("# ") or line.strip() == "---":
            continue
        if line.startswith("|"):
            flush_para()
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            table_rows.append(cells)
            continue
        flush_table()
        if not line.strip():
            flush_para()
            continue
        if line.startswith("- "):
            flush_para()
            blocks[key].append(("bullet", line[2:].strip()))
            continue
        para_lines.append(line)
    flush_para(); flush_table()
    return blocks


def block(blocks: dict, prefix: str) -> list:
    hits = [k for k in blocks if k.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"STOP: F4 block '{prefix}' matched {len(hits)} headings: {hits}")
    return blocks[hits[0]]


def paras(items: list) -> list[str]:
    return [t for kind, t in items if kind == "p"]


def tables(items: list) -> list[list[list[str]]]:
    return [t for kind, t in items if kind == "table"]


def bullets(items: list) -> list[str]:
    return [t for kind, t in items if kind == "bullet"]


# ----------------------------------------------------------------------------------------------
# Placeholder / numbering resolution (single pass per family, no chained replacements)
# ----------------------------------------------------------------------------------------------
def renumber(text: str) -> str:
    def tab(m):
        lab = m.group(2)
        if lab in ("A1", "A.1"):          # already final (T5.3): idempotent
            return f"{m.group(1)} A.1"
        if lab not in TABLE_MAP:
            raise SystemExit(f"STOP: unmapped table label '{lab}' in: {text[:80]}")
        return f"{m.group(1)} {TABLE_MAP[lab]}"

    def fig(m):
        lab = m.group(2).strip("[]")
        if lab not in FIGURE_MAP:
            raise SystemExit(f"STOP: unmapped figure label '{lab}' in: {text[:80]}")
        return f"{m.group(1)} {FIGURE_MAP[lab]}"

    text = re.sub(r"\b(Table) (A\.1|A1|B1|A|6b|7b|\d+)(?![0-9A-Za-z])", tab, text)
    text = re.sub(r"\b(Figure) (\[t\d+\]|\d+)(?![0-9A-Za-z])", fig, text)
    text = text.replace("Appendix Table A.1", "Appendix A, Table A.1")
    for old, new in SECTION_MAP_TEXT.items():
        text = text.replace(f"Section {old}", f"Section {new}")
    # response letter only
    text = text.replace("Tables 4, A, 7 and 8", "Tables 4, 6, 9 and 11")
    return text


def check_no_placeholders(text: str, where: str) -> None:
    bad = re.findall(r"\[t\d+\]|\[Section 4\.x\]|\b3\.x\b|\[Figure tNN\]|\[Table X\]|\[title as submitted\]", text)
    if bad:
        raise SystemExit(f"STOP: unresolved placeholders {bad} in {where}: {text[:100]}")


# ----------------------------------------------------------------------------------------------
# docx helpers
# ----------------------------------------------------------------------------------------------
def body_items(doc):
    out = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            out.append(Paragraph(child, doc))
        elif tag == "tbl":
            out.append(Table(child, doc))
    return out


def find_par(doc, prefix: str) -> Paragraph:
    hits = [p for p in doc.paragraphs if p.text.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"STOP: target '{prefix}' matched {len(hits)} base paragraphs")
    return hits[0]


def find_table_after(doc, par: Paragraph) -> Table:
    nxt = par._p.getnext()
    while nxt is not None and nxt.tag.split("}")[1] != "tbl":
        if nxt.tag.split("}")[1] == "p" and Paragraph(nxt, doc).text.strip():
            raise SystemExit(f"STOP: no table directly after '{par.text[:60]}'")
        nxt = nxt.getnext()
    if nxt is None:
        raise SystemExit(f"STOP: no table after '{par.text[:60]}'")
    return Table(nxt, doc)


def set_text(par: Paragraph, text: str, keep_first_run: bool = True) -> None:
    """Replace the paragraph text keeping the formatting of the first run (and a bold lead run if present)."""
    runs = par.runs
    if not runs:
        par.add_run(text)
        return
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    runs[0].text = text


def set_lead_text(par: Paragraph, lead: str, text: str) -> None:
    """Paragraph with a bold run-in lead ("Principal findings. ") followed by normal text."""
    runs = par.runs
    if len(runs) >= 2:
        for r in runs[2:]:
            r._r.getparent().remove(r._r)
        runs[0].text = lead
        runs[1].text = text
    else:
        set_text(par, "")
        runs = par.runs
        runs[0].text = lead
        runs[0].bold = True
        r2 = par.add_run(text)
        r2.bold = None


def clone_paragraph(template: Paragraph, text: str, lead: str | None = None) -> OxmlElement:
    """Deep copy of a template paragraph element with its runs replaced by the given text."""
    new_p = copy.deepcopy(template._p)
    for r in new_p.findall(qn("w:r")):
        new_p.remove(r)
    for h in new_p.findall(qn("w:hyperlink")):
        new_p.remove(h)
    tmpl_runs = template._p.findall(qn("w:r"))
    rpr = tmpl_runs[0].find(qn("w:rPr")) if tmpl_runs else None
    if lead is not None:
        r = OxmlElement("w:r")
        rp = OxmlElement("w:rPr"); b = OxmlElement("w:b"); rp.append(b); r.append(rp)
        t = OxmlElement("w:t"); t.text = lead; t.set(qn("xml:space"), "preserve"); r.append(t)
        new_p.append(r)
        rpr = tmpl_runs[-1].find(qn("w:rPr")) if len(tmpl_runs) > 1 else None
    r = OxmlElement("w:r")
    if rpr is not None:
        r.append(copy.deepcopy(rpr))
    t = OxmlElement("w:t"); t.text = text; t.set(qn("xml:space"), "preserve"); r.append(t)
    new_p.append(r)
    return new_p


def insert_after(anchor, element):
    """Insert an oxml element after anchor (Paragraph, Table or oxml element); returns the element."""
    a = anchor._p if isinstance(anchor, Paragraph) else anchor._tbl if isinstance(anchor, Table) else anchor
    a.addnext(element)
    return element


def insert_before(anchor, element):
    a = anchor._p if isinstance(anchor, Paragraph) else anchor._tbl if isinstance(anchor, Table) else anchor
    a.addprevious(element)
    return element


def set_cell(cell, text: str, bold: bool | None = None, size_pt: float = 10, align=None) -> None:
    p = cell.paragraphs[0]
    for r in p.runs:
        r._r.getparent().remove(r._r)
    run = p.add_run(text)
    run.font.size = Pt(size_pt)
    if bold:
        run.bold = True
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    pf.space_after = Pt(0); pf.space_before = Pt(0); pf.line_spacing = 1.0


def style_table_kbs(tbl: Table) -> None:
    """KBS table format: horizontal rules only (top, header bottom, bottom), no vertical lines, no shading."""
    tblPr = tbl._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge, val in (("top", "single"), ("left", "nil"), ("bottom", "single"), ("right", "nil"),
                      ("insideH", "nil"), ("insideV", "nil")):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), val)
        if val != "nil":
            e.set(qn("w:sz"), "6"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "auto")
        borders.append(e)
    tblPr.append(borders)
    rows = tbl.rows
    for ri, row in enumerate(rows):
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            for old in tcPr.findall(qn("w:shd")) + tcPr.findall(qn("w:tcBorders")):
                tcPr.remove(old)
            if ri == 0:
                tcb = OxmlElement("w:tcBorders")
                b = OxmlElement("w:bottom")
                b.set(qn("w:val"), "single"); b.set(qn("w:sz"), "4"); b.set(qn("w:space"), "0"); b.set(qn("w:color"), "auto")
                tcb.append(b)
                tcPr.append(tcb)


def build_table(doc, rows: list[list[str]], size_pt: float = 10, first_col_left: bool = True) -> Table:
    ncols = max(len(r) for r in rows)
    tbl = doc.add_table(rows=len(rows), cols=ncols)
    tbl.autofit = True
    for ri, row in enumerate(rows):
        for ci in range(ncols):
            txt = row[ci] if ci < len(row) else ""
            align = WD_ALIGN_PARAGRAPH.LEFT if (ci == 0 and first_col_left) else WD_ALIGN_PARAGRAPH.CENTER
            set_cell(tbl.cell(ri, ci), txt, bold=(ri == 0), size_pt=size_pt, align=align)
    style_table_kbs(tbl)
    return tbl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------
# Main manuscript
# ----------------------------------------------------------------------------------------------
def t11_cells() -> dict[str, str]:
    df = pd.read_csv(T11_SOURCE)
    row = df[df["model"] == "persistence"].iloc[0]
    return {"F1 Rest": f"{row['f1_rest']:.3f}", "F1 Normal": f"{row['f1_normal']:.3f}"}


def md_table_to_rows(tbl: list[list[str]], t11: dict[str, str] | None = None) -> list[list[str]]:
    rows = []
    for r in tbl:
        cells = [renumber(c) for c in r]
        if t11 and "[t11]" in " ".join(cells):
            metric = cells[0]
            if metric not in t11:
                raise SystemExit(f"STOP: [t11] cell in unexpected row '{metric}'")
            cells = [t11[metric] if c == "[t11]" else c for c in cells]
        rows.append(cells)
    return rows


def apply_v8_wording(doc, v8_path: Path) -> None:
    """T4.1-C introduced the v8 wording of eight paragraphs; since T4.1-E the base IS v8, so this step only verifies
    that those paragraphs are identical to the v8 file (no-op) and that the nine method citations are present."""
    v8 = docx.Document(str(v8_path))
    for _base_prefix_v2, v8_prefix in V8_WORDING:
        hits = [p.text for p in v8.paragraphs if p.text.startswith(v8_prefix)]
        if len(hits) != 1:
            raise SystemExit(f"STOP: v8 paragraph '{v8_prefix}' matched {len(hits)} paragraphs in the v8 file")
        p = find_par(doc, v8_prefix)
        if p.text.strip() != hits[0].strip():
            raise SystemExit(f"STOP: base paragraph '{v8_prefix}' differs from the v8 file")
    body = "\n".join(p.text for p in doc.paragraphs)
    missing = [c for c in V8_EXPECTED_CITATIONS if c not in body]
    if missing:
        raise SystemExit(f"STOP: expected v8 in-text citations missing from the base: {missing}")
    plan("(v8 wording, verified no-op)", "8 paragraphs listed in V8_WORDING", "verified identical to the v8 file", "unchanged")
    DEVIATIONS.append("base = v8, the first author's working assumption on 2026-10-19; confirmation of the submitted title in Editorial Manager is pending (co-author). "
                      "If the submitted file turns out to be v2, rerun T4.1 with BASE_MANUSCRIPT.docx (v2) as base; all replacement targets exist in both versions "
                      "(verified at T4.1-C and T4.1-E). (T4.1-E, updated T4.1-G): BASE_MANUSCRIPT_v8.docx is ESWA_Manuscript_Anonymized_v8.docx. "
                      "The eight paragraphs that T4.1-C had taken from v8 (Related Work x2, 3.4, 3.5, 3.8, 3.13, Figure 1 legend, 4.10) are now base text and are verified identical to the v8 file; "
                      "the nine method citations (" + ", ".join(V8_EXPECTED_CITATIONS) + ") are therefore in the submitted version, not additions of the revision. "
                      "v8 embeds Figures 1-3 as images (kept in place; the new Figures 2-5 are separate files, legends in text), has the 3.6 equations as OMML objects (untouched) "
                      "and already omits the four orphan references of v2 (De La Torre, Ferreira, Gu & Qian, Liang). BASE_MANUSCRIPT.docx (v2) is kept for the record only.")


def assemble_manuscript(blocks: dict, v8_path: Path | None = None) -> None:
    doc = docx.Document(str(BASE))
    if v8_path is not None:
        apply_v8_wording(doc, v8_path)
    body_tmpl = find_par(doc, "The dataset comprises physiological data from a single participant")
    h2_tmpl = find_par(doc, "3.7. Explainability")
    cap_tmpl = find_par(doc, "Table 4. Detailed metrics on 19 features")
    note_tmpl = find_par(doc, "* Sets T4 and T6 have only 3 Active samples")
    legend_tmpl = find_par(doc, "Figure 2. Global feature importance")
    disc_tmpl = find_par(doc, "Principal findings. Within a single-subject design")
    unnumbered_h_tmpl = find_par(doc, "References")

    def new_body(anchor, text, lead=None, before=False):
        check_no_placeholders(text, "body")
        el = clone_paragraph(body_tmpl if lead is None else disc_tmpl, text, lead)
        return Paragraph((insert_before if before else insert_after)(anchor, el), doc)

    def new_h2(anchor, text, before=False):
        el = clone_paragraph(h2_tmpl, text)
        return Paragraph((insert_before if before else insert_after)(anchor, el), doc)

    def new_caption(anchor, text, before=False):
        check_no_placeholders(text, "caption")
        el = clone_paragraph(cap_tmpl, text)
        return Paragraph((insert_before if before else insert_after)(anchor, el), doc)

    def new_note(anchor, text, before=False):
        el = clone_paragraph(note_tmpl, text)
        p = Paragraph((insert_before if before else insert_after)(anchor, el), doc)
        for r in p.runs:
            r.italic = True
        return p

    def new_legend(anchor, text, before=False):
        check_no_placeholders(text, "legend")
        el = clone_paragraph(legend_tmpl, text)
        return Paragraph((insert_before if before else insert_after)(anchor, el), doc)

    def new_table(anchor, rows, size_pt=10, before=False):
        tbl = build_table(doc, rows, size_pt=size_pt)
        (insert_before if before else insert_after)(anchor, tbl._tbl)
        return tbl

    # ---- Abstract and keywords -----------------------------------------------------------------
    abs_items = block(blocks, "ABSTRACT")
    abstract, keywords_raw = paras(abs_items)[0], paras(abs_items)[1]
    p = find_par(doc, "Personalized nutrition increasingly depends on individual physiological data")
    plan("ABSTRACT", p.text, "replace paragraph", "Abstract")
    set_text(p, abstract)
    m = re.match(r"Keywords \(1-7, KBS\): (.*?)\s*\((?:fara punct final; )?decis", keywords_raw)
    if not m:
        raise SystemExit("STOP: keywords line not in the expected form")
    p = find_par(doc, "Keywords: Personalized nutrition; Wearable devices")
    runs = p.runs
    if len(runs) < 2 or runs[0].text.strip() != "Keywords:":
        raise SystemExit("STOP: keywords paragraph not in the expected two-run form")
    lead_ws = re.match(r"\s*", runs[1].text).group(0)
    runs[1].text = lead_ws + m.group(1).rstrip(". ")  # T4.1-D item 10: no final period
    for r in runs[2:]:
        r._r.getparent().remove(r._r)
    plan("Keywords", "Keywords: Personalized nutrition; Wearable devices; Explainable AI; SHAP", "replace keyword list", "Keywords")

    # ---- Introduction --------------------------------------------------------------------------
    pos = paras(block(blocks, "Paragraf de pozitionare"))[0]
    anchor = find_par(doc, "Despite this progress, a gap remains.")
    new_body(anchor, renumber(pos))
    plan("INTRODUCTION / Paragraf de pozitionare", anchor.text, "insert after", "Section 1, paragraph 4")
    contrib = bullets(block(blocks, "Contributiile"))
    old_bullets = [
        "A machine learning pipeline that, to the best of our knowledge",
        "An ablation study that quantifies the complementarity",
        "A methodological finding relevant to any study using Apple Health data",
        "An end-to-end explainable prototype that couples SHAP explanations",
        "A frank single-subject longitudinal evaluation",
    ]
    if len(contrib) != len(old_bullets):
        raise SystemExit("STOP: contribution bullet count differs from the base list")
    old_pars = [find_par(doc, ob) for ob in old_bullets]  # locate all before any text changes
    for ob, p, nb in zip(old_bullets, old_pars, contrib):
        set_text(p, renumber(nb))
        plan("INTRODUCTION / Contributiile", ob, "replace bullet", "Section 1, contribution list")

    # ---- 3.1 Dataset ---------------------------------------------------------------------------
    p = find_par(doc, "The dataset comprises physiological data from a single participant")
    f4 = paras(block(blocks, "3.1 Dataset"))[0]
    assert f4.startswith("...collected over")
    old_phrase = "collected over approximately three years of continuous wear of two commercial wearable devices."
    dup = " Data span June 2023 to March 2026, totaling 1,009 days with valid data from both devices."
    txt = p.text
    assert old_phrase in txt and dup in txt
    txt = txt.replace(old_phrase, f4[3:]).replace(dup, "")
    set_text(p, txt)
    plan("3.1 Dataset", "The dataset comprises physiological data from a single participant", "replace phrase (sentence 1) + 2 new sentences", "3.1")
    DEVIATIONS.append("3.1: the base sentence 'Data span June 2023 to March 2026, totaling 1,009 days with valid data from both devices.' was dropped because the inserted F4 text restates the dates and the day count in the same paragraph.")

    # ---- 3.2 Labeling (T4.1-D: prescriptive block at the end of the paragraph) -----------------
    p = find_par(doc, "Each day is assigned to one of three nutritionally relevant classes")
    f4 = paras(block(blocks, "3.2 Labeling strategy"))[0]
    if "(readiness >= 85 and sleep >= 75), in which" in f4:
        raise SystemExit("STOP: the repeated threshold parenthesis is still in the F4 3.2 block (T4.1-D item 3)")
    dup = " The sensitivity of the results to the threshold choice is analyzed in Section 4.5."
    txt = p.text
    assert txt.endswith(dup.strip())
    txt = txt.replace(dup, "") + " " + f4
    set_text(p, txt)
    plan("3.2 Labeling strategy", "Each day is assigned to one of three nutritionally relevant classes", "append at the end of the paragraph (T4.1-D order)", "3.2")
    DEVIATIONS.append("3.2: the base sentence 'The sensitivity of the results to the threshold choice is analyzed in Section 4.5.' was dropped because the inserted F4 text carries the same cross-reference (now Section 4.7).")

    # ---- 3.4 Classification models -------------------------------------------------------------
    p = find_par(doc, "Three supervised algorithms with distinct inductive biases")
    cut = " Deep neural networks were deliberately excluded:"
    txt = p.text
    assert cut in txt
    set_text(p, txt[: txt.index(cut)])
    f4 = renumber(paras(block(blocks, "3.4 Classification models"))[0])
    f4 = f4.replace("LightGBM and CatBoost (gradient-boosted trees)",
                    "LightGBM (Ke et al., 2017) and CatBoost (Prokhorenkova et al., 2018) (gradient-boosted trees)")
    # T4.1-D item 4: the base hyperparameter paragraph stays right after the first paragraph; the new paragraph follows it
    p_hyper = find_par(doc, "Hyperparameters were as follows.")
    new_body(p_hyper, f4)
    plan("3.4 Classification models", "Hyperparameters were as follows.", "delete last sentence of paragraph 1; insert new paragraph after the hyperparameter paragraph", "3.4, paragraph 3")
    DEVIATIONS.append("3.4: citations '(Ke et al., 2017)' and '(Prokhorenkova et al., 2018)' were inserted after LightGBM and CatBoost so that the two new references are cited in the text (they were listed in F4 REFERINTE NOI but not cited).")

    # ---- 3.5 Baselines + hyperparameter search -------------------------------------------------
    p = find_par(doc, "Data were split temporally: the first 80% of days")
    pb = new_body(p, renumber(paras(block(blocks, "3.5 Baselines"))[0]))
    plan("3.5 Baselines", "Data were split temporally: the first 80% of days", "insert after", "3.5, paragraph 2")
    ph = new_body(pb, renumber(paras(block(blocks, "3.x Hyperparameter search"))[0]))
    plan("3.x Hyperparameter search", "Data were split temporally: the first 80% of days", "insert after (paragraph 3 of 3.5)", "3.5, paragraph 3; 'Section 3.x' -> 'Section 3.5'")
    DEVIATIONS.append("3.x Hyperparameter search: the proposed numbering (3.6 Evaluation ... 3.13 Architecture) leaves no number for a separate subsection, so the paragraph was placed as the third paragraph of 3.5 (Class-Imbalance Handling and Data Splitting, which defines the temporal folds) and 'Section 3.x' resolves to 'Section 3.5'.")

    # ---- 3.6 Evaluation metrics ----------------------------------------------------------------
    p = find_par(doc, "For C classes, macro F1 is defined as")
    p36 = new_body(p, renumber(paras(block(blocks, "3.6 Evaluation metrics"))[0]))
    plan("3.6 Evaluation metrics", "For C classes, macro F1 is defined as", "insert after (end of 3.6)", "3.6, last paragraph")

    # ---- 3.7 Calibration (new) ---------------------------------------------------------------
    h = new_h2(p36, NEW_HEADINGS["3.7"])
    new_body(h, renumber(paras(block(blocks, "3.7 Calibration"))[0]))
    plan("3.7 Calibration", "3.7. Explainability (heading, insert before)", "new subsection", "3.7")

    # ---- 3.8 Explainability --------------------------------------------------------------------
    p = find_par(doc, "SHAP (SHapley Additive exPlanations")
    new_body(p, renumber(paras(block(blocks, "3.8 Explainability"))[0]))
    plan("3.8 Explainability", "SHAP (SHapley Additive exPlanations) was integrated", "insert after", "3.8, paragraph 2")

    # ---- 3.9, 3.10, 3.11 (new, before 3.12 Recommendation module) ----------------------------
    anchor = find_par(doc, "3.8. Nutritional Recommendation Module")
    for num, key in (("3.9", "3.9 Feature-group ablation"), ("3.10", "3.10 Label validation"), ("3.11", "3.11 Temporal robustness")):
        h = new_h2(anchor, NEW_HEADINGS[num], before=True)
        txt = renumber(paras(block(blocks, key))[0])
        if num == "3.10":
            txt = txt.replace("Kruskal-Wallis tests with epsilon-squared,",
                              "Kruskal-Wallis tests with epsilon-squared (Tomczak & Tomczak, 2014),")
        if num == "3.9":
            old = "(identity with Table 7 verified)"
            if old not in txt:
                raise SystemExit("STOP: expected cross-reference to the device-ablation table not found in 3.9")
            txt = txt.replace(old, "(identity with the per-device ablation verified)")
            DEVIATIONS.append("3.9: '(identity with Table 6 verified)' became '(identity with the per-device ablation verified)': citing the device-ablation table (final Table 7) in Methods would place its first citation before Tables 2 to 6 and break the order-of-appearance rule; the same paragraph already points to Section 4.6.")
        new_body(anchor, txt, before=True)
        plan(key, "3.8. Nutritional Recommendation Module (heading, insert before)", "new subsection", num)
    DEVIATIONS.append("3.10: citation '(Tomczak & Tomczak, 2014)' inserted after 'epsilon-squared' so that the new reference is cited in the text.")

    # ---- heading renumbering (sections 3 and 4) ------------------------------------------------
    for old, new in HEADING_MAP.items():
        p = find_par(doc, old)
        set_text(p, new)
        plan("(heading)", old, "renumber heading", new)

    # ---- 4.2 Detailed results ------------------------------------------------------------------
    items = block(blocks, "4.2 Detailed results")
    p = find_par(doc, "Table 4 reports the full metric set on the 19-feature configuration")
    set_text(p, renumber(paras(items)[0]))
    plan("4.2 Detailed results (text)", "Table 4 reports the full metric set on the 19-feature configuration", "replace paragraph", "4.2")
    cap = find_par(doc, "Table 4. Detailed metrics on 19 features")
    old_tbl = find_table_after(doc, cap)
    set_text(cap, renumber(paras(items)[1]))
    old_tbl._tbl.getparent().remove(old_tbl._tbl)
    new_table(cap, md_table_to_rows(tables(items)[0], t11_cells()))
    plan("4.2 Detailed results (Table 4)", "Table 4. Detailed metrics on 19 features", "replace caption + table", "Table 4; [t11] cells from results/persistence_baseline.csv")

    # ---- 4.3 Statistical uncertainty and comparison ------------------------------------------
    cap = find_par(doc, "Table 5. Bootstrap 95% confidence intervals")
    tbl5 = find_table_after(doc, cap)
    p43 = new_body(tbl5, renumber(paras(block(blocks, "4.3 Statistical uncertainty"))[0]))
    plan("4.3 Statistical uncertainty and model comparison", "Table 5. Bootstrap 95% confidence intervals (table, insert after)", "insert after Table 5", "4.3, paragraph 2")

    # ---- 4.4 Additional classifiers (new) and 4.5 Calibration (new), before 4.6 heading -----
    anchor = find_par(doc, HEADING_MAP["4.4. Device-Complementarity Ablation"])
    items = block(blocks, "4.4 Additional classifiers")
    ps, tb = paras(items), tables(items)[0]
    new_h2(anchor, NEW_HEADINGS["4.4"], before=True)
    new_body(anchor, renumber(ps[0]), before=True)
    new_caption(anchor, renumber(ps[1]), before=True)
    new_table(anchor, md_table_to_rows(tb), before=True)
    new_note(anchor, renumber(ps[2]), before=True)
    new_body(anchor, renumber(ps[3]), before=True)
    plan("4.4 Additional classifiers and hyperparameter search", "4.6. Ablation by Device and by Feature Group (heading, insert before)", "new subsection (2 paragraphs, Table A, note)", "4.4; Table A -> Table 6; Table B1 -> Table A.1")
    items = block(blocks, "4.5 Calibration")
    ps = paras(items)
    new_h2(anchor, NEW_HEADINGS["4.5"], before=True)
    new_body(anchor, renumber(ps[0]), before=True)
    new_legend(anchor, renumber(ps[1]), before=True)
    plan("4.5 Calibration", "4.6. Ablation by Device and by Feature Group (heading, insert before)", "new subsection (paragraph + legend)", "4.5; Figure [t13] -> Figure 2")

    # ---- 4.6 Ablation: add group ablation after Table 7 (device) -------------------------------
    cap = find_par(doc, "Table 6. Per-device ablation")  # renumbered below via BASE_XREFS
    tbl7 = find_table_after(doc, cap)
    items = block(blocks, "4.6 Ablation")
    ps, tb = paras(items), tables(items)[0]
    txt = renumber(ps[0])
    assert txt.startswith("Table 8 extends the device ablation")
    txt = txt.replace("Table 8 extends the device ablation", "Table 8 and Figure 3 extend the device ablation", 1)
    p46 = new_body(tbl7, txt)
    c46 = new_caption(p46, renumber(ps[1]))
    t46 = new_table(c46, md_table_to_rows(tb))
    new_legend(t46, LEGEND_FIG3)
    plan("4.6 Ablation (group ablation)", "Table 6. Per-device ablation (table, insert after)", "insert paragraph + Table 6b + legend", "4.6; Table 6b -> Table 8; Figure t15 -> Figure 3")
    DEVIATIONS.append("4.6: F4 had no citation of the group-ablation figure; 'Table 8 extends' became 'Table 8 and Figure 3 extend' so that Figure 3 is cited in the text, and the legend of Figure 3 was written at assembly (not in F4).")

    # ---- 4.7 Label validation (replaces 4.5 base) ----------------------------------------------
    items = block(blocks, "4.7 Label validation")
    ps, tbs = paras(items), tables(items)
    p = find_par(doc, "Seven threshold sets were tested (Table 7).")
    lead, rest = "Threshold sensitivity. ", renumber(ps[0])
    assert rest.startswith(lead)
    set_lead_text(p, lead, rest[len(lead):])
    cap = find_par(doc, "Table 7. Labeling-threshold sensitivity.")
    old_tbl = find_table_after(doc, cap)
    set_text(cap, renumber(ps[1]))
    old_tbl._tbl.getparent().remove(old_tbl._tbl)
    t9 = new_table(cap, md_table_to_rows(tbs[0]), size_pt=9)
    note = find_par(doc, "* Sets T4 and T6 have only 3 Active samples")
    set_text(note, renumber(ps[2]))
    for r in note.runs:
        r.italic = True
    lead2 = "Agreement with label-independent indicators. "
    rest2 = renumber(ps[3])
    assert rest2.startswith(lead2)
    pa = Paragraph(insert_after(note, clone_paragraph(disc_tmpl, rest2[len(lead2):], lead2)), doc)
    c10 = new_caption(pa, renumber(ps[4]))
    t10 = new_table(c10, md_table_to_rows(tbs[1]), size_pt=9)
    if len(ps) > 5:  # T4.1-D item 9: note under Table 10 (n varies by indicator)
        new_note(t10, renumber(ps[5]))
    plan("4.7 Label validation (threshold paragraph)", "Seven threshold sets were tested (Table 7).", "replace paragraph", "4.7; Table 7 -> Table 9")
    plan("4.7 Label validation (Table 7)", "Table 7. Labeling-threshold sensitivity.", "replace caption + table + note", "Table 9")
    plan("4.7 Label validation (agreement)", "* Sets T4 and T6 have only 3 Active samples (note, insert after)", "insert paragraph + Table 7b", "4.7; Table 7b -> Table 10")

    # ---- 4.8 Temporal robustness (new), before 4.9 heading -------------------------------------
    anchor = find_par(doc, HEADING_MAP["4.6. Class-Imbalance Strategies"])
    items = block(blocks, "4.8 Temporal robustness")
    ps, tb = paras(items), tables(items)[0]
    new_h2(anchor, NEW_HEADINGS["4.8"], before=True)
    new_body(anchor, renumber(ps[0]), before=True)
    new_caption(anchor, renumber(ps[1]), before=True)
    new_table(anchor, md_table_to_rows(tb), size_pt=8, before=True)
    new_legend(anchor, LEGEND_FIG4, before=True)
    plan("4.8 Temporal robustness", "4.9. Class-Imbalance Strategies (heading, insert before)", "new subsection (paragraph, Table 8, legend)", "4.8; Table 8 -> Table 11; Figure t33 -> Figure 4")
    DEVIATIONS.append("4.8: the legend of Figure 4 (expanding window) was written at assembly (F4 cites Figure [t33] but has no legend).")

    # ---- 4.11 SHAP -----------------------------------------------------------------------------
    items = block(blocks, "4.11 SHAP")
    ps = paras(items)
    p = find_par(doc, "The SHAP analysis was replicated across all configurations")
    set_text(p, renumber(ps[0]))
    leg6 = find_par(doc, "Figure 2. Global feature importance for the Random Forest")
    new_legend(p, renumber(ps[2]))  # Figure 5 legend (t14) right after the paragraph, before Figure 6 legend
    m = re.match(r"Figure 2 \(legenda, adaugat la final\)\. (.*)", ps[1])
    if not m:
        raise SystemExit("STOP: Figure 2 legend addition not in the expected form")
    set_text(leg6, leg6.text + " " + m.group(1))
    plan("4.11 SHAP (text)", "The SHAP analysis was replicated across all configurations", "replace paragraph", "4.11")
    plan("4.11 SHAP (Figure 2 legend)", "Figure 2. Global feature importance for the Random Forest", "append sentence; renumber", "Figure 6")
    plan("4.11 SHAP (Figure t14 legend)", "The SHAP analysis was replicated across all configurations (insert after)", "insert legend", "Figure 5")

    # ---- 4.12 Safety layer ---------------------------------------------------------------------
    p = find_par(doc, "The confidence-threshold mechanism bounds the system's maximum caloric error")
    set_text(p, renumber(paras(block(blocks, "4.12 Safety layer"))[0]))
    plan("4.12 Safety layer", "The confidence-threshold mechanism bounds the system's maximum caloric error", "replace paragraph", "4.12; Figure 3 -> Figure 7")

    # ---- base cross-references -----------------------------------------------------------------
    for prefix, old, new in BASE_XREFS:
        p = find_par(doc, prefix)
        txt = p.text
        if txt.count(old) != 1:
            raise SystemExit(f"STOP: cross-reference '{old}' occurs {txt.count(old)} times in '{prefix}'")
        # keep run structure when the replacement lies inside one run
        done = False
        for r in p.runs:
            if old in r.text:
                r.text = r.text.replace(old, new); done = True; break
        if not done:
            set_text(p, txt.replace(old, new))
        plan("(cross-reference)", prefix, f"'{old}' -> '{new}'", new)
    DEVIATIONS.append("T6.1 (F5, 2026-10-20, KBS p. 25): the seven positional cross-references ('above' / 'below' standing for a section, "
                      "table or figure number) were replaced: in the base text 'All results below use' -> 'All results in Section 4 use' (4), "
                      "'reported above' -> 'reported in Section 4.2' (4.10), and the T4.1-D wording 'the gap identified above' reverted to the "
                      "submitted 'that gap' (Introduction, same section); in F4 text 'the three classifiers above' -> 'Random Forest, XGBoost and "
                      "the SVM' (3.4), 'computed as above' -> 'computed as in Sections 3.5 and 3.6' (3.11), 'the two results above' -> 'the two "
                      "results (Table 7 and Table 8)' (4.6), '(Temporal robustness, above)' -> '(Section 4.8)' (Discussion). The twelve numeric "
                      "comparisons ('7.8 points above', 'below persistence', ...) are unchanged.")
    DEVIATIONS.append("T7.2 (2026-10-22, internal review of 22 Oct): in the retained base paragraph of 3.12 the sentence 'The predicted state then "
                      "shifts this target by a fixed offset of -200 kcal for Rest, 0 for Normal and +200 kcal for Active; these offsets are "
                      "prescriptive rules, not measured effects (Limitation 3, Section 5), and define the caloric error reported in Section 4.12.' "
                      "was inserted after '...adjusted by the declared goal (maintenance, deficit, or surplus).' (BASE_XREFS tuple). The offsets "
                      "were verified in the application code before writing the sentence: nutrition_recommender.py lines 93-96, 131 and 163-164 "
                      "(base_cal = round(25 * weight_kg); Rest base_cal - 200, Normal base_cal, Active base_cal + 200) and confidence_threshold.py "
                      "line 58 (CALORIC_MAP = {0: -200, 1: 0, 2: 200}, the caloric error of Section 4.12); 200 kcal is the maximum caloric error of "
                      "the safety layer in CIFRE_CANONICE.md.")

    # ---- Discussion ----------------------------------------------------------------------------
    p = find_par(doc, "Principal findings. Within a single-subject design")
    set_lead_text(p, "Principal findings. ", renumber(paras(block(blocks, "Principal findings"))[0]))
    plan("DISCUSSION / Principal findings", "Principal findings. Within a single-subject design", "replace paragraph", "Section 5, paragraph 1")
    last = p
    for key, lead in (("Temporal robustness (paragraf nou", "Temporal robustness. "),
                      ("Choice of classifier", "Choice of classifier. "),
                      ("Generalization from a single subject", "Generalization from a single subject. ")):
        last = new_body(last, renumber(paras(block(blocks, key))[0]), lead=lead)
        plan(f"DISCUSSION / {key.split(' (')[0]}", "Principal findings. Within a single-subject design (insert after, in order)", "insert paragraph", "Section 5")
    p = find_par(doc, "Comparison with the state of the art. Table")
    sent = renumber(paras(block(blocks, "Comparison with the state of the art"))[0])
    p.runs[-1].text = p.runs[-1].text.rstrip() + " " + sent
    plan("DISCUSSION / Comparison with the state of the art", "Comparison with the state of the art. Table 10 positions", "append sentence", "Section 5")
    cap = find_par(doc, "Table 14. Positioning against representative")
    t14 = find_table_after(doc, cap)
    row = [r for r in t14.rows if r.cells[0].text.strip() == "This study"]
    if len(row) != 1:
        raise SystemExit("STOP: 'This study' row not unique in Table 14")
    hdr = [c.text.strip() for c in t14.rows[0].cells]
    row = row[0]
    row.cells[hdr.index("Algorithm")].paragraphs[0].runs[0].text = "RF (reported); XGBoost, SVM, LightGBM, CatBoost, TabNet, FT-Transformer compared"
    for r in row.cells[hdr.index("Algorithm")].paragraphs[0].runs[1:]:
        r._r.getparent().remove(r._r)
    row.cells[hdr.index("N")].paragraphs[0].runs[0].text = "1 (958 days)"
    for r in row.cells[hdr.index("N")].paragraphs[0].runs[1:]:
        r._r.getparent().remove(r._r)
    plan("DISCUSSION / Table 10 row 'This study'", "Table 10. Positioning against representative (table)", "update cells Algorithm, N", "Table 14")
    p = find_par(doc, "Limitations. Six limitations bound the interpretation")
    set_lead_text(p, "Limitations. ", renumber(paras(block(blocks, "Limitations (inlocuieste integral"))[0]))
    plan("DISCUSSION / Limitations", "Limitations. Six limitations bound the interpretation", "replace paragraph", "Section 5")
    p = find_par(doc, "Practical implications. Despite these limitations")
    set_lead_text(p, "Practical implications. ", renumber(paras(block(blocks, "Practical implications"))[0]))
    plan("DISCUSSION / Practical implications", "Practical implications. Despite these limitations", "replace paragraph", "Section 5")

    # ---- Conclusions ---------------------------------------------------------------------------
    ps = paras(block(blocks, "CONCLUSIONS"))
    p = find_par(doc, "This paper presented and evaluated an explainable AI system")
    set_text(p, renumber(ps[0]))
    p2 = find_par(doc, "Future work should prioritize multi-subject validation")
    set_text(p2, renumber(ps[1]))
    plan("CONCLUSIONS", "This paper presented and evaluated an explainable AI system", "replace 2 paragraphs", "Section 6")

    # ---- GenAI declaration (T4.1-D item 13) ----------------------------------------------------
    p = find_par(doc, "During the preparation of this work the author(s) used")
    set_text(p, GENAI_DECLARATION)
    plan("GenAI declaration (T4.1-D)", "During the preparation of this work the author(s) used", "replace paragraph", "Declaration of generative AI")
    DEVIATIONS.append("GenAI declaration (T4.1-D item 13, T4.1-E): text supplied by the first author on 2026-10-19. Relative to the v8 base (which already names the declared assistant) "
                      "the changes are 'author(s)' -> 'authors' and the closing words 'content of the publication' -> 'content of the published article' (KBS template). "
                      "The KBS Guide for Authors page could not be read by the assembly session (KBS_journal.pdf has no text layer and no PDF renderer was available).")

    # ---- T6.2 (F5): declaration sections before the GenAI declaration (DRAFT for the coauthor) ---------
    genai_h = find_par(doc, "Declaration of generative AI")
    decl = paras(block(blocks, "DECLARATIONS"))
    if len(decl) != 6:
        raise SystemExit(f"STOP: DECLARATIONS block must hold 6 paragraphs (3 headings + 3 texts), found {len(decl)}")
    for i in range(0, 6, 2):
        head, text = decl[i], decl[i + 1]
        hp = Paragraph(insert_before(genai_h, clone_paragraph(unnumbered_h_tmpl, head)), doc)
        bp = new_body(genai_h, text, before=True)
        note = ""
        if head.startswith("CRediT"):
            doc.add_comment(runs=bp.runs, text="confirm roles with coauthor", author="T6.2 draft", initials="T62")
            doc.add_comment(runs=hp.runs, text="remove if KBS requires anonymized main document; check Editorial Manager",
                            author="T6.2 draft", initials="T62")
            note = " + comments 'confirm roles with coauthor' (text), 'remove if KBS requires anonymized main document; check Editorial Manager' (heading)"
        elif head.startswith("Funding"):
            doc.add_comment(runs=bp.runs, text="confirm no funding with coauthor", author="T6.2 draft", initials="T62")
            note = " + comment 'confirm no funding with coauthor'"
        plan("DECLARATIONS", "Declaration of generative AI (heading, insert before)", "new unnumbered section (heading + paragraph)" + note, head)
    DEVIATIONS.append("T6.2 (F5, 2026-10-20, KBS pp. 26-27): three declaration sections (CRediT authorship contribution statement, Declaration of "
                      "competing interest, Funding) inserted after Conclusions and before the GenAI declaration as a DRAFT for the coauthor (text in "
                      "F4_manuscript_changes.md, block DECLARATIONS), with Word comments 'confirm roles with coauthor', 'confirm no funding with "
                      "coauthor' and, because the v8 base is anonymized (no authors on the title page), 'remove if KBS requires anonymized main "
                      "document; check Editorial Manager' on the CRediT heading. Final order: CRediT, competing interest, Funding, GenAI declaration, "
                      "Data availability, Appendix A, References. No Acknowledgements section (nobody to acknowledge; added only if the coauthor asks).")

    # ---- Appendix A before References ----------------------------------------------------------
    refs_h = find_par(doc, "References")
    items = block(blocks, "APPENDIX")
    ps, tb = paras(items), tables(items)[0]
    hA = Paragraph(insert_before(refs_h, clone_paragraph(unnumbered_h_tmpl, "Appendix A. Hyperparameter search")), doc)
    new_caption(refs_h, renumber(ps[0]), before=True)
    new_table(refs_h, md_table_to_rows(tb), size_pt=7, before=True)
    new_note(refs_h, renumber(ps[1]), before=True)
    plan("APPENDIX / Table B1", "References (heading, insert before)", "new appendix (heading, caption, table, note)", "Appendix A, Table A.1")

    # ---- References: 6 new entries, alphabetical, with Word comments ---------------------------
    ref_pars = []
    started = False
    for p in doc.paragraphs:
        if p.text == "References":
            started = True; continue
        if started and p.text.strip():
            ref_pars.append(p)
    ref_tmpl = ref_pars[0]
    for ref in NEW_REFERENCES:
        key = ref.split("(")[0].strip().lower()
        after = None
        for rp in ref_pars:
            if rp.text.split("(")[0].strip().lower() < key:
                after = rp
        if after is None:
            newp = Paragraph(insert_before(ref_pars[0], clone_paragraph(ref_tmpl, ref)), doc)
        else:
            newp = Paragraph(insert_after(after, clone_paragraph(ref_tmpl, ref)), doc)
        doc.add_comment(runs=newp.runs, text=REF_COMMENT, author="T4.1 assembly", initials="T41")
        ref_pars.append(newp)
        ref_pars.sort(key=lambda q: q.text.split("(")[0].strip().lower())
        plan("REFERINTE NOI", f"References: {ref[:40]}", "insert alphabetically + comment 'verify in Zotero'", "References")

    # ---- uncited base references (T4.1-B) ------------------------------------------------------
    uncited = find_uncited_references(doc)
    removed, kept = [], []
    for rp in uncited:
        if rp.text.startswith(UNCITED_REFS_TO_REMOVE):
            removed.append(rp.text[:60])
            rp._p.getparent().remove(rp._p)
        else:
            kept.append(rp.text[:60])
    DEVIATIONS.append("References not cited anywhere in the assembled text (base list of v8): " + str(len(uncited)) + ". Removed: "
                      + ("; ".join(removed) if removed else "none (the v8 base already omits the four orphan entries of v2)") + ". Kept although uncited: " + ("; ".join(kept) if kept else "none") + ".")
    for r in removed:
        plan("(uncited reference)", r, "delete reference entry", "References")

    # ---- restyle all tables (KBS: no vertical lines, no shading) -------------------------------
    for t in doc.tables:
        style_table_kbs(t)
    DEVIATIONS.append("All tables (base and new) were restyled to the KBS format (horizontal rules only, no vertical lines, no shading); cell contents of base tables are unchanged.")

    doc.save(str(OUT_MAN))


def find_uncited_references(doc) -> list[Paragraph]:
    """Reference entries whose first author surname does not appear with the entry's year anywhere in the text
    (paragraphs and table cells before the References heading)."""
    body_text, refs, started = [], [], False
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            p = Paragraph(child, doc)
            if p.text.strip() == "References":
                started = True; continue
            (refs if started else body_text).append(p if started else p.text)
        elif tag == "tbl" and not started:
            for row in Table(child, doc).rows:
                body_text.append(" ".join(c.text for c in row.cells))
    text = "\n".join(body_text)
    out = []
    for rp in refs:
        if not rp.text.strip():
            continue
        surname = rp.text.split(",")[0].strip()
        m = re.search(r"\((\d{4})", rp.text)
        year = m.group(1) if m else None
        # surname followed, within 80 characters and before any closing parenthesis, by the entry's year
        pat = re.escape(surname) + r"[^)]{0,80}?" + (re.escape(year) if year else "")
        if year is None or re.search(pat, text) is None:
            out.append(rp)
    return out


# ----------------------------------------------------------------------------------------------
# Highlights, Supplementary, Response letter (built on the base styles)
# ----------------------------------------------------------------------------------------------
class Templates:
    def __init__(self, doc):
        self.doc = doc
        self.title = find_par(doc, "Explainable Artificial Intelligence for Personalized Nutrition")
        self.h_unnum = find_par(doc, "References")
        self.h1 = find_par(doc, "1. Introduction")
        self.h2 = find_par(doc, "3.7. Explainability")
        self.body = find_par(doc, "The dataset comprises physiological data from a single participant")
        self.cap = find_par(doc, "Table 4. Detailed metrics on 19 features")
        self.legend = find_par(doc, "Figure 2. Global feature importance")
        self.bullet = find_par(doc, "A machine learning pipeline that, to the best of our knowledge")
        self.lead = find_par(doc, "Principal findings. Within a single-subject design")
        # detach copies before the body is cleared
        for name in ("title", "h_unnum", "h1", "h2", "body", "cap", "legend", "bullet", "lead"):
            setattr(self, name, Paragraph(copy.deepcopy(getattr(self, name)._p), doc))


def empty_doc_from_base():
    doc = docx.Document(str(BASE))
    tm = Templates(doc)
    body = doc.element.body
    for child in list(body.iterchildren()):
        if child.tag.split("}")[1] != "sectPr":
            body.remove(child)
    # v8 embeds three images; drop their relationships so the media are not carried into the derived files
    for rid, rel in list(doc.part.rels.items()):
        if rel.reltype.endswith("/image"):
            doc.part.drop_rel(rid)
    return doc, tm


def append(doc, tmpl: Paragraph, text: str, lead: str | None = None, italic=False, indent_cm=None) -> Paragraph:
    el = clone_paragraph(tmpl, text, lead)
    sect = doc.element.body.find(qn("w:sectPr"))
    sect.addprevious(el)
    p = Paragraph(el, doc)
    if italic:
        for r in p.runs:
            r.italic = True
    if indent_cm is not None:
        p.paragraph_format.left_indent = Cm(indent_cm)
    return p


def append_table(doc, rows, size_pt=9) -> Table:
    tbl = build_table(doc, rows, size_pt=size_pt)
    sect = doc.element.body.find(qn("w:sectPr"))
    sect.addprevious(tbl._tbl)
    return tbl


def append_picture(doc, tmpl: Paragraph, path: Path, width_cm=16.0) -> None:
    p = append(doc, tmpl, "")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].add_picture(str(path), width=Cm(width_cm))


def make_highlights(blocks: dict) -> list[str]:
    hl = bullets(block(blocks, "HIGHLIGHTS"))
    if not 3 <= len(hl) <= 5:
        raise SystemExit("STOP: highlights count outside 3-5")
    doc, tm = empty_doc_from_base()
    append(doc, tm.h_unnum, "Highlights")
    append(doc, tm.body, TITLE)
    for h in hl:
        append(doc, tm.bullet, h)
    doc.save(str(OUT_HIGH))
    return hl


def fmt(v, nd):
    return f"{float(v):.{nd}f}"


def fmt_ci(lo, hi, nd):
    return f"[{fmt(lo, nd)}, {fmt(hi, nd)}]"


def make_supplementary() -> None:
    doc, tm = empty_doc_from_base()
    append(doc, tm.h_unnum, "Supplementary Material")
    append(doc, tm.body, TITLE)
    append(doc, tm.body, "This file contains the full statistical test tables (S1), the calibration metrics (S2), "
           "the agreement analysis for the ring's raw signals and the lag-1 indicators (S3), and three figures "
           "referenced in the main text. All values were produced by the analysis scripts of the revision with seed 42; "
           "table and figure numbers in cross-references are those of the revised manuscript.")

    # ---- S1: McNemar, Cochran's Q, paired bootstrap -----------------------------------------
    append(doc, tm.h1, "S1. Statistical comparison between classifiers")
    mc = pd.read_csv(RES / "t12_mcnemar.csv", keep_default_na=False)
    cq = pd.read_csv(RES / "t12_cochran_q.csv")
    pb = pd.read_csv(RES / "t12_paired_bootstrap.csv", keep_default_na=False)
    append(doc, tm.cap, "Table S1. (a) Pairwise McNemar tests on per-day correctness (192 test days). n01 = first model "
           "correct and second wrong, n10 = the reverse. Chi-square with continuity correction; the exact binomial p is "
           "given for reference; Holm adjustment over the family of three model pairs. The comparison with the persistence "
           "baseline lies outside the Holm family and is unadjusted.")
    rows = [["Pair", "n01", "n10", "Chi-square (cc)", "p (cc)", "p (exact)", "Holm-adjusted p"]]
    for _, r in mc.iterrows():
        holm = fmt(r["p_cc_holm"], 4) if r["p_cc_holm"] not in ("", "n/a") else "n/a (outside family)"
        rows.append([r["pair"], str(r["n01"]), str(r["n10"]), fmt(r["chi2_cc"], 4), fmt(r["p_cc"], 4), fmt(r["p_exact"], 4), holm])
    append_table(doc, rows)
    q = cq.iloc[0]
    append(doc, tm.legend, f"Cochran's Q over the three original classifiers (RF, XGBoost, SVM; per-day correctness): "
           f"Q = {fmt(q['statistic'], 4)}, df = {int(q['df'])}, p = {fmt(q['p'], 4)}.", italic=True)
    append(doc, tm.cap, "Table S1. (b) Paired bootstrap differences (first minus second model) on macro F1 and balanced "
           "accuracy, 1,000 resamples of the 192 test days on the shared resampling matrix (seed 42); 95% percentile "
           "interval of the difference and proportion of resamples with a positive difference (descriptive, not a p-value). "
           "No resample was skipped.")
    rows = [["Pair", "Metric", "Difference", "95% CI of the difference", "Proportion of resamples > 0"]]
    for _, r in pb.iterrows():
        metric = "macro F1" if r["metric"] == "f1_macro" else "balanced accuracy"
        rows.append([r["pair"], metric, fmt(r["diff_point"], 4), fmt_ci(r["ci_low"], r["ci_high"], 4), fmt(r["prop_diff_gt_0"], 3)])
    append_table(doc, rows)

    # ---- S2: calibration -----------------------------------------------------------------------
    append(doc, tm.h1, "S2. Calibration of the random forest")
    cal = pd.read_csv(RES / "t13_calibration.csv", keep_default_na=False)
    append(doc, tm.cap, "Table S2. (a) Calibration metrics of the random forest on the 192 test days: raw probabilities and "
           "one-versus-rest Platt and isotonic calibrators fitted on out-of-fold training probabilities (fit on folds 1-4, "
           "612 days, primary; fit on folds 2-4, 459 days, sensitivity). ECE and MCE are top-label with 10 equal-width bins; "
           "the multiclass Brier score is the sum over classes (range 0 to 2). Point estimate [95% bootstrap interval] on "
           "the shared resampling matrix where available.")
    cal_test = cal[cal["split"] == "test"]

    def pick(fit, var, metric, ci=False):
        d = cal_test[(cal_test["fit_folds"].astype(str) == fit) & (cal_test["variant"] == var) & (cal_test["metric"] == metric)]
        if len(d) != 1:
            raise SystemExit(f"STOP: calibration row ({fit}, {var}, {metric}) not unique: {len(d)}")
        r = d.iloc[0]
        if ci:
            return f"{fmt(r['point'], 4)} {fmt_ci(r['ci_low'], r['ci_high'], 4)}"
        return fmt(r["point"], 4)

    rows = [["Fit folds", "Variant", "ECE top-label [95% CI]", "MCE top-label", "Brier multiclass [95% CI]", "Log loss [95% CI]"]]
    rows_b = [["Fit folds", "Variant", "ECE Active", "Brier Active [95% CI]", "ECE Rest", "Brier Rest", "ECE Normal", "Brier Normal"]]
    for fit in ("1-4", "2-4"):
        for var in ("raw", "platt", "isotonic"):
            rows.append([fit, var, pick(fit, var, "ece_top", True), pick(fit, var, "mce_top"),
                         pick(fit, var, "brier_multiclass", True), pick(fit, var, "log_loss", True)])
            rows_b.append([fit, var, pick(fit, var, "ece_active"), pick(fit, var, "brier_active", True),
                           pick(fit, var, "ece_rest"), pick(fit, var, "brier_rest"), pick(fit, var, "ece_normal"), pick(fit, var, "brier_normal")])
    append_table(doc, rows, size_pt=8)
    append(doc, tm.cap, "Table S2. (b) Per-class calibration on the test set. ECE for the Active class is a point estimate "
           "only (8 positives).")
    append_table(doc, rows_b, size_pt=8)
    sl = pd.read_csv(RES / "t13_safety_layer_variants.csv", keep_default_na=False)
    append(doc, tm.cap, "Table S2. (c) Safety layer at the 0.50 confidence threshold with raw and calibrated probabilities "
           "(192 test days). Coverage = share of days that keep the model's prediction; mean caloric error against the "
           "offset of the true label (Rest -200, Normal 0, Active +200 kcal); states issued after fallback.")
    rows = [["Variant", "Coverage (%)", "Days overridden", "Mean caloric error (kcal)", "Maximum error (kcal)",
             "Macro F1", "Accuracy", "F1 Active", "States issued R/N/A"]]
    for var in ("raw", "platt", "isotonic"):
        d = sl[(sl["variant"] == var) & (sl["threshold"].astype(float) == 0.5)]
        if len(d) != 1:
            raise SystemExit("STOP: safety layer row at 0.50 not unique")
        r = d.iloc[0]
        rows.append([var, fmt(r["coverage_pct"], 1), str(int(r["n_overridden"])), fmt(r["mean_cal_error"], 1),
                     str(int(float(r["max_cal_error"]))), fmt(r["f1_macro"], 4), fmt(r["accuracy"], 4), fmt(r["f1_active"], 4),
                     f"{int(r['n_pred_rest'])}/{int(r['n_pred_normal'])}/{int(r['n_pred_active'])}"])
    append_table(doc, rows, size_pt=8)

    # ---- S3: agreement, level B and lag-1 ------------------------------------------------------
    append(doc, tm.h1, "S3. Agreement of the labels with ring signals and lag-1 indicators")
    tests = pd.read_csv(RES / "t32_tests.csv", keep_default_na=False)
    desc = pd.read_csv(RES / "t32_descriptives.csv", keep_default_na=False)
    append(doc, tm.body, "Level B indicators are raw ring signals that do not enter the labeling rule but feed the "
           "readiness score, so they show the internal consistency of that score rather than independent validation. "
           "The lag-1 rows compare the value of day t-1 with the class of day t. Expected directions were fixed before the "
           "analysis. p-values are approximate because consecutive days are autocorrelated; inference rests on the effect "
           "sizes.")
    append(doc, tm.cap, "Table S3. Ring signals (level B) and lag-1 activity indicators by class on the 958 labeled days "
           "(median [IQR]), Kruskal-Wallis epsilon-squared, and Cliff's delta Active versus Rest with 95% bootstrap interval.")
    rows = [["Indicator", "Rest", "Normal", "Active", "eps2", "Cliff's delta Active vs Rest [95% CI]", "Expected", "Result"]]
    unit_nd = {"ms": 1, "bpm": 1, "deg C": 2, "steps": 0, "kcal": 0}
    display_names = {
        "hrv_mean": "Nocturnal HRV, ring (RMSSD, ms)", "rhr_lowest": "Lowest nocturnal heart rate, ring (bpm)",
        "temp_deviation": "Skin-temperature deviation, ring (deg C)", "steps_oura": "Steps, ring",
        "aw_active_calories_lag1": "Active calories, watch, day t-1 (kcal)", "aw_steps_lag1": "Steps, watch, day t-1",
        "steps_oura_lag1": "Steps, ring, day t-1",
    }
    exp_map = {"+": "higher", "-": "lower", "0": "none"}

    def med_iqr(var, cls, nd):
        d = desc[(desc["variable"] == var) & (desc["cls"] == cls)]
        if len(d) != 1:
            raise SystemExit(f"STOP: descriptives row ({var}, {cls}) not unique")
        r = d.iloc[0]
        if nd == 0:
            return f"{float(r['median']):,.0f} [{float(r['p25']):,.0f}, {float(r['p75']):,.0f}]"
        return f"{fmt(r['median'], nd)} [{fmt(r['p25'], nd)}, {fmt(r['p75'], nd)}]"

    for _, r in tests[tests["level"].isin(["B", "A_lag1", "B_lag1"])].iterrows():
        var = r["variable"]
        nd = unit_nd[r["unit"]]
        delta = f"{float(r['cliff_active_rest']):+.2f} [{float(r['cliff_active_rest_ci_low']):+.2f}, {float(r['cliff_active_rest_ci_high']):+.2f}]"
        rows.append([display_names[var], med_iqr(var, "Rest", nd), med_iqr(var, "Normal", nd), med_iqr(var, "Active", nd),
                     fmt(r["epsilon2"], 2), delta, exp_map[str(r["expected_active_vs_rest"])], r["agreement"]])
    append_table(doc, rows, size_pt=8)

    # ---- Figures S1-S3 ------------------------------------------------------------------------
    append(doc, tm.h1, "Supplementary figures")
    for fname, legend in (("t31_threshold_grid.png", LEGEND_S1), ("t14_shap_rank_stability_seeds.png", LEGEND_S2),
                          ("t13_safety_layer_coverage.png", LEGEND_S3)):
        append_picture(doc, tm.body, FIGS / fname, width_cm=16.0 if "coverage" not in fname else 12.0)
        append(doc, tm.legend, legend)
    doc.save(str(OUT_SUPP))


def make_response() -> None:
    doc, tm = empty_doc_from_base()
    lines = LETTER.read_text(encoding="utf-8").splitlines()
    para_buf: list[str] = []

    def flush():
        nonlocal para_buf
        if not para_buf:
            return
        text = renumber(" ".join(s.strip() for s in para_buf))
        text = text.replace("[title as submitted]", TITLE)
        check_no_placeholders(text, "response letter")
        if text.startswith("Response. "):
            append(doc, tm.lead, text[len("Response. "):], lead="Response. ")
        elif text.startswith("Changes. "):
            append(doc, tm.lead, text[len("Changes. "):], lead="Changes. ")
        else:
            append(doc, tm.body, text)
        para_buf = []

    quote_buf: list[str] = []

    def flush_quote():
        nonlocal quote_buf
        if quote_buf:
            text = renumber(" ".join(quote_buf))
            append(doc, tm.body, text, italic=True, indent_cm=1.0)
            quote_buf = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("# "):
            flush(); flush_quote()
            append(doc, tm.h_unnum, "Response to Reviewers (Revision 1, Knowledge-Based Systems)")
            continue
        if line.startswith("Stare:") or line.strip() == "---":
            flush(); flush_quote(); continue
        if line.startswith("## "):
            flush(); flush_quote(); append(doc, tm.h1, line[3:].strip()); continue
        if line.startswith("### "):
            flush(); flush_quote(); append(doc, tm.h2, line[4:].strip()); continue
        if line.startswith(">"):
            flush(); quote_buf.append(line.lstrip(">").strip()); continue
        if line.startswith("- "):
            flush(); flush_quote()
            text = renumber(line[2:].strip()); check_no_placeholders(text, "response bullets")
            append(doc, tm.bullet, text); continue
        if not line.strip():
            flush(); flush_quote(); continue
        flush_quote()
        para_buf.append(line)
    flush(); flush_quote()
    doc.save(str(OUT_RESP))


# ----------------------------------------------------------------------------------------------
# Marked copy (T4.1-B): yellow highlight on new / changed text relative to the base, removed text listed at the end
# ----------------------------------------------------------------------------------------------
def _tokens(s: str) -> list[str]:
    return re.findall(r"\S+|\s+", s)


def _inserted_spans(new: str, old: str) -> list[tuple[int, int]]:
    """Character spans of `new` that are not in `old` (word-level SequenceMatcher)."""
    import difflib
    a, b = _tokens(old), _tokens(new)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pos, spans = [], []
    off = 0
    for t in b:
        pos.append(off); off += len(t)
    pos.append(off)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert") and j2 > j1:
            spans.append((pos[j1], pos[j2]))
    return spans


def _deleted_segments(new: str, old: str, min_words: int = 6) -> list[str]:
    import difflib
    a, b = _tokens(old), _tokens(new)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            seg = "".join(a[i1:i2]).strip()
            if len(seg.split()) >= min_words:
                out.append(seg)
    return out


def _highlight_run(run) -> None:
    from docx.enum.text import WD_COLOR_INDEX
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def _highlight_spans(par: Paragraph, spans: list[tuple[int, int]]) -> None:
    """Split the runs of `par` so that exactly the characters inside `spans` are highlighted."""
    if not spans:
        return
    runs = list(par.runs)
    start = 0
    for run in runs:
        text = run.text
        end = start + len(text)
        cuts = {0, len(text)}
        for s, e in spans:
            for c in (s - start, e - start):
                if 0 < c < len(text):
                    cuts.add(c)
        cuts = sorted(cuts)
        pieces = [(text[c0:c1], c0 + start) for c0, c1 in zip(cuts[:-1], cuts[1:]) if c1 > c0]
        if len(pieces) <= 1:
            if any(s <= start and end <= e for s, e in spans) and text:
                _highlight_run(run)
            start = end
            continue
        run.text = pieces[0][0]
        prev = run
        for ptxt, poff in pieces[1:]:
            new_r = copy.deepcopy(run._r)
            prev._r.addnext(new_r)
            from docx.text.run import Run
            nr = Run(new_r, par)
            nr.text = ptxt
            nr.font.highlight_color = None
            prev = nr
        # apply highlight piece by piece
        for r in [run] + [Run(x, par) for x in [prev._r] if False]:
            pass
        cur = start
        node = run._r
        from docx.text.run import Run as _Run
        for ptxt, poff in pieces:
            rr = _Run(node, par)
            if any(s <= poff and poff + len(ptxt) <= e for s, e in spans):
                _highlight_run(rr)
            else:
                rr.font.highlight_color = None
            node = node.getnext()
            cur += len(ptxt)
        start = end


def make_marked_copy() -> dict:
    import difflib
    base = docx.Document(str(BASE))
    doc = docx.Document(str(OUT_MAN))
    base_pars = [p.text for p in base.paragraphs if p.text.strip()]
    base_tables = [[[c.text for c in r.cells] for r in t.rows] for t in base.tables]
    removed: list[str] = []
    matched_base: set[int] = set()
    n_marked_pars = 0
    for p in doc.paragraphs:
        txt = p.text
        if not txt.strip():
            continue
        if txt in base_pars:
            matched_base.add(base_pars.index(txt))
            continue
        # best base paragraph by the fraction of the base text that survives in the new paragraph (word tokens)
        best, best_i = 0.0, -1
        new_tok = _tokens(txt)
        for i, bt in enumerate(base_pars):
            old_tok = _tokens(bt)
            sm = difflib.SequenceMatcher(None, old_tok, new_tok, autojunk=False)
            # only runs of at least 4 consecutive tokens count, so shared common words do not pair unrelated paragraphs
            kept = sum(b.size for b in sm.get_matching_blocks() if b.size >= 4) / max(1, len(old_tok))
            if kept > best:
                best, best_i = kept, i
        if best >= 0.6:
            matched_base.add(best_i)
            _highlight_spans(p, _inserted_spans(txt, base_pars[best_i]))
            for seg in _deleted_segments(txt, base_pars[best_i]):
                removed.append(f"From the paragraph beginning '{base_pars[best_i][:50]}...': \"{seg}\"")
        else:
            for r in p.runs:
                _highlight_run(r)
        n_marked_pars += 1
    for i, bt in enumerate(base_pars):
        if i not in matched_base and bt.strip() not in ("Abstract", "References"):
            removed.append(f"Paragraph: \"{bt}\"")
    n_marked_cells = 0
    for t in doc.tables:
        rows = [[c.text for c in r.cells] for r in t.rows]
        if rows in base_tables:
            continue
        same_shape = [bt for bt in base_tables if len(bt) == len(rows) and len(bt[0]) == len(rows[0]) and bt[0] == rows[0]]
        ref = same_shape[0] if same_shape else None
        for ri, row in enumerate(t.rows):
            for ci, cell in enumerate(row.cells):
                if ref is None or ref[ri][ci] != cell.text:
                    for par in cell.paragraphs:
                        for r in par.runs:
                            _highlight_run(r)
                    n_marked_cells += 1
    # removed-text section at the very end
    tmpl_h = find_par(doc, "References")
    tmpl_body = find_par(doc, "Ring data were retrieved through a cloud REST API")
    sect = doc.element.body.find(qn("w:sectPr"))
    sect.addprevious(clone_paragraph(tmpl_h, "Removed from the submitted version"))
    sect.addprevious(clone_paragraph(tmpl_body, "Text of the submitted manuscript (BASE_MANUSCRIPT_v8.docx) that does not appear in the revised version. "
                                     "Yellow highlighting in the body marks text that is new or changed relative to the submitted version."))
    for item in removed:
        sect.addprevious(clone_paragraph(tmpl_body, item))
    doc.save(str(OUT_MARKED))
    return {"paragraphs_marked": n_marked_pars, "cells_marked": n_marked_cells, "removed_items": len(removed)}


# ----------------------------------------------------------------------------------------------
# Figures folder
# ----------------------------------------------------------------------------------------------
def make_figures(legacy_dir: Path) -> list[tuple[str, str, str]]:
    OUT_FIGDIR.mkdir(exist_ok=True)
    mapping = [
        ("Figure1_architecture", legacy_dir / "Figure_1_architecture_EN"),
        ("Figure2_t13_reliability_rf", FIGS / "t13_reliability_rf"),
        ("Figure3_t15_group_ablation", FIGS / "t15_group_ablation"),
        ("Figure4_t33_expanding_window", FIGS / "t33_expanding_window"),
        ("Figure5_t14_shap_rank_stability_bootstrap", FIGS / "t14_shap_rank_stability_bootstrap"),
        ("Figure6_shap_bar_rf", legacy_dir / "Figure2_shap_bar_rf"),
        ("Figure7_confusion_matrices", legacy_dir / "Figure3_confusion_matrices"),
        ("FigureS1_t31_threshold_grid", FIGS / "t31_threshold_grid"),
        ("FigureS2_t14_shap_rank_stability_seeds", FIGS / "t14_shap_rank_stability_seeds"),
        ("FigureS3_t13_safety_layer_coverage", FIGS / "t13_safety_layer_coverage"),
    ]
    out = []
    for name, src in mapping:
        for ext in (".pdf", ".png"):
            s = src.with_suffix(ext)
            if not s.exists():
                raise SystemExit(f"STOP: figure source missing: {s}")
            d = OUT_FIGDIR / f"{name}{ext}"
            shutil.copyfile(s, d)
            out.append((d.name, str(s), sha256(d)[:16]))
    return out


# ----------------------------------------------------------------------------------------------
# assembly_plan.md
# ----------------------------------------------------------------------------------------------
def write_plan(fig_rows, highlights, base_sha, legacy_dir) -> None:
    L = []
    L.append("# assembly_plan.md - T4.1-A, Revision 1 (KBS)\n")
    L.append(f"Generated by `revision/scripts/assemble_manuscript.py` on {RUN_DATE}. Base: `revision/manuscript/BASE_MANUSCRIPT_v8.docx` "
             f"(copy of `ESWA_Manuscript_Anonymized_v8.docx`, the file submitted to KBS, title confirmed in Editorial Manager; sha256 {base_sha}; "
             f"T4.1-A to T4.2-A used v2, `BASE_MANUSCRIPT.docx`, kept for the record). "
             "Text source: `F4_manuscript_changes.md` (verbatim; only placeholders and numbers of sections, tables and figures are resolved). "
             "Output: `KBS_Manuscript_R1.docx`, `KBS_Highlights_R1.docx`, `KBS_Supplementary_R1.docx`, `KBS_Response_R1.docx`, `figures/`.\n")
    L.append("## Final numbering\n")
    L.append("Methods: 3.1 Dataset, 3.2 Labeling, 3.3 Feature engineering, 3.4 Classification models, 3.5 Data splitting, class imbalance, baselines and hyperparameter search (heading renamed at T4.1-B), "
             "3.6 Evaluation metrics, 3.7 Calibration (new), 3.8 Explainability, 3.9 Feature-group ablation (new), 3.10 Label validation (new), 3.11 Temporal robustness (new), "
             "3.12 Nutritional recommendation module, 3.13 System architecture.\n")
    L.append("Results: 4.1 Feature set, 4.2 Detailed results, 4.3 Statistical uncertainty and model comparison, 4.4 Additional classifiers and hyperparameter search (new), "
             "4.5 Calibration (new), 4.6 Ablation by device and by feature group, 4.7 Label validation, 4.8 Temporal robustness (new), 4.9 Class-imbalance strategies, "
             "4.10 XML versus JSON, 4.11 SHAP robustness, 4.12 Safety layer.\n")
    L.append("| F4 / base label | Final | Location |\n|---|---|---|")
    tab_loc = {"1": "3.1", "2": "3.12", "3": "4.1", "4": "4.2", "5": "4.3", "A": "4.4", "6": "4.6", "6b": "4.6", "7": "4.7", "7b": "4.7", "8": "4.8", "9": "4.10", "10": "5", "B1": "Appendix A"}
    for k, v in TABLE_MAP.items():
        L.append(f"| Table {k} (F4 numbering) | Table {v} | Section {tab_loc[k]} |")
    L.append("| Table 8 (base numbering, imbalance strategies) | Table 12 | Section 4.9 |")
    fig_loc = {"1": "3.13", "t13": "4.5", "t15": "4.6", "t33": "4.8", "t14": "4.11", "2": "4.11", "3": "4.12"}
    for k, v in FIGURE_MAP.items():
        L.append(f"| Figure {k} | Figure {v} | Section {fig_loc[k]} (first citation) |")
    L.append("| Supplementary Table S1 / S2 / S3 | S1 (4.3), S2 (4.5), S3 (4.7) | KBS_Supplementary_R1.docx |")
    L.append("| Supplementary Figure S1 / S2 / S3 | S1 (4.7), S2 (4.11), S3 (4.12) | KBS_Supplementary_R1.docx |")
    L.append("| Section 3.x (hyperparameter search) | Section 3.5, paragraph 3 | see deviations |\n")
    L.append("Figures are numbered by first citation in the assembled text: Figure 1 (3.13), reliability (4.5), group ablation (4.6), expanding window (4.8), "
             "SHAP rank stability (4.11, cited before the SHAP bar plot in the same paragraph), SHAP bar plot (4.11, base Figure 2 -> Figure 6), confusion matrices (4.12, base Figure 3 -> Figure 7).\n")
    L.append("## Blocks: F4 block -> base target (first 60 characters) -> operation -> final number\n")
    L.append("| # | F4 block | Base target | Operation | Final |\n|---|---|---|---|---|")
    for i, r in enumerate(PLAN_ROWS, 1):
        L.append(f"| {i} | {r['block']} | {r['target']} | {r['op']} | {r['final']} |")
    L.append("\nEvery target was matched by a unique paragraph prefix in the base (the script stops otherwise). Base sections not listed above "
             "(1 Introduction except the two marked blocks, 3.3, 4.1, 4.9, declarations, data availability) are unchanged apart from section, table and figure numbers; "
             "the '(v8 wording)' rows take the v8 text of the paragraph (T4.1-C).\n")
    L.append("## Cross-references resolved in F4 text\n")
    L.append("- `Figure [t13]` -> Figure 2; `Figure [t15]` -> Figure 3 (citation added, see deviations); `Figure [t33]` -> Figure 4; `Figure [t14]` -> Figure 5; `Figure 2` (SHAP bar) -> Figure 6; `Figure 3` (confusion) -> Figure 7.")
    L.append("- `Table A` -> Table 6; `Table 6` (device ablation) -> Table 7; `Table 6b` -> Table 8; `Table 7` -> Table 9; `Table 7b` -> Table 10; `Table 8` (expanding window) -> Table 11; `Table 10` -> Table 14; `Table B1` / `Appendix Table B1` -> Table A.1 / Appendix A, Table A.1.")
    L.append("- `Section 3.x` -> Section 3.5. Sections 4.2 to 4.12, 3.2, 3.8, 3.9, 3.13, 4.6, 4.7, 4.8, 4.11, 4.12 and 5 cited in F4 already use the final numbering and were verified against the final headings.")
    L.append("- `[t11]` -> F1 Rest 0.205 and F1 Normal 0.707 for persistence (results/persistence_baseline.csv, f1_rest 0.2045 / f1_normal 0.7071; results/t11_extended_metrics.csv has no per-class F1).")
    L.append("- Base cross-references renumbered: " + "; ".join(f"'{o}' -> '{n}' in '{p[:40]}...'" for p, o, n in BASE_XREFS) + ".\n")
    L.append("## Deviations and text added at assembly (for review)\n")
    for d in DEVIATIONS:
        L.append(f"- {d}")
    L.append("- Supplementary: captions of Tables S1-S3, the introductory sentences of the Supplementary file and the legends of Figures S1-S3 were written at assembly (not in F4). Table values are formatted from the results csv files (4 decimals for S1 and S2, as in the source tables; medians and Cliff's delta in S3 as in main Table 10).")
    L.append("- Response letter: the heading '(draft, ...)' was rendered as '(Revision 1, Knowledge-Based Systems)'; '[title as submitted]' -> the base title; reviewer comments are italic indented paragraphs; placeholders resolved with the same mapping as the manuscript.")
    L.append("- Discussion paragraphs 'Temporal robustness', 'Choice of classifier' and 'Generalization from a single subject' received a bold run-in lead (base convention); in 4.7 the run-in leads 'Threshold sensitivity.' and 'Agreement with label-independent indicators.' present in F4 were set in bold.")
    L.append("- Keywords: 'Keywords:' label kept from the base, list replaced by the F4 list (Random Forest -> Decision support systems).")
    L.append("- New references: six APA 7 entries inserted alphabetically, each carrying the Word comment 'verify in Zotero' (kept until the Zotero import); T4.1-F (2026-10-20): authors, titles, volumes, pages, editors, DOIs / stable proceedings URLs verified on the publishers' pages (revision/references_new_verified.md, revision/references_new.bib); NeurIPS and AAAI papers cited as proceedings, not arXiv.")
    L.append("- T4.1-B: base paragraph of 4.3 'Table 5 reports bootstrap confidence intervals for the best model' -> 'for the reported model'; Abstract: first 'SHAP' expanded to 'SHAP (Shapley additive explanations)' in F4_manuscript_changes.md (248 words); heading 3.5 renamed; "
             "the two persistence per-class F1 cells of Table 4 now have a row in CIFRE_REVIZIE.md (T0.4).")
    L.append("- T4.1-D (editorial fixes after the read-through, 2026-10-19): Discussion 'Temporal robustness' shortened in F4 (figures stay in 4.8, 'canonical' -> 'reported'); "
             "3.2 reordered (thresholds, calibration, distribution, pseudo-labels, then the prescriptive block without the repeated threshold parenthesis); 3.4 hyperparameter paragraph "
             "kept directly after the first paragraph; 3.1 second paragraph shortened to the Table 1 sentence; 'the gap identified above' in the Introduction; Table 2 caption "
             "'by predicted physiological state'; Table 7 caption '(Random Forest, reported labels)'; Table 10 header 'Normal (n = 696 to 704)' with a note on missing values; "
             "keywords without final period; 3.8 'the three original algorithms'. Items checked with no change needed: no paragraph containing 'Internal' exists in any docx "
             "(0 found); the Table 9 footnote is a single italic run with one asterisk; Figure 1 (figures/Figure1_architecture.png, from the submitted figure set) shows the "
             "confidence diamond and the Normal fallback, so the v8 legend matches the image (the repo copy under outputs/figures/ differs only by 'hydration' in the recommendation box and is not used).")
    L.append("- Marked copy `KBS_Manuscript_R1_marked.docx`: every run of text that is new or changed relative to BASE_MANUSCRIPT.docx is highlighted in yellow (word-level diff per paragraph; whole new tables and changed table cells highlighted); "
             "base paragraphs and sentences that no longer appear are listed at the end under 'Removed from the submitted version'. It is a reading aid, not Word track changes.\n")
    L.append("## Figures folder (revision/manuscript/figures/)\n")
    L.append(f"Legacy figures 1-3 copied from `{legacy_dir}` (the figure files of the submitted manuscript; outside the repo).\n")
    L.append("| File | Source | sha256 (16) |\n|---|---|---|")
    for name, src, h in fig_rows:
        L.append(f"| {name} | {src} | {h} |")
    L.append("\n## Highlights (KBS_Highlights_R1.docx)\n")
    for h in highlights:
        L.append(f"- {h} ({len(h)} characters)")
    L.append("")
    OUT_PLAN.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-figures-dir", default=r"C:\Users\Adrian\Downloads\assa",
                    help="folder with Figure_1_architecture_EN, Figure2_shap_bar_rf, Figure3_confusion_matrices (.pdf + .png)")
    ap.add_argument("--v8-docx", default=r"C:\Users\Adrian\Downloads\ESWA_Manuscript_Anonymized_v8.docx",
                    help="v8 of the ESWA manuscript, source of the in-text citations and wording listed in V8_WORDING (T4.1-C)")
    ap.add_argument("--only", choices=["response"], default=None,
                    help="rebuild only one output (T4.2-A: 'response' rebuilds KBS_Response_R1.docx from F4_response_letter.md and touches nothing else)")
    args = ap.parse_args()
    if args.only == "response":
        make_response()
        print(f"OK: {OUT_RESP.name} rebuilt from {LETTER.name}; no other file touched")
        return
    base_sha = sha256(BASE)
    blocks = parse_md_blocks(F4)
    assemble_manuscript(blocks, Path(args.v8_docx))
    highlights = make_highlights(blocks)
    make_supplementary()
    make_response()
    fig_rows = make_figures(Path(args.legacy_figures_dir))
    marked = make_marked_copy()
    DEVIATIONS.append(f"Marked copy statistics: {marked['paragraphs_marked']} paragraphs with highlighted text, {marked['cells_marked']} highlighted table cells, "
                      f"{marked['removed_items']} removed items listed at the end.")
    write_plan(fig_rows, highlights, base_sha, args.legacy_figures_dir)
    if sha256(BASE) != base_sha:
        raise SystemExit("STOP: BASE_MANUSCRIPT.docx changed during assembly")
    print(f"OK: {OUT_MAN.name}, {OUT_MARKED.name}, {OUT_HIGH.name}, {OUT_SUPP.name}, {OUT_RESP.name}, {len(fig_rows)} figure files, {len(PLAN_ROWS)} plan rows; marked {marked}")


if __name__ == "__main__":
    main()
