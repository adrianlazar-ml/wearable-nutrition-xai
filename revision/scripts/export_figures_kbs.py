"""T5.2 (F5): export the manuscript figures as separate files for KBS (Guide for Authors p. 20):
vector PDF with embedded TrueType fonts (matplotlib pdf.fonttype 42) and PNG at 300 dpi, named
Figure_2.pdf / Figure_2.png ... Figure_S3.pdf / Figure_S3.png in revision/figures/.

The figure CONTENT is not changed: the same make_figure() functions of the Phase 1-3 scripts are called
with the same csv results they produced (seed 42 runs recorded in CIFRE_REVIZIE.md); only the PDF font
type differs from the Phase 1-3 files (Type 3 -> Type 42). The PNG files must be byte-identical to the
files in revision/results/figures/ (checked here), which proves that the content is unchanged.

Figures 2 (calibration) and S3 (safety-layer coverage) are drawn inside main() of their scripts; their
plotting code is copied verbatim below and fed with the saved csv files (t13_reliability_bins.csv,
t13_safety_layer_variants.csv). Figures 1, 6 and 7 are the figures of the submitted manuscript
(matplotlib 3.11 files, produced outside the frozen environment, no script in the repo reproduces them
byte for byte): they are copied unchanged from revision/manuscript/figures/ and only reported.

Run from the repo root:  python revision/scripts/export_figures_kbs.py
Writes revision/figures/*.pdf|png and revision/figures/FIGURES_MANIFEST.md.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pypdf  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "revision" / "results"
RES_FIG = RES / "figures"
MAN_FIG = ROOT / "revision" / "manuscript" / "figures"
OUT = ROOT / "revision" / "figures"
TMP = OUT / "_tmp"
sys.path.insert(0, str(HERE))

import calibration_rf as cal  # noqa: E402
import expanding_window as ew  # noqa: E402
import group_ablation as ga  # noqa: E402
import shap_stability as ss  # noqa: E402
import threshold_sensitivity as ts  # noqa: E402
from common import FEATURES  # noqa: E402

PDF_META = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}
FULL_PAGE_PX, ONE_COL_PX = 2244, 1063


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- Figure 2: copied from calibration_rf.py (stage B)
def fig_calibration(bins_df: pd.DataFrame, stem: Path) -> None:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    colors = {"raw": "#555555", "platt": "#2196F3", "isotonic": "#E65100"}
    markers = {"raw": "o", "platt": "s", "isotonic": "^"}
    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.1], hspace=0.08, wspace=0.25)
    panels = [("(a) Top-label reliability, test set\n(n = 192; calibrators fit on OOF folds 1-4)", "test", "top_label",
               "Confidence (max p)", "Accuracy in bin"),
              ("(b) Class Active reliability, OOF in-sample\n(n = 612, 33 Active; fit folds 1-4)", "oof", "active",
               "Predicted p(Active)", "Observed Active fraction in bin")]
    for col, (title, split, kind, xlabel, ylabel) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        axh = fig.add_subplot(gs[1, col], sharex=ax)
        ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1, label="perfect calibration")
        width = 0.1 / 3.6
        for k, v in enumerate(cal.VARIANTS):
            t = bins_df[(bins_df.split == split) & (bins_df.fit_folds == "1-4") & (bins_df.variant == v)
                        & (bins_df.kind == kind) & (bins_df.n_b > 0)]
            ax.plot(t.conf_b, t.acc_b, marker=markers[v], color=colors[v], lw=1.4, ms=5, label=v)
            full = bins_df[(bins_df.split == split) & (bins_df.fit_folds == "1-4") & (bins_df.variant == v)
                           & (bins_df.kind == kind)]
            centers = (full.bin_low + full.bin_high) / 2 + (k - 1) * width
            axh.bar(centers, full.n_b, width=width, color=colors[v], alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=9, loc="upper left")
        axh.set_xlabel(xlabel, fontsize=10)
        axh.set_ylabel("n days", fontsize=9)
        axh.set_xticks(cal.EDGES)
        axh.grid(alpha=0.3, axis="y")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata=PDF_META)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


# ---------------------------------------------------------------- Figure S3: copied from safety_layer_calibrated.py
VARIANTS = ["raw", "platt", "isotonic"]
THRESHOLDS = [0.00, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80]
CANONICAL_THRESHOLD = 0.50


def fig_safety(df: pd.DataFrame, stem: Path) -> None:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    colors = {"raw": "#555555", "platt": "#2196F3", "isotonic": "#E65100"}
    markers = {"raw": "o", "platt": "s", "isotonic": "^"}
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for v in VARIANTS:
        sub = df[df.variant == v]
        ax.plot(sub.threshold, sub.coverage_pct, marker=markers[v], color=colors[v], lw=1.4, ms=5, label=v)
    ax.axvline(CANONICAL_THRESHOLD, color="#999999", ls=":", lw=1)
    ax.text(CANONICAL_THRESHOLD + 0.01, 3, "canonical 0.50", fontsize=8, color="#666666")
    ax.set_xlabel("Confidence threshold (max p)", fontsize=10)
    ax.set_ylabel("Coverage (% of test days)", fontsize=10)
    ax.set_title("Safety-layer coverage vs threshold, test (n = 192)", fontsize=10, fontweight="bold")
    ax.set_xticks(THRESHOLDS)
    ax.set_xticklabels([f"{t:.2f}" for t in THRESHOLDS], rotation=45, ha="right")
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata=PDF_META)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def ranks_from_csv(mode: str) -> tuple[np.ndarray, np.ndarray]:
    long = pd.read_csv(RES / f"t14_shap_values_{mode}.csv")
    piv = long.pivot(index="run", columns="feature", values="rank").sort_index()[FEATURES]
    ref = pd.read_csv(RES / "t14_shap_reference.csv").set_index("feature").loc[FEATURES, "rank"]
    return piv.to_numpy(dtype=int), ref.to_numpy(dtype=int)


def pdf_fonts(p: Path) -> list[str]:
    out = set()
    for page in pypdf.PdfReader(str(p)).pages:
        fd = (page.get("/Resources") or {}).get("/Font") or {}
        for k in fd:
            fo = fd[k].get_object()
            sub = str(fo.get("/Subtype"))
            desc = fo.get("/DescendantFonts")
            if desc:
                d0 = desc[0].get_object()
                fdsc = d0.get("/FontDescriptor").get_object()
                emb = any("FontFile" in x for x in fdsc.keys())
                out.add(f"{str(fo.get('/BaseFont')).split('+')[-1]} ({sub}/{d0.get('/Subtype')}, embedded={emb})")
            else:
                fdsc = fo.get("/FontDescriptor")
                emb = any("FontFile" in x for x in fdsc.get_object().keys()) if fdsc else (sub == "/Type3")
                out.add(f"{str(fo.get('/BaseFont')).split('+')[-1]} ({sub}, embedded={emb})")
    return sorted(out)


def captions() -> dict[str, str]:
    import docx
    out = {}
    for f in ("KBS_Manuscript_R1.docx", "KBS_Supplementary_R1.docx"):
        d = docx.Document(str(ROOT / "revision" / "manuscript" / f))
        for p in d.paragraphs:
            m = re.match(r"Figure (S?\d+)\. (.*)", p.text)
            if m:
                out[m.group(1)] = m.group(2).replace("|", "/")
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    TMP.mkdir(exist_ok=True)
    ga.FIG = TMP
    ss.FIG = TMP
    ew.FIG = TMP
    ts.FIG = TMP

    # ---- regenerated figures (same data, same functions) ----
    logo = pd.read_csv(RES / "t15_logo.csv")
    kogi = pd.read_csv(RES / "t15_kogi.csv")
    ga.make_figure(logo[logo.config == "logo"], kogi, ga.CANON_FULL["f1_macro"], ga.build_groups(ga.SLEEP_SCORE_GROUP))
    folds = pd.read_csv(RES / "t33_expanding_window.csv")
    ew.make_figure(folds[folds.trainable.eq(True)], TMP / "t33_expanding_window")  # pooled row excluded (as in main)
    grid = pd.read_csv(RES / "t31_threshold_grid.csv")
    ts.make_figure(grid, TMP / "t31_threshold_grid")
    for mode in ("bootstrap", "seeds"):
        r, ref = ranks_from_csv(mode)
        ss.make_figure(r, ref, mode, r.shape[0])
    fig_calibration(pd.read_csv(RES / "t13_reliability_bins.csv"), TMP / "t13_reliability_rf")
    fig_safety(pd.read_csv(RES / "t13_safety_layer_variants.csv"), TMP / "t13_safety_layer_coverage")

    generated = {  # final name -> (tmp stem, source script, manuscript figure stem)
        "Figure_2": ("t13_reliability_rf", "revision/scripts/calibration_rf.py (stage B; plotting code copied into export_figures_kbs.py)", "Figure2_t13_reliability_rf"),
        "Figure_3": ("t15_group_ablation", "revision/scripts/group_ablation.py make_figure()", "Figure3_t15_group_ablation"),
        "Figure_4": ("t33_expanding_window", "revision/scripts/expanding_window.py make_figure()", "Figure4_t33_expanding_window"),
        "Figure_5": ("t14_shap_rank_stability_bootstrap", "revision/scripts/shap_stability.py make_figure(), mode bootstrap", "Figure5_t14_shap_rank_stability_bootstrap"),
        "Figure_S1": ("t31_threshold_grid", "revision/scripts/threshold_sensitivity.py make_figure()", "FigureS1_t31_threshold_grid"),
        "Figure_S2": ("t14_shap_rank_stability_seeds", "revision/scripts/shap_stability.py make_figure(), mode seeds", "FigureS2_t14_shap_rank_stability_seeds"),
        "Figure_S3": ("t13_safety_layer_coverage", "revision/scripts/safety_layer_calibrated.py (plotting code copied into export_figures_kbs.py)", "FigureS3_t13_safety_layer_coverage"),
    }
    legacy = {
        "Figure_1": ("Figure1_architecture", "not generated in the revision: file of the submitted manuscript (Downloads/assa/Figure_1_architecture_EN; matplotlib 3.11 PDF; the untracked repo script eswa_architecture.py renders a variant with a hydration box and is not used)"),
        "Figure_6": ("Figure6_shap_bar_rf", "not generated in the revision: file of the submitted manuscript (Downloads/assa/Figure2_shap_bar_rf; matplotlib 3.11, DejaVu Serif, 1000 dpi; no script in the repo reproduces it at these settings, the candidates cap5_compare_algorithms*.py write 150 dpi PNG only)"),
        "Figure_7": ("Figure7_confusion_matrices", "not generated in the revision: file of the submitted manuscript (Downloads/assa/Figure3_confusion_matrices; matplotlib 3.11, DejaVu Serif, 1000 dpi; same remark as Figure 6)"),
    }
    rows = []
    fails = []
    for name, (stem, script, man_stem) in generated.items():
        for ext in ("pdf", "png"):
            shutil.copyfile(TMP / f"{stem}.{ext}", OUT / f"{name}.{ext}")
        png, pdf = OUT / f"{name}.png", OUT / f"{name}.pdf"
        same_png = sha(png) == sha(RES_FIG / f"{stem}.png") == sha(MAN_FIG / f"{man_stem}.png")
        if not same_png:
            fails.append(f"{name}: PNG differs from results/figures or manuscript/figures")
        rows.append((name, script, f"{stem}.pdf / .png (revision/results/figures)", png, pdf, same_png))
    for name, (man_stem, script) in legacy.items():
        for ext in ("pdf", "png"):
            shutil.copyfile(MAN_FIG / f"{man_stem}.{ext}", OUT / f"{name}.{ext}")
        rows.append((name, script, f"{man_stem}.pdf / .png (revision/manuscript/figures, copied unchanged)",
                     OUT / f"{name}.png", OUT / f"{name}.pdf", None))
    shutil.rmtree(TMP)
    rows.sort(key=lambda r: (r[0].startswith("Figure_S"), int(re.sub(r"\D", "", r[0]))))

    # ---- manifest ----
    L = [f"# FIGURES_MANIFEST.md - figuri exportate pentru KBS (T5.2), {date.today().isoformat()}", "",
         "Generat de `revision/scripts/export_figures_kbs.py` (seed 42 in toate scripturile sursa; datele din csv-urile T1.3 / T1.4 / T1.5 / T3.1 / T3.3 inregistrate in CIFRE_REVIZIE.md). "
         "PDF vector cu fonturi TrueType embedded (matplotlib pdf.fonttype 42), PNG la 300 dpi. "
         "Continutul nu s-a schimbat: PNG-urile regenerate sunt byte-identice cu cele din revision/results/figures/ si revision/manuscript/figures/ (coloana 'PNG identic'). "
         "Figurile 1, 6 si 7 sunt fisierele manuscrisului submis, copiate neschimbate (nu exista script in repo care sa le reproduca; vezi coloana Script). "
         f"Praguri KBS: >= {FULL_PAGE_PX} px latime pentru figuri pe toata pagina, >= {ONE_COL_PX} px pentru o coloana.", "",
         "| Figura | Script / sursa | Fisier sursa | PDF (puncte, fonturi) | PNG (px, dpi) | Latime | PNG identic cu sursa | Caption (din manuscris) |",
         "|---|---|---|---|---|---|---|---|"]
    caps = captions()
    for name, script, src, png, pdf, same in rows:
        im = Image.open(png)
        w, h = im.size
        dpi = im.info.get("dpi", (None, None))[0]
        r = pypdf.PdfReader(str(pdf))
        mb = r.pages[0].mediabox
        fonts = "; ".join(pdf_fonts(pdf))
        width_ok = ("pagina" if w >= FULL_PAGE_PX else "coloana" if w >= ONE_COL_PX else "SUB PRAG")
        cap = caps.get(name.replace("Figure_", ""), "NEGASIT")
        L.append(f"| {name} | {script} | {src} | {float(mb.width):.0f} x {float(mb.height):.0f} pt; {fonts} | {w} x {h} px, {round(dpi) if dpi else 'n/a'} dpi | {width_ok} ({w} px) | "
                 f"{'da' if same else ('n/a (copiat)' if same is None else 'NU')} | {cap} |")
    L += ["", "Verificari: " + ("toate PNG-urile regenerate identice cu sursa" if not fails else "; ".join(fails)) + "."]
    (OUT / "FIGURES_MANIFEST.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))
    if fails:
        sys.exit("FAIL: " + "; ".join(fails))


if __name__ == "__main__":
    main()
