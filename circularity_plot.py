"""
Vizualizare rezultate circularity sanity check pentru Cap.6 / anexe.
Output: outputs/figures/experiments/ (consistent cu celelalte experimente de validare)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import numpy as np

# Font care suporta diacritice romanesti
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Formatter pentru virgula decimala (convenția RO)
ro_formatter = ticker.FuncFormatter(lambda x, _: f"{x:.1f}".replace(".", ","))

# Output folder (consistent cu celelalte experimente de validare)
OUTPUT_DIR = "outputs/figures/experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load results
df = pd.read_csv("outputs/tables/circularity_sanity_check.csv")

# ============================================================
# FIGURA 1: Circularity sanity check (F1 Macro + F1 per clasă)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# LEFT: F1 Macro pe cele 3 variante
variants = ["A: Referință\n(19 variabile)",
            "B: Fără readiness+\nsleep_score (17)",
            "C: Strict fiziologic\n(14 variabile)"]
f1_test = df["f1_macro"].values
f1_cv = df["f1_cv_mean"].values
f1_cv_std = df["f1_cv_std"].values

x = np.arange(len(variants))
width = 0.35

bars1 = axes[0].bar(x - width/2, f1_test, width, label="F1 Macro (test)", color="#3b7ea1")
bars2 = axes[0].bar(x + width/2, f1_cv, width, yerr=f1_cv_std, label="F1 Macro (CV 5-fold)",
                    color="#f4a261", capsize=5)

for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}".replace(".", ","),
                 ha="center", va="bottom", fontsize=9)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}".replace(".", ","),
                 ha="center", va="bottom", fontsize=9)

axes[0].set_xticks(x)
axes[0].set_xticklabels(variants, fontsize=9)
axes[0].set_ylabel("F1 Macro")
axes[0].set_title("Impactul eliminării variabilelor corelate cu eticheta\n"
                  "(RF, split temporal 80/20 + SMOTE k=3)")
axes[0].legend(loc="lower left")
axes[0].set_ylim(0, 0.65)
axes[0].grid(axis="y", alpha=0.3)
axes[0].yaxis.set_major_formatter(ro_formatter)
axes[0].axhline(y=0.22, color="red", linestyle="--", alpha=0.5, label="Clasificator trivial")
axes[0].text(2.5, 0.23, "Clasificator trivial (~0,22)", color="red", fontsize=8)

# RIGHT: F1 per clasă pe variante
rest = df["f1_rest"].values
normal = df["f1_normal"].values
active = df["f1_active"].values

x = np.arange(len(variants))
width = 0.27

axes[1].bar(x - width, rest, width, label="Rest (n=202)", color="#e76f51")
axes[1].bar(x, normal, width, label="Normal (n=710)", color="#2a9d8f")
axes[1].bar(x + width, active, width, label="Active (n=46)", color="#e9c46a")

axes[1].set_xticks(x)
axes[1].set_xticklabels(variants, fontsize=9)
axes[1].set_ylabel("F1 per clasă")
axes[1].set_title("F1 per clasă vs. variantă de variabile\n(n test ~192 zile)")
axes[1].legend(loc="lower left", fontsize=8)
axes[1].set_ylim(0, 0.85)
axes[1].grid(axis="y", alpha=0.3)
axes[1].yaxis.set_major_formatter(ro_formatter)

plt.suptitle("Verificare empirică: circularitatea pseudo-etichetelor vs. variabile\n"
             "Etichetă = f(readiness, sleep_score) | Variabile predictor = versiuni lag-1 ale scorurilor + alte variabile",
             fontsize=11, y=1.02)
plt.tight_layout()

out_path_1 = os.path.join(OUTPUT_DIR, "circularity_sanity_check.png")
plt.savefig(out_path_1, dpi=150, bbox_inches="tight")
print(f"Salvat: {out_path_1}")
plt.close()

# ============================================================
# FIGURA 2: Autocorrelation plot
# ============================================================
autocorr = {
    "readiness_score": 0.234,
    "sleep_score": 0.054,
    "hrv_mean": 0.329,
    "hrv_balance": 0.781,
    "recovery_index": 0.108,
    "activity_score": 0.811,
}

fig, ax = plt.subplots(figsize=(9, 4.5))
vars_sorted = sorted(autocorr.items(), key=lambda x: x[1])
names = [x[0] for x in vars_sorted]
values = [x[1] for x in vars_sorted]

# Color code: rosu >= 0.7, portocaliu 0.4-0.7, verde < 0.4
colors = ["#2a9d8f" if v < 0.4 else "#f4a261" if v < 0.7 else "#e76f51" for v in values]

bars = ax.barh(names, values, color=colors)
for bar, v in zip(bars, values):
    ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
            f"{v:.3f}".replace(".", ","), va="center", fontsize=10)

ax.axvline(x=0.4, color="gray", linestyle="--", alpha=0.5)
ax.axvline(x=0.7, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Autocorelație (lag-1)")
ax.set_title("Autocorelația zi T vs. zi T-1 pentru variabilele din generarea etichetei\n"
             "Verde: r < 0,4 (risc scăzut) | Portocaliu: 0,4 ≤ r < 0,7 (risc mediu) | Roșu: r ≥ 0,7 (risc ridicat)",
             fontsize=11)
ax.set_xlim(0, 1.0)
ax.grid(axis="x", alpha=0.3)
ax.xaxis.set_major_formatter(ro_formatter)

plt.tight_layout()

out_path_2 = os.path.join(OUTPUT_DIR, "autocorrelation_label_vars.png")
plt.savefig(out_path_2, dpi=150, bbox_inches="tight")
print(f"Salvat: {out_path_2}")
plt.close()