"""
_check_t11_outputs.py - T1.1 Task 4 pass check: every cell of
t11_extended_metrics_table.md equals the corresponding row of
t11_extended_metrics.csv at 4 decimals, and every csv figure appears in
CIFRE_REVIZIE.md. Evidence script, kept on purpose.

Run from repo root:  python revision/scripts/_check_t11_outputs.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
ORDER = ["balanced_accuracy", "cohen_kappa", "g_mean", "roc_auc_ovr_macro", "pr_auc_macro",
         "f1_macro", "accuracy", "mcc", "qwk"]
SYS = {"RF": "rf", "XGB": "xgb", "SVM": "svm", "Persistence": "persistence", "Majority": "majority"}
CELL = re.compile(r"(-?\d\.\d{4}) \[(-?\d\.\d{4}), (-?\d\.\d{4})\]")


def main() -> None:
    csv = pd.read_csv(RES / "t11_extended_metrics.csv")
    md = (RES / "t11_extended_metrics_table.md").read_text(encoding="utf-8").splitlines()
    rows = [l for l in md if l.startswith("| ") and l.split("|")[1].strip() in SYS]
    fails = 0
    n_cells = 0
    for line in rows:
        cells = [c.strip() for c in line.strip("|").split("|")]
        system = SYS[cells[0]]
        for metric, cell in zip(ORDER, cells[1:]):
            r = csv[(csv.system == system) & (csv.metric == metric)].iloc[0]
            n_cells += 1
            if cell == "n/a":
                ok = np.isnan(r.value)
            else:
                m = CELL.fullmatch(cell)
                ok = m is not None and all(
                    abs(float(m.group(i + 1)) - round(float(v), 4)) < 1e-9
                    for i, v in enumerate((r.value, r.ci_low, r.ci_high)))
            if not ok:
                fails += 1
                print(f"MISMATCH {system}/{metric}: md '{cell}' vs csv {r.value}, {r.ci_low}, {r.ci_high}")
    print(f"md vs csv: {n_cells} cells compared, {fails} mismatches")

    reg = (ROOT / "revision" / "CIFRE_REVIZIE.md").read_text(encoding="utf-8")
    missing = []
    for r in csv.itertuples():
        if np.isnan(r.value):
            continue
        if f"{r.value:.4f}" not in reg or f"{r.ci_low:.4f}" not in reg or f"{r.ci_high:.4f}" not in reg:
            missing.append(f"{r.system}/{r.metric}")
    print(f"csv figures present in CIFRE_REVIZIE.md: {len(csv) - csv.value.isna().sum() - len(missing)}"
          f"/{len(csv) - csv.value.isna().sum()}; missing: {missing}")
    ok = fails == 0 and not missing
    print("ALL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
