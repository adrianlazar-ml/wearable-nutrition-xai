"""
_check_t12_outputs.py - T1.2 Task 2 pass criteria + Task 4 (determinism, LF).

1. Markdown tables carry exactly the csv values (floats compared at the 4
   decimals printed in the md; integers and strings compared verbatim; nan in
   csv <-> "n/a" in md).
2. Every text output of statistical_tests.py (csv, md, json) and
   CIFRE_REVIZIE.md contain no CRLF.
3. Determinism: given a directory with a previous copy of the outputs
   (--previous DIR), sha256 must be identical for every file, except
   t12_run_meta.json which is compared without the run_date field.

Exits 1 on any FAIL. Evidence script, kept.
Run from repo root:
    python revision/scripts/_check_t12_outputs.py [--previous DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
TABLES = {"t12_mcnemar.csv": "t12_mcnemar_table.md",
          "t12_paired_bootstrap.csv": "t12_paired_bootstrap_table.md"}
OUTPUTS = ["t12_mcnemar.csv", "t12_mcnemar_table.md", "t12_paired_bootstrap.csv",
           "t12_paired_bootstrap_table.md", "t12_cochran_q.csv", "t12_run_meta.json"]
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def md_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]
    return header, rows


def compare(csv_name: str, md_name: str) -> None:
    df = pd.read_csv(RES / csv_name)
    header, rows = md_rows(RES / md_name)
    check(header == list(df.columns), f"{md_name}: header == csv columns")
    check(len(rows) == len(df), f"{md_name}: {len(rows)} rows == csv {len(df)}")
    bad = 0
    for i, row in enumerate(rows):
        for c, cell in zip(header, row):
            v = df.iloc[i][c]
            if isinstance(v, (float, np.floating)):
                if cell == "n/a":
                    ok = bool(pd.isna(v))
                else:
                    ok = (not pd.isna(v)) and f"{float(v):.4f}" == cell
            else:
                ok = str(v) == cell
            if not ok:
                bad += 1
                print(f"   mismatch {md_name} row {i} col {c}: md '{cell}' vs csv '{v}'")
    check(bad == 0, f"{md_name}: all cells == csv values ({bad} mismatches)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--previous", type=Path, default=None)
    args = ap.parse_args()

    print("=== 1. md == csv ===")
    for csv_name, md_name in TABLES.items():
        compare(csv_name, md_name)

    print("\n=== 2. LF only ===")
    for name in OUTPUTS:
        check(b"\r\n" not in (RES / name).read_bytes(), f"no CRLF in results/{name}")
    check(b"\r\n" not in (ROOT / "revision" / "CIFRE_REVIZIE.md").read_bytes(), "no CRLF in CIFRE_REVIZIE.md")

    if args.previous is not None:
        print(f"\n=== 3. determinism vs {args.previous} ===")
        for name in OUTPUTS:
            cur, prev = RES / name, args.previous / name
            if name == "t12_run_meta.json":
                a = json.loads(cur.read_text(encoding="utf-8"))
                b = json.loads(prev.read_text(encoding="utf-8"))
                a.pop("run_date"), b.pop("run_date")
                check(a == b, f"{name}: identical without run_date")
            else:
                check(sha256(cur) == sha256(prev), f"{name}: sha256 identical ({sha256(cur)[:16]})")

    print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}"))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
