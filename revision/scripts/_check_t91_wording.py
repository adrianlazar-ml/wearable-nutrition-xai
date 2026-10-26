"""T9.1 T2 evidence (2026-10-26): the provenance wording check of _check_t41_outputs.py (PROVENANCE_FORBIDDEN in the
Abstract, Introduction, Conclusions, Highlights and Response letter) applied to a folder holding KBS_Manuscript_R1.docx,
KBS_Highlights_R1.docx and KBS_Response_R1.docx. Expected: FAIL on the copies saved at T0 (before T1), PASS on
revision/manuscript/ after the T1 reassembly. Same code as the permanent check (imported), so the two runs are comparable.

Usage (from the repo root): python revision/scripts/_check_t91_wording.py [DIR]   (default: revision/manuscript)
Exit 1 when a forbidden formulation is found.
"""
from __future__ import annotations

import sys
from pathlib import Path

import docx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _check_t41_outputs import PROVENANCE_FORBIDDEN, PROVENANCE_SCOPE, provenance_hits, provenance_sections  # noqa: E402


def main() -> None:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "manuscript"
    man = docx.Document(str(d / "KBS_Manuscript_R1.docx"))
    hl = docx.Document(str(d / "KBS_Highlights_R1.docx"))
    resp = docx.Document(str(d / "KBS_Response_R1.docx"))
    secs = provenance_sections(man, hl, resp)
    print(f"# provenance wording check on {d}")
    for name, text in secs.items():
        print(f"  {name}: {len(text.split())} words")
    hits = provenance_hits(secs)
    print(f"  forbidden: {', '.join(repr(s) for s in PROVENANCE_FORBIDDEN)} in the {PROVENANCE_SCOPE}")
    for h in hits:
        print(f"  HIT {h}")
    print(f"[{'FAIL' if hits else 'PASS'}] {len(hits)} hits")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
