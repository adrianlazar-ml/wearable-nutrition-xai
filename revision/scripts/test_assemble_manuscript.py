"""T11.4 (2026-11-03): the two keyword lists of assemble_manuscript.py (KEYWORDS['manuscript'], the Revision 1 list, and
KEYWORDS['em'], the Editorial Manager record of the original submission) each hold 1 to 7 non-empty, unique keywords;
KEYWORDS_MODE selects one of them; the 'manuscript' list equals the F4 source line (F4_manuscript_changes.md, ABSTRACT).

Run from the repo root: python -m pytest revision/scripts/test_assemble_manuscript.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import assemble_manuscript as am  # noqa: E402


def test_keyword_lists_have_1_to_7_entries_and_manuscript_list_equals_f4():
    assert set(am.KEYWORDS) == {"manuscript", "em"}
    for mode, kws in am.KEYWORDS.items():
        assert 1 <= len(kws) <= 7, mode
        assert all(k and k == k.strip() for k in kws), mode
        assert len(set(kws)) == len(kws), mode
        assert not any(k.endswith(".") for k in kws), mode  # T4.1-D item 10: no final period
    assert am.KEYWORDS_MODE in am.KEYWORDS
    text = am.F4.read_text(encoding="utf-8")
    m = re.search(r"Keywords \(1-7, KBS\): (.*?)\s*\((?:fara punct final; )?decis", text)
    assert m is not None
    assert [k.strip() for k in m.group(1).rstrip(". ").split(";")] == am.KEYWORDS["manuscript"]
