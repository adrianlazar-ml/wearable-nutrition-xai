"""T11.4 (2026-11-03): the keyword lists of assemble_manuscript.py (KEYWORDS['manuscript'], the Revision 1 list of 19 Oct;
KEYWORDS['em'], the Editorial Manager record of the original submission; KEYWORDS['em7'], T11b.1: the EM list in EM order
completed with 'Decision support systems' and 'SHAP') each hold 1 to 7 non-empty, unique keywords; KEYWORDS_MODE selects
one of them; the F4 source line (F4_manuscript_changes.md, ABSTRACT) equals the ACTIVE list (T11b.1: em7).

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

EM = ["Personalized nutrition", "Wearable devices", "Multimodal physiological data", "Knowledge-based decision support",
      "Explainable artificial intelligence (XAI)"]


def _f4_keywords() -> list[str]:
    text = am.F4.read_text(encoding="utf-8")
    m = re.search(r"Keywords \(1-7, KBS\): (.*?)\s*\((?:fara punct final; )?decis", text)
    assert m is not None
    return [k.strip() for k in m.group(1).rstrip(". ").split(";")]


def test_keyword_lists_have_1_to_7_entries_and_f4_line_equals_active_list():
    assert set(am.KEYWORDS) == {"manuscript", "em", "em7"}
    for mode, kws in am.KEYWORDS.items():
        assert 1 <= len(kws) <= 7, mode
        assert all(k and k == k.strip() for k in kws), mode
        assert len(set(kws)) == len(kws), mode
        assert not any(k.endswith(".") for k in kws), mode  # T4.1-D item 10: no final period
    assert am.KEYWORDS_MODE in am.KEYWORDS
    assert _f4_keywords() == am.KEYWORDS[am.KEYWORDS_MODE]


def test_em7_is_the_em_list_in_em_order_plus_decision_support_systems_and_shap_and_is_active():
    assert am.KEYWORDS["em"] == EM
    assert am.KEYWORDS["em7"] == [*EM, "Decision support systems", "SHAP"]
    assert len(am.KEYWORDS["em7"]) == 7
    assert am.KEYWORDS_MODE == "em7"
    for absent in ("Explainable AI", "Physiological state classification", "mHealth", "Random Forest"):
        assert absent not in am.KEYWORDS["em7"]
    assert _f4_keywords() == am.KEYWORDS["em7"]
