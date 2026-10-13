"""
t3_labels.py - Schema canonica de etichetare T3 (Rest / Normal / Active)
========================================================================
Sursa unica de adevar pentru pragurile T3 si maparea etichetelor.

Praguri canonice (Cap.5 + CIFRE_CANONICE.md sectiunea 14):
  Rest   (0): readiness_score < 60 OR sleep_score < 60
  Active (2): readiness_score >= 85 AND sleep_score >= 75
  Normal (1): restul zilelor

Revision 1 (KBS, T0.3): thresholds are now optional keyword arguments with
canonical defaults, so the sensitivity grid (T3.1) can reuse this module
instead of re-implementing the rule. Default behaviour is bit-identical to
the frozen version (verified by revision/scripts/test_t3_labels.py).

Canonical thresholds do NOT change (Ground Truth in the tooling context file).

Autor: Adrian Lazar | Disertatie UPB 2026
"""

from typing import Optional

import numpy as np
import pandas as pd

# Canonical T3 thresholds
T3_REST_CUTOFF = 60        # readiness OR sleep below this -> Rest
T3_ACTIVE_READINESS = 85   # readiness >= this AND ...
T3_ACTIVE_SLEEP = 75       # ... sleep >= this -> Active

# Canonical label -> name / colour mapping (FND-215)
LABEL_NAMES = {0: "Rest", 1: "Normal", 2: "Active"}
LABEL_COLORS = {0: "🔴", 1: "🟡", 2: "🟢"}


def classify_day(
    readiness: float,
    sleep: float,
    rest_cutoff: int = T3_REST_CUTOFF,
    active_readiness: int = T3_ACTIVE_READINESS,
    active_sleep: int = T3_ACTIVE_SLEEP,
    sleep_rest_cutoff: Optional[int] = None,
) -> int:
    """T3 classification on scalar values. Returns int 0/1/2.

    Does not handle NaN (used on direct input, e.g. UI sliders).

    Args:
        readiness: Oura readiness score.
        sleep: Oura sleep score.
        rest_cutoff: Rest threshold on readiness (and on sleep unless
            sleep_rest_cutoff is given).
        active_readiness: Active threshold on readiness.
        active_sleep: Active threshold on sleep.
        sleep_rest_cutoff: Optional separate Rest threshold on sleep;
            defaults to rest_cutoff (canonical rule uses one value for both).
    """
    if sleep_rest_cutoff is None:
        sleep_rest_cutoff = rest_cutoff
    if readiness < rest_cutoff or sleep < sleep_rest_cutoff:
        return 0
    elif readiness >= active_readiness and sleep >= active_sleep:
        return 2
    else:
        return 1


def assign_nutrition_label(row: pd.Series, **thresholds) -> float:
    """T3 classification on a DataFrame row (for df.apply(..., axis=1)).

    Returns 0/1/2, or np.nan if readiness_score / sleep_score is missing.
    Extra keyword arguments are forwarded to classify_day.
    """
    if pd.isna(row["readiness_score"]) or pd.isna(row["sleep_score"]):
        return np.nan
    return classify_day(row["readiness_score"], row["sleep_score"], **thresholds)


def assign_labels(df: pd.DataFrame, **thresholds) -> pd.Series:
    """Vectorised T3 labelling of a whole DataFrame.

    Equivalent to df.apply(assign_nutrition_label, axis=1, **thresholds)
    but ~100x faster; convenient for the threshold grid (T3.1).
    Returns a float Series (NaN where scores are missing).
    """
    rest_cutoff = thresholds.get("rest_cutoff", T3_REST_CUTOFF)
    sleep_rest_cutoff = thresholds.get("sleep_rest_cutoff") or rest_cutoff
    active_readiness = thresholds.get("active_readiness", T3_ACTIVE_READINESS)
    active_sleep = thresholds.get("active_sleep", T3_ACTIVE_SLEEP)

    r = df["readiness_score"]
    s = df["sleep_score"]
    missing = r.isna() | s.isna()
    is_rest = (r < rest_cutoff) | (s < sleep_rest_cutoff)
    is_active = (r >= active_readiness) & (s >= active_sleep)

    label = pd.Series(1.0, index=df.index)
    label[is_active] = 2.0
    label[is_rest] = 0.0          # Rest has priority, as in classify_day
    label[missing] = np.nan
    return label
