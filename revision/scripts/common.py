"""
common.py - Shared data loading and canonical split for Revision 1 scripts.
===========================================================================
Reproduces load_and_prepare() and the preprocessing block of
cap5_compare_algorithms_json_ext.py (the canonical comparison script)
so every revision script works on the same X/y/split.

Canonical preprocessing (must stay bit-identical):
  - temporal split 80/20 on the 958 labelled days -> 766/192
  - SimpleImputer(median) fit on train, applied to test
  - tree models: SMOTE(random_state=42, k_neighbors=3) on imputed train
  - SVM: StandardScaler fit on imputed train (pre-SMOTE), then SMOTE on
    scaled train; test scaled with the same scaler (FND-105)

Autor: Adrian Lazar | Disertatie UPB 2026 | KBS Revision 1
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision"))
import t3_labels  # noqa: E402

SEED = 42
TRAIN_RATIO = 0.8
LAG_FEATURES = [
    "readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
    "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
    "steps_oura", "respiratory_rate", "temp_deviation",
    "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2",
    "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate",
]
FEATURES = [f"{f}_lag1" for f in LAG_FEATURES]

CANON = dict(n_total=958, n_train=766, n_test=192, test_dist={0: 44, 1: 140, 2: 8})


def load_raw(root: Path = ROOT) -> pd.DataFrame:
    """Merged daily Oura + Apple Watch (JSON source), sorted by date.
    Identical to cap5_compare_algorithms_json_ext.load_and_prepare()."""
    oura = pd.read_csv(root / "data" / "oura_combined.csv")
    oura["date"] = pd.to_datetime(oura["date"])
    aw = pd.read_csv(root / "apple_watch_api" / "apple_health_daily.csv")
    aw["date"] = pd.to_datetime(aw["date"])
    aw["aw_active_calories"] = (aw["aw_active_energy_kj"] / 4.184).round(1)
    df = oura
    for col in ["aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories",
                "aw_spo2", "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate"]:
        df = df.merge(aw[["date", col]].dropna(subset=[col]), on="date", how="left")
    return df.sort_values("date").reset_index(drop=True)


def build_dataset(df: pd.DataFrame, **thresholds):
    """Lag-1 features + T3 labels (thresholds optional, canonical by default).
    Returns X (958x19), y (int), dates. Rows without a label are dropped;
    rows with NaN features are kept (imputed later), as in the canonical script."""
    df = df.copy()
    for f in LAG_FEATURES:
        df[f"{f}_lag1"] = df[f].shift(1)
    df["label"] = t3_labels.assign_labels(df, **thresholds)
    dm = df[["date"] + FEATURES + ["label"]].dropna(subset=["label"])
    dm = dm.sort_values("date").reset_index(drop=True)
    return dm[FEATURES], dm["label"].astype(int), dm["date"]


@dataclass
class Split:
    X_train: pd.DataFrame          # imputed, real 766 rows
    X_test: pd.DataFrame           # imputed
    y_train: pd.Series
    y_test: pd.Series
    X_train_sm: pd.DataFrame       # imputed + SMOTE (trees)
    y_train_sm: pd.Series
    X_train_scaled_sm: pd.DataFrame  # imputed + scaled + SMOTE (SVM)
    y_train_scaled_sm: pd.Series
    X_test_scaled: pd.DataFrame
    dates_test: pd.Series
    imputer: SimpleImputer
    scaler: StandardScaler


def canonical_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                    k_neighbors: int = 3, seed: int = SEED) -> Split:
    """Temporal 80/20 split + imputer/SMOTE/scaler exactly as the canonical script."""
    i = int(len(X) * TRAIN_RATIO)
    X_tr, X_te, y_tr, y_te = X.iloc[:i], X.iloc[i:], y.iloc[:i], y.iloc[i:]

    imputer = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=FEATURES)
    X_te_imp = pd.DataFrame(imputer.transform(X_te), columns=FEATURES)

    X_tr_sm, y_tr_sm = SMOTE(random_state=seed, k_neighbors=k_neighbors).fit_resample(X_tr_imp, y_tr)

    scaler = StandardScaler()
    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr_imp), columns=FEATURES)
    X_te_scaled = pd.DataFrame(scaler.transform(X_te_imp), columns=FEATURES)
    X_tr_scaled_sm, y_tr_scaled_sm = SMOTE(random_state=seed, k_neighbors=k_neighbors).fit_resample(X_tr_scaled, y_tr)

    return Split(X_tr_imp, X_te_imp, y_tr.reset_index(drop=True), y_te.reset_index(drop=True),
                 X_tr_sm, pd.Series(y_tr_sm), X_tr_scaled_sm, pd.Series(y_tr_scaled_sm),
                 X_te_scaled, dates.iloc[i:].reset_index(drop=True), imputer, scaler)


def check_canonical_sizes(X, y, split: Split) -> None:
    """Exit with code 1 if the split does not match CIFRE_CANONICE.md."""
    found = dict(n_total=len(X), n_train=len(split.y_train), n_test=len(split.y_test),
                 test_dist=split.y_test.value_counts().sort_index().to_dict())
    if found != CANON:
        sys.exit(f"FAIL canonical sizes: expected {CANON}, found {found}")
    print(f"PASS canonical sizes {found}")


def load_canonical(root: Path = ROOT, check: bool = True):
    """One-call helper: raw -> dataset -> split, with the size check."""
    X, y, dates = build_dataset(load_raw(root))
    split = canonical_split(X, y, dates)
    if check:
        check_canonical_sizes(X, y, split)
    return X, y, dates, split
