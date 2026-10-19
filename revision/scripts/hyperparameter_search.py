"""
hyperparameter_search.py - T2.2 (Reviewer 1, R1.4): reportable hyperparameter search for the
seven models (RF, XGB, SVM, LightGBM, CatBoost, TabNet, FT-Transformer), selection on temporal
CV of the training part only, test set touched once per model with the winning configuration.

Design (fixed before running, unchanged after):
  Folds: TimeSeriesSplit(n_splits=3, test_size=154) on the 766 canonical train rows, in temporal
  order: validation rows 304-457 / 458-611 / 612-765, train rows 0-303 / 0-457 / 0-611 (304 / 458 /
  612 rows). Active in train 10 / 17 / 25, in validation 7 / 8 / 13 (asserted; the brief quoted
  11 / 6 for fold 1 from the T1.3 TimeSeriesSplit(4) boundaries, whose first cut is row 307; with
  test_size 154 the first cut is row 304, which moves one Active day into the validation block).
  Per fold common.preprocess_split: imputer, SMOTE (k from threshold_sensitivity.smote_k, = 3 on
  every fold) and scaler fit on the fold's train part only; validation rows are real data.
  Search spaces: ParameterSampler(random_state=42, n_iter=20) per tabular model, generated ONCE
  and saved to results/t22_search_space.json before any fit; fixed grids of 5 for the networks
  (TabNet: n_d = n_a, n_steps, batch_size, virtual_batch_size, lr, drop_last False explicit;
  FT-Transformer: n_blocks, d_block, lr; 8 heads everywhere, 64 and 128 are divisible by 8).
  Reference rows (NOT eligible): the canonical configuration of RF / XGB / SVM (build_models())
  and the T2.1 defaults of the four new models, scored on the same folds.
  Networks per fold: max 200 epochs, early stopping on validation macro F1, patience 16, the
  T2.1 functions (extra_models.tabnet_fit / ftt_build / ftt_train); fold score = validation
  macro F1 at best_epoch. Tabular models: validation macro F1 of the fitted model.
  Selection: argmax of the mean validation macro F1 over the 3 folds among eligible
  configurations; ties -> smaller config_id. No look at the test set before --mode test.
  Final evaluation (--mode test): the winner of each model is trained on the 766 days (canonical
  SMOTE k = 3, seed 42) and evaluated ONCE on the 192 test days; networks follow the T2.1
  protocol (early stop on the inner 612 / 154 split with the winning configuration, refit on
  766 for best_epoch epochs). Predictions in results/t22_predictions_{model}.csv.
  Sacrifice order (declared; applied only with --reduced, recorded in run_meta): n_iter 10 for
  the tabular models (the first 10 of the SAME sampler) and grid indices 0, 2, 4 for the networks;
  folds and epochs are never reduced.

Modes:
  --mode cv     results/t22_search_space.json, t22_cv_results.csv, t22_cv_summary.md,
                t22_run_meta_cv.json
  --mode test   t22_predictions_{rf,xgb,svm,lgbm,catboost,tabnet,ftt}.csv, t22_test_results.csv /
                _table.md (one row per model: winner, CV F1, test metrics with 95 percent CI on the
                shared matrix, T2.1-default and canonical test F1), t22_test_metrics.csv (long, 11
                metrics x 7 tuned models), t22_comparisons.csv / _table.md (paired bootstrap:
                (a) tuned minus RF canonical, (b) tuned minus own default / canonical, (c) tuned
                minus RF tuned, (d) tuned minus persistence; macro F1 and balanced accuracy),
                t22_mcnemar.csv / _table.md (tuned vs RF canonical, Holm over the family of 7),
                t22_run_meta_test.json
  --mode seeds  winners x seeds {42, 0, 1, 2, 3} (SMOTE fixed at 42), t22_seeds.csv / _table.md,
                t22_run_meta_seeds.json; informative, NON-canonical

Sanity (abort on FAIL): fold rows and Active counts; RF canonical marginal CI on macro F1 ==
T1.1 row of CIFRE_REVIZIE.md (read at run time); point differences == csv differences; 0 skipped
resamples; argmax == y_pred (SVM: predict(), canonical convention, agreement reported); identity
with the reference prediction files when a winner coincides with its reference configuration;
seed 42 in --mode seeds == --mode test.

Rule: if the test hierarchy changes versus the defaults, BOTH hierarchies are reported with
paired CIs; RF canonical (0.5063) remains the system's model whatever happens here.
Run from repo root under `python -W error`. Exits 1 on any failed check. Nothing canonical changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "vendor"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import loguniform  # noqa: E402
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

import extra_models as em  # noqa: E402
from common import FEATURES, SEED, Split, preprocess_split  # noqa: E402
from extended_metrics import PROBA_COLS  # noqa: E402
from extended_metrics import fmt as fmt_ci  # noqa: E402
from group_ablation import f1_macro, read_t11_rf_f1_ci, save_predictions, sha256  # noqa: E402
from group_ablation import metrics as point_metrics  # noqa: E402
from save_test_predictions import build_models  # noqa: E402
from statistical_tests import (DISCORDANT_MIN_FOR_CC, INDEX_FILE, LABELS, md_table, mcnemar_counts,  # noqa: E402
                               mcnemar_tests, paired_bootstrap, write_text)
from statistical_tests import METRICS as PB_METRICS  # noqa: E402
from threshold_sensitivity import label_counts, smote_k  # noqa: E402

RES = ROOT / "revision" / "results"
N_TRAIN, N_TEST, N_BOOT = em.N_TRAIN, em.N_TEST, em.N_BOOT
N_SPLITS, VAL_SIZE, N_ITER = 3, 154, 20
MAX_EPOCHS, PATIENCE = em.MAX_EPOCHS, em.PATIENCE
MODELS = ["rf", "xgb", "svm", "lgbm", "catboost", "tabnet", "ftt"]
TABULAR = ["rf", "xgb", "svm", "lgbm", "catboost"]
NETWORKS = ["tabnet", "ftt"]
CANONICAL_MODELS = ["rf", "xgb", "svm"]
SCALED = {"rf": False, "xgb": False, "svm": True, "lgbm": False, "catboost": False, "tabnet": True, "ftt": True}
LABEL = dict(em.LABEL)
SEEDS = em.SEEDS
REF_ID = "ref"
EXPECT_FOLDS = [(0, 304, 458), (0, 458, 612), (0, 612, 766)]          # (train_start, train_end = val_start, val_end)
EXPECT_ACTIVE_TRAIN, EXPECT_ACTIVE_VAL = (10, 17, 25), (7, 8, 13)
REDUCED_N_ITER, REDUCED_GRID = 10, (0, 2, 4)
CANON_TEST_F1 = {"rf": 0.5063, "xgb": 0.4760, "svm": 0.4347}
T11_RF_BAL_ACC_CI = em.T11_RF_BAL_ACC_CI
COMPARISONS = {"a": "tuned_minus_rf_canonical", "b": "tuned_minus_own_reference", "c": "tuned_minus_rf_tuned",
               "d": "tuned_minus_persistence"}

# ------------------------------------------------------------------ search spaces (fixed)
SEARCH_SPACES: dict[str, dict] = {
    "rf": {"n_estimators": [100, 200, 500], "max_depth": [4, 6, 8, 10, None], "min_samples_leaf": [1, 3, 5, 10],
           "max_features": ["sqrt", 0.5, None]},
    "xgb": {"n_estimators": [100, 200, 500], "max_depth": [2, 3, 4, 6], "learning_rate": loguniform(0.01, 0.3),
            "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5],
            "reg_lambda": [1, 5, 10]},
    "svm": {"C": loguniform(0.01, 100), "gamma": loguniform(1e-4, 1)},
    "lgbm": {"n_estimators": [100, 200, 500], "num_leaves": [7, 15, 31], "max_depth": [-1, 4, 6],
             "learning_rate": loguniform(0.01, 0.3), "min_child_samples": [5, 10, 20, 40], "subsample": [0.6, 0.8, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0, 1, 10]},
    "catboost": {"iterations": [200, 500, 1000], "depth": [4, 6, 8], "learning_rate": loguniform(0.01, 0.3),
                 "l2_leaf_reg": [1, 3, 10], "random_strength": [1, 5]},
}
LGBM_FIXED = {"subsample_freq": 1}                   # with every sampled LightGBM configuration (bagging active)
NET_GRIDS: dict[str, list[dict]] = {
    "tabnet": [dict(n_d=8, n_steps=3, batch_size=128, virtual_batch_size=32, lr=2e-2),
               dict(n_d=16, n_steps=3, batch_size=128, virtual_batch_size=32, lr=2e-2),
               dict(n_d=32, n_steps=3, batch_size=128, virtual_batch_size=32, lr=2e-2),
               dict(n_d=16, n_steps=5, batch_size=128, virtual_batch_size=32, lr=2e-2),
               dict(n_d=16, n_steps=3, batch_size=256, virtual_batch_size=64, lr=1e-2)],
    "ftt": [dict(n_blocks=1, d_block=64, lr=1e-3), dict(n_blocks=2, d_block=64, lr=1e-3),
            dict(n_blocks=2, d_block=128, lr=3e-4), dict(n_blocks=3, d_block=128, lr=3e-4),
            dict(n_blocks=3, d_block=192, lr=1e-4)],
}
# Reference configurations (not eligible): canonical build_models() for RF / XGB / SVM, T2.1 defaults otherwise.
REFERENCE: dict[str, dict] = {
    "rf": dict(n_estimators=100, max_depth=10, min_samples_leaf=5, max_features="sqrt"),
    "xgb": dict(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, min_child_weight=1,
                reg_lambda=1),
    "svm": dict(C=1.0, gamma="scale"),
    "lgbm": {}, "catboost": {},
    "tabnet": dict(n_d=8, n_steps=3, batch_size=1024, virtual_batch_size=128, lr=2e-2, drop_last=True),
    "ftt": dict(em.FTT_DEFAULT_CONFIG),
}


# ------------------------------------------------------------------ pure helpers (unit-tested)
def py(v):
    """numpy scalar -> python scalar (json / csv friendly)."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def space_description(space: dict) -> dict:
    """Serialisable description of a search space (lists as is, loguniform as its bounds)."""
    out = {}
    for k, v in space.items():
        if isinstance(v, list):
            out[k] = [py(x) for x in v]
        else:
            a, b = v.support()
            out[k] = {"loguniform": [float(a), float(b)]}
    return out


def sample_configs(model: str, n_iter: int = N_ITER, seed: int = SEED) -> list[dict]:
    """ParameterSampler(random_state=seed) over the fixed space; prefix-stable in n_iter."""
    return [{k: py(v) for k, v in sorted(c.items())} for c in ParameterSampler(SEARCH_SPACES[model], n_iter=n_iter, random_state=seed)]


def build_search_space(reduced: bool = False) -> dict:
    """All configurations (ids 0..n-1) for all models, generated once from the fixed spaces / grids."""
    out: dict = {"seed": SEED, "n_iter": REDUCED_N_ITER if reduced else N_ITER, "reduced": reduced, "models": {}}
    for m in TABULAR:
        cfgs = sample_configs(m, out["n_iter"])
        out["models"][m] = {"space": space_description(SEARCH_SPACES[m]), "fixed": LGBM_FIXED if m == "lgbm" else {},
                            "configs": cfgs, "reference": REFERENCE[m]}
    for m in NETWORKS:
        grid = [dict(c) for c in NET_GRIDS[m]]
        ids = list(REDUCED_GRID) if reduced else list(range(len(grid)))
        out["models"][m] = {"grid": grid, "grid_ids_used": ids, "configs": [grid[i] for i in ids],
                            "config_ids": ids, "reference": REFERENCE[m]}
    return out


def config_key(cfg: dict) -> str:
    return json.dumps({k: py(v) for k, v in sorted(cfg.items())}, sort_keys=True)


def fold_bounds(n: int = N_TRAIN, n_splits: int = N_SPLITS, val_size: int = VAL_SIZE) -> list[tuple[int, int, int]]:
    """(train_start, train_end, val_end) of TimeSeriesSplit(n_splits, test_size=val_size) on n rows."""
    out = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits, test_size=val_size).split(np.zeros(n)):
        if tr[0] != 0 or tr[-1] + 1 != va[0] or va[-1] - va[0] + 1 != val_size:
            raise AssertionError("TimeSeriesSplit folds are not contiguous expanding blocks")
        out.append((int(tr[0]), int(va[0]), int(va[-1]) + 1))
    return out


def select_winner(cv: pd.DataFrame, model: str) -> tuple[int, float]:
    """argmax of f1_mean among eligible rows of `model`; ties -> smaller config_id."""
    sub = cv[(cv.model == model) & (cv.eligible)].copy()
    if sub.empty:
        raise ValueError(f"no eligible configuration for {model}")
    sub["config_id"] = sub["config_id"].astype(int)
    sub = sub.sort_values(["f1_mean", "config_id"], ascending=[False, True], kind="stable")
    r = sub.iloc[0]
    return int(r.config_id), float(r.f1_mean)


def reference_rank(cv: pd.DataFrame, model: str) -> int:
    """1 + number of eligible configurations with a strictly larger mean than the reference row."""
    sub = cv[cv.model == model]
    ref = float(sub[~sub.eligible].iloc[0].f1_mean)
    return int(1 + (sub[sub.eligible].f1_mean > ref).sum())


def mcnemar_safe(n00: int, n01: int, n10: int, n11: int) -> dict[str, float]:
    """mcnemar_tests of T1.2, with the degenerate case n01 + n10 == 0 (identical predictions)
    defined as statistic 0, p 1 (no discordant day: no evidence of a difference)."""
    if n01 + n10 == 0:
        return {"chi2_cc": 0.0, "p_cc": 1.0, "stat_exact": 0.0, "p_exact": 1.0}
    return mcnemar_tests(n00, n01, n10, n11)


# ------------------------------------------------------------------ data and folds
def make_folds(X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> list[dict]:
    """The 3 temporal folds on the train part, preprocessed with common.preprocess_split (imputer /
    SMOTE / scaler fit on the fold's train only). Asserts rows and Active counts."""
    bounds = fold_bounds()
    if bounds != EXPECT_FOLDS:
        raise AssertionError(f"fold bounds {bounds} != {EXPECT_FOLDS}")
    folds = []
    for k, (a, b, c) in enumerate(bounds, start=1):
        y_tr, y_va = y.iloc[a:b], y.iloc[b:c]
        ktr = smote_k(min(label_counts(y_tr)))
        if ktr is None:
            raise AssertionError(f"fold {k}: SMOTE not possible")
        s = preprocess_split(X.iloc[a:b], X.iloc[b:c], y_tr, y_va, dates.iloc[b:c], k_neighbors=ktr, seed=SEED)
        n_act = (int((y_tr == 2).sum()), int((y_va == 2).sum()))
        if n_act != (EXPECT_ACTIVE_TRAIN[k - 1], EXPECT_ACTIVE_VAL[k - 1]):
            raise AssertionError(f"fold {k}: Active train / val {n_act} != expected")
        if len(s.y_test) != VAL_SIZE or len(s.X_test) != VAL_SIZE or len(s.y_train) != b - a:
            raise AssertionError(f"fold {k}: sizes changed by preprocessing")
        folds.append(dict(fold=k, bounds=(a, b, c), split=s, smote_k=ktr, n_train=b - a, n_val=c - b,
                          n_train_smote=int(len(s.y_train_sm)), active_train=n_act[0], active_val=n_act[1],
                          counts_train=label_counts(y_tr), counts_val=label_counts(y_va)))
    return folds


# ------------------------------------------------------------------ model construction
def build_tabular(name: str, cfg: dict, seed: int):
    """Canonical base object (build_models) with the configuration applied; seed on the model only."""
    cfg = {k: py(v) for k, v in cfg.items()}
    if name == "rf":
        return build_models()["rf"].set_params(random_state=seed, **cfg)
    if name == "xgb":
        return build_models()["xgb"].set_params(random_state=seed, **cfg)
    if name == "svm":
        return build_models()["svm"].set_params(random_state=seed, **cfg)
    if name == "lgbm":
        return em.make_tree("lgbm", seed, **cfg)
    if name == "catboost":
        return em.make_tree("catboost", seed, **cfg)
    raise ValueError(name)


def tabnet_kwargs(cfg: dict) -> tuple[dict, dict]:
    """(constructor overrides, fit overrides) from a grid / reference configuration."""
    ctor = dict(n_d=int(cfg["n_d"]), n_a=int(cfg["n_d"]), n_steps=int(cfg["n_steps"]),
                optimizer_params=dict(lr=float(cfg["lr"])))
    fit = dict(batch_size=int(cfg["batch_size"]), virtual_batch_size=int(cfg["virtual_batch_size"]),
               drop_last=bool(cfg.get("drop_last", False)))
    return ctor, fit


def fit_predict(name: str, cfg: dict, seed: int, X_tr, y_tr, X_ev, y_ev=None, early_stop: bool = False) -> dict:
    """Fit one configuration on (X_tr, y_tr) and predict X_ev.
    Tabular: plain fit; returns proba, y_pred (SVM: predict(), else argmax), argmax agreement.
    Networks with early_stop: (X_ev, y_ev) is the validation set, early stopping on its macro F1
    (T2.1 functions); returns best_epoch, best_val_f1, epochs_run.
    Networks without early_stop: cfg must carry 'epochs' (the refit); returns proba, y_pred."""
    t0 = time.time()
    out: dict = {}
    if name in TABULAR:
        em.seed_everything(seed)
        m = build_tabular(name, cfg, seed)
        m.fit(X_tr, y_tr)
        proba = np.asarray(m.predict_proba(X_ev), dtype=np.float64)
        pred_api = np.asarray(m.predict(X_ev)).reshape(-1).astype(int)
        if [int(c) for c in m.classes_] != LABELS:
            raise AssertionError(f"{name}: classes_ != {LABELS}")
        out.update(proba=proba, y_pred=pred_api if name == "svm" else proba.argmax(1).astype(int),
                   argmax_agreement=float(np.mean(proba.argmax(1) == pred_api)), n_iter_effective=
                   int(m.tree_count_) if name == "catboost" else None)
    elif name == "tabnet":
        ctor, fit_kw = tabnet_kwargs(cfg)
        Xtr, Xev = np.asarray(X_tr, dtype=np.float32), np.asarray(X_ev, dtype=np.float32)
        if early_stop:
            m = em.tabnet_fit(Xtr, np.asarray(y_tr, dtype=int), seed, MAX_EPOCHS, PATIENCE, Xev, np.asarray(y_ev, dtype=int),
                              fit_overrides=fit_kw, **ctor)
            out.update(best_epoch=int(m.best_epoch) + 1, best_val_f1=float(m.best_cost), epochs_run=em.tabnet_epochs_run(m),
                       val_history=[float(v) for v in m.history["val_f1_macro"]],
                       n_batches_per_epoch=int(len(y_tr) // fit_kw["batch_size"]) if fit_kw["drop_last"]
                       else -(-len(y_tr) // fit_kw["batch_size"]))
        else:
            m = em.tabnet_fit(Xtr, np.asarray(y_tr, dtype=int), seed, int(cfg["epochs"]), 0, fit_overrides=fit_kw, **ctor)
            if em.tabnet_epochs_run(m) != int(cfg["epochs"]):
                raise AssertionError("tabnet refit epochs != best_epoch")
            proba = np.asarray(m.predict_proba(Xev), dtype=np.float64)
            pred_api = np.asarray(m.predict(Xev)).reshape(-1).astype(int)
            out.update(proba=proba, y_pred=proba.argmax(1).astype(int), argmax_agreement=float(np.mean(proba.argmax(1) == pred_api)))
    elif name == "ftt":
        arch = {k: cfg[k] for k in ("n_blocks", "d_block", "lr")}
        model, opt = em.ftt_build(X_tr.shape[1], seed, arch)
        if early_stop:
            r = em.ftt_train(model, opt, np.asarray(X_tr), np.asarray(y_tr, dtype=int), seed, MAX_EPOCHS, em.FTT_BATCH,
                             np.asarray(X_ev), np.asarray(y_ev, dtype=int), PATIENCE)
            out.update(best_epoch=int(r["best_epoch"]), best_val_f1=float(r["best_val_f1"]), epochs_run=r["epochs_run"],
                       val_history=r["val_history"], n_batches_per_epoch=r["n_batches_per_epoch"])
        else:
            r = em.ftt_train(model, opt, np.asarray(X_tr), np.asarray(y_tr, dtype=int), seed, int(cfg["epochs"]), em.FTT_BATCH)
            if r["epochs_run"] != int(cfg["epochs"]):
                raise AssertionError("ftt refit epochs != best_epoch")
            proba = em.ftt_proba(model, np.asarray(X_ev))
            out.update(proba=proba, y_pred=proba.argmax(1).astype(int), argmax_agreement=1.0)
        out["n_parameters"] = sum(int(p.numel()) for p in model.parameters())
    else:
        raise ValueError(name)
    out["fit_seconds"] = round(time.time() - t0, 1)
    return out


def fold_matrices(name: str, s: Split) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Train (SMOTE) and evaluation matrices of a Split for the model's input flow (scaled or not)."""
    if SCALED[name]:
        return s.X_train_scaled_sm, s.y_train_scaled_sm, s.X_test_scaled, s.y_test
    return s.X_train_sm, s.y_train_sm, s.X_test, s.y_test


def score_config_on_folds(name: str, cfg: dict, folds: list[dict], seed: int = SEED) -> dict:
    """Validation macro F1 on each fold (+ best_epoch for the networks); mean and sample sd."""
    f1s, epochs, secs, extra = [], [], [], []
    for f in folds:
        X_tr, y_tr, X_va, y_va = fold_matrices(name, f["split"])
        if name in NETWORKS:
            r = fit_predict(name, cfg, seed, X_tr, y_tr, X_va, y_va, early_stop=True)
            f1s.append(r["best_val_f1"])
            epochs.append(r["best_epoch"])
            extra.append({k: r[k] for k in ("epochs_run", "n_batches_per_epoch")})
        else:
            r = fit_predict(name, cfg, seed, X_tr, y_tr, X_va)
            f1s.append(f1_macro(y_va.to_numpy(dtype=int), r["y_pred"]))
            epochs.append(None)
            extra.append({"argmax_agreement": r["argmax_agreement"]})
        secs.append(r["fit_seconds"])
    arr = np.asarray(f1s, dtype=float)
    return dict(f1_folds=[float(v) for v in f1s], f1_mean=float(arr.mean()), f1_sd=float(arr.std(ddof=1)),
                best_epochs=epochs, fit_seconds=secs, extra=extra)


def final_fit(name: str, cfg: dict, seed: int, split: Split, inner: Split) -> dict:
    """Winner trained on the 766 days (networks: early stop on the inner 612 / 154 split with this
    configuration, then refit for best_epoch) and predicted ONCE on the 192 test days."""
    X_tr, y_tr, X_te, _ = fold_matrices(name, split)
    info: dict = {}
    if name in NETWORKS:
        X_in, y_in, X_val, y_val = fold_matrices(name, inner)
        a = fit_predict(name, cfg, seed, X_in, y_in, X_val, y_val, early_stop=True)
        info.update(best_epoch=a["best_epoch"], val_f1_macro_at_best=a["best_val_f1"], stage_a_epochs_run=a["epochs_run"],
                    val_history=a["val_history"], stage_a_seconds=a["fit_seconds"])
        cfg = dict(cfg, epochs=a["best_epoch"])
    r = fit_predict(name, cfg, seed, X_tr, y_tr, X_te)
    info.update(proba=r["proba"], y_pred=r["y_pred"], argmax_agreement=r["argmax_agreement"], fit_seconds=r["fit_seconds"],
                n_parameters=r.get("n_parameters"), n_iter_effective=r.get("n_iter_effective"))
    return info


# ------------------------------------------------------------------ tables
CV_COLS = ["model", "config_id", "eligible", "params", "f1_fold1", "f1_fold2", "f1_fold3", "f1_mean", "f1_sd",
           "best_epoch_fold1", "best_epoch_fold2", "best_epoch_fold3"]
CV_FMT = {c: "%.4f" for c in ("f1_fold1", "f1_fold2", "f1_fold3", "f1_mean", "f1_sd")}


def cv_summary(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in [x for x in MODELS if x in set(cv.model)]:
        sub = cv[cv.model == m]
        wid, wmean = select_winner(cv, m)
        w = sub[(sub.eligible) & (sub.config_id.astype(str) == str(wid))].iloc[0]
        ref = sub[~sub.eligible].iloc[0]
        rows.append(dict(model=m, n_configs=int(sub.eligible.sum()), best_config_id=wid, best_params=w.params,
                         best_f1_mean=float(w.f1_mean), best_f1_sd=float(w.f1_sd),
                         best_f1_folds=f"{float(w.f1_fold1):.4f} / {float(w.f1_fold2):.4f} / {float(w.f1_fold3):.4f}",
                         best_epochs="" if m not in NETWORKS else f"{w.best_epoch_fold1} / {w.best_epoch_fold2} / {w.best_epoch_fold3}",
                         ref_f1_mean=float(ref.f1_mean), ref_f1_sd=float(ref.f1_sd), ref_rank=reference_rank(cv, m),
                         gain_cv=float(w.f1_mean) - float(ref.f1_mean)))
    return pd.DataFrame(rows)


CV_SUMMARY_FMT = {c: "%.4f" for c in ("best_f1_mean", "best_f1_sd", "ref_f1_mean", "ref_f1_sd", "gain_cv")}
TEST_FMT_COLS = ["cv_f1_mean", "cv_f1_sd", "f1_macro", "f1_macro_ci_low", "f1_macro_ci_high", "f1_rest", "f1_normal",
                 "f1_active", "f1_active_ci_low", "f1_active_ci_high", "accuracy", "balanced_accuracy", "mcc", "mcc_ci_low",
                 "mcc_ci_high", "cohen_kappa", "g_mean", "roc_auc_ovr_macro", "pr_auc_macro", "val_f1_inner", "default_t21_test_f1",
                 "canonical_test_f1", "own_reference_test_f1", "delta_vs_own_reference", "argmax_agreement"]
COMP_FMT_COLS = ["value_tuned", "value_other", "diff_point", "ci_low", "ci_high", "prop_gt_0"]
MCN_FMT_COLS = ["chi2_cc", "p_cc", "p_exact", "p_cc_holm", "p_exact_holm"]


def cv_md_lines(cv: pd.DataFrame) -> list[str]:
    return md_table(cv_summary(cv), CV_SUMMARY_FMT)


def test_md_lines(t: pd.DataFrame) -> list[str]:
    return md_table(t, {c: "%.4f" for c in TEST_FMT_COLS})


def comparisons_md_lines(c: pd.DataFrame) -> list[str]:
    return md_table(c, {c_: "%.4f" for c_ in COMP_FMT_COLS})


def mcnemar_md_lines(mc: pd.DataFrame) -> list[str]:
    return md_table(mc, {c: "%.4f" for c in MCN_FMT_COLS})


def seeds_md_lines(df: pd.DataFrame) -> list[str]:
    return em.seeds_md_lines(em.seeds_summary(df, MODELS))


# ------------------------------------------------------------------ modes
def _checker():
    fails: list[str] = []

    def check(cond: bool, msg: str, abort: bool = False) -> None:
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)
            if abort:
                sys.exit(f"ABORT: {msg}")
    return fails, check


def run_cv(res: Path, reduced: bool) -> None:
    t0 = time.time()
    fails, check = _checker()
    vers = em.versions()
    check(em.VENDOR.resolve() in Path(vers["pytorch_tabnet_path"]).parents, "pytorch_tabnet imported from revision/vendor", abort=True)
    space = build_search_space(reduced)
    em.write_json(res / "t22_search_space.json", em.jsonable(space))
    print(f"search space written ({'REDUCED: ' + str(REDUCED_N_ITER) + ' / grid ' + str(REDUCED_GRID) if reduced else 'full: 20 / 5'})")
    data = em.load_data()
    folds = make_folds(*data)
    for f in folds:
        print(f"PASS fold {f['fold']}: train rows {f['bounds'][0]}-{f['bounds'][1] - 1} ({f['n_train']}), val rows "
              f"{f['bounds'][1]}-{f['bounds'][2] - 1} ({f['n_val']}); Active {f['active_train']} / {f['active_val']}; "
              f"SMOTE k {f['smote_k']}, train rows after SMOTE {f['n_train_smote']}")
    rows: list[dict] = []
    timing: dict = {}
    failed: list[dict] = []
    for name in MODELS:
        tm = time.time()
        spec = space["models"][name]
        cfg_ids = spec.get("config_ids", list(range(len(spec["configs"]))))
        todo = [(str(i), c, True) for i, c in zip(cfg_ids, spec["configs"])] + [(REF_ID, spec["reference"], False)]
        timing[name] = {"configs": {}}
        for cid, cfg, eligible in todo:
            full_cfg = dict(cfg, **LGBM_FIXED) if (name == "lgbm" and eligible) else dict(cfg)
            try:
                r = score_config_on_folds(name, full_cfg, folds)
            except Exception as e:                       # reported, never replaced
                failed.append(dict(model=name, config_id=cid, params=config_key(full_cfg), error=repr(e)))
                print(f"FAIL {name} config {cid}: {e!r}")
                continue
            be = r["best_epochs"]
            rows.append(dict(model=name, config_id=cid, eligible=eligible, params=config_key(full_cfg),
                             f1_fold1=r["f1_folds"][0], f1_fold2=r["f1_folds"][1], f1_fold3=r["f1_folds"][2],
                             f1_mean=r["f1_mean"], f1_sd=r["f1_sd"],
                             best_epoch_fold1=be[0] if be[0] is not None else "", best_epoch_fold2=be[1] if be[1] is not None else "",
                             best_epoch_fold3=be[2] if be[2] is not None else ""))
            timing[name]["configs"][cid] = {"fit_seconds": r["fit_seconds"], "extra": r["extra"]}
            print(f"  {LABEL[name]:<15} {cid:>3} {'ref' if not eligible else '   '} F1 {r['f1_mean']:.4f} +- {r['f1_sd']:.4f} "
                  f"({' / '.join(f'{v:.4f}' for v in r['f1_folds'])})"
                  + (f"  best_epoch {be}" if name in NETWORKS else "") + f"  {sum(r['fit_seconds']):.1f} s")
        timing[name]["elapsed_seconds"] = round(time.time() - tm, 1)
        print(f"  {LABEL[name]} done in {timing[name]['elapsed_seconds']} s (total so far {time.time() - t0:.0f} s)")
    cv = pd.DataFrame(rows)[CV_COLS]
    n_expected = sum(len(space["models"][m]["configs"]) + 1 for m in MODELS)
    check(len(cv) == n_expected and not failed, f"{n_expected} configurations x 3 folds scored, 0 failed fits ({len(failed)} failed)")
    check(bool(((cv.f1_mean >= 0) & (cv.f1_mean <= 1)).all()), "all mean F1 in [0, 1]")
    for m in NETWORKS:
        sub = cv[cv.model == m]
        be = sub[["best_epoch_fold1", "best_epoch_fold2", "best_epoch_fold3"]].astype(int).to_numpy()
        check(bool(((be >= 1) & (be <= MAX_EPOCHS)).all()), f"{m}: best_epoch per fold in [1, {MAX_EPOCHS}]")
    summary = cv_summary(cv)
    for r in summary.itertuples():
        print(f"  winner {LABEL[r.model]:<15} config {r.best_config_id} F1 {r.best_f1_mean:.4f} +- {r.best_f1_sd:.4f}; "
              f"reference {r.ref_f1_mean:.4f} (rank {r.ref_rank} of {r.n_configs + 1}); {r.best_params}")
    run_date = datetime.now().isoformat(timespec="seconds")
    day = run_date[:10]
    cv.to_csv(res / "t22_cv_results.csv", index=False, float_format="%.10g", lineterminator="\n")
    md = cv_md_lines(cv)
    md += ["",
           f"Temporal CV on the 766 train days: TimeSeriesSplit(n_splits={N_SPLITS}, test_size={VAL_SIZE}); validation rows "
           "304-457 / 458-611 / 612-765 (Active 7 / 8 / 13), train rows 0-303 / 0-457 / 0-611 (Active 10 / 17 / 25); imputer, "
           "SMOTE (k = 3) and scaler fit on the fold's train only. Score = validation macro F1 (labels [0, 1, 2], zero_division 0); "
           "mean and sample sd over the 3 folds; selection = argmax of the mean among eligible configurations (ties: smaller "
           f"config_id). Tabular models: ParameterSampler(random_state={SEED}, n_iter={space['n_iter']}) over the fixed spaces of "
           "t22_search_space.json; networks: fixed grid " + ("indices 0, 2, 4" if reduced else "of 5") + " with early stopping per "
           f"fold (max {MAX_EPOCHS} epochs, patience {PATIENCE}, macro F1). ref = reference row, NOT eligible: canonical "
           "build_models() configuration (RF 100 / 10 / 5, XGB 100 / 6 / 0.1, SVM C 1 / gamma scale) or the T2.1 defaults; ref_rank = "
           "1 + number of eligible configurations with a larger mean. No test-set information is used here. "
           f"Script: revision/scripts/hyperparameter_search.py --mode cv, run {day}."]
    write_text(res / "t22_cv_summary.md", "\n".join(md) + "\n")
    meta = {"run_date": run_date, "elapsed_seconds": round(time.time() - t0, 1), "timing": timing,
            "task": "T2.2", "mode": "cv", "seed": SEED, "reduced": reduced, "n_iter": space["n_iter"],
            "sacrifice_order": "declared: n_iter 10 (first 10 of the same sampler) and network grid indices 0, 2, 4; "
                               + ("APPLIED" if reduced else "NOT applied"),
            "folds": [{k: v for k, v in f.items() if k != "split"} for f in folds],
            "fold_note": "brief quoted Active 11 / 6 for fold 1 from the T1.3 TimeSeriesSplit(4) boundary (row 307); with "
                         "test_size 154 the first cut is row 304: 10 / 7",
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "versions": vers, "torch_threads": torch.get_num_threads(),
            "search_space_sha256": sha256(res / "t22_search_space.json"),
            "winners": em.jsonable(summary.to_dict(orient="records")), "failed_fits": failed,
            "files_sha256": {f: sha256(res / f) for f in ("t22_cv_results.csv", "t22_cv_summary.md")},
            "sanity_checks_failed": fails}
    em.write_json(res / "t22_run_meta_cv.json", em.jsonable(meta))
    print(f"\nelapsed {time.time() - t0:.1f}s; outputs in {res}")
    if fails:
        print(f"{len(fails)} check(s) FAILED: {fails}")
        sys.exit(1)
    print("ALL CHECKS PASS")


def load_winners(res: Path) -> tuple[pd.DataFrame, dict[str, dict]]:
    cv = pd.read_csv(res / "t22_cv_results.csv", keep_default_na=False)
    cv["eligible"] = cv["eligible"].astype(str) == "True"
    winners = {}
    for m in MODELS:
        wid, _ = select_winner(cv, m)
        row = cv[(cv.model == m) & (cv.eligible) & (cv.config_id.astype(str) == str(wid))].iloc[0]
        winners[m] = dict(config_id=wid, params=json.loads(row.params), cv_f1_mean=float(row.f1_mean), cv_f1_sd=float(row.f1_sd))
    return cv, winners


def reference_predictions(res: Path, m: str) -> pd.DataFrame:
    return pd.read_csv(res / (f"test_predictions_{m}.csv" if m in CANONICAL_MODELS else f"t21_predictions_{m}.csv"))


def run_test(res: Path) -> None:
    t0 = time.time()
    fails, check = _checker()
    vers = em.versions()
    cv, winners = load_winners(res)
    data = em.load_data()
    X, y, dates = data
    split, k = em.canonical_preprocess(X, y, dates)
    inner, _ = em.inner_preprocess(X, y, dates)
    idx = em.load_index(res)
    t11_ci = read_t11_rf_f1_ci()
    y_true, preds, probas, _ = em.load_saved_systems(res)
    check(np.array_equal(split.y_test.to_numpy(dtype=int), y_true), "y_true == saved systems", abort=True)
    t21 = pd.read_csv(res / "t21_extra_models.csv")
    t21_f1 = {r.system: float(r.value) for r in t21[t21.metric == "f1_macro"].itertuples()}
    print("\n=== T2.2 test: winners trained on 766 days, evaluated once ===")
    results: dict[str, dict] = {}
    pred_files: dict[str, str] = {}
    for m in MODELS:
        w = winners[m]
        cfg = dict(w["params"])
        r = final_fit(m, cfg, SEED, split, inner)
        results[m] = r
        f = res / f"t22_predictions_{m}.csv"
        save_predictions(f, split.dates_test, y_true, r["y_pred"], r["proba"])
        pred_files[f.name] = sha256(f)
        max_dev = float(np.abs(r["proba"].sum(1) - 1).max())
        check(max_dev < 1e-6, f"{m}: probabilities sum to 1 (max dev {max_dev:.1e})", abort=True)
        if m != "svm":
            check(np.array_equal(r["proba"].argmax(1), r["y_pred"]), f"{m}: argmax == y_pred", abort=True)
        else:
            print(f"INFO svm: y_pred = predict() (canonical convention); argmax agreement {r['argmax_agreement']:.4f}")
        same_as_ref = config_key({k_: v for k_, v in cfg.items() if k_ != "epochs"}) == config_key(
            dict(REFERENCE[m], **LGBM_FIXED) if m == "lgbm" else REFERENCE[m])
        if same_as_ref:
            ref = reference_predictions(res, m)
            check(np.array_equal(ref.y_pred.to_numpy(), r["y_pred"]) and float(np.abs(ref[PROBA_COLS].to_numpy() - r["proba"]).max()) < 1e-9,
                  f"{m}: winner == reference configuration -> predictions identical to the reference file", abort=True)
        pm = point_metrics(y_true, r["y_pred"])
        r["point"] = pm
        r["same_as_reference"] = bool(same_as_ref)
        print(f"  {LABEL[m]:<15} config {w['config_id']:>2} CV {w['cv_f1_mean']:.4f}  test F1 {pm['f1_macro']:.4f}  Active {pm['f1_active']:.4f}  "
              f"acc {pm['accuracy']:.4f}  MCC {pm['mcc']:.4f}" + (f"  best_epoch {r['best_epoch']}" if m in NETWORKS else "")
              + f"  fit {r['fit_seconds']} s")
    # ---- metrics with CI (7 tuned models)
    systems = {m: (results[m]["y_pred"], results[m]["proba"]) for m in MODELS}
    table = em.metrics_with_ci(systems, y_true, idx)
    table["system"] = table["system"].map(lambda s: f"{s}_tuned")
    g = {(r.system, r.metric): r for r in table.itertuples()}
    check(int(table.n_skipped.sum()) == 0, "0 resamples skipped on the 7 tuned models", abort=True)
    check(int(table.n_warnings.sum()) == 0, "0 sklearn warnings inside the bootstrap loops")
    # RF canonical marginal CI on the same matrix (sanity)
    b0 = paired_bootstrap(y_true, preds["rf"], results["rf"]["y_pred"], PB_METRICS["f1_macro"], idx)
    got = (round(b0["ci_A_low"], 4), round(b0["ci_A_high"], 4))
    check(got == t11_ci, f"sanity: RF canonical marginal CI {got} == T1.1 {t11_ci}", abort=True)
    # ---- wide test table
    trows = []
    for m in MODELS:
        w, r, pm = winners[m], results[m], results[m]["point"]
        own_ref = CANON_TEST_F1.get(m, t21_f1.get(m))
        trows.append(dict(model=m, label=LABEL[m], best_config_id=w["config_id"], best_params=config_key(w["params"]),
                          cv_f1_mean=w["cv_f1_mean"], cv_f1_sd=w["cv_f1_sd"],
                          best_epoch=r.get("best_epoch", "n/a"), val_f1_inner=r.get("val_f1_macro_at_best", np.nan),
                          f1_macro=pm["f1_macro"], f1_macro_ci_low=g[f"{m}_tuned", "f1_macro"].ci_low, f1_macro_ci_high=g[f"{m}_tuned", "f1_macro"].ci_high,
                          f1_rest=pm["f1_rest"], f1_normal=pm["f1_normal"], f1_active=pm["f1_active"],
                          f1_active_ci_low=g[f"{m}_tuned", "f1_active"].ci_low, f1_active_ci_high=g[f"{m}_tuned", "f1_active"].ci_high,
                          accuracy=pm["accuracy"], balanced_accuracy=pm["balanced_accuracy"], mcc=pm["mcc"],
                          mcc_ci_low=g[f"{m}_tuned", "mcc"].ci_low, mcc_ci_high=g[f"{m}_tuned", "mcc"].ci_high,
                          cohen_kappa=g[f"{m}_tuned", "cohen_kappa"].value, g_mean=g[f"{m}_tuned", "g_mean"].value,
                          roc_auc_ovr_macro=g[f"{m}_tuned", "roc_auc_ovr_macro"].value, pr_auc_macro=g[f"{m}_tuned", "pr_auc_macro"].value,
                          default_t21_test_f1=t21_f1.get(m, np.nan) if m not in CANONICAL_MODELS else np.nan,
                          canonical_test_f1=CANON_TEST_F1.get(m, np.nan), own_reference_test_f1=own_ref,
                          delta_vs_own_reference=pm["f1_macro"] - own_ref, same_as_reference=r["same_as_reference"],
                          argmax_agreement=r["argmax_agreement"], n_active_pred=int((r["y_pred"] == 2).sum()),
                          cm=" ".join(str(pm[f"cm_{a}{b}"]) for a in "rna" for b in "rna")))
    tt = pd.DataFrame(trows)
    # ---- comparisons (a)-(d), paired bootstrap on the shared matrix
    crows = []
    for m in MODELS:
        yp = results[m]["y_pred"]
        others = {"a": ("rf_canonical", preds["rf"]), "b": (f"{m}_reference", reference_predictions(res, m).y_pred.to_numpy(dtype=int)),
                  "c": ("rf_tuned", results["rf"]["y_pred"]), "d": ("persistence", preds["persistence"])}
        for key, (oname, op) in others.items():
            if key == "c" and m == "rf":
                continue
            for mname, mfn in PB_METRICS.items():
                b = paired_bootstrap(y_true, yp, op, mfn, idx)
                exp = mfn(y_true, yp) - mfn(y_true, op)
                check(abs(b["diff_point"] - exp) < 1e-12, f"{m} ({key}) {mname}: diff_point == difference of the point values", abort=True)
                check(b["n_skipped"] == 0, f"{m} ({key}) {mname}: 0 resamples skipped", abort=True)
                crows.append(dict(model=m, comparison=key, comparison_name=COMPARISONS[key], other=oname, metric=mname,
                                  value_tuned=mfn(y_true, yp), value_other=mfn(y_true, op), diff_point=b["diff_point"],
                                  ci_low=b["ci_low"], ci_high=b["ci_high"], include_zero=bool(b["ci_low"] <= 0.0 <= b["ci_high"]),
                                  prop_gt_0=b["prop_diff_gt_0"], n_skipped=b["n_skipped"]))
    comp = pd.DataFrame(crows)
    ca = comp[(comp.comparison == "a") & (comp.metric == "f1_macro")].set_index("model")
    check(abs(float(ca.loc["rf", "value_other"]) - 0.5063) < 5e-5, "comparison (a): RF canonical point == 0.5063")
    # ---- McNemar tuned vs RF canonical, Holm over the family of 7
    mrows = []
    for m in MODELS:
        n00, n01, n10, n11 = mcnemar_counts(y_true, results[m]["y_pred"], preds["rf"])
        t = mcnemar_safe(n00, n01, n10, n11)
        disc = n01 + n10
        mrows.append(dict(model=m, pair=f"{LABEL[m]} tuned - RF canonical", n00=n00, n01=n01, n10=n10, n11=n11, chi2_cc=t["chi2_cc"],
                          p_cc=t["p_cc"], p_exact=t["p_exact"], p_cc_holm=np.nan, p_exact_holm=np.nan,
                          recommended="exact" if disc < DISCORDANT_MIN_FOR_CC else "cc", discordant=disc))
        check(n00 + n01 + n10 + n11 == N_TEST, f"{m}: McNemar cells sum to {N_TEST}")
    mc = pd.DataFrame(mrows)
    mc["p_cc_holm"] = multipletests(mc["p_cc"].to_numpy(), method="holm")[1]
    mc["p_exact_holm"] = multipletests(mc["p_exact"].to_numpy(), method="holm")[1]
    # ---- hierarchy report
    order_tuned = tt.sort_values("f1_macro", ascending=False).model.tolist()
    defaults = {m: CANON_TEST_F1.get(m, t21_f1.get(m)) for m in MODELS}
    order_default = sorted(MODELS, key=lambda m_: -defaults[m_])
    above_rf_canon = [m for m in MODELS if results[m]["point"]["f1_macro"] > 0.5063]
    print(f"\n  hierarchy defaults / canonical: {' > '.join(LABEL[m] for m in order_default)}")
    print(f"  hierarchy tuned:               {' > '.join(LABEL[m] for m in order_tuned)}")
    print(f"  tuned models above RF canonical (0.5063) at the point: {[LABEL[m] for m in above_rf_canon] or 'none'}")
    for r in ca.itertuples():
        print(f"  (a) {LABEL[r.Index]:<15} tuned minus RF canonical {r.diff_point:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] "
              f"(include 0: {r.include_zero}, prop > 0 {r.prop_gt_0:.3f})")
    # ---- outputs
    run_date = datetime.now().isoformat(timespec="seconds")
    day = run_date[:10]
    table.to_csv(res / "t22_test_metrics.csv", index=False, float_format="%.10g", lineterminator="\n")
    tt.to_csv(res / "t22_test_results.csv", index=False, float_format="%.10g", lineterminator="\n")
    md = test_md_lines(tt)
    md += ["",
           "One row per model: the configuration selected on temporal CV (t22_cv_results.csv, mean validation macro F1 over 3 "
           "folds) trained on the 766 train days (imputer + SMOTE k = 3 fit on train, seed 42; networks: early stopping on the "
           "inner 612 / 154 split with that configuration, then refit for best_epoch epochs) and evaluated ONCE on the 192 test "
           f"days. [ci_low, ci_high] = bootstrap 95 percent CI, {N_BOOT} resamples on the shared matrix {INDEX_FILE}; the same "
           "metric functions as T1.1 / T2.1. default_t21_test_f1 = T2.1 default-parameter test macro F1 (new models); "
           "canonical_test_f1 = CIFRE_CANONICE figures (RF / XGB / SVM); own_reference_test_f1 = whichever applies; "
           "delta_vs_own_reference = tuned minus reference at the point (paired CI in t22_comparisons.csv, comparison b). "
           "same_as_reference = the winner coincides with the reference configuration (then the predictions equal the reference "
           "file). SVM: y_pred = predict() (canonical convention). cm = confusion counts rr rn ra nr nn na ar an aa. "
           "RF canonical (0.5063) remains the system's reported model regardless of this table. "
           f"Script: revision/scripts/hyperparameter_search.py --mode test, run {day}."]
    write_text(res / "t22_test_results_table.md", "\n".join(md) + "\n")
    comp.to_csv(res / "t22_comparisons.csv", index=False, float_format="%.10g", lineterminator="\n")
    cmd = comparisons_md_lines(comp)
    cmd += ["",
            "diff = metric(tuned model) - metric(other); other: (a) RF canonical (test_predictions_rf.csv), (b) the model's own "
            "reference predictions (canonical test_predictions_*.csv for RF / XGB / SVM, T2.1 default t21_predictions_*.csv for "
            "the four new models), (c) RF tuned (this run), (d) persistence. [ci_low, ci_high] = percentile 2.5 / 97.5 of the "
            f"paired bootstrap over {N_BOOT} resamples ({INDEX_FILE}, same indices for both prediction vectors); prop_gt_0 "
            "descriptive (not a p-value); include_zero = ci_low <= 0 <= ci_high. CIs are not adjusted for the number of "
            f"comparisons. Script: revision/scripts/hyperparameter_search.py --mode test, run {day}."]
    write_text(res / "t22_comparisons_table.md", "\n".join(cmd) + "\n")
    mc.to_csv(res / "t22_mcnemar.csv", index=False, float_format="%.10g", lineterminator="\n")
    mmd = mcnemar_md_lines(mc)
    mmd += ["",
            "McNemar on correct / incorrect per day, A = tuned model, B = RF canonical (n01 = tuned correct and RF wrong, n10 = "
            "tuned wrong and RF correct); chi2_cc / p_cc with continuity correction, p_exact binomial; recommended = exact when "
            f"n01 + n10 < {DISCORDANT_MIN_FOR_CC}; Holm over the family of 7 comparisons (p_cc and p_exact separately). A pair with "
            "0 discordant days (identical predictions) is defined as statistic 0, p 1. McNemar compares per-day accuracy, not "
            f"macro F1 (T1.2 note). Script: revision/scripts/hyperparameter_search.py --mode test, run {day}."]
    write_text(res / "t22_mcnemar_table.md", "\n".join(mmd) + "\n")
    meta = {"run_date": run_date, "elapsed_seconds": round(time.time() - t0, 1),
            "fit_seconds": {m: {"final": results[m]["fit_seconds"], "stage_a": results[m].get("stage_a_seconds")} for m in MODELS},
            "task": "T2.2", "mode": "test", "seed": SEED, "smote_seed": SEED, "smote_k": k, "n_resamples": N_BOOT,
            "versions": vers, "torch_threads": torch.get_num_threads(),
            "winners": em.jsonable(winners),
            "models": {m: {k_: v for k_, v in results[m].items() if k_ not in ("proba", "y_pred", "fit_seconds", "stage_a_seconds")}
                       for m in MODELS},
            "hierarchy_default_or_canonical": order_default, "hierarchy_tuned": order_tuned,
            "tuned_above_rf_canonical": above_rf_canon,
            "t11_rf_f1_macro_ci_from_cifre_revizie": list(t11_ci), "rf_canonical_marginal_ci": [b0["ci_A_low"], b0["ci_A_high"]],
            "cv_results_sha256": sha256(res / "t22_cv_results.csv"), "bootstrap_indices_sha256": sha256(res / INDEX_FILE),
            "prediction_files_sha256": pred_files,
            "table_files_sha256": {f: sha256(res / f) for f in ("t22_test_results.csv", "t22_test_results_table.md", "t22_test_metrics.csv",
                                                                "t22_comparisons.csv", "t22_comparisons_table.md", "t22_mcnemar.csv",
                                                                "t22_mcnemar_table.md")},
            "sanity_checks_failed": fails}
    em.write_json(res / "t22_run_meta_test.json", em.jsonable(meta))
    print(f"\nelapsed {time.time() - t0:.1f}s; outputs in {res}")
    if fails:
        print(f"{len(fails)} check(s) FAILED: {fails}")
        sys.exit(1)
    print("ALL CHECKS PASS")


def run_seeds(res: Path, seeds: list[int] = SEEDS) -> None:
    t0 = time.time()
    fails, check = _checker()
    vers = em.versions()
    cv, winners = load_winners(res)
    data = em.load_data()
    X, y, dates = data
    split, _ = em.canonical_preprocess(X, y, dates)
    inner, _ = em.inner_preprocess(X, y, dates)
    y_true = split.y_test.to_numpy(dtype=int)
    for m in MODELS:
        check((res / f"t22_predictions_{m}.csv").exists(), f"--mode test predictions exist for {m}", abort=True)
    rows, timing, identity = [], {}, {}
    print(f"\n=== T2.2 seeds {seeds} x winners (SMOTE seed {SEED} fixed; model seed varies) ===")
    for m in MODELS:
        timing[m] = {}
        for seed in seeds:
            r = final_fit(m, dict(winners[m]["params"]), seed, split, inner)
            timing[m][str(seed)] = r["fit_seconds"]
            pm = point_metrics(y_true, r["y_pred"])
            if seed == SEED:
                ref = pd.read_csv(res / f"t22_predictions_{m}.csv")
                same = np.array_equal(ref.y_pred.to_numpy(), r["y_pred"])
                max_dp = float(np.abs(ref[PROBA_COLS].to_numpy() - r["proba"]).max())
                identity[m] = {"y_pred_identical": bool(same), "n_days_y_pred_differ": int((ref.y_pred.to_numpy() != r["y_pred"]).sum()),
                               "max_abs_proba_diff": max_dp}
                check(same and max_dp < 1e-9, f"{m} seed {SEED}: identical to --mode test (max abs proba diff {max_dp:.2e})", abort=True)
            rows.append(dict(model=m, seed=seed, f1_macro=pm["f1_macro"], f1_active=pm["f1_active"], accuracy=pm["accuracy"],
                             mcc=pm["mcc"], balanced_accuracy=pm["balanced_accuracy"], best_epoch=r.get("best_epoch", np.nan),
                             val_f1_macro_at_best=r.get("val_f1_macro_at_best", np.nan)))
            print(f"  {LABEL[m]:<15} seed {seed:>2}  F1 macro {pm['f1_macro']:.4f}  Active {pm['f1_active']:.4f}  acc {pm['accuracy']:.4f}  "
                  f"MCC {pm['mcc']:.4f}" + (f"  best_epoch {r['best_epoch']}" if m in NETWORKS else "") + f"  {r['fit_seconds']} s")
    df = pd.DataFrame(rows)
    check(len(df) == len(MODELS) * len(seeds), f"{len(MODELS) * len(seeds)} runs", abort=True)
    run_date = datetime.now().isoformat(timespec="seconds")
    day = run_date[:10]
    df.to_csv(res / "t22_seeds.csv", index=False, float_format="%.10g", lineterminator="\n")
    md = seeds_md_lines(df)
    md += ["",
           f"Informative, NON-canonical: the winning configuration of each model refit with model seeds {seeds} (SMOTE fixed at "
           f"seed {SEED}; networks: early stop + refit per seed), evaluated on the 192 test days; mean / min / max / p2_5 / p97_5 "
           "are descriptive statistics of the 5 values. The reported figure is the seed-42 run (== --mode test, identity checked). "
           f"Script: revision/scripts/hyperparameter_search.py --mode seeds, run {day}."]
    write_text(res / "t22_seeds_table.md", "\n".join(md) + "\n")
    meta = {"run_date": run_date, "elapsed_seconds": round(time.time() - t0, 1), "fit_seconds": timing, "task": "T2.2",
            "mode": "seeds", "seeds": seeds, "smote_seed": SEED, "versions": vers, "identity_seed42_vs_test": identity,
            "winners": em.jsonable(winners), "summary": em.jsonable(em.seeds_summary(df, MODELS).to_dict(orient="records")),
            "files_sha256": {f: sha256(res / f) for f in ("t22_seeds.csv", "t22_seeds_table.md")}, "sanity_checks_failed": fails}
    em.write_json(res / "t22_run_meta_seeds.json", em.jsonable(meta))
    print("\n" + "\n".join(seeds_md_lines(df)))
    print(f"\nelapsed {time.time() - t0:.1f}s; outputs in {res}")
    if fails:
        print(f"{len(fails)} check(s) FAILED: {fails}")
        sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="T2.2 reportable hyperparameter search (temporal CV on train, test once)")
    ap.add_argument("--mode", choices=["cv", "test", "seeds"], default="cv")
    ap.add_argument("--reduced", action="store_true", help="declared sacrifice order: n_iter 10, network grid indices 0, 2, 4")
    ap.add_argument("--results", default=RES, type=Path)
    a = ap.parse_args()
    {"cv": lambda: run_cv(a.results, a.reduced), "test": lambda: run_test(a.results), "seeds": lambda: run_seeds(a.results)}[a.mode]()
