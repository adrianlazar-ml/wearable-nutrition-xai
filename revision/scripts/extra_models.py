"""
extra_models.py - T2.1 (Reviewer 1, R1.2): LightGBM, CatBoost, TabNet and FT-Transformer with
default parameters in the canonical pipeline, each evaluated ONCE on the 192 test days (seed 42).

Pipeline (identical for every model, one function fit_eval_model(name, seed)):
  958 labelled days (canonical thresholds 60 / 85 / 75), temporal cut 766 / 192,
  common.preprocess_split: SimpleImputer(median) fit on train, SMOTE (k from
  threshold_sensitivity.smote_k, = 3; random_state 42 ALWAYS, also in --mode seeds) on train,
  StandardScaler fit on the imputed train (pre-SMOTE) and a second SMOTE on the scaled train.
  Trees (RF, LightGBM, CatBoost) receive the UNSCALED matrix (Split.X_train_sm / X_test), as RF
  and XGB do canonically; the networks (TabNet, FT-Transformer) receive the SCALED matrix
  (Split.X_train_scaled_sm / X_test_scaled), as the canonical SVM does (FND-105).

Models, package defaults plus ONLY seed / threads / verbosity (effective parameters are read
back from the fitted objects into t21_run_meta.json):
  lgbm      LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1)
  catboost  CatBoostClassifier(random_seed=seed, thread_count=1, verbose=0, allow_writing_files=False)
  tabnet    TabNetClassifier(seed=seed, device_name="cpu", verbose=0)  (n_d 8, n_a 8, n_steps 3,
            gamma 1.3, lambda_sparse 1e-3, Adam lr 2e-2, batch 1024, virtual batch 128, drop_last
            True, weights 0 = no class balancing, cross-entropy)
  ftt       rtdl_revisiting_models.FTTransformer(n_cont_features 19, no categoricals, d_out 3,
            **FTTransformer.get_default_kwargs()) (3 blocks, d_block 192, 8 heads, ReGLU),
            FTTransformer.make_default_optimizer() (AdamW lr 1e-4, weight decay 1e-5, no decay on
            embeddings / normalization / biases), CrossEntropyLoss, batch 64.

Network protocol "early stop, then refit" (the trees do not have steps a-b: asymmetry documented):
  (a) inner temporal split of the 766 train days: rows 0-611 inner-train (612), rows 612-765
      validation (154, real rows, no SMOTE); imputer / SMOTE / scaler fit on the inner-train only
      (common.preprocess_split again); asserted 25 Active in inner-train, 13 in validation.
  (b) up to MAX_EPOCHS = 200 epochs, early stopping on macro F1 of the validation rows with
      patience 16 (same rule for both networks: TabNet through eval_set + a custom Metric,
      FT-Transformer through its own loop and EarlyStopper, which replicates
      pytorch_tabnet.callbacks.EarlyStopping with tol 0); best_epoch (1-based) and the
      validation macro F1 at best_epoch are recorded.
  (c) REFIT from scratch on the full 766-day train (canonical SMOTE k = 3) for exactly
      best_epoch epochs, no validation; this refit model is the one evaluated on test.
Determinism: before every model construction random.seed / np.random.seed /
torch.manual_seed(seed), torch.use_deterministic_algorithms(True), torch.set_num_threads(1);
the FT-Transformer DataLoader uses torch.Generator().manual_seed(seed) and shuffle=True on
train only; TabNet's DataLoader (inside the package) shuffles with the global torch RNG, which
torch.manual_seed seeds. CPU only.

Identity T2.1.0 (abort on FAIL): fit_eval_model("rf", 42) through the SAME function and data flow
reproduces results/test_predictions_rf.csv (y_pred element-wise, probabilities at 1e-9) and
0.5063 / 0.6406 / 0.1416.

--mode main (default): T2.1.0, the 4 new models once with seed 42, predictions in
  results/t21_predictions_{lgbm,catboost,tabnet,ftt}.csv (schema of test_predictions_*.csv,
  y_pred = argmax(proba)); metrics with 95 percent bootstrap CIs on the shared matrix
  results/bootstrap_indices_1000x192_seed42.npy (loaded and verified byte-identical with
  default_rng(42), not regenerated) for the 4 new models PLUS RF, XGB, SVM, persistence,
  majority read from test_predictions_*.csv (NOT retrained), same functions as
  extended_metrics.py (t21_extra_models.csv / _table.md); comparisons (t21_comparisons.csv /
  _table.md): paired bootstrap RF minus model on macro F1 and balanced accuracy, paired
  bootstrap model minus persistence on macro F1 (same indices, CI and prop_gt_0 descriptive),
  McNemar model vs RF (cc and exact, n01 + n10 < 25 rule) with Holm over the family of 4
  comparisons with RF; t21_run_meta.json.
--mode seeds: the 4 new models x seeds {42, 0, 1, 2, 3} (SMOTE stays at seed 42; only the model
  seed varies; for the networks early stop + refit per seed); results/t21_seeds.csv, _table.md
  (mean, min, max, [p2.5, p97.5] descriptive over the 5 values), t21_run_meta_seeds.json. The
  seed-42 run must be identical to the main-mode prediction files (abort on FAIL). Informative,
  NON-canonical: the reported figure is seed 42, fixed before running.

Internal sanity (abort on FAIL): RF marginal CI on macro F1 == T1.1 row of CIFRE_REVIZIE.md
(read at run time); point differences == differences of the csv figures; 0 skipped resamples
for the 4 new models; argmax == y_pred; predicted Active count == confusion matrix.

Warnings: run under `python -W error`. The only warnings excluded, BY NAME, are the two
informational UserWarnings of pytorch-tabnet's training callbacks (see
revision/vendor/pytorch_tabnet/PATCH.md). pytorch_tabnet is imported from revision/vendor.

Run from repo root:
    python -W error revision/scripts/extra_models.py --mode main
    python -W error revision/scripts/extra_models.py --mode seeds
Exits 1 on any failed check. Nothing canonical is modified; all figures are additive.
"""

from __future__ import annotations

import argparse
import importlib.metadata as im
import json
import platform
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "revision" / "vendor"
sys.path.insert(0, str(VENDOR))                                  # vendored pytorch_tabnet (PATCH.md)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import catboost  # noqa: E402
import imblearn  # noqa: E402
import lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytorch_tabnet  # noqa: E402
import rtdl_revisiting_models  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
import statsmodels  # noqa: E402
import torch  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from pytorch_tabnet.metrics import Metric as TabNetMetric  # noqa: E402
from pytorch_tabnet.tab_model import TabNetClassifier  # noqa: E402
from rtdl_revisiting_models import FTTransformer  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from common import FEATURES, SEED, TRAIN_RATIO, Split, build_dataset, load_raw, preprocess_split  # noqa: E402
from extended_metrics import PROBA_COLS, bootstrap, label_metrics, proba_metrics  # noqa: E402
from extended_metrics import fmt as fmt_ci  # noqa: E402
from extended_metrics import load_inputs as load_saved_systems  # noqa: E402
from group_ablation import f1_class, f1_macro, read_t11_rf_f1_ci, save_predictions, sha256  # noqa: E402
from group_ablation import metrics as point_metrics  # noqa: E402
from save_test_predictions import build_models  # noqa: E402
from statistical_tests import (DISCORDANT_MIN_FOR_CC, INDEX_FILE, LABELS, md_table, mcnemar_counts,  # noqa: E402
                               mcnemar_tests, paired_bootstrap, write_text)
from statistical_tests import METRICS as PB_METRICS  # noqa: E402
from threshold_sensitivity import label_counts, smote_k  # noqa: E402

RES = ROOT / "revision" / "results"

N_TOTAL, N_TRAIN, N_TEST, N_BOOT = 958, 766, 192, 1000
INNER_TRAIN, INNER_VAL = 612, 154
EXPECT_INNER_ACTIVE = (25, 13)               # T1.3 fold 4 convention: 25 Active in inner-train, 13 in validation
CANONICAL_THRESHOLDS = dict(rest_cutoff=60, active_readiness=85, active_sleep=75)
MAX_EPOCHS, PATIENCE, FTT_BATCH = 200, 16, 64
N_CLASSES = 3
NEW_MODELS = ["lgbm", "catboost", "tabnet", "ftt"]
NETWORKS = ("tabnet", "ftt")
SCALED = {"rf": False, "lgbm": False, "catboost": False, "tabnet": True, "ftt": True}
SYSTEMS = ["rf", "xgb", "svm"] + NEW_MODELS + ["persistence", "majority"]
LABEL = {"rf": "RF", "xgb": "XGB", "svm": "SVM", "lgbm": "LightGBM", "catboost": "CatBoost", "tabnet": "TabNet",
         "ftt": "FT-Transformer", "persistence": "Persistence", "majority": "Majority"}
SEEDS = [42, 0, 1, 2, 3]
CANON_RF = dict(f1_macro=0.5063, accuracy=0.6406, mcc=0.1416)
T11_RF_BAL_ACC_CI = (0.3851, 0.6634)
# pytorch-tabnet informational warnings excluded BY NAME (revision/vendor/pytorch_tabnet/PATCH.md)
TABNET_WARNINGS = ["Best weights from best epoch are automatically used!",
                   "No early stopping will be performed, last training weights will be used."]
METRIC_ORDER = ["f1_macro", "f1_rest", "f1_normal", "f1_active", "accuracy", "balanced_accuracy", "mcc",
                "cohen_kappa", "g_mean", "roc_auc_ovr_macro", "pr_auc_macro"]
METRIC_TITLE = {"f1_macro": "Macro F1", "f1_rest": "F1 Rest", "f1_normal": "F1 Normal", "f1_active": "F1 Active",
                "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "mcc": "MCC",
                "cohen_kappa": "Cohen's kappa", "g_mean": "G-mean", "roc_auc_ovr_macro": "ROC-AUC (OvR, macro)",
                "pr_auc_macro": "PR-AUC (macro)"}
SEED_METRICS = ["f1_macro", "f1_active", "accuracy", "mcc"]


# ------------------------------------------------------------------ pure helpers (unit-tested)
def seed_everything(seed: int) -> None:
    """Determinism block run before every model construction (CPU, one thread)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


class EarlyStopper:
    """Early stopping on a maximised score, replicating pytorch_tabnet.callbacks.EarlyStopping
    (tol 0): an epoch improves when score > best (strict); after an improvement wait = 1; on a
    non-improving epoch training stops when wait >= patience, else wait += 1. So training stops
    on the `patience`-th consecutive non-improving epoch. Epochs are 1-based here."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = -np.inf
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0

    def update(self, epoch: int, score: float) -> bool:
        """Record the score of `epoch`; return True when training must stop."""
        if score > self.best:
            self.best, self.best_epoch, self.wait = float(score), int(epoch), 1
            return False
        if self.wait >= self.patience:
            self.stopped_epoch = int(epoch)
            return True
        self.wait += 1
        return False


class F1MacroMetric(TabNetMetric):
    """Custom TabNet eval metric: macro F1 with labels [0, 1, 2], zero_division 0 (revision convention)."""

    def __init__(self) -> None:
        self._name = "f1_macro"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return f1_macro(np.asarray(y_true, dtype=int), np.asarray(y_score).argmax(1))


def jsonable(obj):
    """Recursively convert an object of effective parameters to JSON-serialisable values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


def inner_bounds(n_train: int = N_TRAIN, n_val: int = INNER_VAL) -> tuple[int, int]:
    """(n_inner_train, n_val): the last n_val train rows are the validation block."""
    if n_val <= 0 or n_val >= n_train:
        raise ValueError(f"validation block {n_val} must be in (0, {n_train})")
    return n_train - n_val, n_val


# ------------------------------------------------------------------ data
def load_data(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """958 labelled days with the canonical thresholds, 19 lag-1 features, sorted by date."""
    if df is None:
        df = load_raw()
    X, y, dates = build_dataset(df, **CANONICAL_THRESHOLDS)
    if len(X) != N_TOTAL or list(X.columns) != FEATURES:
        raise ValueError(f"dataset {X.shape} differs from the canonical 958 x 19")
    if int(len(X) * TRAIN_RATIO) != N_TRAIN:
        raise ValueError("canonical cut != 766")
    if label_counts(y) != (202, 710, 46) or label_counts(y.iloc[:N_TRAIN]) != (158, 570, 38):
        raise ValueError(f"label distribution differs from canonical: {label_counts(y)} / {label_counts(y.iloc[:N_TRAIN])}")
    return X, y, dates


def canonical_preprocess(X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> tuple[Split, int]:
    """common.preprocess_split on the canonical cut 766 / 192 with the Table 7 SMOTE k convention."""
    k = smote_k(min(label_counts(y.iloc[:N_TRAIN])))
    if k is None:
        raise ValueError("SMOTE not possible on the canonical train")
    s = preprocess_split(X.iloc[:N_TRAIN], X.iloc[N_TRAIN:], y.iloc[:N_TRAIN], y.iloc[N_TRAIN:], dates.iloc[N_TRAIN:],
                         k_neighbors=k, seed=SEED)
    return s, k


def inner_preprocess(X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> tuple[Split, int]:
    """Inner temporal split of the 766 train days (612 inner-train / 154 validation, real rows);
    imputer / SMOTE / scaler fit on the inner-train only through the same common.preprocess_split.
    Asserts the Active counts (25 / 13) and that the validation block has no synthetic rows."""
    n_in, n_val = inner_bounds()
    y_in, y_val = y.iloc[:n_in], y.iloc[n_in:N_TRAIN]
    k = smote_k(min(label_counts(y_in)))
    if k is None:
        raise ValueError("SMOTE not possible on the inner train")
    s = preprocess_split(X.iloc[:n_in], X.iloc[n_in:N_TRAIN], y_in, y_val, dates.iloc[n_in:N_TRAIN],
                         k_neighbors=k, seed=SEED)
    got = (label_counts(y_in)[2], label_counts(y_val)[2])
    if got != EXPECT_INNER_ACTIVE:
        raise AssertionError(f"inner split Active counts {got} != {EXPECT_INNER_ACTIVE}")
    if len(s.y_test) != n_val or len(s.X_test) != n_val or len(s.X_test_scaled) != n_val:
        raise AssertionError("validation block changed size (SMOTE must not touch it)")
    if len(s.y_train) != n_in:
        raise AssertionError("inner-train size")
    return s, k


# ------------------------------------------------------------------ model wrappers
def make_tree(name: str, seed: int, **overrides):
    """Package defaults plus seed / threads / verbosity only (overrides exist for unit tests)."""
    if name == "rf":
        return build_models()["rf"]
    if name == "lgbm":
        return LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1, **overrides)
    if name == "catboost":
        return CatBoostClassifier(random_seed=seed, thread_count=1, verbose=0, allow_writing_files=False, **overrides)
    raise ValueError(name)


def tabnet_fit(X: np.ndarray, y: np.ndarray, seed: int, max_epochs: int, patience: int,
               X_val: np.ndarray | None = None, y_val: np.ndarray | None = None, **overrides) -> TabNetClassifier:
    """TabNetClassifier(seed, cpu, verbose 0) with the package fit defaults (batch 1024, virtual batch
    128, drop_last True, weights 0); with a validation set: early stopping on macro F1 (custom
    Metric, patience); without: exactly max_epochs epochs, last weights (the refit)."""
    seed_everything(seed)
    m = TabNetClassifier(seed=seed, device_name="cpu", verbose=0, **overrides)
    kw = dict(max_epochs=max_epochs, patience=patience)
    if X_val is not None:
        kw.update(eval_set=[(X_val.astype(np.float32), np.asarray(y_val, dtype=int))], eval_name=["val"],
                  eval_metric=[F1MacroMetric])
    with warnings.catch_warnings():
        for msg in TABNET_WARNINGS:                      # excluded BY NAME, nothing else
            warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        m.fit(X.astype(np.float32), np.asarray(y, dtype=int), **kw)
    return m


def tabnet_epochs_run(m: TabNetClassifier) -> int:
    return len(m.history["loss"])


def ftt_build(n_features: int, seed: int) -> tuple[FTTransformer, torch.optim.AdamW]:
    """Default FT-Transformer of the package and its default optimizer (seeded construction)."""
    seed_everything(seed)
    model = FTTransformer(n_cont_features=n_features, cat_cardinalities=[], d_out=N_CLASSES,
                          **FTTransformer.get_default_kwargs())
    return model, model.make_default_optimizer()


def ftt_proba(model: FTTransformer, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(np.asarray(X), dtype=torch.float32), None)
    return torch.softmax(out, dim=1).numpy().astype(np.float64)


def ftt_train(model: FTTransformer, opt: torch.optim.Optimizer, X: np.ndarray, y: np.ndarray, seed: int,
              max_epochs: int, batch: int = FTT_BATCH, X_val: np.ndarray | None = None,
              y_val: np.ndarray | None = None, patience: int | None = None) -> dict:
    """Own loop: CrossEntropyLoss, DataLoader(shuffle=True, generator seeded, train only); with a
    validation set: macro F1 after every epoch and EarlyStopper(patience); without: exactly
    max_epochs epochs. Returns epochs_run, best_epoch, best_val_f1, val_history."""
    loss_fn = torch.nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.tensor(np.asarray(X), dtype=torch.float32),
                                        torch.tensor(np.asarray(y, dtype=int), dtype=torch.long))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))
    stopper = EarlyStopper(patience) if X_val is not None else None
    history: list[float] = []
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb, None), yb).backward()
            opt.step()
        epochs_run = epoch
        if stopper is not None:
            val = f1_macro(np.asarray(y_val, dtype=int), ftt_proba(model, X_val).argmax(1))
            history.append(val)
            if stopper.update(epoch, val):
                break
    return dict(epochs_run=epochs_run, best_epoch=stopper.best_epoch if stopper else epochs_run,
                best_val_f1=stopper.best if stopper else None, stopped_epoch=stopper.stopped_epoch if stopper else 0,
                val_history=history, n_batches_per_epoch=len(loader))


# ------------------------------------------------------------------ the single fit / eval function
def fit_eval_model(name: str, seed: int, data: tuple[pd.DataFrame, pd.Series, pd.Series]) -> dict:
    """One protocol for every model: canonical data flow (common.preprocess_split), package
    defaults + seed / threads / verbosity, test set touched once at the end. SMOTE seed is
    always SEED (42); `seed` applies to the model only. Returns predictions, probabilities,
    effective parameters and (networks) best_epoch / validation F1."""
    X, y, dates = data
    split, k = canonical_preprocess(X, y, dates)
    y_true = split.y_test.to_numpy(dtype=int)
    info: dict = dict(model=name, seed=seed, smote_seed=SEED, smote_k=k, scaled_input=SCALED[name],
                      n_train_rows_after_smote=int(len(split.y_train_sm)), n_test=int(len(y_true)))
    X_tr = split.X_train_scaled_sm if SCALED[name] else split.X_train_sm
    y_tr = split.y_train_scaled_sm if SCALED[name] else split.y_train_sm
    X_te = split.X_test_scaled if SCALED[name] else split.X_test
    t0 = time.time()
    if name in ("rf", "lgbm", "catboost"):
        seed_everything(seed)
        m = make_tree(name, seed)
        m.fit(X_tr, y_tr)                              # DataFrames with the 19 feature names
        proba = np.asarray(m.predict_proba(X_te), dtype=np.float64)
        pred_api = np.asarray(m.predict(X_te)).reshape(-1).astype(int)
        classes = [int(c) for c in m.classes_]
        params = m.get_all_params() if name == "catboost" else m.get_params()
        if name == "catboost":                         # get_all_params() = effective values (iterations, learning_rate, depth, ...);
            params = dict(params, tree_count=int(m.tree_count_), thread_count=m.get_params()["thread_count"])
    elif name in NETWORKS:
        inner, k_in = inner_preprocess(X, y, dates)
        X_in, y_in = inner.X_train_scaled_sm.to_numpy(), inner.y_train_scaled_sm.to_numpy(dtype=int)
        X_val, y_val = inner.X_test_scaled.to_numpy(), inner.y_test.to_numpy(dtype=int)
        info.update(inner_train_rows=int(len(inner.y_train)), inner_val_rows=int(len(y_val)), inner_smote_k=k_in,
                    inner_train_rows_after_smote=int(len(y_in)),
                    inner_active=[int((inner.y_train == 2).sum()), int((y_val == 2).sum())],
                    max_epochs=MAX_EPOCHS, patience=PATIENCE)
        if name == "tabnet":
            m_a = tabnet_fit(X_in, y_in, seed, MAX_EPOCHS, PATIENCE, X_val, y_val)
            best_epoch, val_f1 = int(m_a.best_epoch) + 1, float(m_a.best_cost)     # package best_epoch is 0-based
            info.update(stage_a_epochs_run=tabnet_epochs_run(m_a), val_history=[float(v) for v in m_a.history["val_f1_macro"]])
            m = tabnet_fit(X_tr.to_numpy(), y_tr.to_numpy(dtype=int), seed, best_epoch, 0)
            if tabnet_epochs_run(m) != best_epoch:
                raise AssertionError(f"refit epochs {tabnet_epochs_run(m)} != best_epoch {best_epoch}")
            Xte = X_te.to_numpy().astype(np.float32)
            proba = np.asarray(m.predict_proba(Xte), dtype=np.float64)
            pred_api = np.asarray(m.predict(Xte)).reshape(-1).astype(int)
            classes = [int(c) for c in m.classes_]
            params = dict(m.get_params(), batch_size=m.batch_size, virtual_batch_size=m.virtual_batch_size,
                          drop_last=m.drop_last, weights=0, loss_fn=getattr(m.loss_fn, "__name__", repr(m.loss_fn)),
                          refit_max_epochs=m.max_epochs,
                          n_train_batches_per_epoch=int(len(y_tr) // m.batch_size) if m.drop_last else -(-len(y_tr) // m.batch_size))
        else:
            model_a, opt_a = ftt_build(X_in.shape[1], seed)
            res_a = ftt_train(model_a, opt_a, X_in, y_in, seed, MAX_EPOCHS, FTT_BATCH, X_val, y_val, PATIENCE)
            best_epoch, val_f1 = int(res_a["best_epoch"]), float(res_a["best_val_f1"])
            info.update(stage_a_epochs_run=res_a["epochs_run"], val_history=res_a["val_history"])
            m, opt = ftt_build(X_tr.shape[1], seed)
            res_b = ftt_train(m, opt, X_tr.to_numpy(), y_tr.to_numpy(dtype=int), seed, best_epoch, FTT_BATCH)
            if res_b["epochs_run"] != best_epoch:
                raise AssertionError(f"refit epochs {res_b['epochs_run']} != best_epoch {best_epoch}")
            proba = ftt_proba(m, X_te.to_numpy())
            pred_api = proba.argmax(1)
            classes = LABELS
            groups = [{"n_params": sum(int(p.numel()) for p in g["params"]), "lr": g["lr"], "weight_decay": g["weight_decay"]}
                      for g in opt.param_groups]
            params = dict(FTTransformer.get_default_kwargs(), n_cont_features=X_tr.shape[1], cat_cardinalities=[],
                          d_out=N_CLASSES, optimizer=type(opt).__name__, optimizer_param_groups=groups,
                          n_parameters=sum(int(p.numel()) for p in m.parameters()), loss="CrossEntropyLoss",
                          batch_size=FTT_BATCH, n_train_batches_per_epoch=res_b["n_batches_per_epoch"])
        info.update(best_epoch=best_epoch, val_f1_macro_at_best=val_f1)
    else:
        raise ValueError(name)
    fit_seconds = time.time() - t0
    if classes != LABELS:
        raise AssertionError(f"{name}: classes_ {classes} != {LABELS}")
    if proba.shape != (N_TEST, N_CLASSES):
        raise AssertionError(f"{name}: proba shape {proba.shape}")
    y_pred = proba.argmax(1).astype(int)
    if not np.array_equal(y_pred, pred_api):
        raise AssertionError(f"{name}: argmax(proba) != predict() on {int((y_pred != pred_api).sum())} days")
    max_dev = float(np.abs(proba.sum(1) - 1).max())
    if max_dev > 1e-6:
        raise AssertionError(f"{name}: probabilities do not sum to 1 (max dev {max_dev:.2e})")
    info.update(y_true=y_true, y_pred=y_pred, proba=proba, dates_test=split.dates_test, fit_seconds=round(fit_seconds, 1),
                params=jsonable(params), torch_threads=torch.get_num_threads(), max_proba_sum_dev=max_dev)
    return info


# ------------------------------------------------------------------ metrics and tables
def metric_functions() -> tuple[dict, dict]:
    """Label metrics (extended_metrics + per-class F1 of group_ablation) and probability metrics."""
    lm = label_metrics()
    label = {"f1_macro": lm["f1_macro"], "f1_rest": lambda y, p: f1_class(y, p, 0),
             "f1_normal": lambda y, p: f1_class(y, p, 1), "f1_active": lambda y, p: f1_class(y, p, 2),
             "accuracy": lm["accuracy"], "balanced_accuracy": lm["balanced_accuracy"], "mcc": lm["mcc"],
             "cohen_kappa": lm["cohen_kappa"], "g_mean": lm["g_mean"]}
    return label, proba_metrics()


def metrics_with_ci(systems: dict[str, tuple[np.ndarray, np.ndarray | None]], y_true: np.ndarray,
                    idx: np.ndarray) -> pd.DataFrame:
    """One row per (system, metric): point value, percentile CI on the shared matrix, skips, warnings."""
    lm, pm = metric_functions()
    rows = []
    for sysname, (yp, pr) in systems.items():
        for name, fn in lm.items():
            v = fn(y_true, yp)
            lo, hi, used, skip, nwarn = bootstrap(lambda r, fn=fn, yp=yp: fn(y_true[r], yp[r]), idx)
            rows.append(dict(system=sysname, metric=name, value=v, ci_low=lo, ci_high=hi, n_resamples_used=used,
                             n_skipped=skip, n_warnings=nwarn))
        for name, fn in pm.items():
            if pr is None:
                rows.append(dict(system=sysname, metric=name, value=np.nan, ci_low=np.nan, ci_high=np.nan,
                                 n_resamples_used=0, n_skipped=0, n_warnings=0))
                continue
            v = fn(y_true, pr)
            lo, hi, used, skip, nwarn = bootstrap(lambda r, fn=fn, pr=pr: fn(y_true[r], pr[r]), idx)
            rows.append(dict(system=sysname, metric=name, value=v, ci_low=lo, ci_high=hi, n_resamples_used=used,
                             n_skipped=skip, n_warnings=nwarn))
    return pd.DataFrame(rows)


def metrics_md_lines(table: pd.DataFrame) -> list[str]:
    """Wide markdown table (systems on rows, metrics on columns) from the long csv; the checker
    rebuilds these lines from the csv and compares them verbatim."""
    lines = ["| System | " + " | ".join(METRIC_TITLE[m] for m in METRIC_ORDER) + " |", "|---|" + "---|" * len(METRIC_ORDER)]
    for sysname in [s for s in SYSTEMS if s in set(table.system)]:
        cells = []
        for m in METRIC_ORDER:
            r = table[(table.system == sysname) & (table.metric == m)].iloc[0]
            cells.append("n/a" if pd.isna(r.value) else fmt_ci(float(r.value), float(r.ci_low), float(r.ci_high)))
        lines.append(f"| {LABEL[sysname]} | " + " | ".join(cells) + " |")
    return lines


COMP_FMT_COLS = ["f1_macro_model", "f1_macro_rf", "delta_rf_minus_model_f1_macro", "delta_rf_minus_model_f1_macro_ci_low",
                 "delta_rf_minus_model_f1_macro_ci_high", "delta_rf_minus_model_f1_macro_prop_gt_0",
                 "balanced_accuracy_model", "balanced_accuracy_rf", "delta_rf_minus_model_balanced_accuracy",
                 "delta_rf_minus_model_balanced_accuracy_ci_low", "delta_rf_minus_model_balanced_accuracy_ci_high",
                 "delta_rf_minus_model_balanced_accuracy_prop_gt_0", "f1_macro_persistence",
                 "delta_model_minus_persistence_f1_macro", "delta_model_minus_persistence_f1_macro_ci_low",
                 "delta_model_minus_persistence_f1_macro_ci_high", "delta_model_minus_persistence_f1_macro_prop_gt_0",
                 "chi2_cc", "p_cc", "p_exact", "p_cc_holm", "p_exact_holm"]


def comparisons_md_lines(comp: pd.DataFrame) -> list[str]:
    return md_table(comp, {c: "%.4f" for c in COMP_FMT_COLS})


def seeds_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per model x metric over the 5 seeds: seed-42 value, mean, min, max, p2.5, p97.5 (descriptive);
    plus the best_epoch range of the networks."""
    rows = []
    for name in [m for m in NEW_MODELS if m in set(df.model)]:
        sub = df[df.model == name]
        for metric in SEED_METRICS:
            v = sub[metric].to_numpy(dtype=float)
            rows.append(dict(model=name, metric=metric, n_seeds=int(len(v)),
                             seed42=float(sub.loc[sub.seed == SEED, metric].iloc[0]), mean=float(v.mean()),
                             min=float(v.min()), max=float(v.max()), p2_5=float(np.percentile(v, 2.5)),
                             p97_5=float(np.percentile(v, 97.5))))
        if name in NETWORKS:
            be = sub["best_epoch"].to_numpy(dtype=float)
            rows.append(dict(model=name, metric="best_epoch", n_seeds=int(len(be)),
                             seed42=float(sub.loc[sub.seed == SEED, "best_epoch"].iloc[0]), mean=float(be.mean()),
                             min=float(be.min()), max=float(be.max()), p2_5=float(np.percentile(be, 2.5)),
                             p97_5=float(np.percentile(be, 97.5))))
    return pd.DataFrame(rows)


def seeds_md_lines(summary: pd.DataFrame) -> list[str]:
    return md_table(summary, {c: "%.4f" for c in ("seed42", "mean", "min", "max", "p2_5", "p97_5")})


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "imbalanced_learn": imblearn.__version__, "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__, "torch": torch.__version__, "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__, "pytorch_tabnet": im.version("pytorch-tabnet"),
            "rtdl_revisiting_models": im.version("rtdl_revisiting_models"),
            "pytorch_tabnet_path": str(Path(list(pytorch_tabnet.__path__)[0]).resolve()),
            "rtdl_path": str(Path(rtdl_revisiting_models.__file__).resolve()),
            "torch_cuda_available": bool(torch.cuda.is_available())}


def load_index(res: Path) -> np.ndarray:
    idx = np.load(res / INDEX_FILE)
    regen = np.random.default_rng(SEED).integers(0, N_TEST, size=(N_BOOT, N_TEST))
    if idx.shape != (N_BOOT, N_TEST) or idx.dtype != regen.dtype or not np.array_equal(idx, regen):
        sys.exit(f"ABORT: {INDEX_FILE} is not byte-identical to default_rng({SEED})")
    return idx


def write_json(path: Path, meta: dict) -> None:
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")


# ------------------------------------------------------------------ modes
def run_main(res: Path = RES) -> None:
    t0 = time.time()
    fails: list[str] = []

    def check(cond: bool, msg: str, abort: bool = False) -> None:
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)
            if abort:
                sys.exit(f"ABORT: {msg}")

    vers = versions()
    check(VENDOR.resolve() in Path(vers["pytorch_tabnet_path"]).parents, f"pytorch_tabnet imported from revision/vendor", abort=True)
    check(vers["torch_cuda_available"] is False, "torch.cuda.is_available() == False", abort=True)
    data = load_data()
    idx = load_index(res)
    print(f"PASS {INDEX_FILE} loaded, byte-identical to default_rng({SEED})")
    t11_ci = read_t11_rf_f1_ci()
    y_saved, preds, probas, max_dev_saved = load_saved_systems(res)        # rf, xgb, svm, persistence, majority
    print(f"PASS saved systems loaded (rf, xgb, svm, persistence, majority), max |sum(proba) - 1| = {max_dev_saved:.2e}")

    # ---- T2.1.0 identity
    print("\n=== T2.1.0 identity: fit_eval_model('rf', 42) ===")
    rf = fit_eval_model("rf", SEED, data)
    ref = pd.read_csv(res / "test_predictions_rf.csv")
    check(np.array_equal(ref["y_true"].to_numpy(), rf["y_true"]) and np.array_equal(rf["y_true"], y_saved),
          "y_true == test_predictions_rf.csv", abort=True)
    check(np.array_equal(ref["y_pred"].to_numpy(), rf["y_pred"]), "y_pred element-wise == test_predictions_rf.csv", abort=True)
    max_dp = float(np.abs(ref[PROBA_COLS].to_numpy() - rf["proba"]).max())
    check(max_dp < 1e-9, f"probabilities == csv (max abs diff {max_dp:.2e} < 1e-9)", abort=True)
    check(list(pd.to_datetime(ref["date"]).dt.strftime("%Y-%m-%d")) == list(rf["dates_test"].dt.strftime("%Y-%m-%d")),
          "test dates == csv", abort=True)
    m_rf = point_metrics(rf["y_true"], rf["y_pred"])
    for k, v in CANON_RF.items():
        check(round(m_rf[k], 4) == v, f"RF {k} {m_rf[k]:.6f} == canonical {v}", abort=True)
    y_true = rf["y_true"]

    # ---- the four new models, once, seed 42
    print("\n=== T2.1.1 new models, default parameters, seed 42 ===")
    results: dict[str, dict] = {}
    pred_files: dict[str, str] = {}
    for name in NEW_MODELS:
        r = fit_eval_model(name, SEED, data)
        results[name] = r
        f = res / f"t21_predictions_{name}.csv"
        save_predictions(f, r["dates_test"], r["y_true"], r["y_pred"], r["proba"])
        pred_files[f.name] = sha256(f)
        m = point_metrics(y_true, r["y_pred"])
        r["point"] = m
        check(m["cm_ra"] + m["cm_na"] + m["cm_aa"] == int((r["y_pred"] == 2).sum()),
              f"{name}: predicted Active count == confusion matrix column ({int((r['y_pred'] == 2).sum())})", abort=True)
        extra = (f"  best_epoch {r['best_epoch']} (val F1 {r['val_f1_macro_at_best']:.4f}, stage A ran {r['stage_a_epochs_run']} epochs)"
                 if name in NETWORKS else "")
        print(f"  {LABEL[name]:<15} F1 macro {m['f1_macro']:.4f}  F1 Active {m['f1_active']:.4f}  acc {m['accuracy']:.4f}  "
              f"MCC {m['mcc']:.4f}  bal acc {m['balanced_accuracy']:.4f}  fit {r['fit_seconds']} s{extra}")

    # ---- metrics with CI for the 9 systems
    print("\n=== T2.1.2 metrics with 95 percent CI (shared matrix) ===")
    systems: dict[str, tuple[np.ndarray, np.ndarray | None]] = {
        "rf": (preds["rf"], probas["rf"]), "xgb": (preds["xgb"], probas["xgb"]), "svm": (preds["svm"], probas["svm"])}
    for name in NEW_MODELS:
        systems[name] = (results[name]["y_pred"], results[name]["proba"])
    systems["persistence"] = (preds["persistence"], None)
    systems["majority"] = (preds["majority"], None)
    check(np.array_equal(preds["rf"], rf["y_pred"]), "saved RF y_pred == T2.1.0 y_pred", abort=True)
    table = metrics_with_ci(systems, y_true, idx)
    g = {(r.system, r.metric): r for r in table.itertuples()}
    got = (round(g["rf", "f1_macro"].ci_low, 4), round(g["rf", "f1_macro"].ci_high, 4))
    check(got == t11_ci, f"sanity: RF macro F1 marginal CI {got} == T1.1 CIFRE_REVIZIE.md {t11_ci}", abort=True)
    check(round(g["rf", "f1_macro"].value, 4) == CANON_RF["f1_macro"] and round(g["xgb", "f1_macro"].value, 4) == 0.4760
          and round(g["svm", "f1_macro"].value, 4) == 0.4347 and round(g["persistence", "f1_macro"].value, 4) == 0.3039
          and round(g["majority", "f1_macro"].value, 4) == 0.2811, "sanity: saved systems reproduce the canonical macro F1 points", abort=True)
    new_skipped = int(table[table.system.isin(NEW_MODELS)].n_skipped.sum())
    check(new_skipped == 0, f"0 resamples skipped on the 4 new models ({new_skipped})", abort=True)
    check(int(table.n_warnings.sum()) == 0, f"0 sklearn warnings inside the bootstrap loops ({int(table.n_warnings.sum())})")
    for name in NEW_MODELS:
        check(abs(g[name, "f1_macro"].value - results[name]["point"]["f1_macro"]) < 1e-12, f"{name}: table macro F1 == point_metrics")
    for name in NEW_MODELS:
        r = g[name, "f1_macro"]
        print(f"  {LABEL[name]:<15} macro F1 {fmt_ci(r.value, r.ci_low, r.ci_high)}  F1 Active "
              f"{fmt_ci(g[name, 'f1_active'].value, g[name, 'f1_active'].ci_low, g[name, 'f1_active'].ci_high)}  "
              f"MCC {fmt_ci(g[name, 'mcc'].value, g[name, 'mcc'].ci_low, g[name, 'mcc'].ci_high)}")

    # ---- comparisons with RF and persistence
    print("\n=== T2.1.3 comparisons (paired bootstrap, McNemar + Holm over the 4 comparisons with RF) ===")
    comp_rows = []
    for name in NEW_MODELS:
        yp = results[name]["y_pred"]
        row: dict = dict(model=name, label=LABEL[name])
        for mname, mfn in PB_METRICS.items():                       # f1_macro, balanced_accuracy
            b = paired_bootstrap(y_true, preds["rf"], yp, mfn, idx)
            exp = g["rf", mname].value - g[name, mname].value
            check(abs(b["diff_point"] - exp) < 1e-12, f"{name} {mname}: diff_point RF minus model == csv difference ({b['diff_point']:+.4f})",
                  abort=True)
            ci_rf = (round(b["ci_A_low"], 4), round(b["ci_A_high"], 4))
            check(ci_rf == (t11_ci if mname == "f1_macro" else T11_RF_BAL_ACC_CI), f"{name} {mname}: RF marginal CI {ci_rf} == T1.1")
            check(b["n_skipped"] == 0, f"{name} {mname}: 0 resamples skipped in the paired bootstrap")
            row.update({f"{mname}_model": g[name, mname].value, f"{mname}_rf": g["rf", mname].value,
                        f"delta_rf_minus_model_{mname}": b["diff_point"], f"delta_rf_minus_model_{mname}_ci_low": b["ci_low"],
                        f"delta_rf_minus_model_{mname}_ci_high": b["ci_high"],
                        f"delta_rf_minus_model_{mname}_include_zero": bool(b["ci_low"] <= 0.0 <= b["ci_high"]),
                        f"delta_rf_minus_model_{mname}_prop_gt_0": b["prop_diff_gt_0"], f"n_skipped_{mname}": b["n_skipped"]})
        b = paired_bootstrap(y_true, yp, preds["persistence"], PB_METRICS["f1_macro"], idx)
        exp = g[name, "f1_macro"].value - g["persistence", "f1_macro"].value
        check(abs(b["diff_point"] - exp) < 1e-12, f"{name}: diff_point model minus persistence == csv difference ({b['diff_point']:+.4f})", abort=True)
        row.update(f1_macro_persistence=g["persistence", "f1_macro"].value, delta_model_minus_persistence_f1_macro=b["diff_point"],
                   delta_model_minus_persistence_f1_macro_ci_low=b["ci_low"], delta_model_minus_persistence_f1_macro_ci_high=b["ci_high"],
                   delta_model_minus_persistence_f1_macro_include_zero=bool(b["ci_low"] <= 0.0 <= b["ci_high"]),
                   delta_model_minus_persistence_f1_macro_prop_gt_0=b["prop_diff_gt_0"], n_skipped_persistence=b["n_skipped"])
        n00, n01, n10, n11 = mcnemar_counts(y_true, yp, preds["rf"])        # A = model, B = RF
        t = mcnemar_tests(n00, n01, n10, n11)
        disc = n01 + n10
        check(n00 + n01 + n10 + n11 == N_TEST, f"{name}: McNemar cells sum to {N_TEST}")
        row.update(n00=n00, n01=n01, n10=n10, n11=n11, chi2_cc=t["chi2_cc"], p_cc=t["p_cc"], p_exact=t["p_exact"],
                   p_cc_holm=np.nan, p_exact_holm=np.nan, recommended="exact" if disc < DISCORDANT_MIN_FOR_CC else "cc",
                   model_gt_rf_f1_macro=bool(g[name, "f1_macro"].value > g["rf", "f1_macro"].value),
                   model_gt_persistence_f1_macro=bool(g[name, "f1_macro"].value > g["persistence", "f1_macro"].value))
        comp_rows.append(row)
    comp = pd.DataFrame(comp_rows)
    comp["p_cc_holm"] = multipletests(comp["p_cc"].to_numpy(), method="holm")[1]
    comp["p_exact_holm"] = multipletests(comp["p_exact"].to_numpy(), method="holm")[1]
    for c in ("p_cc", "p_exact", "p_cc_holm", "p_exact_holm"):
        check(bool(((comp[c] >= 0) & (comp[c] <= 1)).all()), f"McNemar {c} in [0, 1]")
    for r in comp.itertuples():
        print(f"  {r.label:<15} RF minus model F1 {r.delta_rf_minus_model_f1_macro:+.4f} [{r.delta_rf_minus_model_f1_macro_ci_low:+.4f}, "
              f"{r.delta_rf_minus_model_f1_macro_ci_high:+.4f}] (prop > 0 {r.delta_rf_minus_model_f1_macro_prop_gt_0:.3f}); "
              f"model minus persistence {r.delta_model_minus_persistence_f1_macro:+.4f} [{r.delta_model_minus_persistence_f1_macro_ci_low:+.4f}, "
              f"{r.delta_model_minus_persistence_f1_macro_ci_high:+.4f}]; McNemar n01 / n10 {r.n01} / {r.n10}, p_cc {r.p_cc:.4f}, "
              f"p_exact {r.p_exact:.4f}, Holm {r.p_cc_holm:.4f} / {r.p_exact_holm:.4f} ({r.recommended})")
    above = [LABEL[n] for n in NEW_MODELS if g[n, "f1_macro"].value > g["rf", "f1_macro"].value]
    print(f"  models above RF on macro F1 at the point: {above if above else 'none'} (reported; nothing canonical changes)")

    # ---- outputs
    run_date = datetime.now().isoformat(timespec="seconds")
    day = run_date[:10]
    table.to_csv(res / "t21_extra_models.csv", index=False, float_format="%.10g", lineterminator="\n")
    md = metrics_md_lines(table)
    md += ["",
           f"Test set n = {N_TEST} (Rest / Normal / Active = 44 / 140 / 8). Point estimate [bootstrap 95 percent CI], {N_BOOT} "
           f"resamples of the 192 test days on the shared matrix {INDEX_FILE} (loaded, not regenerated), the same functions "
           "as extended_metrics.py (T1.1) plus per-class F1 (labels fixed, zero_division 0). RF / XGB / SVM / persistence / "
           "majority are read from results/test_predictions_*.csv (not retrained). LightGBM, CatBoost, TabNet and "
           "FT-Transformer: package defaults plus seed 42 / one thread / verbosity only, canonical pipeline (766 / 192, "
           "median imputer + SMOTE k = 3 fit on train; trees on the unscaled matrix like RF / XGB, networks on the scaled "
           "matrix like SVM), each evaluated once on test; y_pred = argmax(proba). Networks: early stopping on validation "
           f"macro F1 (inner temporal split 612 / 154 of the train, patience {PATIENCE}, max {MAX_EPOCHS} epochs), then refit "
           "on the full train for best_epoch epochs: "
           + "; ".join(f"{LABEL[n]} best_epoch {results[n]['best_epoch']} (validation macro F1 {results[n]['val_f1_macro_at_best']:.4f})"
                       for n in NETWORKS)
           + ". ROC-AUC and PR-AUC n/a for persistence and majority (hard labels only). Resamples skipped: "
           f"{int(table.n_skipped.sum())} (summed over systems and metrics). All figures of the new models are additive; the "
           f"canonical RF figures (0.5063 / 0.6406 / 0.1416) remain. Script: revision/scripts/extra_models.py --mode main, run {day}."]
    write_text(res / "t21_extra_models_table.md", "\n".join(md) + "\n")
    comp.to_csv(res / "t21_comparisons.csv", index=False, float_format="%.10g", lineterminator="\n")
    cmd = comparisons_md_lines(comp)
    cmd += ["",
            "delta_rf_minus_model = metric(RF) - metric(model) (positive = RF better), delta_model_minus_persistence = "
            "metric(model) - metric(persistence); [ci_low, ci_high] = percentile 2.5 / 97.5 of the paired bootstrap of the "
            f"difference over {N_BOOT} resamples ({INDEX_FILE}, same indices applied to y_true and both prediction vectors); "
            "prop_gt_0 = proportion of resamples with the difference > 0 (descriptive, NOT a p-value); include_zero uses "
            "ci_low <= 0 <= ci_high. McNemar: A = model, B = RF on correct / incorrect per day (n01 = model correct and RF "
            "wrong, n10 = model wrong and RF correct); chi2_cc / p_cc with continuity correction, p_exact binomial; "
            f"recommended = exact when n01 + n10 < {DISCORDANT_MIN_FOR_CC}; Holm over the family of the 4 comparisons with RF "
            "(p_cc and p_exact separately). The paired-bootstrap CIs are not adjusted. "
            f"Script: revision/scripts/extra_models.py --mode main, run {day}."]
    write_text(res / "t21_comparisons_table.md", "\n".join(cmd) + "\n")

    meta = {
        "run_date": run_date, "elapsed_seconds": round(time.time() - t0, 1),
        "fit_seconds": {n: results[n]["fit_seconds"] for n in NEW_MODELS} | {"rf_identity": rf["fit_seconds"]},
        "task": "T2.1", "mode": "main", "seed": SEED, "smote_seed": SEED, "n_resamples": N_BOOT, "n_test": N_TEST,
        "n_train": N_TRAIN, "n_total": N_TOTAL, "versions": vers, "torch_threads": torch.get_num_threads(),
        "protocol": {"trees": "fit once on the full 766-day train (imputer + SMOTE k = 3), no validation, no early stopping",
                     "networks": f"inner temporal split {INNER_TRAIN} / {INNER_VAL} of the train (imputer / SMOTE / scaler "
                                 f"fit on the inner-train only; validation = real rows), early stopping on validation macro F1 "
                                 f"(patience {PATIENCE}, max {MAX_EPOCHS} epochs, strict improvement, tol 0; same rule for both "
                                 "networks), then refit from scratch on the full train (SMOTE k = 3) for exactly best_epoch "
                                 "epochs; asymmetry with the trees documented",
                     "input_matrix": {n: ("scaled (Split.X_train_scaled_sm / X_test_scaled, SVM flow)" if SCALED[n]
                                          else "unscaled (Split.X_train_sm / X_test, RF / XGB flow)") for n in ["rf"] + NEW_MODELS},
                     "determinism": "random / numpy / torch.manual_seed(seed), torch.use_deterministic_algorithms(True), "
                                    "torch.set_num_threads(1) before every model construction; FT-Transformer DataLoader with "
                                    "torch.Generator().manual_seed(seed), shuffle on train only; TabNet DataLoader shuffles with "
                                    "the global torch RNG (seeded); CPU only",
                     "warnings_excluded_by_name": TABNET_WARNINGS,
                     "y_pred": "argmax(proba) (checked equal to predict() where the API has one)"},
        "t210_identity": {k: m_rf[k] for k in ("f1_macro", "accuracy", "mcc", "f1_active")} | {"max_abs_proba_diff_vs_csv": max_dp},
        "t11_rf_f1_macro_ci_from_cifre_revizie": list(t11_ci),
        "rf_f1_macro_marginal_ci": [g["rf", "f1_macro"].ci_low, g["rf", "f1_macro"].ci_high],
        "models": {n: {k: results[n][k] for k in results[n] if k not in ("y_true", "y_pred", "proba", "dates_test", "fit_seconds")}
                   | {"point": results[n]["point"]} for n in NEW_MODELS},
        "models_above_rf_f1_macro": above,
        "bootstrap_indices_sha256": sha256(res / INDEX_FILE),
        "prediction_files_sha256": pred_files,
        "table_files_sha256": {f: sha256(res / f) for f in ("t21_extra_models.csv", "t21_extra_models_table.md",
                                                            "t21_comparisons.csv", "t21_comparisons_table.md")},
        "sanity_checks_failed": fails,
    }
    write_json(res / "t21_run_meta.json", jsonable(meta))
    print(f"\nelapsed {time.time() - t0:.1f}s; outputs in {res}")
    if fails:
        print(f"{len(fails)} check(s) FAILED: {fails}")
        sys.exit(1)
    print("ALL CHECKS PASS")


def run_seeds(res: Path = RES, seeds: list[int] = SEEDS) -> None:
    t0 = time.time()
    fails: list[str] = []

    def check(cond: bool, msg: str, abort: bool = False) -> None:
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)
            if abort:
                sys.exit(f"ABORT: {msg}")

    vers = versions()
    check(VENDOR.resolve() in Path(vers["pytorch_tabnet_path"]).parents, "pytorch_tabnet imported from revision/vendor", abort=True)
    data = load_data()
    for name in NEW_MODELS:
        check((res / f"t21_predictions_{name}.csv").exists(), f"main-mode predictions exist for {name} (run --mode main first)", abort=True)
    rows = []
    fit_seconds: dict[str, dict[str, float]] = {}
    identity: dict[str, dict] = {}
    print(f"\n=== T2.1.4 seeds {seeds} x {NEW_MODELS} (SMOTE seed {SEED} fixed; model seed varies) ===")
    for name in NEW_MODELS:
        fit_seconds[name] = {}
        for seed in seeds:
            r = fit_eval_model(name, seed, data)
            fit_seconds[name][str(seed)] = r["fit_seconds"]
            m = point_metrics(r["y_true"], r["y_pred"])
            if seed == SEED:
                ref = pd.read_csv(res / f"t21_predictions_{name}.csv")
                same_pred = np.array_equal(ref["y_pred"].to_numpy(), r["y_pred"])
                max_dp = float(np.abs(ref[PROBA_COLS].to_numpy() - r["proba"]).max())
                identity[name] = {"y_pred_identical": bool(same_pred), "n_days_y_pred_differ": int((ref["y_pred"].to_numpy() != r["y_pred"]).sum()),
                                  "max_abs_proba_diff": max_dp}
                check(same_pred and max_dp < 1e-9, f"{name} seed {SEED}: identical to --mode main (y_pred identical, max abs proba diff "
                      f"{max_dp:.2e} < 1e-9)", abort=True)
            rows.append(dict(model=name, seed=seed, f1_macro=m["f1_macro"], f1_active=m["f1_active"], accuracy=m["accuracy"],
                             mcc=m["mcc"], balanced_accuracy=m["balanced_accuracy"],
                             best_epoch=r.get("best_epoch", np.nan), val_f1_macro_at_best=r.get("val_f1_macro_at_best", np.nan)))
            extra = f"  best_epoch {r['best_epoch']} (val F1 {r['val_f1_macro_at_best']:.4f})" if name in NETWORKS else ""
            print(f"  {LABEL[name]:<15} seed {seed:>2}  F1 macro {m['f1_macro']:.4f}  F1 Active {m['f1_active']:.4f}  acc {m['accuracy']:.4f}  "
                  f"MCC {m['mcc']:.4f}  fit {r['fit_seconds']} s{extra}")
    df = pd.DataFrame(rows)
    check(len(df) == len(NEW_MODELS) * len(seeds), f"{len(NEW_MODELS) * len(seeds)} runs ({len(df)})", abort=True)
    summary = seeds_summary(df)
    run_date = datetime.now().isoformat(timespec="seconds")
    day = run_date[:10]
    df.to_csv(res / "t21_seeds.csv", index=False, float_format="%.10g", lineterminator="\n")
    md = seeds_md_lines(summary)
    md += ["",
           f"Informative, NON-canonical: the 4 new models refit with model seeds {seeds} (SMOTE stays at seed {SEED}, as in T1.4 "
           "mode seeds; only the model varies; networks: early stop + refit per seed), each evaluated on the 192 test days. "
           "mean / min / max / p2_5 / p97_5 are descriptive statistics of the 5 values (not a CI). The reported figure is the "
           f"seed-42 run (== --mode main, identity checked). best_epoch rows only for the networks. "
           f"Script: revision/scripts/extra_models.py --mode seeds, run {day}."]
    write_text(res / "t21_seeds_table.md", "\n".join(md) + "\n")
    meta = {"run_date": run_date, "elapsed_seconds": round(time.time() - t0, 1), "fit_seconds": fit_seconds,
            "task": "T2.1", "mode": "seeds", "seeds": seeds, "smote_seed": SEED, "versions": vers,
            "torch_threads": torch.get_num_threads(), "identity_seed42_vs_main": identity,
            "summary": jsonable(summary.to_dict(orient="records")),
            "files_sha256": {f: sha256(res / f) for f in ("t21_seeds.csv", "t21_seeds_table.md")},
            "sanity_checks_failed": fails}
    write_json(res / "t21_run_meta_seeds.json", jsonable(meta))
    print("\n" + "\n".join(seeds_md_lines(summary)))
    print(f"\nelapsed {time.time() - t0:.1f}s; outputs in {res}")
    if fails:
        print(f"{len(fails)} check(s) FAILED: {fails}")
        sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="T2.1 extra models (LightGBM, CatBoost, TabNet, FT-Transformer), defaults")
    ap.add_argument("--mode", choices=["main", "seeds"], default="main")
    ap.add_argument("--results", default=RES, type=Path)
    a = ap.parse_args()
    run_main(a.results) if a.mode == "main" else run_seeds(a.results)
