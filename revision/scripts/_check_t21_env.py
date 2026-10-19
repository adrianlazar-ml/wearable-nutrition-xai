"""
_check_t21_env.py - T2.1 Task 0: environment smoke test for the four extra models
(LightGBM, CatBoost, TabNet, FT-Transformer), run under `python -W error`.

Checks:
  1. imports of torch, lightgbm, catboost, pytorch_tabnet (the VENDORED copy in
     revision/vendor, see revision/vendor/pytorch_tabnet/PATCH.md) and rtdl_revisiting_models;
     versions printed and compared with the frozen environment (python / sklearn / numpy /
     pandas / imblearn / shap / xgboost / scipy / statsmodels must be EXACTLY the T0.1 versions).
  2. torch.cuda.is_available() == False (CPU wheel), one thread.
  3. fit + predict_proba on synthetic data (200 x 19, 3 classes, default_rng(42)) for each of
     the four models with a fixed seed, CPU, one thread; probabilities sum to 1, argmax ==
     predict where the API has predict; a second fit with the same seed gives byte-identical
     probabilities (determinism on the synthetic data).
  4. every warning emitted inside the fits is recorded (catch_warnings(record=True) with
     simplefilter("always") inside the block, so nothing is raised there) and must be in the
     NAMED allow-list below; anything else is a FAIL.

The TabNet smoke fit uses batch_size 100 / virtual_batch_size 50 (not the 1024 / 128 defaults)
only because 200 rows < 1024 and the package default drop_last=True would train on zero
batches; extra_models.py keeps the package defaults on the real data (1,710 / 1,356 rows).

Exits 1 on any FAIL. Evidence script, kept.
Run from repo root:  python -W error revision/scripts/_check_t21_env.py
"""

from __future__ import annotations

import importlib.metadata as im
import platform
import random
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "revision" / "vendor"
sys.path.insert(0, str(VENDOR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
import imblearn  # noqa: E402
import shap  # noqa: E402
import xgboost  # noqa: E402
import statsmodels  # noqa: E402
import torch  # noqa: E402
import lightgbm  # noqa: E402
import catboost  # noqa: E402
import pytorch_tabnet  # noqa: E402
import rtdl_revisiting_models  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from pytorch_tabnet.tab_model import TabNetClassifier  # noqa: E402
from rtdl_revisiting_models import FTTransformer  # noqa: E402

FROZEN = {"python": "3.11.9", "scikit-learn": "1.8.0", "numpy": "2.4.3", "pandas": "2.3.3",
          "imbalanced-learn": "0.14.1", "shap": "0.51.0", "xgboost": "3.2.0", "scipy": "1.17.1",
          "statsmodels": "0.15.0"}
NEW = ["torch", "lightgbm", "catboost", "pytorch-tabnet", "rtdl_revisiting_models"]
# Warnings allowed BY NAME (message prefix); everything else fails the smoke test.
ALLOWED_WARNINGS = {
    "tabnet": ["Best weights from best epoch are automatically used!",
               "No early stopping will be performed, last training weights will be used."],
}
SEED = 42
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def synthetic(n: int = 200, p: int = 19, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    score = X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(scale=0.5, size=n)
    y = np.digitize(score, np.quantile(score, [0.2, 0.9]))   # 3 classes, imbalanced (~20 / 70 / 10 %)
    return X.astype(np.float64), y.astype(int)


def fit_lgbm(X, y, seed):
    # DataFrame with named columns for fit AND predict: lightgbm 4.6 always exposes feature_names_in_
    # (auto "Column_i" on numpy input), so sklearn's validate_data(reset=False) warns on numpy at predict;
    # the real pipeline passes DataFrames (common.Split), so the smoke test does the same.
    Xd = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    m = LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1)
    m.fit(Xd, y)
    return m.predict_proba(Xd), m.predict(Xd), list(m.classes_)


def fit_catboost(X, y, seed):
    m = CatBoostClassifier(random_seed=seed, thread_count=1, verbose=0, allow_writing_files=False)
    m.fit(X, y)
    return m.predict_proba(X), m.predict(X).reshape(-1).astype(int), [int(c) for c in m.classes_]


def fit_tabnet(X, y, seed):
    m = TabNetClassifier(seed=seed, device_name="cpu", verbose=0)
    m.fit(X.astype(np.float32), y, eval_set=[(X.astype(np.float32), y)], eval_metric=["accuracy"],
          max_epochs=3, patience=2, batch_size=100, virtual_batch_size=50)
    return m.predict_proba(X.astype(np.float32)), m.predict(X.astype(np.float32)).astype(int), list(m.classes_)


def fit_ftt(X, y, seed, n_epochs: int = 3, batch: int = 64):
    model = FTTransformer(n_cont_features=X.shape[1], cat_cardinalities=[], d_out=3, **FTTransformer.get_default_kwargs())
    opt = model.make_default_optimizer()
    loss_fn = torch.nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt), batch_size=batch, shuffle=True, generator=gen)
    for _ in range(n_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb, None), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(Xt, None), dim=1).numpy().astype(np.float64)
    return proba, proba.argmax(1), [0, 1, 2]


def main() -> None:
    print("=== versions ===")
    versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                "pandas": pd.__version__, "imbalanced-learn": imblearn.__version__, "shap": shap.__version__,
                "xgboost": xgboost.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__}
    for k, v in FROZEN.items():
        check(versions[k] == v, f"{k} == {v} (found {versions[k]})")
    for k in NEW:
        print(f"  {k} {im.version(k)}")
    check(torch.__version__.endswith("+cpu") or torch.version.cuda is None, f"torch CPU build ({torch.__version__})")
    check(torch.cuda.is_available() is False, "torch.cuda.is_available() == False")
    tabnet_path = Path(list(pytorch_tabnet.__path__)[0]).resolve()   # namespace package (no __init__.py): __file__ is None
    check(VENDOR.resolve() in tabnet_path.parents, f"pytorch_tabnet imported from revision/vendor ({tabnet_path})")
    check((VENDOR / "pytorch_tabnet" / "PATCH.md").exists(), "revision/vendor/pytorch_tabnet/PATCH.md present")
    print(f"  rtdl_revisiting_models from {Path(rtdl_revisiting_models.__file__).name}; "
          f"FTTransformer.get_default_kwargs() = {FTTransformer.get_default_kwargs()}")

    print("\n=== fit + predict_proba on synthetic data (200 x 19, 3 classes), seed 42, CPU, 1 thread ===")
    X, y = synthetic()
    check(sorted(set(y.tolist())) == [0, 1, 2], f"synthetic labels 0/1/2, counts {np.bincount(y).tolist()}")
    fits = {"lgbm": fit_lgbm, "catboost": fit_catboost, "tabnet": fit_tabnet, "ftt": fit_ftt}
    all_warnings: dict[str, list[str]] = {}
    for name, fn in fits.items():
        t0 = time.time()
        runs = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                seed_everything(SEED)
                runs.append(fn(X, y, SEED))
        msgs = sorted({str(w.message) for w in caught})
        all_warnings[name] = msgs
        (proba, pred, classes), (proba2, _, _) = runs
        check(proba.shape == (200, 3), f"{name}: predict_proba shape {proba.shape}")
        check(float(np.abs(proba.sum(1) - 1).max()) < 1e-6, f"{name}: probabilities sum to 1 (max dev {np.abs(proba.sum(1) - 1).max():.1e})")
        check(np.array_equal(proba.argmax(1), pred), f"{name}: argmax(proba) == predict")
        check(classes == [0, 1, 2], f"{name}: classes_ == [0, 1, 2] ({classes})")
        check(np.array_equal(proba, proba2), f"{name}: second fit with the same seed is byte-identical "
              f"(max abs diff {np.abs(proba - proba2).max():.1e})")
        allowed = ALLOWED_WARNINGS.get(name, [])
        unexpected = [m for m in msgs if not any(m.startswith(a) for a in allowed)]
        check(not unexpected, f"{name}: warnings emitted: {len(msgs)}; all in the named allow-list"
              + (f"; UNEXPECTED: {unexpected}" if unexpected else ""))
        for m in msgs:
            print(f"      warning: {m!r}")
        print(f"  {name}: {time.time() - t0:.1f} s for 2 fits; threads torch {torch.get_num_threads()}")

    print(f"\n{'ALL PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
