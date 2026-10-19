"""
test_hyperparameter_search.py - unit tests for hyperparameter_search.py (T2.2, Task 4).

- folds: rows exactly 0-303 / 0-457 / 0-611 train and 304-457 / 458-611 / 612-765 validation,
  Active 10 / 17 / 25 in train and 7 / 8 / 13 in validation (the design's true counts; the brief's
  11 / 6 for fold 1 comes from the T1.3 cut at row 307), SMOTE does not touch the validation block
  (154 rows before and after, non-missing values untouched)
- ParameterSampler with random_state 42 regenerates exactly results/t22_search_space.json
  (when present) and is prefix-stable (reduced = first 10 of the same sampler)
- selection: argmax of the mean, smaller config_id on ties; reference rows are not eligible;
  reference rank
- networks: best_epoch per fold in [1, 200] on a tiny early-stopped run; FT-T grid d_block divisible
  by the heads; TabNet grid -> constructor / fit kwargs; McNemar degenerate case
- fixed design constants

Run from repo root:  python -m pytest revision/scripts/test_hyperparameter_search.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parents[0]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import hyperparameter_search as hs  # noqa: E402
from common import FEATURES  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return hs.em.load_data()


@pytest.fixture(scope="module")
def folds(data):
    return hs.make_folds(*data)


def test_fold_rows_and_active(data, folds) -> None:
    X, y, dates = data
    assert hs.fold_bounds() == [(0, 304, 458), (0, 458, 612), (0, 612, 766)]
    assert [f["bounds"] for f in folds] == hs.EXPECT_FOLDS
    assert [f["active_train"] for f in folds] == [10, 17, 25] and [f["active_val"] for f in folds] == [7, 8, 13]
    assert [f["n_train"] for f in folds] == [304, 458, 612] and all(f["n_val"] == 154 for f in folds)
    assert all(f["smote_k"] == 3 for f in folds)
    for f in folds:
        a, b, c = f["bounds"]
        s = f["split"]
        assert len(s.y_test) == 154 and len(s.X_test) == 154 and len(s.X_test_scaled) == 154       # SMOTE never touches validation
        assert len(s.y_train_sm) == 3 * int((y.iloc[a:b] == 1).sum()) == f["n_train_smote"]
        assert list(s.dates_test) == list(dates.iloc[b:c])
        raw = X.iloc[b:c].reset_index(drop=True)
        mask = raw.notna().to_numpy()
        assert np.allclose(s.X_test.to_numpy()[mask], raw.to_numpy()[mask])
        assert np.array_equal(s.y_test.to_numpy(), y.iloc[b:c].to_numpy())
    assert folds[2]["bounds"] == (0, 612, 766)                      # fold 3 == the T2.1 inner split


def test_fold_bounds_generic() -> None:
    assert hs.fold_bounds(20, 2, 5) == [(0, 10, 15), (0, 15, 20)]
    with pytest.raises(ValueError):
        hs.fold_bounds(10, 3, 5)                                      # not enough rows


def test_sampler_reproduces_search_space_json() -> None:
    sp = hs.build_search_space()
    assert {m: len(v["configs"]) for m, v in sp["models"].items()} == {"rf": 20, "xgb": 20, "svm": 20, "lgbm": 20, "catboost": 20,
                                                                      "tabnet": 5, "ftt": 5}
    assert sp == hs.build_search_space()                              # deterministic
    red = hs.build_search_space(reduced=True)
    for m in hs.TABULAR:
        assert red["models"][m]["configs"] == sp["models"][m]["configs"][:10]      # prefix-stable sampler
    for m in hs.NETWORKS:
        assert red["models"][m]["config_ids"] == [0, 2, 4] and red["models"][m]["configs"] == [sp["models"][m]["grid"][i] for i in (0, 2, 4)]
    saved = hs.RES / "t22_search_space.json"
    if saved.exists():
        assert json.loads(saved.read_text(encoding="utf-8")) == json.loads(json.dumps(hs.em.jsonable(sp)))
    for m in hs.TABULAR:                                             # every sampled value is inside its space
        desc = sp["models"][m]["space"]
        for cfg in sp["models"][m]["configs"]:
            for k, v in cfg.items():
                if isinstance(desc[k], list):
                    assert v in desc[k]
                else:
                    lo, hi = desc[k]["loguniform"]
                    assert lo <= v <= hi
    assert "gamma" in sp["models"]["svm"]["space"] and all(isinstance(c["gamma"], float) for c in sp["models"]["svm"]["configs"])


def test_selection_ties_and_reference_not_eligible() -> None:
    cv = pd.DataFrame([dict(model="m", config_id="0", eligible=True, f1_mean=0.5, f1_sd=0.0),
                       dict(model="m", config_id="1", eligible=True, f1_mean=0.6, f1_sd=0.0),
                       dict(model="m", config_id="2", eligible=True, f1_mean=0.6, f1_sd=0.0),
                       dict(model="m", config_id="10", eligible=True, f1_mean=0.6, f1_sd=0.0),
                       dict(model="m", config_id="ref", eligible=False, f1_mean=0.9, f1_sd=0.0)])
    assert hs.select_winner(cv, "m") == (1, 0.6)                     # tie 1 / 2 / 10 -> smallest id (numeric, not lexicographic)
    assert hs.reference_rank(cv, "m") == 1
    cv.loc[cv.config_id == "ref", "f1_mean"] = 0.55
    assert hs.reference_rank(cv, "m") == 4                            # three eligible means strictly above 0.55
    with pytest.raises(ValueError):
        hs.select_winner(cv, "other")


def test_grid_validity_and_kwargs() -> None:
    for cfg in hs.NET_GRIDS["ftt"]:
        assert cfg["d_block"] % 8 == 0
        model, opt = hs.em.ftt_build(19, 0, cfg)
        assert opt.param_groups[0]["lr"] == cfg["lr"] and opt.param_groups[0]["weight_decay"] == 1e-5
        assert len(model.backbone.blocks) == cfg["n_blocks"]
    assert hs.NET_GRIDS["ftt"][4] == hs.em.FTT_DEFAULT_CONFIG == hs.REFERENCE["ftt"]
    m_def, o_def = hs.em.ftt_build(19, 0, hs.NET_GRIDS["ftt"][4])
    assert m_def._is_default and o_def.param_groups[0]["lr"] == 1e-4
    ctor, fit = hs.tabnet_kwargs(hs.NET_GRIDS["tabnet"][0])
    assert ctor == dict(n_d=8, n_a=8, n_steps=3, optimizer_params=dict(lr=0.02)) and fit == dict(batch_size=128, virtual_batch_size=32, drop_last=False)
    ctor, fit = hs.tabnet_kwargs(hs.REFERENCE["tabnet"])
    assert fit == dict(batch_size=1024, virtual_batch_size=128, drop_last=True)


def test_network_best_epoch_in_range() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 19)); y = (X[:, 0] > 0.5).astype(int) + (X[:, 1] > 0.8).astype(int)
    Xv = rng.normal(size=(60, 19)); yv = (Xv[:, 0] > 0.5).astype(int) + (Xv[:, 1] > 0.8).astype(int)
    for name, cfg in (("tabnet", hs.NET_GRIDS["tabnet"][0]), ("ftt", hs.NET_GRIDS["ftt"][0])):
        r = hs.fit_predict(name, cfg, 0, X, y, Xv, yv, early_stop=True)
        assert 1 <= r["best_epoch"] <= r["epochs_run"] <= hs.MAX_EPOCHS and len(r["val_history"]) == r["epochs_run"]
        r2 = hs.fit_predict(name, dict(cfg, epochs=r["best_epoch"]), 0, X, y, Xv)
        assert r2["proba"].shape == (60, 3) and np.array_equal(r2["proba"].argmax(1), r2["y_pred"])


def test_tabular_builders_and_mcnemar_degenerate() -> None:
    assert hs.build_tabular("rf", hs.REFERENCE["rf"], 42).get_params()["max_depth"] == 10
    assert hs.build_tabular("svm", dict(C=3.0, gamma=0.01), 7).get_params()["gamma"] == 0.01
    assert hs.build_tabular("lgbm", dict(hs.SEARCH_SPACES["lgbm"] and {"n_estimators": 100, "num_leaves": 7}, **hs.LGBM_FIXED), 1).get_params()["subsample_freq"] == 1
    assert hs.mcnemar_safe(100, 0, 0, 92) == {"chi2_cc": 0.0, "p_cc": 1.0, "stat_exact": 0.0, "p_exact": 1.0}
    assert 0 < hs.mcnemar_safe(44, 25, 14, 109)["p_cc"] < 1


def test_fixed_design() -> None:
    assert (hs.N_SPLITS, hs.VAL_SIZE, hs.N_ITER, hs.MAX_EPOCHS, hs.PATIENCE) == (3, 154, 20, 200, 16)
    assert hs.MODELS == ["rf", "xgb", "svm", "lgbm", "catboost", "tabnet", "ftt"] and hs.SEEDS == [42, 0, 1, 2, 3]
    assert (hs.REDUCED_N_ITER, hs.REDUCED_GRID) == (10, (0, 2, 4))
    assert hs.CANON_TEST_F1 == {"rf": 0.5063, "xgb": 0.4760, "svm": 0.4347}
    assert hs.SCALED["svm"] and hs.SCALED["tabnet"] and hs.SCALED["ftt"] and not hs.SCALED["rf"]
    assert list(hs.COMPARISONS) == ["a", "b", "c", "d"]
