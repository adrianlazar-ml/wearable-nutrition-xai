"""
test_extra_models.py - unit tests for extra_models.py (T2.1, Task 5).

- inner temporal split 612 / 154 with 25 / 13 Active; SMOTE fit on the inner-train only (the
  validation block keeps 154 real rows before and after; non-missing values untouched)
- EarlyStopper replicates pytorch_tabnet.callbacks.EarlyStopping (random score sequences)
- refit with exactly best_epoch epochs: epoch counter == best_epoch (FT-Transformer loop and
  TabNet history), early-stopped stage records best_epoch <= epochs_run
- predict_proba sums to 1 and argmax == predict for every wrapper (LightGBM, CatBoost, TabNet,
  FT-Transformer); class mapping [0, 1, 2] preserved for CatBoost and TabNet when the first
  rows of y are not in sorted class order
- the metrics function reproduces the RF row of t11_extended_metrics.csv at 4 decimals
- seeds summary and fixed design constants

Run from repo root:  python -m pytest revision/scripts/test_extra_models.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parents[0]))        # revision/ (t3_labels) before the frozen root copies
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import extra_models as em  # noqa: E402
from common import FEATURES  # noqa: E402
from pytorch_tabnet.callbacks import EarlyStopping  # noqa: E402  (vendored copy, via extra_models' sys.path)


@pytest.fixture(scope="module")
def data():
    return em.load_data()


def synthetic(n: int, seed: int = 0, first_rows_order=(2, 0, 1)) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    score = X[:, 0] - 0.8 * X[:, 1] + rng.normal(scale=0.5, size=n)
    y = np.digitize(score, np.quantile(score, [0.25, 0.85])).astype(int)
    # put one row of each class first, in a NON-sorted order, so the wrappers must sort classes_
    for j, c in enumerate(first_rows_order):
        k = int(np.where(y == c)[0][0])
        X[[j, k]] = X[[k, j]]
        y[[j, k]] = y[[k, j]]
    assert list(y[:3]) == list(first_rows_order)
    return X, y


# ------------------------------------------------------------------ inner split and preprocessing
def test_inner_split_sizes_and_active(data) -> None:
    X, y, dates = data
    assert em.inner_bounds() == (612, 154)
    s, k = em.inner_preprocess(X, y, dates)
    assert k == 3
    assert len(s.y_train) == 612 and len(s.y_test) == 154 and len(s.X_test) == 154 and len(s.X_test_scaled) == 154
    assert int((s.y_train == 2).sum()) == 25 and int((s.y_test == 2).sum()) == 13
    # SMOTE only on the inner-train: 3 x majority (452 Normal) synthetic-augmented rows, validation untouched
    assert int((s.y_train == 1).sum()) == 452 and len(s.y_train_sm) == 1356 and len(s.y_train_scaled_sm) == 1356
    assert list(s.dates_test) == list(dates.iloc[612:766])
    raw = X.iloc[612:766].reset_index(drop=True)
    mask = raw.notna().to_numpy()
    assert np.allclose(s.X_test.to_numpy()[mask], raw.to_numpy()[mask])      # imputation fills only the NaNs
    assert np.array_equal(s.y_test.to_numpy(), y.iloc[612:766].to_numpy())


def test_canonical_preprocess_sizes(data) -> None:
    X, y, dates = data
    s, k = em.canonical_preprocess(X, y, dates)
    assert k == 3 and len(s.y_train) == 766 and len(s.y_test) == 192
    assert len(s.y_train_sm) == 3 * 570 == 1710 and len(s.y_train_scaled_sm) == 1710
    assert tuple(int((s.y_test == c).sum()) for c in (0, 1, 2)) == (44, 140, 8)


def test_inner_bounds_validation() -> None:
    assert em.inner_bounds(10, 3) == (7, 3)
    with pytest.raises(ValueError):
        em.inner_bounds(10, 0)
    with pytest.raises(ValueError):
        em.inner_bounds(10, 10)


# ------------------------------------------------------------------ early stopping rule
class _DummyNet:
    def state_dict(self):
        return {}


class _DummyTrainer:
    def __init__(self) -> None:
        self.network = _DummyNet()
        self._stop_training = False


def _run_tabnet_rule(scores, patience: int) -> tuple[int, int]:
    tr = _DummyTrainer()
    es = EarlyStopping(early_stopping_metric="m", is_maximize=True, patience=patience)
    es.set_trainer(tr)
    stopped = 0
    for i, v in enumerate(scores):
        es.on_epoch_end(i, {"m": v})
        if tr._stop_training:
            stopped = i + 1
            break
    return es.best_epoch + 1, stopped          # 1-based like EarlyStopper


def test_early_stopper_simple_sequence() -> None:
    st = em.EarlyStopper(patience=3)
    stops = [st.update(e, v) for e, v in enumerate([0.1, 0.2, 0.2, 0.2, 0.2], start=1)]
    assert stops == [False, False, False, False, True]
    assert st.best_epoch == 2 and st.best == 0.2 and st.stopped_epoch == 5
    st2 = em.EarlyStopper(patience=2)
    assert [st2.update(e, v) for e, v in enumerate([0.5, 0.6, 0.7], start=1)] == [False, False, False]
    assert st2.best_epoch == 3 and st2.stopped_epoch == 0


def test_early_stopper_matches_tabnet_rule() -> None:
    rng = np.random.default_rng(7)
    for trial in range(40):
        patience = int(rng.integers(1, 6))
        scores = np.round(rng.random(30), 1).tolist()      # ties are frequent: strict improvement matters
        st = em.EarlyStopper(patience)
        stopped = 0
        for e, v in enumerate(scores, start=1):
            if st.update(e, v):
                stopped = e
                break
        assert (st.best_epoch, stopped) == _run_tabnet_rule(scores, patience), (trial, patience, scores)


# ------------------------------------------------------------------ wrappers
def test_ftt_refit_epoch_counter_and_proba() -> None:
    X, y = synthetic(96)
    model, opt = em.ftt_build(19, 0)
    r = em.ftt_train(model, opt, X, y, 0, max_epochs=3, batch=32)
    assert r["epochs_run"] == 3 and r["best_epoch"] == 3 and r["val_history"] == [] and r["n_batches_per_epoch"] == 3
    proba = em.ftt_proba(model, X)
    assert proba.shape == (96, 3) and float(np.abs(proba.sum(1) - 1).max()) < 1e-6
    assert np.array_equal(proba.argmax(1), em.ftt_proba(model, X).argmax(1))
    Xv, yv = synthetic(40, seed=1)
    model2, opt2 = em.ftt_build(19, 0)
    r2 = em.ftt_train(model2, opt2, X, y, 0, max_epochs=40, batch=32, X_val=Xv, y_val=yv, patience=2)
    assert 1 <= r2["best_epoch"] <= r2["epochs_run"] <= 40 and len(r2["val_history"]) == r2["epochs_run"]
    assert r2["best_val_f1"] == max(r2["val_history"])
    # the refit for best_epoch epochs runs exactly that many epochs
    model3, opt3 = em.ftt_build(19, 0)
    r3 = em.ftt_train(model3, opt3, X, y, 0, max_epochs=r2["best_epoch"], batch=32)
    assert r3["epochs_run"] == r2["best_epoch"]


def test_ftt_build_is_default_and_seeded() -> None:
    m1, o1 = em.ftt_build(19, 3)
    m2, o2 = em.ftt_build(19, 3)
    assert m1._is_default and o1.param_groups[0]["lr"] == 1e-4 and o1.param_groups[0]["weight_decay"] == 1e-5
    assert o1.param_groups[1]["weight_decay"] == 0.0
    for p, q in zip(m1.parameters(), m2.parameters()):
        assert np.array_equal(p.detach().numpy(), q.detach().numpy())


def test_tabnet_refit_epoch_counter_classes_and_proba() -> None:
    X, y = synthetic(1100)                                   # >= 1024 rows: one real batch per epoch with the defaults
    Xv, yv = synthetic(200, seed=1)
    m_a = em.tabnet_fit(X, y, 0, max_epochs=4, patience=2, X_val=Xv, y_val=yv)
    assert 1 <= m_a.best_epoch + 1 <= em.tabnet_epochs_run(m_a) <= 4
    assert len(m_a.history["val_f1_macro"]) == em.tabnet_epochs_run(m_a)
    best_epoch = int(m_a.best_epoch) + 1
    m = em.tabnet_fit(X, y, 0, max_epochs=best_epoch, patience=0)
    assert em.tabnet_epochs_run(m) == best_epoch                  # refit counter == best_epoch
    assert list(m.classes_) == [0, 1, 2] and m.target_mapper == {0: 0, 1: 1, 2: 2}
    proba = m.predict_proba(X.astype(np.float32))
    assert proba.shape == (1100, 3) and float(np.abs(proba.sum(1) - 1).max()) < 1e-6
    assert np.array_equal(proba.argmax(1), m.predict(X.astype(np.float32)).astype(int))
    assert m.batch_size == 1024 and m.virtual_batch_size == 128 and m.drop_last is True


def test_lgbm_and_catboost_wrappers() -> None:
    X, y = synthetic(300)
    Xd = pd.DataFrame(X, columns=FEATURES)
    lg = em.make_tree("lgbm", 0)
    lg.fit(Xd, y)
    p = lg.predict_proba(Xd)
    assert list(lg.classes_) == [0, 1, 2] and float(np.abs(p.sum(1) - 1).max()) < 1e-6
    assert np.array_equal(p.argmax(1), lg.predict(Xd))
    assert lg.get_params()["n_jobs"] == 1 and lg.get_params()["random_state"] == 0 and lg.get_params()["verbose"] == -1
    cb = em.make_tree("catboost", 0, iterations=20)         # override only to keep the test fast
    cb.fit(Xd, y)
    p = cb.predict_proba(Xd)
    assert [int(c) for c in cb.classes_] == [0, 1, 2] and float(np.abs(p.sum(1) - 1).max()) < 1e-6
    assert np.array_equal(p.argmax(1), np.asarray(cb.predict(Xd)).reshape(-1).astype(int))
    ap = cb.get_all_params()                                 # effective parameters (no thread_count key there)
    assert cb.get_params()["thread_count"] == 1 and ap["random_seed"] == 0 and ap["iterations"] == 20 and cb.tree_count_ == 20
    assert em.make_tree("rf", 0).get_params()["random_state"] == 42           # canonical RF, not reseeded
    with pytest.raises(ValueError):
        em.make_tree("svm", 0)


def test_tabnet_metric_and_jsonable() -> None:
    m = em.F1MacroMetric()
    assert m._name == "f1_macro" and m._maximize is True
    y = np.array([0, 1, 2, 2])
    s = np.eye(3)[[0, 1, 2, 2]]
    assert m(y, s) == 1.0
    out = em.jsonable({"a": np.int64(3), "b": np.float32(0.5), "c": [np.bool_(True), None], "d": object})
    assert out["a"] == 3 and out["b"] == 0.5 and out["c"] == [True, None] and isinstance(out["d"], str)


# ------------------------------------------------------------------ metrics and summaries
def test_metrics_reproduce_t11_rf_row() -> None:
    idx = np.load(em.RES / em.INDEX_FILE)
    y_true, preds, probas, _ = em.load_saved_systems(em.RES)
    table = em.metrics_with_ci({"rf": (preds["rf"], probas["rf"])}, y_true, idx)
    t11 = pd.read_csv(em.RES / "t11_extended_metrics.csv")
    t11 = t11[t11.system == "rf"].set_index("metric")
    got = table.set_index("metric")
    for metric in ("balanced_accuracy", "cohen_kappa", "g_mean", "roc_auc_ovr_macro", "pr_auc_macro", "f1_macro", "accuracy", "mcc"):
        for col in ("value", "ci_low", "ci_high"):
            assert round(float(got.loc[metric, col]), 4) == round(float(t11.loc[metric, col]), 4), (metric, col)
    assert round(float(got.loc["f1_active", "value"]), 4) == 0.4444
    assert int(table.n_skipped.sum()) == 0 and int(table.n_warnings.sum()) == 0
    lines = em.metrics_md_lines(table)
    assert lines[2].startswith("| RF | 0.5063 [0.3784, 0.6130] |")


def test_seeds_summary() -> None:
    df = pd.DataFrame([dict(model="lgbm", seed=s, f1_macro=v, f1_active=v / 2, accuracy=v, mcc=v, best_epoch=np.nan)
                       for s, v in zip([42, 0, 1, 2, 3], [0.5, 0.4, 0.6, 0.45, 0.55])]
                      + [dict(model="ftt", seed=s, f1_macro=0.3, f1_active=0.1, accuracy=0.3, mcc=0.0, best_epoch=e)
                         for s, e in zip([42, 0, 1, 2, 3], [10, 12, 8, 20, 10])])
    sm = em.seeds_summary(df).set_index(["model", "metric"])
    r = sm.loc[("lgbm", "f1_macro")]
    assert r.seed42 == 0.5 and r["mean"] == pytest.approx(0.5) and r["min"] == 0.4 and r["max"] == 0.6 and r.n_seeds == 5
    assert ("lgbm", "best_epoch") not in sm.index
    b = sm.loc[("ftt", "best_epoch")]
    assert b.seed42 == 10 and b["min"] == 8 and b["max"] == 20
    assert len(em.seeds_md_lines(em.seeds_summary(df))) == 2 + len(sm)


def test_fixed_design() -> None:
    assert (em.N_TOTAL, em.N_TRAIN, em.N_TEST, em.N_BOOT, em.SEED) == (958, 766, 192, 1000, 42)
    assert (em.INNER_TRAIN, em.INNER_VAL, em.EXPECT_INNER_ACTIVE) == (612, 154, (25, 13))
    assert (em.MAX_EPOCHS, em.PATIENCE, em.FTT_BATCH) == (200, 16, 64)
    assert em.NEW_MODELS == ["lgbm", "catboost", "tabnet", "ftt"] and em.SEEDS == [42, 0, 1, 2, 3]
    assert em.SCALED == {"rf": False, "lgbm": False, "catboost": False, "tabnet": True, "ftt": True}
    assert em.CANONICAL_THRESHOLDS == dict(rest_cutoff=60, active_readiness=85, active_sleep=75)
    assert em.CANON_RF == dict(f1_macro=0.5063, accuracy=0.6406, mcc=0.1416)
    assert Path(list(sys.modules["pytorch_tabnet"].__path__)[0]).resolve().parent == (SCRIPTS.parents[0] / "vendor").resolve()
