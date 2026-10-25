"""
_check_f5_empirical.py - evidence for the T2 questions of audit slice F5 that need code execution.

Imports the revision scripts in memory (extra_models, hyperparameter_search, the vendored
pytorch_tabnet) and calls none of their main() / run_*() functions. Nothing under
revision/results/ is written; the regenerated search space goes to a temporary directory.

E1  EarlyStopper (FT-Transformer loop) vs pytorch_tabnet.callbacks.EarlyStopping on the same
    score sequences: the two recorded T2.1 validation histories (t21_run_meta.json) and 400
    random sequences with ties, patience in {1, 2, 3, 5, 16}.                           (Q3)
E2  TabNet DataLoader batches per epoch with the package defaults (batch 1024, drop_last True)
    on 1356 / 1710 / 669 / 1008 rows, through the vendored create_dataloaders.           (Q7)
E3  FT-Transformer construction twice through em.ftt_build(19, 42): identical state_dict
    (stage A and the refit start from the same weights); batches per epoch 22 vs 27.   (Q5)
E4  hs.build_search_space() written with em.write_json to a temporary file: sha256 == the
    committed t22_search_space.json.                                                   (Q12)
E5  hs.make_folds and em.inner_preprocess: validation blocks are the raw rows b..c-1 after
    imputation only (no synthetic rows), imputer / scaler statistics equal those refit on the
    fold's train part; fold 3 == inner split.                                       (Q1, Q2, Q11)
E6  CatBoost default through em.fit_eval_model("catboost", 42, data): y_pred and probabilities
    vs t21_predictions_catboost.csv.                                                     (Q8)
E7  LightGBM and CatBoost through em.make_tree on the canonical split: predict() == argmax of
    predict_proba on the 192 test days.                                                 (Q10)
E8  Diagnostic: the tuned LightGBM configuration with and without the fixed subsample_freq = 1
    that Appendix B1 does not mention (sizing a documentation gap; two fits under 1 s).  (Q15)
Run:  python revision/audit/_check_f5_empirical.py > revision/audit/_check_f5_empirical_output.txt
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(newline="\n")                      # LF output on Windows
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "scripts"))
import extra_models as em  # noqa: E402
import hyperparameter_search as hs  # noqa: E402
import torch  # noqa: E402
from pytorch_tabnet.callbacks import EarlyStopping  # noqa: E402  (vendored, via extra_models' sys.path)
from pytorch_tabnet.utils import create_dataloaders  # noqa: E402

RES = ROOT / "revision" / "results"
fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


print("pytorch_tabnet from:", Path(list(sys.modules["pytorch_tabnet"].__path__)[0]).resolve())
check(str(ROOT / "revision" / "vendor") in str(Path(list(sys.modules["pytorch_tabnet"].__path__)[0]).resolve()), "pytorch_tabnet imported from revision/vendor")

# ------------------------------------------------------------------ E1  early stopping equivalence
print("\n--- E1. EarlyStopper vs pytorch_tabnet EarlyStopping on the same score sequences ---")


class _Trainer:
    def __init__(self) -> None:
        self.network = torch.nn.Linear(1, 1)
        self._stop_training = False


def tabnet_rule(scores, patience):
    """Feed the sequence to the vendored callback exactly as fit() does (epoch 0-based,
    on_epoch_end after every epoch, break when _stop_training). Returns (best_epoch 1-based,
    stopped epoch 1-based or 0, epochs run)."""
    tr = _Trainer()
    es = EarlyStopping(early_stopping_metric="val_f1_macro", is_maximize=True, patience=patience)
    es.set_trainer(tr)
    run = 0
    for i, s in enumerate(scores):
        run = i + 1
        es.on_epoch_end(i, {"val_f1_macro": float(s)})
        if tr._stop_training:
            break
    # on_train_end() only copies es.best_epoch / es.best_loss to the trainer (and prints); read them directly
    return es.best_epoch + 1, (es.stopped_epoch + 1 if tr._stop_training else 0), run


def ftt_rule(scores, patience):
    st = em.EarlyStopper(patience)
    run = 0
    for i, s in enumerate(scores, start=1):
        run = i
        if st.update(i, float(s)):
            break
    return st.best_epoch, st.stopped_epoch, run


meta21 = json.load(open(RES / "t21_run_meta.json", encoding="utf-8"))
for name, exp in (("tabnet", (5, 21, 21)), ("ftt", (11, 27, 27))):
    hist = meta21["models"][name]["val_history"]
    a, b = tabnet_rule(hist, em.PATIENCE), ftt_rule(hist, em.PATIENCE)
    check(a == b == exp, f"recorded {name} validation history ({len(hist)} epochs): TabNet rule {a} == FT-T rule {b} == recorded (best_epoch, stopped, run) {exp}")
rng = np.random.default_rng(0)
n_same, n_tot = 0, 0
for trial in range(400):
    L = int(rng.integers(1, 60))
    seq = rng.choice(np.round(np.linspace(0, 1, 7), 3), size=L, replace=True)   # coarse grid -> many ties
    for patience in (1, 2, 3, 5, 16):
        n_tot += 1
        if tabnet_rule(seq, patience) == ftt_rule(seq, patience):
            n_same += 1
check(n_same == n_tot, f"{n_same} / {n_tot} random sequences (ties included) x patience in (1, 2, 3, 5, 16): identical (best_epoch, stopped_epoch, epochs_run)")
seq = [0.3, 0.3, 0.3, 0.3]
check(tabnet_rule(seq, 2) == ftt_rule(seq, 2) == (1, 3, 3), f"tie sequence [0.3 x 4], patience 2: strict improvement, stop at epoch 3 -> {ftt_rule(seq, 2)}")

# ------------------------------------------------------------------ E2  TabNet batches per epoch
print("\n--- E2. TabNet batches per epoch with the package defaults (batch 1024, drop_last True) ---")
for n in (669, 1008, 1356, 1710):
    X = np.zeros((n, 19), dtype=np.float32)
    yv = np.resize(np.array([0, 1, 2]), n)
    loader, _ = create_dataloaders(X, yv, [], 0, 1024, 0, True, False)
    rows = sum(len(b[0]) for b in loader)
    print(f"  n = {n}: {len(loader)} batch(es) per epoch, {rows} rows seen per epoch")
    check(len(loader) == n // 1024 and rows == 1024 * (n // 1024), f"n = {n}: batches {len(loader)} == {n // 1024}")
loader, _ = create_dataloaders(np.zeros((1710, 19), dtype=np.float32), np.resize(np.array([0, 1, 2]), 1710), [], 0, 128, 0, False, False)
print(f"  tuned TabNet (batch 128, drop_last False) on 1710 rows: {len(loader)} batches, {sum(len(b[0]) for b in loader)} rows per epoch")

# ------------------------------------------------------------------ E3  FT-T construction identity
print("\n--- E3. FT-Transformer: two constructions through em.ftt_build(19, 42) ---")
m1, _ = em.ftt_build(19, 42)
m2, _ = em.ftt_build(19, 42)
sd1, sd2 = m1.state_dict(), m2.state_dict()
same = all(torch.equal(sd1[k], sd2[k]) for k in sd1) and list(sd1) == list(sd2)
check(same, f"state_dict identical on {len(sd1)} tensors ({sum(int(v.numel()) for v in sd1.values()):,} parameters): stage A and the refit start from the same weights")
g1 = torch.Generator().manual_seed(42)
g2 = torch.Generator().manual_seed(42)
p1, p2 = torch.randperm(1356, generator=g1), torch.randperm(1710, generator=g2)
print(f"  DataLoader order: randperm(1356) vs randperm(1710) from the same seeded generator: first 8 indices {p1[:8].tolist()} vs {p2[:8].tolist()}")
print(f"  batches per epoch at batch 64: ceil(1356 / 64) = {-(-1356 // 64)} (stage A), ceil(1710 / 64) = {-(-1710 // 64)} (refit); "
      f"optimizer steps at best_epoch 11: {11 * -(-1356 // 64)} vs {11 * -(-1710 // 64)}")
check(meta21["models"]["ftt"]["params"]["n_train_batches_per_epoch"] == 27, "t21_run_meta.json: FT-T refit 27 batches per epoch")

# ------------------------------------------------------------------ E4  search space regeneration
print("\n--- E4. Search space regenerated and written with the script's own writer ---")
tmp = Path(tempfile.mkdtemp(prefix="audit_f5_"))
em.write_json(tmp / "t22_search_space.json", em.jsonable(hs.build_search_space()))
check(sha16(tmp / "t22_search_space.json") == sha16(RES / "t22_search_space.json") == "79c1aed12f0151e7",
      f"regenerated file sha256 {sha16(tmp / 't22_search_space.json')} == committed {sha16(RES / 't22_search_space.json')} (byte-identical)")
print("  run_cv order: write_json at hyperparameter_search.py:450, load_data :452, make_folds :453, first fit inside the loop :461-470")

# ------------------------------------------------------------------ E5  folds and inner split
print("\n--- E5. Fold validation blocks and the inner split ---")
t0 = time.time()
data = em.load_data()
X, y, dates = data
folds = hs.make_folds(X, y, dates)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
for f in folds:
    a, b, c = f["bounds"]
    s = f["split"]
    imp = SimpleImputer(strategy="median").fit(X.iloc[a:b])
    sc = StandardScaler().fit(imp.transform(X.iloc[a:b]))
    val_imp = imp.transform(X.iloc[b:c])
    ok = (np.array_equal(s.X_test.to_numpy(), val_imp) and np.array_equal(s.y_test.to_numpy(), y.iloc[b:c].to_numpy())
          and np.array_equal(s.X_test_scaled.to_numpy(), sc.transform(val_imp)) and len(s.y_test) == 154
          and np.array_equal(s.imputer.statistics_, imp.statistics_) and np.allclose(s.scaler.mean_, sc.mean_, rtol=0, atol=0)
          and len(s.y_train) == b - a and len(s.y_train_sm) == f["n_train_smote"])
    check(ok, f"fold {f['fold']}: validation = raw rows {b}-{c - 1} imputed with the fold-train imputer (154 real rows, no SMOTE); "
              f"imputer / scaler statistics == refit on rows {a}-{b - 1}; train {b - a} -> {f['n_train_smote']} after SMOTE k {f['smote_k']}")
    print(f"       validation dates {dates.iloc[b].date()} .. {dates.iloc[c - 1].date()}; last train date {dates.iloc[b - 1].date()}")
inner, k_in = em.inner_preprocess(X, y, dates)
s3 = folds[2]["split"]
check(np.array_equal(inner.X_test.to_numpy(), s3.X_test.to_numpy()) and np.array_equal(inner.X_train_scaled_sm.to_numpy(), s3.X_train_scaled_sm.to_numpy())
      and np.array_equal(inner.y_test.to_numpy(), y.iloc[612:766].to_numpy()) and len(inner.y_train_scaled_sm) == 1356 and k_in == 3,
      "inner split (612 / 154) == fold 3 matrices; validation = rows 612-765 real; inner-train after SMOTE 1,356")
check(bool((dates.iloc[612:766].to_numpy() > dates.iloc[611]).all()) and dates.is_monotonic_increasing, "validation dates strictly after the last inner-train date; dates monotonic")
print(f"  (E5 {time.time() - t0:.1f} s)")

# ------------------------------------------------------------------ E6  CatBoost identity through fit_eval_model
print("\n--- E6. CatBoost default through em.fit_eval_model('catboost', 42, data) ---")
t0 = time.time()
r = em.fit_eval_model("catboost", 42, data)
ref = pd.read_csv(RES / "t21_predictions_catboost.csv")
n_diff = int((ref.y_pred.to_numpy() != r["y_pred"]).sum())
max_dp = float(np.abs(ref[["p_rest", "p_normal", "p_active"]].to_numpy() - r["proba"]).max())
check(n_diff == 0 and max_dp < 1e-9, f"y_pred identical on 192 days ({n_diff} differ), max abs proba diff {max_dp:.2e}; fit {r['fit_seconds']} s")
f1 = em.f1_macro(r["y_true"], r["y_pred"])
check(round(f1, 4) == 0.4980, f"F1 macro {f1:.4f} (claim 0.4980); effective params iterations {r['params']['iterations']}, depth {r['params']['depth']}, lr {r['params']['learning_rate']:.4f}, tree_count {r['params']['tree_count']}")
print(f"  (E6 {time.time() - t0:.1f} s)")

# ------------------------------------------------------------------ E7  predict() vs argmax for the tree libraries
print("\n--- E7. LightGBM / CatBoost: predict() vs argmax(predict_proba) on the canonical split ---")
split, k = em.canonical_preprocess(X, y, dates)
for name in ("lgbm", "catboost"):
    t0 = time.time()
    em.seed_everything(42)
    m = em.make_tree(name, 42)
    m.fit(split.X_train_sm, split.y_train_sm)
    pr = np.asarray(m.predict_proba(split.X_test))
    pa = np.asarray(m.predict(split.X_test)).reshape(-1).astype(int)
    saved = pd.read_csv(RES / f"t21_predictions_{name}.csv")
    check(np.array_equal(pr.argmax(1), pa) and np.array_equal(pa, saved.y_pred.to_numpy()),
          f"{name}: predict() == argmax(proba) on 192 / 192 days and == t21_predictions_{name}.csv ({time.time() - t0:.1f} s)")

# ------------------------------------------------------------------ E8  diagnostic: the LightGBM winner as described in the manuscript
print("\n--- E8. Diagnostic (sizing only, not a proposed figure): LightGBM winner with / without the fixed subsample_freq = 1 ---")
w = json.load(open(RES / "t22_run_meta_test.json", encoding="utf-8"))["winners"]["lgbm"]["params"]
ref = pd.read_csv(RES / "t22_predictions_lgbm.csv")
for label, cfg in (("as run (json / csv params, subsample_freq 1)", dict(w)),
                   ("as described in Appendix B1 (no subsample_freq)", {k_: v for k_, v in w.items() if k_ != "subsample_freq"})):
    em.seed_everything(42)
    m = em.make_tree("lgbm", 42, **cfg).fit(split.X_train_sm, split.y_train_sm)
    p = np.asarray(m.predict(split.X_test)).astype(int)
    f1 = em.f1_macro(split.y_test.to_numpy(int), p)
    same = np.array_equal(p, ref.y_pred.to_numpy())
    print(f"  {label:<50} subsample_freq {m.get_params()['subsample_freq']}  test macro F1 {f1:.4f}  y_pred == t22_predictions_lgbm.csv: {same}")
    if "subsample_freq" in cfg:
        check(same and round(f1, 4) == 0.5739, "winner as run reproduces t22_predictions_lgbm.csv (0.5739)")
    else:
        print(f"  INFO without subsample_freq the LightGBM default (0) disables bagging: subsample 0.6 has no effect; F1 {f1:.4f}")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}"))
