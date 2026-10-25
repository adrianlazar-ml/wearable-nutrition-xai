"""_check_f2_empirical.py - AUDIT F2 (T2 evidence): function-level behaviour of
revision/scripts/extended_metrics.py and statistical_tests.py, plus the Q6 / Q14
direct counts. Imports the two modules under audit (no retraining, no writes).
Run from repo root: python revision/audit/_check_f2_empirical.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "scripts"))
import extended_metrics as em  # noqa: E402
import statistical_tests as st  # noqa: E402

RES = ROOT / "revision" / "results"
L = [0, 1, 2]
pred = {k: pd.read_csv(RES / f"test_predictions_{k}.csv") for k in ["rf", "xgb", "svm", "persistence"]}
y = pred["rf"]["y_true"].to_numpy(int)
yp = {k: d["y_pred"].to_numpy(int) for k, d in pred.items()}
P = {k: pred[k][["p_rest", "p_normal", "p_active"]].to_numpy(float) for k in ["rf", "xgb", "svm"]}
idx = np.load(RES / "bootstrap_indices_1000x192_seed42.npy")

# ---- Q1: balanced accuracy = mean recall over the 3 fixed classes, on all 1000 resamples
mx = 0.0
for r in idx:
    rec = recall_score(y[r], yp["rf"][r], labels=L, average=None, zero_division=0)
    mx = max(mx, abs(rec.mean() - balanced_accuracy_score(y[r], yp["rf"][r])))
print(f"Q1 max |sklearn balanced_accuracy - mean recall(labels=[0,1,2])| over 1000 RF resamples: {mx:.2e}")
# behaviour when a class is absent from y_true but predicted (does not occur in the 1000 resamples)
yt = np.array([0, 0, 1, 1])
pp = np.array([0, 1, 1, 2])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    ba = balanced_accuracy_score(yt, pp)
print(f"Q1 sklearn balanced_accuracy with Active absent from y_true but predicted: {ba:.4f} "
      f"({len(w)} warning(s); classes with 0 true samples are dropped, i.e. NOT fixed labels); gmean() returns {em.gmean(yt, pp, L)}")

# ---- Q2: kappa row is unweighted; QWK is a different row
lm = em.label_metrics()
print(f"Q2 label_metrics()['cohen_kappa'](RF) = {lm['cohen_kappa'](y, yp['rf']):.6f} == cohen_kappa_score(weights=None) "
      f"{cohen_kappa_score(y, yp['rf']):.6f}; ['qwk'] = {lm['qwk'](y, yp['rf']):.6f} == weights='quadratic' "
      f"{cohen_kappa_score(y, yp['rf'], weights='quadratic'):.6f}")

# ---- Q3: gmean with an absent class -> nan -> skipped by bootstrap()
print(f"Q3 gmean(y_true without Active) = {em.gmean(np.array([0, 1, 1]), np.array([0, 1, 2]), L)}")
non_active = np.array([i for i in range(192) if y[i] != 2])
fake_idx = np.random.default_rng(0).choice(non_active, size=(1, 192), replace=True)   # one row with no Active
lo, hi, used, skipped, nwarn = em.bootstrap(lambda r: em.gmean(y[r], yp["rf"][r], L), np.vstack([idx[:3], fake_idx]))
print(f"Q3 bootstrap() on 3 real rows + 1 row without Active: used {used}, skipped {skipped}, warnings {nwarn}")
pm = em.proba_metrics()
print(f"Q3/Q4 proba_metrics roc/pr on a resample without Active: "
      f"{pm['roc_auc_ovr_macro'](y[fake_idx[0]], P['rf'][fake_idx[0]])}, {pm['pr_auc_macro'](y[fake_idx[0]], P['rf'][fake_idx[0]])}")

# ---- Q6: SVM hard labels: predict() vs argmax(proba); McNemar counts both ways
agree = {k: float((P[k].argmax(1) == yp[k]).mean()) for k in ["rf", "xgb", "svm"]}
print(f"Q6 argmax(proba) == y_pred agreement: {agree}")
svm_arg = P["svm"].argmax(1)
for a, name in [("rf", "RF-SVM"), ("xgb", "XGB-SVM")]:
    n00, n01, n10, n11 = st.mcnemar_counts(y, yp[a], yp["svm"])
    t = st.mcnemar_tests(n00, n01, n10, n11)
    m00, m01, m10, m11 = st.mcnemar_counts(y, yp[a], svm_arg)
    t2 = st.mcnemar_tests(m00, m01, m10, m11)
    print(f"Q6 {name}: y_pred(predict): n01 {n01} n10 {n10} p_cc {t['p_cc']:.4f} | argmax(proba): n01 {m01} n10 {m10} p_cc {t2['p_cc']:.4f}")
print(f"Q6 SVM days where predict != argmax(proba): {int((svm_arg != yp['svm']).sum())}; "
      f"SVM accuracy predict {float((yp['svm'] == y).mean()):.4f} vs argmax {float((svm_arg == y).mean()):.4f}")

# ---- Q7: n01 / n10 from direct counting for RF-XGB vs mcnemar_counts()
ca, cb = yp["rf"] == y, yp["xgb"] == y
direct = dict(n01=int((ca & ~cb).sum()), n10=int((~ca & cb).sum()), n11=int((ca & cb).sum()), n00=int((~ca & ~cb).sum()))
n00, n01, n10, n11 = st.mcnemar_counts(y, yp["rf"], yp["xgb"])
print(f"Q7 RF-XGB direct {direct} | mcnemar_counts (n00,n01,n10,n11) = {(n00, n01, n10, n11)} | csv:")
print(pd.read_csv(RES / "t12_mcnemar.csv").loc[0, ["pair", "n00", "n01", "n10", "n11"]].to_dict())

# ---- Q11: same indices; sign convention
r = st.paired_bootstrap(y, yp["rf"], yp["xgb"], st.METRICS["f1_macro"], idx[:50])
r_rev = st.paired_bootstrap(y, yp["xgb"], yp["rf"], st.METRICS["f1_macro"], idx[:50])
print(f"Q11 paired_bootstrap(RF, XGB) diff_point {r['diff_point']:+.6f}; (XGB, RF) {r_rev['diff_point']:+.6f}; "
      f"CI(RF,XGB) [{r['ci_low']:.4f}, {r['ci_high']:.4f}] vs -CI(XGB,RF) [{-r_rev['ci_high']:.4f}, {-r_rev['ci_low']:.4f}]")

# ---- Q14: persistence multiset identity, counted directly
pc = [int((yp["persistence"] == c).sum()) for c in L]
sup = [int((y == c).sum()) for c in L]
print(f"Q14 persistence predicted counts {pc} == true support {sup}: {pc == sup}")
print(f"Q14 last train label (snapshot) = {em.last_train_label()}; last test label = {int(y[-1])}; "
      f"persistence[0] = {int(yp['persistence'][0])}")
prec = np.diag(em.confusion_matrix(y, yp["persistence"], labels=L)) / np.array(pc)
rec = np.diag(em.confusion_matrix(y, yp["persistence"], labels=L)) / np.array(sup)
print(f"Q14 per-class precision {np.round(prec, 6).tolist()} == recall {np.round(rec, 6).tolist()}: {np.allclose(prec, rec)}")
print(f"Q14 kappa {cohen_kappa_score(y, yp['persistence']):.10f} vs MCC {matthews_corrcoef(y, yp['persistence']):.10f}")
# does the identity survive resampling? (it should not: multiset equality is broken by resampling)
eq = sum(1 for rr in idx if abs(balanced_accuracy_score(y[rr], yp["persistence"][rr])
                                 - st.METRICS["f1_macro"](y[rr], yp["persistence"][rr])) < 1e-12)
print(f"Q14 resamples (of 1000) in which balanced_accuracy == f1_macro still holds for persistence: {eq}")

# ---- Q13: any train-side input? (grep-level evidence is in the report; here: the only non-test read)
print(f"Q13 extended_metrics.SNAPSHOT = {em.SNAPSHOT.relative_to(ROOT)} (read only for the last train label); "
      f"statistical_tests reads: {[st.INDEX_FILE] + ['test_predictions_' + k + '.csv' for k in st.MODELS + [st.BASELINE]]}")
