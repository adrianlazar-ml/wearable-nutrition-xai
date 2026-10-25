"""recompute_f2.py - AUDIT F2 (T3): independent recomputation of T1.1 / T1.2 / T1.2b.
Imports only numpy, pandas, scipy, sklearn, statsmodels. Nothing from revision/scripts/.
Inputs : revision/results/test_predictions_{rf,xgb,svm,persistence}.csv,
         revision/results/bootstrap_indices_1000x192_seed42.npy (comparison only; the
         matrix used below is regenerated with np.random.default_rng(42)),
         revision/results/t11_extended_metrics.csv, t12_mcnemar.csv,
         t12_paired_bootstrap.csv, t12_cochran_q.csv (claims to compare against).
Run from repo root: python revision/audit/recompute_f2.py
Every comparison is at 4 decimals (round(x, 4) == round(claim, 4)).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (auc, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.preprocessing import label_binarize
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "revision" / "results"
L = [0, 1, 2]
PCOLS = ["p_rest", "p_normal", "p_active"]
N_BOOT, N_TEST, SEED = 1000, 192, 42

results: list[tuple[str, bool, str]] = []


def cmp(name: str, got: float, claim: float) -> None:
    ok = round(float(got), 4) == round(float(claim), 4)
    results.append((name, ok, f"got {got:.6f} vs claim {claim:.4f}"))
    print(f"{'PASS' if ok else 'FAIL'} {name}: recomputed {got:.6f} | claimed {claim:.4f}")


# ----------------------------------------------------------------------------- inputs
pred = {k: pd.read_csv(RES / f"test_predictions_{k}.csv") for k in ["rf", "xgb", "svm", "persistence"]}
y = pred["rf"]["y_true"].to_numpy(int)
for k, df in pred.items():
    assert len(df) == N_TEST and np.array_equal(df["y_true"].to_numpy(int), y), k
yp = {k: df["y_pred"].to_numpy(int) for k, df in pred.items()}
P = {k: pred[k][PCOLS].to_numpy(float) for k in ["rf", "xgb", "svm"]}
print("rows", N_TEST, "| support Rest/Normal/Active", [int((y == c).sum()) for c in L])

t11 = pd.read_csv(RES / "t11_extended_metrics.csv").set_index(["system", "metric"])
t12m = pd.read_csv(RES / "t12_mcnemar.csv").set_index("pair")
t12p = pd.read_csv(RES / "t12_paired_bootstrap.csv").set_index(["pair", "metric"])
t12q = pd.read_csv(RES / "t12_cochran_q.csv").iloc[0]

# ----------------------------------------------------------------------------- bootstrap matrix
idx = np.random.default_rng(SEED).integers(0, N_TEST, size=(N_BOOT, N_TEST))
saved = np.load(RES / "bootstrap_indices_1000x192_seed42.npy")
same = np.array_equal(idx, saved)
results.append(("bootstrap matrix regenerated == saved npy", same, f"shape {idx.shape}, dtype {idx.dtype}"))
print(f"{'PASS' if same else 'FAIL'} regenerated default_rng(42) matrix == saved npy ({idx.shape}, {idx.dtype})")
n_missing = int(sum(len(np.unique(y[r])) < 3 for r in idx))
print("resamples with a class absent from y_true:", n_missing)


# ----------------------------------------------------------------------------- metric definitions (manual)
def recalls(yt, yp_):
    cm = confusion_matrix(yt, yp_, labels=L)
    sup = cm.sum(axis=1)
    if (sup == 0).any():
        return None
    return np.diag(cm) / sup


def bal_acc(yt, yp_):
    r = recalls(yt, yp_)
    return np.nan if r is None else float(r.mean())


def kappa_manual(yt, yp_):
    cm = confusion_matrix(yt, yp_, labels=L).astype(float)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = float((cm.sum(axis=1) * cm.sum(axis=0)).sum()) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def gmean(yt, yp_):
    r = recalls(yt, yp_)
    return np.nan if r is None else float(np.prod(r) ** (1 / 3))


def roc_manual(yt, pr):
    if len(np.unique(yt)) < 3:
        return np.nan
    return float(np.mean([roc_auc_score((yt == c).astype(int), pr[:, i]) for i, c in enumerate(L)]))


def ap_manual(yt, pr):
    if len(np.unique(yt)) < 3:
        return np.nan
    return float(np.mean([average_precision_score((yt == c).astype(int), pr[:, i]) for i, c in enumerate(L)]))


def prauc_trapezoid(yt, pr):
    """Diagnostic only (Q5 naming): area under the interpolated PR curve, macro."""
    out = []
    for i, c in enumerate(L):
        p_, r_, _ = precision_recall_curve((yt == c).astype(int), pr[:, i])
        out.append(auc(r_, p_))
    return float(np.mean(out))


def f1m(yt, yp_):
    return float(f1_score(yt, yp_, labels=L, average="macro", zero_division=0))


def ci(fn, *arrays):
    vals = []
    for r in idx:
        v = fn(*(a[r] for a in arrays))
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), N_BOOT - len(vals)


# ----------------------------------------------------------------------------- T1.1: RF point + CI
print("\n== T1.1 RF, five extended metrics (point and 95% percentile CI, 1000 x 192, seed 42) ==")
rf_metrics = {
    "balanced_accuracy": (bal_acc, (y, yp["rf"])),
    "cohen_kappa": (kappa_manual, (y, yp["rf"])),
    "g_mean": (gmean, (y, yp["rf"])),
    "roc_auc_ovr_macro": (roc_manual, (y, P["rf"])),
    "pr_auc_macro": (ap_manual, (y, P["rf"])),
}
for m, (fn, arrs) in rf_metrics.items():
    point = fn(*arrs)
    lo, hi, skipped = ci(fn, *arrs)
    row = t11.loc[("rf", m)]
    cmp(f"RF {m} point", point, row["value"])
    cmp(f"RF {m} ci_low", lo, row["ci_low"])
    cmp(f"RF {m} ci_high", hi, row["ci_high"])
    if skipped:
        print(f"   note: {skipped} resamples skipped for {m}")

# cross-checks of the manual definitions against sklearn on the full test set
print("\n-- definition cross-checks on the full test set (RF)")
print(f"   sklearn balanced_accuracy_score {balanced_accuracy_score(y, yp['rf']):.10f} vs manual mean recall {bal_acc(y, yp['rf']):.10f}")
print(f"   sklearn cohen_kappa_score(weights=None) {cohen_kappa_score(y, yp['rf']):.10f} vs manual {kappa_manual(y, yp['rf']):.10f}")
print(f"   sklearn cohen_kappa_score(weights='quadratic') {cohen_kappa_score(y, yp['rf'], weights='quadratic'):.10f}  (QWK, must NOT equal the kappa row)")
print(f"   sklearn roc_auc_score(multi_class='ovr', average='macro') {roc_auc_score(y, P['rf'], multi_class='ovr', average='macro'):.10f} vs manual per-class mean {roc_manual(y, P['rf']):.10f}")
yb = label_binarize(y, classes=L)
print(f"   sklearn average_precision_score(macro) {average_precision_score(yb, P['rf'], average='macro'):.10f} vs manual per-class mean {ap_manual(y, P['rf']):.10f}")
print(f"   DIAGNOSTIC (naming, Q5): trapezoidal area under interpolated PR curve, macro = {prauc_trapezoid(y, P['rf']):.6f} "
      f"(XGB {prauc_trapezoid(y, P['xgb']):.6f}, SVM {prauc_trapezoid(y, P['svm']):.6f}); AP macro RF {ap_manual(y, P['rf']):.6f}")
# column-order sensitivity (Q4): swap p_normal and p_active
Pswap = P["rf"][:, [0, 2, 1]]
print(f"   ROC-AUC with columns [p_rest, p_active, p_normal] = {roc_manual(y, Pswap):.6f} (order matters; csv order gives the claimed value)")
print(f"   argmax(proba) == y_pred: RF {(P['rf'].argmax(1) == yp['rf']).mean():.4f}, XGB {(P['xgb'].argmax(1) == yp['xgb']).mean():.4f}, SVM {(P['svm'].argmax(1) == yp['svm']).mean():.4f}")

# ----------------------------------------------------------------------------- T1.2: McNemar
print("\n== T1.2 McNemar ==")


def counts(a, b):
    ca, cb = yp[a] == y, yp[b] == y
    return int((ca & ~cb).sum()), int((~ca & cb).sum()), int((ca & cb).sum()), int((~ca & ~cb).sum())


def mcnemar_manual(n01, n10):
    chi2_cc = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_cc = float(stats.chi2.sf(chi2_cc, 1))
    p_exact = float(stats.binomtest(min(n01, n10), n01 + n10, 0.5, alternative="two-sided").pvalue)
    return chi2_cc, p_cc, p_exact


pairs = [("rf", "xgb", "RF-XGB"), ("rf", "svm", "RF-SVM"), ("xgb", "svm", "XGB-SVM"), ("rf", "persistence", "RF-persistence")]
p_cc_fam, p_ex_fam = [], []
for a, b, name in pairs:
    n01, n10, n11, n00 = counts(a, b)
    chi2_cc, p_cc, p_exact = mcnemar_manual(n01, n10)
    row = t12m.loc[name]
    print(f"{name}: n01 (A correct, B wrong) {n01}, n10 {n10}, n11 {n11}, n00 {n00} | csv n01 {row.n01} n10 {row.n10} n11 {row.n11} n00 {row.n00}")
    ok = (n01, n10, n11, n00) == (row.n01, row.n10, row.n11, row.n00)
    results.append((f"McNemar {name} counts", ok, f"{(n01, n10, n11, n00)} vs csv {(row.n01, row.n10, row.n11, row.n00)}"))
    print(f"{'PASS' if ok else 'FAIL'} {name} counts")
    cmp(f"McNemar {name} chi2_cc (manual formula)", chi2_cc, row.chi2_cc)
    cmp(f"McNemar {name} p_cc (scipy chi2.sf)", p_cc, row.p_cc)
    cmp(f"McNemar {name} p_exact (scipy binomtest)", p_exact, row.p_exact)
    sm = mcnemar([[n11, n01], [n10, n00]], exact=False, correction=True)
    sm_t = mcnemar([[n11, n10], [n01, n00]], exact=False, correction=True)   # transposed discordant cells
    print(f"   statsmodels p_cc {sm.pvalue:.6f}; with n01/n10 swapped {sm_t.pvalue:.6f} (statistic symmetric in n01, n10)")
    if name != "RF-persistence":
        p_cc_fam.append(p_cc)
        p_ex_fam.append(p_exact)
    rec = "exact" if n01 + n10 < 25 else "cc"
    ok = rec == row.recommended
    results.append((f"McNemar {name} recommended", ok, f"{rec} vs csv {row.recommended} (discordant {n01 + n10})"))
    print(f"{'PASS' if ok else 'FAIL'} {name} recommended = {rec} (discordant {n01 + n10})")


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, p[i] * (m - rank))
        adj[i] = min(1.0, running)
    return adj


print("\n-- Holm (manual) on the 3 model pairs")
h_cc, h_ex = holm(p_cc_fam), holm(p_ex_fam)
h_sm = multipletests(p_cc_fam, method="holm")[1]
for i, (_, _, name) in enumerate(pairs[:3]):
    cmp(f"Holm p_cc {name}", h_cc[i], t12m.loc[name].p_cc_holm)
    cmp(f"Holm p_exact {name}", h_ex[i], t12m.loc[name].p_exact_holm)
print("   statsmodels multipletests(holm) on p_cc:", np.round(h_sm, 6), "| manual:", np.round(h_cc, 6))

# ----------------------------------------------------------------------------- T1.2: Cochran's Q
print("\n== T1.2 Cochran's Q (manual formula) ==")
X = np.column_stack([yp[k] == y for k in ["rf", "xgb", "svm"]]).astype(int)
k = X.shape[1]
C = X.sum(axis=0)
R = X.sum(axis=1)
N = X.sum()
Q = (k - 1) * (k * (C ** 2).sum() - N ** 2) / (k * N - (R ** 2).sum())
pQ = float(stats.chi2.sf(Q, k - 1))
print(f"matrix {X.shape}; column totals (correct days) RF/XGB/SVM = {C.tolist()}")
cmp("Cochran Q statistic", Q, t12q.statistic)
cmp("Cochran Q p", pQ, t12q.p)
results.append(("Cochran Q df", int(t12q.df) == k - 1, f"df csv {int(t12q.df)} vs k-1 = {k - 1}"))
print(f"{'PASS' if int(t12q.df) == k - 1 else 'FAIL'} Cochran df = {int(t12q.df)} (k-1 = {k - 1})")
sq = cochrans_q(X)
print(f"   statsmodels cochrans_q: {sq.statistic:.6f}, df {sq.df}, p {sq.pvalue:.6f}")

# ----------------------------------------------------------------------------- T1.2 / T1.2b: paired bootstrap
print("\n== T1.2 / T1.2b paired bootstrap (same row of indices on y_true, y_pred_A, y_pred_B) ==")


def paired(a, b, fn):
    d, va, vb = [], [], []
    for r in idx:
        yt = y[r]
        if len(np.unique(yt)) < 3:
            continue
        ma, mb = fn(yt, yp[a][r]), fn(yt, yp[b][r])
        va.append(ma)
        vb.append(mb)
        d.append(ma - mb)
    d = np.asarray(d)
    return (fn(y, yp[a]) - fn(y, yp[b]), *np.percentile(d, [2.5, 97.5]), float((d > 0).mean()),
            *np.percentile(va, [2.5, 97.5]), *np.percentile(vb, [2.5, 97.5]))


for a, b, name, m, fn in [("rf", "xgb", "RF-XGB", "f1_macro", f1m),
                          ("rf", "persistence", "RF-persistence", "f1_macro", f1m),
                          ("rf", "persistence", "RF-persistence", "balanced_accuracy", bal_acc)]:
    pt, lo, hi, prop, alo, ahi, blo, bhi = paired(a, b, fn)
    row = t12p.loc[(name, m)]
    cmp(f"paired {name} {m} diff_point", pt, row.diff_point)
    cmp(f"paired {name} {m} ci_low", lo, row.ci_low)
    cmp(f"paired {name} {m} ci_high", hi, row.ci_high)
    cmp(f"paired {name} {m} prop_diff_gt_0", prop, row.prop_diff_gt_0)
    cmp(f"paired {name} {m} marginal A low (vs t11 {a})", alo, t11.loc[(a, m)]["ci_low"])
    cmp(f"paired {name} {m} marginal A high (vs t11 {a})", ahi, t11.loc[(a, m)]["ci_high"])
    cmp(f"paired {name} {m} marginal B low (vs t11 {b})", blo, t11.loc[(b, m)]["ci_low"])
    cmp(f"paired {name} {m} marginal B high (vs t11 {b})", bhi, t11.loc[(b, m)]["ci_high"])

# ----------------------------------------------------------------------------- persistence / majority identities (T0.4 note)
print("\n== persistence / majority identities ==")
cmp("persistence balanced_accuracy", bal_acc(y, yp["persistence"]), 0.3039)
cmp("persistence f1_macro", f1m(y, yp["persistence"]), 0.3039)
cmp("persistence kappa", kappa_manual(y, yp["persistence"]), -0.0566)
cmp("persistence g_mean", gmean(y, yp["persistence"]), 0.0)
maj = np.ones(N_TEST, int)
cmp("majority balanced_accuracy", bal_acc(y, maj), 1 / 3)
cmp("majority kappa", kappa_manual(y, maj), 0.0)
cmp("majority g_mean", gmean(y, maj), 0.0)
pc = [int((yp["persistence"] == c).sum()) for c in L]
print(f"persistence predicted counts {pc} vs true support {[int((y == c).sum()) for c in L]}")

# ----------------------------------------------------------------------------- verdict
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{len(results) - n_fail} / {len(results)} comparisons identical at 4 decimals; {n_fail} FAIL")
for name, ok, detail in results:
    if not ok:
        print("  FAIL", name, detail)
print("T3 VERDICT:", "PASS" if n_fail == 0 else "FAIL")
