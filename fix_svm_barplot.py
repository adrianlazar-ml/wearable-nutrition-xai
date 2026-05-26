"""
fix_svm_barplot.py — Generează SHAP bar plot pentru SVM
Rulare: python fix_svm_barplot.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, shap
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings; warnings.filterwarnings("ignore")

print("Loading + training SVM...")
oura = pd.read_csv("data/oura_combined.csv")
oura["date"] = pd.to_datetime(oura["date"])
aw = pd.read_csv("apple_watch_api/apple_health_daily.csv")
aw["date"] = pd.to_datetime(aw["date"])
aw["aw_active_calories"] = (aw["aw_active_energy_kj"] / 4.184).round(1)

df = oura
for c in ["aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories",
          "aw_spo2", "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate"]:
    df = df.merge(aw[["date", c]].dropna(subset=[c]), on="date", how="left")
df = df.sort_values("date").reset_index(drop=True)

LAG = ["readiness_score", "sleep_score", "hrv_mean", "hrv_balance",
       "recovery_index", "rhr_lowest", "rhr_avg", "activity_score",
       "steps_oura", "respiratory_rate", "temp_deviation",
       "aw_hr_mean", "aw_hrv_mean", "aw_steps", "aw_active_calories", "aw_spo2",
       "aw_sleep_deep_hours", "aw_sleep_rem_hours", "aw_resp_rate"]
FEAT = [f"{f}_lag1" for f in LAG]
SHORT = [f.replace("_lag1", "").replace("aw_", "AW:") for f in FEAT]

for f in LAG:
    df[f"{f}_lag1"] = df[f].shift(1)

def lbl(r):
    if pd.isna(r["readiness_score"]) or pd.isna(r["sleep_score"]):
        return np.nan
    if r["readiness_score"] < 60 or r["sleep_score"] < 60:
        return 0
    elif r["readiness_score"] >= 85 and r["sleep_score"] >= 75:
        return 2
    return 1

df["label"] = df.apply(lbl, axis=1)
dm = df[["date"] + FEAT + ["label"]].dropna(subset=["label"]).sort_values("date").reset_index(drop=True)
X, y = dm[FEAT], dm["label"].astype(int)
si = int(len(X) * 0.8)

imp = SimpleImputer(strategy="median")
Xtr = pd.DataFrame(imp.fit_transform(X.iloc[:si]), columns=FEAT)
Xte = pd.DataFrame(imp.transform(X.iloc[si:]), columns=FEAT)
sm = SMOTE(random_state=42, k_neighbors=3)
Xts, yts = sm.fit_resample(Xtr, y.iloc[:si])
sc = StandardScaler()
Xtrs = pd.DataFrame(sc.fit_transform(Xts), columns=FEAT)
Xtes = pd.DataFrame(sc.transform(Xte), columns=FEAT)

mdl = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42, probability=True)
mdl.fit(Xtrs, yts)

print("Computing SHAP (KernelExplainer)...")
bg = shap.kmeans(Xtrs, 50)
exp = shap.KernelExplainer(mdl.predict_proba, bg)
sv = exp.shap_values(Xtes, nsamples=100)

# sv shape: (191, 19, 3) = (samples, features, classes)
sv_array = np.array(sv)
print(f"sv_array shape: {sv_array.shape}")
ms = np.abs(sv_array).mean(axis=(0, 2))  # media peste samples și clase
print(f"ms shape: {ms.shape}, SHORT len: {len(SHORT)}")

sdf = pd.DataFrame({"feature": SHORT, "mean_abs_shap": ms}).sort_values("mean_abs_shap", ascending=True)

fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(sdf["feature"], sdf["mean_abs_shap"], color="#E53935")
ax.set_xlabel("Mean |SHAP value|", fontsize=11)
ax.set_title("SHAP Feature Importance — SVM (RBF), 19 features", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/figures/json_extended_pipeline/ext_shap_bar_svm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: data/ext_shap_bar_svm.png")

print("\nTop 10 features (SVM SHAP):")
for _, r in sdf.sort_values("mean_abs_shap", ascending=False).head(10).iterrows():
    feat = r["feature"]
    val = r["mean_abs_shap"]
    print(f"  {feat:<25} {val:.4f}")
