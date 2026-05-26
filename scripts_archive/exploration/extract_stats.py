import pandas as pd, numpy as np

oura = pd.read_csv('data/oura_combined.csv')
oura['date'] = pd.to_datetime(oura['date'])
aw = pd.read_csv('apple_watch_api/apple_health_daily.csv')
aw['date'] = pd.to_datetime(aw['date'])
aw['aw_active_calories'] = (aw['aw_active_energy_kj'] / 4.184).round(1)

df = oura.copy()
for col in ['aw_hr_mean','aw_hrv_mean','aw_steps','aw_active_calories','aw_spo2','aw_sleep_deep_hours','aw_sleep_rem_hours','aw_resp_rate']:
    if col in aw.columns:
        df = df.merge(aw[['date',col]].dropna(subset=[col]), on='date', how='left')

feats = ['readiness_score','sleep_score','hrv_mean','hrv_balance','recovery_index','rhr_lowest','rhr_avg','activity_score','steps_oura','respiratory_rate','temp_deviation','aw_hr_mean','aw_hrv_mean','aw_steps','aw_active_calories','aw_spo2','aw_sleep_deep_hours','aw_sleep_rem_hours','aw_resp_rate']
print(df[feats].describe().round(2).to_string())
