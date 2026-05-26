import pandas as pd, json

# 1. Exemplu date integrate (1 zi per clasa)
oura = pd.read_csv('data/oura_combined.csv')
oura['date'] = pd.to_datetime(oura['date'])
aw = pd.read_csv('apple_watch_api/apple_health_daily.csv')
aw['date'] = pd.to_datetime(aw['date'])
aw['aw_active_calories'] = (aw['aw_active_energy_kj'] / 4.184).round(1)

import numpy as np
df = oura.copy()
for col in ['aw_hr_mean','aw_hrv_mean','aw_steps','aw_active_calories','aw_spo2','aw_sleep_deep_hours','aw_sleep_rem_hours','aw_resp_rate']:
    if col in aw.columns:
        df = df.merge(aw[['date',col]].dropna(subset=[col]), on='date', how='left')

def label(row):
    if pd.isna(row['readiness_score']) or pd.isna(row['sleep_score']): return np.nan
    if row['readiness_score'] < 60 or row['sleep_score'] < 60: return 'Rest'
    elif row['readiness_score'] >= 85 and row['sleep_score'] >= 75: return 'Active'
    return 'Normal'

df['label'] = df.apply(label, axis=1)
cols = ['date','readiness_score','sleep_score','hrv_mean','rhr_avg','activity_score','steps_oura','aw_hr_mean','aw_steps','aw_sleep_deep_hours','aw_sleep_rem_hours','label']
for lbl in ['Rest','Normal','Active']:
    sample = df[df['label']==lbl].dropna(subset=cols).tail(2)
    print(f'\n=== {lbl} ===')
    print(sample[cols].to_string(index=False))
