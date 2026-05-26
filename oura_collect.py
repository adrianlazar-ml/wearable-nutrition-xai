"""
Colectare date din Oura Ring API v2.
Trage date din endpoints: daily_sleep, daily_readiness, daily_activity,
daily_stress, sleep (detaliat cu HRV, RHR, respiratory rate).

Rulare: py oura_collect.py
"""

import os
from datetime import datetime
import requests
import pandas as pd
from dotenv import load_dotenv


def collect_oura_data(start_date="2025-07-01", end_date=None):
    load_dotenv()

    token = os.getenv("OURA_ACCESS_TOKEN")
    if not token:
        print("EROARE: OURA_ACCESS_TOKEN nu e setat in .env. Ruleaza oura_auth.py.")
        return

    headers = {"Authorization": f"Bearer {token}"}
    base_url = "https://api.ouraring.com/v2/usercollection"

    # FIX: END_DATE dinamic daca nu e specificat
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    def fetch(endpoint):
        r = requests.get(
            f"{base_url}/{endpoint}",
            headers=headers,
            params={"start_date": start_date, "end_date": end_date}
        )
        if r.status_code != 200:
            print(f"  EROARE {endpoint}: {r.status_code} - {r.text[:200]}")
            return []
        return r.json().get("data", [])

    print(f"Colectez date din Oura Ring ({start_date} -> {end_date})...")

    # Daily endpoints
    sleep_df = pd.DataFrame(fetch("daily_sleep"))
    readiness_df = pd.DataFrame(fetch("daily_readiness"))
    activity_df = pd.DataFrame(fetch("daily_activity"))
    stress_df = pd.DataFrame(fetch("daily_stress"))

    # Sleep detaliat - contine average_hrv, lowest_heart_rate, average_breath
    sleep_detailed = pd.DataFrame(fetch("sleep"))

    # Agregare metrici per zi din sleep detaliat
    sleep_metrics = pd.DataFrame()
    if not sleep_detailed.empty:
        sleep_detailed["date"] = pd.to_datetime(sleep_detailed["day"])
        sleep_metrics = sleep_detailed.groupby("day").agg(
            average_hrv=("average_hrv", "mean"),
            lowest_heart_rate=("lowest_heart_rate", "min"),
            average_breath=("average_breath", "mean"),
            average_heart_rate=("average_heart_rate", "mean")
        ).reset_index()
        sleep_metrics.rename(columns={"day": "date"}, inplace=True)

    # Salvare
    os.makedirs("data", exist_ok=True)
    sleep_df.to_csv("data/sleep.csv", index=False)
    readiness_df.to_csv("data/readiness.csv", index=False)
    activity_df.to_csv("data/activity.csv", index=False)
    stress_df.to_csv("data/stress.csv", index=False)
    sleep_detailed.to_csv("data/sleep_detailed.csv", index=False)

    if not sleep_metrics.empty:
        sleep_metrics.to_csv("data/sleep_metrics.csv", index=False)

    print(f"  Sleep daily:    {len(sleep_df)} zile")
    print(f"  Readiness:      {len(readiness_df)} zile")
    print(f"  Activity:       {len(activity_df)} zile")
    print(f"  Stress:         {len(stress_df)} zile")
    print(f"  Sleep detaliat: {len(sleep_detailed)} sesiuni")
    if not sleep_metrics.empty:
        print(f"  Sleep metrics:  {len(sleep_metrics)} zile")
    print("Done.")


if __name__ == "__main__":
    collect_oura_data()
