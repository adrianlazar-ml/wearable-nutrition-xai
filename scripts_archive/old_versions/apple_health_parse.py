"""
Parser pentru Apple Health export.xml.
Extrage metrici relevante pentru disertatie si le salveaza ca CSV-uri separate.

Metrici extrase: heart_rate, hrv, steps, active_calories, basal_calories,
spo2, sleep, respiratory_rate, weight.

Rulare: py apple_health_parse.py
"""

import xml.etree.ElementTree as ET
import pandas as pd
import os

XML_PATH = "data/apple_health/export.xml"

# Tipurile de date relevante pentru disertatie
TYPES = {
    "HKQuantityTypeIdentifierHeartRate": "heart_rate",
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv",
    "HKQuantityTypeIdentifierStepCount": "steps",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "active_calories",
    "HKQuantityTypeIdentifierBasalEnergyBurned": "basal_calories",
    "HKQuantityTypeIdentifierOxygenSaturation": "spo2",
    "HKCategoryTypeIdentifierSleepAnalysis": "sleep",
    "HKQuantityTypeIdentifierRespiratoryRate": "respiratory_rate",
    "HKQuantityTypeIdentifierBodyMass": "weight",
}


def parse_apple_health(xml_path=XML_PATH):
    if not os.path.exists(xml_path):
        print(f"EROARE: Fisierul {xml_path} nu exista.")
        print("Exporta datele din Health pe iPhone si pune export.xml in data/apple_health/")
        return

    print(f"Parsez {xml_path}... (poate dura 1-2 minute)")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    count = 0
    for record in root.findall("Record"):
        rtype = record.get("type")
        if rtype in TYPES:
            records.append({
                "type": TYPES[rtype],
                "value": record.get("value"),
                "unit": record.get("unit"),
                "startDate": record.get("startDate"),
                "endDate": record.get("endDate"),
                "sourceName": record.get("sourceName"),
            })
        count += 1
        if count % 500_000 == 0:
            print(f"  ... procesat {count:,} records ({len(records):,} relevante)")

    df = pd.DataFrame(records)
    print(f"\nTotal records relevante: {len(df):,} din {count:,} procesate")
    print(df["type"].value_counts())

    # Salvare per tip de metrica
    os.makedirs("data/apple_health", exist_ok=True)
    for metric, group in df.groupby("type"):
        output_path = f"data/apple_health/{metric}.csv"
        group.to_csv(output_path, index=False)
        print(f"  Salvat: {metric}.csv ({len(group):,} records)")

    print("\nDone!")
    return df


if __name__ == "__main__":
    parse_apple_health()
