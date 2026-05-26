import pandas as pd, hashlib

paths = [
    "data/oura_combined.csv",
    "apple_watch_api/apple_health_daily.csv",
    "models/dataset_snapshot_20260331.csv",
]

for path in paths:
    print(f"=== {path} ===")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  EROARE citire: {e}")
        print()
        continue
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"NA per col:\n{df.isna().sum()}")
    if "date" in df.columns:
        print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
        print(f"Duplicates on date: {df['date'].duplicated().sum()}")
    print(f"Head(3):\n{df.head(3).to_string()}")
    print(f"Describe(numerice):\n{df.describe().to_string()}")
    with open(path, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()[:16]
    print(f"SHA256 (16): {h}")
    print()
