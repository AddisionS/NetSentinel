
from pathlib import Path
import pandas as pd

data_dir = Path("data")
csv_files = sorted(data_dir.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found inside data/")

print(f"Found {len(csv_files)} CSV files\n")

for file in csv_files:
    print("=" * 70)
    print(f"FILE: {file.name}")
    print(f"SIZE: {file.stat().st_size / (1024**2):.2f} MB")

    try:
        df = pd.read_csv(
            file,
            encoding="latin1",
            low_memory=False
        )
        df.columns = df.columns.str.strip()

        print(f"SHAPE: {df.shape}")

        if "Label" in df.columns:
            print("\nLABEL COUNTS:")
            print(df["Label"].value_counts().to_string())
        else:
            print("WARNING: Label column not found")

        print(f"\nMissing values: {df.isna().sum().sum()}")
        print(f"Infinite values: {df.select_dtypes(include='number').isin([float('inf'), float('-inf')]).sum().sum()}")

    except Exception as e:
        print(f"ERROR: {e}")

print("\nDataset inspection complete.")
