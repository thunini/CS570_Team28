import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

def convert_csv_to_parquet(
    csv_dir='data/CASAS_smart_watch/',
    out_dir='data/CASAS_smart_watch_parquet/',
    skip_existing=True
):
    """
    Converts all sw1.p*.csv files to Parquet format with debug output.
    """
    os.makedirs(out_dir, exist_ok=True)
    file_paths = [os.path.join(csv_dir, f'sw1.p{i}.csv') for i in range(1, 50)]
    file_paths = [fp for fp in file_paths if os.path.isfile(fp)]

    print(f"🔄 Converting {len(file_paths)} CSV files to Parquet...")

    for csv_path in tqdm(file_paths, desc="📂 Converting files", ncols=100):
        base_name = os.path.basename(csv_path).replace('.csv', '.parquet')
        out_path = os.path.join(out_dir, base_name)

        if skip_existing and os.path.exists(out_path):
            print(f"⚠️ Skipping existing file: {out_path}")
            continue

        try:
            print(f"\n📄 Reading: {csv_path}")
            headers = pd.read_csv(csv_path, nrows=2, header=None)
            field_names = headers.iloc[0].values.tolist()
            print(f"🧾 Column names: {field_names}")

            df = pd.read_csv(csv_path, skiprows=2, header=None, names=field_names)
            print(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns.")

            # Check timestamp format
            if 'stamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['stamp'], errors='coerce')
                print("⏱️ Parsed 'stamp' column as datetime.")
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                print("⏱️ Parsed 'timestamp' column as datetime.")
            else:
                print(f"⚠️ No timestamp column found in {csv_path}.")

            # Drop rows with invalid timestamp
            initial_rows = len(df)
            df.dropna(subset=['timestamp'], inplace=True)
            dropped = initial_rows - len(df)
            if dropped:
                print(f"🚮 Dropped {dropped} rows with invalid timestamps.")

            df.to_parquet(out_path, index=False)
            print(f"💾 Saved to: {out_path}")

        except Exception as e:
            print(f"❌ Error processing {csv_path}: {e}")

    print("✅ Done converting all files.")

convert_csv_to_parquet()