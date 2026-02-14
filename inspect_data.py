
import pandas as pd
import os
import config

print("Inspecting annot.parquet...")
try:
    df_annot = pd.read_parquet(config.ANNOT_FILE)
    print(df_annot.head())
    print(df_annot.columns)
    print(f"Shape: {df_annot.shape}")
except Exception as e:
    print(f"Error reading annot: {e}")

print("\nInspecting img.parquet...")
try:
    df_img = pd.read_parquet(config.IMG_FILE)
    print(df_img.head())
    print(df_img.columns)
    print(f"Shape: {df_img.shape}")
except Exception as e:
    print(f"Error reading immg: {e}")
