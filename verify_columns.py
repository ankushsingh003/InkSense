
import pandas as pd
import config
import os

with open("columns.txt", "w") as f:
    try:
        df_annot = pd.read_parquet(config.ANNOT_FILE)
        f.write(f"ANNOT_COLUMNS: {list(df_annot.columns)}\n")
        f.write(f"ANNOT_SAMPLE: {df_annot.iloc[0].to_dict()}\n")
    except Exception as e:
        f.write(f"ANNOT_ERROR: {e}\n")

    try:
        df_img = pd.read_parquet(config.IMG_FILE)
        f.write(f"IMG_COLUMNS: {list(df_img.columns)}\n")
        f.write(f"IMG_SAMPLE: {df_img.iloc[0].to_dict()}\n")
    except Exception as e:
        f.write(f"IMG_ERROR: {e}\n")
