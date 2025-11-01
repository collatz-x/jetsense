import os
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from sklearn.preprocessing import StandardScaler

def process_gold_table(silver_input, gold_directory, spark):
    """
    Builds the Gold dataset with:
      - RUL computation
      - Feature engineering (rolling, delta, ratios)
      - Normalization (Z-score)
      - Saves full table, feature store, and label store as CSV (no pyarrow needed)
    """

    print("🚀 Processing Gold layer...")

    # -------------------------------------------------------------------------
    # STEP 1: Load Silver
    # -------------------------------------------------------------------------
    if isinstance(silver_input, str):
        if silver_input.endswith(".parquet"):
            df = spark.read.parquet(silver_input)
        elif silver_input.endswith(".csv"):
            df = spark.read.option("header", True).csv(silver_input, inferSchema=True)
        else:
            raise ValueError("Unsupported file type for silver_input")
    else:
        df = silver_input

    # -------------------------------------------------------------------------
    # STEP 2: Compute Remaining Useful Life (RUL)
    # -------------------------------------------------------------------------
    w = Window.partitionBy("unit")
    max_cycle = df.groupBy("unit").agg(F.max("cycle").alias("max_cycle"))
    df = df.join(max_cycle, on="unit", how="left")
    df = df.withColumn("RUL", F.col("max_cycle") - F.col("cycle"))
    df = df.drop("max_cycle")

    # -------------------------------------------------------------------------
    # STEP 3: Convert to pandas for feature engineering
    # -------------------------------------------------------------------------
    pdf = df.toPandas()
    print(f"✅ Loaded Silver → {len(pdf):,} rows, {pdf['unit'].nunique()} engines")

    # Rolling means & deltas
    sensors = ["T24", "T30", "T50", "P15", "P30", "Nf", "Nc"]
    for s in sensors:
        pdf[f"{s}_roll5_mean"] = (
            pdf.groupby("unit")[s].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
        )
        pdf[f"{s}_delta1"] = pdf.groupby("unit")[s].diff().fillna(0)

    # Ratios & health index
    pdf["T_ratio_24_30"] = pdf["T24"] / (pdf["T30"] + 1e-6)
    pdf["T_ratio_30_50"] = pdf["T30"] / (pdf["T50"] + 1e-6)
    pdf["P_ratio_15_30"] = pdf["P15"] / (pdf["P30"] + 1e-6)
    pdf["N_ratio_f_c"]   = pdf["Nf"] / (pdf["Nc"]  + 1e-6)
    pdf["cycle_norm"]    = pdf.groupby("unit")["cycle"].transform(lambda x: x / x.max())
    pdf["health_index"]  = 1 - pdf["cycle_norm"]

    # -------------------------------------------------------------------------
    # STEP 4: Normalize numeric features (Z-score)
    # -------------------------------------------------------------------------
    numeric_cols = pdf.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["unit", "cycle", "RUL"]
    cols_to_scale = [c for c in numeric_cols if c not in exclude_cols]

    scaler = StandardScaler()
    pdf_scaled = pdf.copy()
    pdf_scaled[cols_to_scale] = scaler.fit_transform(pdf[cols_to_scale])

    # -------------------------------------------------------------------------
    # STEP 5: Create feature & label stores
    # -------------------------------------------------------------------------
    feature_store = pdf_scaled.drop(columns=["RUL"])
    label_store = pdf_scaled[["unit", "cycle", "RUL"]]

    # -------------------------------------------------------------------------
    # STEP 6: Save all outputs as CSVs (no pyarrow dependency)
    # -------------------------------------------------------------------------
    os.makedirs(gold_directory, exist_ok=True)
    gold_full_path = os.path.join(gold_directory, "gold_full.csv")
    feature_path = os.path.join(gold_directory, "feature_store.csv")
    label_path = os.path.join(gold_directory, "label_store.csv")
    meta_path = os.path.join(gold_directory, "feature_metadata.csv")

    pdf_scaled.to_csv(gold_full_path, index=False)
    feature_store.to_csv(feature_path, index=False)
    label_store.to_csv(label_path, index=False)

    # Save normalization metadata
    pd.DataFrame({
        "feature": cols_to_scale,
        "mean": scaler.mean_,
        "std": scaler.scale_
    }).to_csv(meta_path, index=False)

    print(f"💾 Saved Gold full table      → {gold_full_path}")
    print(f"💾 Saved Feature Store        → {feature_path}")
    print(f"💾 Saved Label Store          → {label_path}")
    print(f"💾 Saved normalization config → {meta_path}")

    return pdf_scaled, feature_store, label_store