import os
import pandas as pd
import numpy as np
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# ------------------------------------------------------------
# GOLD LAYER PROCESSING (CLEAN + LINEAR REGRESSION READY)
# ------------------------------------------------------------

def calculate_rul(df):
    """Recalculate Remaining Useful Life (RUL) per engine if not already present."""
    if "RUL" in df.columns:
        return df
    window_unit = Window.partitionBy("unit")
    df = df.withColumn("max_cycle", F.max("cycle").over(window_unit))
    df = df.withColumn("RUL", F.col("max_cycle") - F.col("cycle"))
    return df.drop("max_cycle")


def create_gold_features(df):
    """Create engineered features for the Gold table."""
    print("✨ Creating engineered features...")

    window_unit = Window.partitionBy("unit")
    window_5 = Window.partitionBy("unit").orderBy("cycle").rowsBetween(-4, 0)
    window_lag = Window.partitionBy("unit").orderBy("cycle")

    # Normalized cycle
    df = df.withColumn("cycle_norm", F.col("cycle") / F.max("cycle").over(window_unit))

    # Temperature and pressure ratios
    df = df.withColumn("T_ratio_24_30", F.col("T24") / (F.col("T30") + 1e-6))
    df = df.withColumn("T_ratio_30_50", F.col("T30") / (F.col("T50") + 1e-6))
    df = df.withColumn("P_ratio_15_30", F.col("P15") / (F.col("P30") + 1e-6))
    df = df.withColumn("N_ratio_f_c", F.col("Nf") / (F.col("Nc") + 1e-6))

    # Rolling averages (smooth short-term trends)
    for col in ["T24", "T30", "T50", "P15", "P30", "Nf", "Nc"]:
        df = df.withColumn(f"{col}_roll5_mean", F.avg(F.col(col)).over(window_5))

    # Deltas (rate of change)
    for col in ["T30", "Nf", "Nc"]:
        df = df.withColumn(f"{col}_delta1", F.col(col) - F.lag(F.col(col), 1).over(window_lag))

    # Health indicator (simple relative degradation measure)
    df = df.withColumn("health_index", (F.col("T50") - F.col("T30")) / (F.col("T24") + 1e-6))

    # Add timestamp
    df = df.withColumn("processing_timestamp", F.lit(datetime.now()))
    return df


def fill_missing_values(df):
    """Fill NaN/Null values for numeric columns with column mean."""
    print("🧹 Handling missing values in numeric columns...")

    # Compute column means for numeric cols
    numeric_cols = [c for (c, t) in df.dtypes if t in ("double", "float", "int")]
    means = df.select([F.mean(F.col(c)).alias(c) for c in numeric_cols]).collect()[0].asDict()

    # Fill nulls with corresponding column means
    for c in numeric_cols:
        mean_val = means.get(c)
        if mean_val is not None:
            df = df.na.fill({c: mean_val})

    print(f"✅ Filled missing values for {len(numeric_cols)} numeric columns.")
    return df


def process_gold_table(silver_filepath, gold_directory, spark):
    """Main Gold pipeline: read, engineer features, handle NaNs, and save Feature + Label stores."""
    print("\n🟡 Processing Gold Layer...")

    # Step 1: Read Silver
    df = spark.read.parquet(silver_filepath)
    print(f"✅ Loaded Silver data: {df.count():,} rows, {len(df.columns)} columns")

    # Step 2: Compute or reuse RUL
    df = calculate_rul(df)

    # Step 3: Create engineered features
    df = create_gold_features(df)

    # Step 4: Handle missing values (from rolling/lag features)
    df = fill_missing_values(df)

    # Step 5: Define features & labels
    label_cols = ["unit", "cycle", "RUL"]
    exclude_cols = set(label_cols + ["plot_value", "processing_timestamp"])
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    feature_store = df.select(*feature_cols)
    label_store = df.select(*label_cols)

    # Step 6: Save outputs
    os.makedirs(gold_directory, exist_ok=True)
    feature_path = os.path.join(gold_directory, "feature_store.parquet")
    label_path = os.path.join(gold_directory, "label_store.parquet")
    gold_path = os.path.join(gold_directory, "gold_full.parquet")

    feature_store.write.mode("overwrite").parquet(feature_path)
    label_store.write.mode("overwrite").parquet(label_path)
    df.write.mode("overwrite").parquet(gold_path)

    print(f"✅ Feature Store saved to: {feature_path}")
    print(f"✅ Label Store saved to: {label_path}")
    print(f"✅ Full Gold saved to: {gold_path}")

    return df, feature_store, label_store
