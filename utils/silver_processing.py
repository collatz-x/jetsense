import os
import glob
import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StringType, IntegerType, FloatType, DateType, BooleanType, DoubleType
)

# ==============================================================
# SILVER LAYER PROCESSING
# Combines all Bronze CSVs, ensures unique engine IDs, fixes schema, saves to Parquet
# ==============================================================

def process_silver_table(filepaths, bronze_directory, silver_directory, spark):
    """
    Processes Bronze CSVs into a unified Silver-level dataset:
    - Ensures consistent schema
    - Maintains unique engine IDs across files (if multiple)
    - Saves cleaned dataset as Parquet for Gold
    """

    datasets = []
    offset = 0

    # Combine all listed Bronze CSVs
    for i, path in enumerate(filepaths, start=1):
        full_path = os.path.join(bronze_directory, path)

        # Handle both folder-based and direct CSV paths (Spark output fix)
        if os.path.isdir(full_path):
            csv_files = glob.glob(os.path.join(full_path, "part-*.csv"))
            if not csv_files:
                print(f"⚠️ No CSV found inside {full_path}")
                continue
            full_path = csv_files[0]

        # Read using pandas to preserve structure
        df = pd.read_csv(full_path)
        n_engines = df["unit"].nunique()
        print(f"📂 Loaded {path} → {n_engines} engines, {len(df)} rows")

        # If combining multiple subsets, shift unit IDs to keep unique
        df["unit"] = df["unit"] + offset
        offset += n_engines
        datasets.append(df)

    # Combine all into one dataframe
    combined = pd.concat(datasets, ignore_index=True)
    print(f"✅ Combined Silver dataset → {combined['unit'].nunique()} engines, {len(combined)} rows")

    # Convert to Spark DataFrame
    df = spark.createDataFrame(combined)

    # Enforce schema consistency
    column_type_map = {
        'unit': IntegerType(),
        'cycle': IntegerType(),
        'op_setting_1': FloatType(),
        'op_setting_2': FloatType(),
        'op_setting_3': FloatType(),
        'T2': FloatType(),
        'T24': FloatType(),
        'T30': FloatType(),
        'T50': FloatType(),
        'P2': FloatType(),
        'P15': FloatType(),
        'P30': FloatType(),
        'Nf': FloatType(),
        'Nc': FloatType(),
        'epr': FloatType(),
        'Ps30': FloatType(),
        'phi': FloatType(),
        'NRf': FloatType(),
        'NRc': FloatType(),
        'BPR': FloatType(),
        'farB': FloatType(),
        'htBleed': FloatType(),
        'Nf_dmd': FloatType(),
        'PCNfR_dmd': FloatType(),
        'W31': FloatType(),
        'W32': FloatType()
    }

    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))

    # Create silver directory if needed
    os.makedirs(silver_directory, exist_ok=True)

    # Save unified silver parquet
    filename = os.path.join(silver_directory, "silver_feature.parquet")
    df.write.mode("overwrite").parquet(filename)

    print(f"💾 Saved Silver dataset to: {filename}")
    print(f"✅ Silver → {df.select('unit').distinct().count()} engines, {df.count()} rows")

    return df
