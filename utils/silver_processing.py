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
# ==============================================================
# Purpose: Transform Bronze layer (raw ingestion) into Silver layer (cleaned, unified data)
# 
# Key Operations:
# - Consolidates multiple Bronze CSV files into single unified dataset
# - Ensures globally unique engine IDs across all input files
# - Standardizes data types for all sensor measurements
# - Outputs as Parquet for efficient downstream processing
# ==============================================================

def process_silver_table(filepaths, bronze_directory, silver_directory, spark):
    """
    Process Bronze layer CSV files into a unified Silver layer dataset.
    
    The Silver layer represents cleaned, validated, and standardized data ready
    for feature engineering and analytics. This function handles multiple Bronze
    files (e.g., different datasets or time periods) and combines them while
    maintaining data integrity.
    
    Args:
        filepaths (list): List of Bronze CSV filenames/folders to process
        bronze_directory (str): Root directory containing Bronze layer data
        silver_directory (str): Output directory for Silver layer Parquet files
        spark (SparkSession): Active Spark session for distributed processing
    
    Returns:
        pyspark.sql.DataFrame: Unified Silver dataset with standardized schema
    """

    datasets = []  # Stores individual dataframes before combining
    offset = 0     # Running counter to ensure unique engine IDs across files

    # ==================== STEP 1: Load and Combine Bronze Files ====================
    for i, path in enumerate(filepaths, start=1):
        full_path = os.path.join(bronze_directory, path)

        # Handle Spark's folder-based CSV output structure
        # Spark often creates a folder with part-*.csv files inside
        if os.path.isdir(full_path):
            csv_files = glob.glob(os.path.join(full_path, "part-*.csv"))
            if not csv_files:
                print(f"⚠️ No CSV found inside {full_path}")
                continue
            full_path = csv_files[0]  # Use first part file (coalesce(1) should create only one)

        # Read CSV using Pandas for easier manipulation
        df = pd.read_csv(full_path)
        n_engines = df["unit"].nunique()
        print(f"📂 Loaded {path} → {n_engines} engines, {len(df)} rows")

        # ==================== STEP 2: Ensure Unique Engine IDs ====================
        # If combining multiple datasets (e.g., train_FD001, train_FD002),
        # shift unit IDs to prevent collisions. 
        # Example: File 1 has units 1-100, File 2's units become 101-200
        df["unit"] = df["unit"] + offset
        offset += n_engines  # Increment offset for next file
        
        datasets.append(df)

    # ==================== STEP 3: Combine All Datasets ====================
    # Concatenate all dataframes vertically (stack rows)
    combined = pd.concat(datasets, ignore_index=True)
    print(f"✅ Combined Silver dataset → {combined['unit'].nunique()} engines, {len(combined)} rows")

    # ==================== STEP 4: Convert to Spark DataFrame ====================
    # Convert to Spark for scalable processing and Parquet writing
    df = spark.createDataFrame(combined)

    # ==================== STEP 5: Enforce Data Type Schema ====================
    # Define expected data types for all columns to ensure consistency
    # - unit & cycle: Integer identifiers
    # - All sensor readings and operational settings: Float for precision
    column_type_map = {
        'unit': IntegerType(),           # Engine identifier
        'cycle': IntegerType(),          # Time cycle / operational sequence
        'op_setting_1': FloatType(),     # Operational setting 1
        'op_setting_2': FloatType(),     # Operational setting 2
        'op_setting_3': FloatType(),     # Operational setting 3
        'T2': FloatType(),               # Temperature sensor 2
        'T24': FloatType(),              # Temperature sensor 24
        'T30': FloatType(),              # Temperature sensor 30
        'T50': FloatType(),              # Temperature sensor 50
        'P2': FloatType(),               # Pressure sensor 2
        'P15': FloatType(),              # Pressure sensor 15
        'P30': FloatType(),              # Pressure sensor 30
        'Nf': FloatType(),               # Fan speed
        'Nc': FloatType(),               # Core speed
        'epr': FloatType(),              # Engine pressure ratio
        'Ps30': FloatType(),             # Static pressure at station 30
        'phi': FloatType(),              # Ratio of fuel flow to Ps30
        'NRf': FloatType(),              # Corrected fan speed
        'NRc': FloatType(),              # Corrected core speed
        'BPR': FloatType(),              # Bypass ratio
        'farB': FloatType(),             # Burner fuel-air ratio
        'htBleed': FloatType(),          # Bleed enthalpy
        'Nf_dmd': FloatType(),           # Demanded fan speed
        'PCNfR_dmd': FloatType(),        # Demanded corrected fan speed
        'W31': FloatType(),              # HPT coolant bleed
        'W32': FloatType()               # LPT coolant bleed
    }

    # Cast each column to its specified type
    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))

    # ==================== STEP 6: Save to Silver Layer ====================
    # Create output directory if it doesn't exist
    os.makedirs(silver_directory, exist_ok=True)

    # Write as Parquet for efficient storage and fast querying
    # Parquet is columnar format, ideal for analytics workloads
    filename = os.path.join(silver_directory, "silver_feature.parquet")
    df.write.mode("overwrite").parquet(filename)

    # ==================== STEP 7: Verification ====================
    print(f"💾 Saved Silver dataset to: {filename}")
    print(f"✅ Silver → {df.select('unit').distinct().count()} engines, {df.count()} rows")

    return df