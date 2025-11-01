import os
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def process_bronze_table(filepath, bronze_directory, spark):
    """
    Process raw sensor data files into Bronze layer (raw ingestion layer).
    
    This function reads turbofan engine sensor data from text files, converts it to
    a standardized format, and stores it in the Bronze layer for further processing.
    Expected to handle data with 600 unique engine units.
    
    Args:
        filepath (str): Path to the raw input data file
        bronze_directory (str): Directory path for Bronze layer output
        spark (SparkSession): Active Spark session for distributed processing
    
    Returns:
        pyspark.sql.DataFrame: Processed data as Spark DataFrame
    """
    
    # Define column names for the turbofan engine sensor dataset
    # Includes operational settings (op_setting_*) and various sensor measurements
    # (temperature T*, pressure P*, speed N*, etc.)
    header_names = [
        "unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3",
        "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
        "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
        "Nf_dmd", "PCNfR_dmd", "W31", "W32"
    ]

    # ==================== STEP 1: Read Raw File ====================
    # Read space-delimited raw file into Pandas DataFrame
    # The raw files have no headers, so we assign column names manually
    print(f"\n🔍 Reading raw file: {filepath}")
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=header_names)
    
    # Log statistics about the raw input data
    raw_engine_count = df['unit'].nunique()
    raw_row_count = len(df)
    
    print(f"📊 RAW FILE STATS:")
    print(f"   Total Rows: {raw_row_count:,}")
    print(f"   Unique Engines: {raw_engine_count}")
    print(f"   Engine IDs range: {df['unit'].min()} to {df['unit'].max()}")

    # ==================== STEP 2: Convert to Spark ====================
    # Convert Pandas DataFrame to Spark DataFrame for scalable processing
    spark_df = spark.createDataFrame(df)
    
    # Verify data integrity after conversion
    spark_engine_count = spark_df.select('unit').distinct().count()
    spark_row_count = spark_df.count()
    
    print(f"📊 AFTER SPARK CONVERSION:")
    print(f"   Total Rows: {spark_row_count:,}")
    print(f"   Unique Engines: {spark_engine_count}")

    # ==================== STEP 3: Prepare Output Path ====================
    # Extract base filename without extension and create Bronze layer path
    # Example: "train_FD001.txt" becomes "bronze_train_FD001"
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    bronze_path = os.path.join(bronze_directory, f"bronze_{base_name}")

    # Create Bronze directory if it doesn't exist
    os.makedirs(bronze_directory, exist_ok=True)

    print(f"💾 Writing to: {bronze_path}")

    # ==================== STEP 4: Write to Bronze Layer ====================
    # Write Spark DataFrame to CSV in Bronze layer
    # coalesce(1) ensures single output file for easier verification
    # mode("overwrite") replaces existing data if present
    spark_df.coalesce(1).write.mode("overwrite").option("header", True).csv(bronze_path)

    # ==================== STEP 5: Verification ====================
    # Read back the written file to verify data integrity
    written_df = spark.read.option("header", True).csv(bronze_path)
    written_engine_count = written_df.select('unit').distinct().count()
    written_row_count = written_df.count()
    
    print(f"✅ VERIFICATION:")
    print(f"   Written Rows: {written_row_count:,}")
    print(f"   Written Engines: {written_engine_count}")
    
    # Alert if the expected number of engines (600) doesn't match
    if written_engine_count != 600:
        print(f"   ⚠️ WARNING: Expected 600 engines, got {written_engine_count}!")
    
    return spark_df