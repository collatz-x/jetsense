import os
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def process_bronze_table(filepath, bronze_directory, spark):
    header_names = [
        "unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3",
        "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
        "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
        "Nf_dmd", "PCNfR_dmd", "W31", "W32"
    ]

    # 🔍 Read raw file with proper separator
    print(f"\n🔍 Reading raw file: {filepath}")
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=header_names)
    
    raw_engine_count = df['unit'].nunique()
    raw_row_count = len(df)
    
    print(f"📊 RAW FILE STATS:")
    print(f"   Total Rows: {raw_row_count:,}")
    print(f"   Unique Engines: {raw_engine_count}")
    print(f"   Engine IDs range: {df['unit'].min()} to {df['unit'].max()}")

    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(df)
    
    spark_engine_count = spark_df.select('unit').distinct().count()
    spark_row_count = spark_df.count()
    
    print(f"📊 AFTER SPARK CONVERSION:")
    print(f"   Total Rows: {spark_row_count:,}")
    print(f"   Unique Engines: {spark_engine_count}")

    # Define output path
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    bronze_path = os.path.join(bronze_directory, f"bronze_{base_name}")

    os.makedirs(bronze_directory, exist_ok=True)

    print(f"💾 Writing to: {bronze_path}")

    # Write Spark DataFrame to CSV
    spark_df.coalesce(1).write.mode("overwrite").option("header", True).csv(bronze_path)

    # Verify written file
    written_df = spark.read.option("header", True).csv(bronze_path)
    written_engine_count = written_df.select('unit').distinct().count()
    written_row_count = written_df.count()
    
    print(f"✅ VERIFICATION:")
    print(f"   Written Rows: {written_row_count:,}")
    print(f"   Written Engines: {written_engine_count}")
    
    if written_engine_count != 600:
        print(f"   ⚠️ WARNING: Expected 600 engines, got {written_engine_count}!")
    
    return spark_df