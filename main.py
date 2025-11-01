# ==============================================================
# MAIN ETL PIPELINE — Bronze → Silver → Gold
# ==============================================================
import os
import glob
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StringType, IntegerType, FloatType, DateType, BooleanType, DoubleType
)

# Import processing scripts
import utils.bronze_processing as bp
import utils.silver_processing as sp
import utils.gold_processing as gp


# --------------------------------------------------------------
# Initialize SparkSession
# --------------------------------------------------------------
spark = pyspark.sql.SparkSession.builder \
    .appName("jetsense_pipeline") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# --------------------------------------------------------------
# BRONZE LAYER
# --------------------------------------------------------------
train_paths = [
    'raw_data/train_FD001_augmented.txt'
]

print("\n🟤 Processing Bronze Layer...")
for train in train_paths:
    bp.process_bronze_table(train, "datamart/bronze/", spark)


# --------------------------------------------------------------
# SILVER LAYER
# --------------------------------------------------------------
# ✅ FIXED: Match Bronze output folder (Spark writes folder, not .csv)
train_paths = ['bronze_train_FD001_augmented']

print("\n⚪ Processing Silver Layer...")
sp.process_silver_table(train_paths, "datamart/bronze/", "datamart/silver", spark)


# --------------------------------------------------------------
# CLEAN SILVER OUTPUT (Drop constants + correlated features)
# --------------------------------------------------------------
print("\n⚪ Cleaning Silver Output...")

silver_path = "datamart/silver/silver_feature.parquet"
df = spark.read.parquet(silver_path)

# Drop constant / redundant features
cols_to_drop = ['PCNfR_dmd', 'farB', 'Nf_dmd', 'epr', 'P2', 'T2']
existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(*existing_cols_to_drop)
print("Dropped constant or redundant columns:", existing_cols_to_drop)

# Drop highly correlated features
cols_to_drop_corr = ["NRc", "NRf", "Ps30"]
existing_corr_drop = [c for c in cols_to_drop_corr if c in df.columns]
df = df.drop(*existing_corr_drop)
print("Dropped highly correlated columns:", existing_corr_drop)

# Save cleaned Silver dataset
cleaned_path = "datamart/silver/silver_feature_cleaned.parquet"
df.write.mode("overwrite").parquet(cleaned_path)
print(f"✅ Cleaned Silver saved to: {cleaned_path}")

# Quick check on engine count before Gold
engine_count = df.select("unit").distinct().count()
row_count = df.count()
print(f"📊 Silver integrity check → {engine_count} unique engines, {row_count:,} rows")


# --------------------------------------------------------------
# GOLD LAYER (Feature + Label Store)
# --------------------------------------------------------------
print("\n🟡 Processing Gold Layer...")
silver_filepath = cleaned_path
gold_directory = "datamart/gold"

gold_df, feature_store, label_store = gp.process_gold_table(silver_filepath, gold_directory, spark)

# Verify engine counts again
gold_engine_count = gold_df.select("unit").distinct().count()
gold_rows = gold_df.count()
print(f"📊 Gold integrity check → {gold_engine_count} unique engines, {gold_rows:,} rows")

# --------------------------------------------------------------
# COMPLETION SUMMARY
# --------------------------------------------------------------
print("\n✅ FULL ETL PIPELINE COMPLETE!")
print(f"📁 Gold Table:      {gold_directory}/gold_full.parquet")
print(f"📁 Feature Store:   {gold_directory}/feature_store.parquet")
print(f"📁 Label Store:     {gold_directory}/label_store.parquet")
print("--------------------------------------------------------------")
print("🏁 Pipeline finished successfully.")
