# ==============================================================
# MAIN ETL PIPELINE — Bronze → Silver → Gold
# ==============================================================

import os
import pyspark
from pyspark.sql import functions as F

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
train_paths = ['raw_data/train_FD001_augmented.txt']

print("\n🟤 Processing Bronze Layer...")
for train in train_paths:
    bp.process_bronze_table(train, "datamart/bronze/", spark)


# --------------------------------------------------------------
# SILVER LAYER
# --------------------------------------------------------------
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
# GOLD LAYER (Feature Store + Label Store)
# --------------------------------------------------------------
print("\n🟡 Processing Gold Layer...")
silver_filepath = cleaned_path
gold_directory = "datamart/gold"

# ✅ Updated: gold_processing now returns 3 objects
gold_df, feature_store, label_store = gp.process_gold_table(silver_filepath, gold_directory, spark)

# ✅ gold_df is a pandas DataFrame (normalized)
print(f"📊 Gold integrity check → {gold_df['unit'].nunique()} unique engines, {len(gold_df):,} rows")


# --------------------------------------------------------------
# COMPLETION SUMMARY
# --------------------------------------------------------------
print("\n✅ FULL ETL PIPELINE COMPLETE!")
print(f"📁 Gold Table:      {gold_directory}/gold_full.parquet")
print(f"📁 Feature Store:   {gold_directory}/feature_store.parquet")
print(f"📁 Label Store:     {gold_directory}/label_store.parquet")
print(f"📁 Normalization:   {gold_directory}/feature_metadata.csv")
print("--------------------------------------------------------------")
print("🏁 Pipeline finished successfully.")
