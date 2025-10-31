# ==============================================================
# MAIN ETL PIPELINE — Bronze → Silver → Gold
# ==============================================================

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import argparse
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, BooleanType, DoubleType

# Import Bronze, Silver, and Gold processing functions
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
train_paths = ['bronze_train_FD001.csv']

print("\n⚪ Processing Silver Layer...")
sp.process_silver_table(train_paths, "datamart/bronze/", "datamart/silver", spark)


# --------------------------------------------------------------
# CLEAN SILVER OUTPUT (Drop constants + correlated features)
# --------------------------------------------------------------
print("\n⚪ Cleaning Silver Output...")

silver_path = "datamart/silver/silver_feature.parquet"
df = spark.read.parquet(silver_path)

# Columns to drop (constant / redundant)
cols_to_drop = ['PCNfR_dmd', 'farB', 'Nf_dmd', 'epr', 'P2', 'T2']

# Drop safely if exist
existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(*existing_cols_to_drop)
print("Dropped constant or redundant columns:", existing_cols_to_drop)

# Drop correlated features
cols_to_drop_corr = ["NRc", "NRf", "Ps30"]
existing_corr_drop = [c for c in cols_to_drop_corr if c in df.columns]
df = df.drop(*existing_corr_drop)
print("Dropped highly correlated columns:", existing_corr_drop)

# Save cleaned Silver
cleaned_path = "datamart/silver/silver_feature_cleaned.parquet"
df.write.mode("overwrite").parquet(cleaned_path)
print(f"✅ Cleaned Silver saved to: {cleaned_path}")


# --------------------------------------------------------------
# GOLD LAYER (Feature + Label Store)
# --------------------------------------------------------------
print("\n🟡 Processing Gold Layer...")
silver_filepath = cleaned_path
gold_directory = "datamart/gold"

gold_df, feature_store, label_store = gp.process_gold_table(silver_filepath, gold_directory, spark)

print("\n✅ Full ETL Pipeline Complete!")
print(f"Gold Table:      {gold_directory}/gold_full.parquet")
print(f"Feature Store:   {gold_directory}/feature_store.parquet")
print(f"Label Store:     {gold_directory}/label_store.parquet")
