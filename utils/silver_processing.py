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

def process_silver_table(filepaths, bronze_directory, silver_directory,spark):
    """
    Combines all C-MAPSS raw files, ensures unique unit IDs,
    appends failure type, and saves the Bronze table.
    """

    datasets = []
    offset = 0

    # Combine all raw subsets
    for i, path in enumerate(filepaths, start=1):
        df = pd.read_csv(bronze_directory + path)
        df["unit"] = df["unit"] + offset          # ensure unique engine IDs
        offset += df["unit"].nunique()
        datasets.append(df)

    combined = pd.concat(datasets, ignore_index=True)

    df = spark.createDataFrame(combined)

    column_type_map = {'unit': IntegerType(), 
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
        df = df.withColumn(column, col(column).cast(new_type))

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    # save silver table - IRL connect to database to write
    filename = f"{silver_directory}/silver_feature.parquet"
    df.write.mode("overwrite").parquet(filename)
    print('saved to:', filename)
    
    return df

# def process_silver_rolling_mean(filename, silver_directory, spark, rolling_window=3):
#     df = spark.read.parquet(filename)

#     # Identify numeric columns only
#     numeric_cols = [c for c, t in df.dtypes if t in ("double", "float", "int") and c not in ("unit", "cycle")]

#     # Define rolling and lag windows
#     roll_window = (
#         Window.partitionBy("unit")
#               .orderBy("cycle")
#               .rowsBetween(-(rolling_window - 1), 0)
#     )

#     lag_window = (
#         Window.partitionBy("unit")
#               .orderBy("cycle")
#     )

#     for column in numeric_cols:
#         roll_mean = f"{column}_rolling_mean"
#         roll_std  = f"{column}_rolling_std"
#         delta_col = f"{column}_delta"

#         df = (
#             df.withColumn(roll_mean, F.avg(F.col(column)).over(roll_window))
#               .withColumn(roll_std,  F.stddev(F.col(column)).over(roll_window))
#               .withColumn(delta_col, F.col(column) - F.lag(F.col(column), 1).over(lag_window))
#         )

#         # Fill nulls
#         df = (
#             df.withColumn(
#                 roll_mean,
#                 F.when(F.col(roll_mean).isNull(), F.col(column)).otherwise(F.col(roll_mean))
#             )
#             .withColumn(
#                 roll_std,
#                 F.when(F.col(roll_std).isNull(), F.lit(0.0)).otherwise(F.col(roll_std))
#             )
#             .withColumn(
#                 delta_col,
#                 F.when(F.col(delta_col).isNull(), F.lit(0.0)).otherwise(F.col(delta_col))
#             )
#         )

    # Save enriched Silver dataset
    filename = f"{silver_directory}/silver_feature_rolling.parquet"
    df.write.mode("overwrite").parquet(filename)

    print('saved to:', filename)

    return df
