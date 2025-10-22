import os
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def process_bronze_table(filepaths, bronze_directory, spark):
    """
    Combines all C-MAPSS raw files, ensures unique unit IDs,
    appends failure type, and saves the Bronze table.
    """
    
    header_names = [
        "unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3",
        "T2", 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc',
        'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed',
        'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32'
    ]

    datasets = []
    offset = 0

    # Combine all raw subsets
    for i, path in enumerate(filepaths, start=1):
        df = pd.read_csv(path, delim_whitespace=True, header=None, names=header_names)
        df["unit"] = df["unit"] + offset          # ensure unique engine IDs
        offset += df["unit"].nunique()
        df["failure_type"] = i                    # tag each FD subset
        datasets.append(df)

    combined = pd.concat(datasets, ignore_index=True)

    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(combined)

    # Save to Bronze layer
    filename = f"bronze_train.csv" 
    if "test" in path: 
        filename = f"bronze_test.csv"
    bronze_filepath = os.path.join(bronze_directory, filename)

    os.makedirs(bronze_directory, exist_ok=True)
    spark_df.toPandas().to_csv(bronze_filepath, index=False)

    print(f"Combined {len(filepaths)} files and saved Bronze table to:\n{bronze_filepath}")
    return spark_df
