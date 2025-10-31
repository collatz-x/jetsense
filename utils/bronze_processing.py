import os
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def process_bronze_table(filepath, bronze_directory, spark):
    header_names = [
        "unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3",
        "T2", 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc',
        'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed',
        'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32'
    ]

    df = pd.read_csv(filepath, delim_whitespace=True, header=None, names=header_names)

    # # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(df)

    # Save to Bronze layer
    filename = f"bronze_{filepath.split("/")[-1].split(".txt")[0]}.csv" 
    bronze_filepath = os.path.join(bronze_directory, filename)

    os.makedirs(bronze_directory, exist_ok=True)
    spark_df.toPandas().to_csv(bronze_filepath, index=False)

    print(f"File saved Bronze table to:\n{bronze_filepath}")
    return spark_df