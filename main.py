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

# import utils.gold_processing
import utils.bronze_processing as bp
# import utils.silver_processing

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
train = [
    'raw_data/train_FD001.txt',
    'raw_data/train_FD002.txt',
    'raw_data/train_FD003.txt',
    'raw_data/train_FD004.txt'
]

test = [
    'raw_data/test_FD001.txt',
    'raw_data/test_FD002.txt',
    'raw_data/test_FD003.txt',
    'raw_data/test_FD004.txt'
]

bp.process_bronze_table(train, "datamart/bronze/", spark)
bp.process_bronze_table(test, "datamart/bronze/", spark)
