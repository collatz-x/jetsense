import argparse
import os
import pandas as pd
import pickle
import json
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from datetime import datetime


# to call this script: python scripts/inference/inference.py --modelname "engine_rul_prediction_2025-11-02.pkl" --units "1, 5, 10"

def main(modelname, units):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("jetsense_inference") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    
    # --- set up config ---
    config = {}
    config["units_filter"] = units
    config["model_name"] = modelname
    config["model_bank_directory"] = "model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    config["model_meta_filepath"] = config["model_bank_directory"] + config["model_name"].replace('.pkl', '_meta.json')
    
    pprint.pprint(config)
    

    # --- load model artefact from model bank ---
    # Load the model from the pickle file
    with open(config["model_artefact_filepath"], 'rb') as file:
        model = pickle.load(file)
    
    print("Model loaded successfully! " + config["model_artefact_filepath"])
    
    # Load model metadata
    with open(config["model_meta_filepath"], 'r') as file:
        model_metadata = json.load(file)
    
    print("Model metadata loaded successfully! " + config["model_meta_filepath"])


    # --- load feature store ---
    feature_location = "datamart/gold/feature_store.parquet"
    
    # Load Parquet into DataFrame - connect to feature store
    features_store_sdf = spark.read.parquet(feature_location)
    
    
    # extract feature store
    if config["units_filter"] == "all":
        features_sdf = features_store_sdf
        print(f"extracted features_sdf for all units: {features_sdf.count()} rows")
    else:
        unit_list = [int(u.strip()) for u in config["units_filter"].split(",")]
        features_sdf = features_store_sdf.filter(col("unit").isin(unit_list))
        print(f"extracted features_sdf for units {unit_list}: {features_sdf.count()} rows")
    
    features_pdf = features_sdf.toPandas()


    # --- preprocess data for modeling ---
    # prepare X_inference
    feature_cols = model_metadata["features_used"]
    X_inference = features_pdf[feature_cols]
    
    # Note: Model is already a Pipeline with StandardScaler, no separate transform needed
    
    print('X_inference', X_inference.shape[0])


    # --- model prediction inference ---
    # predict model
    y_inference = model.predict(X_inference)
    
    # prepare output
    y_inference_pdf = features_pdf[["unit", "cycle"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_version"] = model_metadata["model_version"]
    y_inference_pdf["RUL_predicted"] = y_inference
    

    # --- save model inference to datamart gold table ---
    gold_directory = f"datamart/gold/model_predictions/{config['model_name'][:-4]}/"
    print(gold_directory)
    
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    # save gold table - IRL connect to database to write
    timestamp_str = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    partition_name = config["model_name"][:-4] + "_predictions_" + timestamp_str + '.parquet'
    filepath = gold_directory + partition_name
    y_inference_pdf.to_parquet(filepath, index=False)
    print('saved to:', filepath)

    
    # --- end spark session --- 
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run inference job")
    parser.add_argument("--modelname", type=str, required=True, help="model filename (e.g., engine_rul_prediction_2025-11-02.pkl)")
    parser.add_argument("--units", type=str, default="all", help="comma-separated unit IDs (e.g., '1,2,3') or 'all'")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.modelname, args.units)

