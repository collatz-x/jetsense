# ==============================================================
# MAIN ETL PIPELINE — Bronze → Silver → Gold
# ==============================================================
# Purpose: Orchestrate end-to-end data transformation for turbofan engine predictive maintenance
# 
# Pipeline Architecture (Medallion/Lakehouse Pattern):
# 1. BRONZE: Raw data ingestion from text files → structured CSVs
# 2. SILVER: Data consolidation, cleaning, and standardization → unified Parquet
# 3. GOLD: Feature engineering, normalization, ML-ready datasets → feature/label stores
# 
# This pipeline transforms raw sensor readings into production-ready features for
# Remaining Useful Life (RUL) prediction models.
# ==============================================================

import os
import sys
import pyspark
from pyspark.sql import functions as F

# Add parent directory to Python path to allow imports from project root
# This ensures 'utils' module can be found regardless of where script is run from
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import custom processing modules for each layer
import utils.bronze_processing as bp  # Raw data ingestion
import utils.silver_processing as sp  # Data cleaning and consolidation
import utils.gold_processing as gp    # Feature engineering and ML preparation


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================


def create_spark_session():
    """Create and return a configured SparkSession."""
    spark = pyspark.sql.SparkSession.builder \
        .appName("jetsense_pipeline") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ==============================================================
# BRONZE LAYER: Raw Data Ingestion
# ==============================================================


def process_bronze_layer(spark, train_paths=None):
    """
    Bronze Layer: Raw Data Ingestion
    
    Purpose: Load raw text files and convert to structured format
    Input: Space-delimited text files with sensor readings
    Output: Structured CSV files in Bronze directory
    
    This layer performs minimal transformation - just schema assignment
    and format conversion for downstream processing
    
    Args:
        spark (SparkSession): Active Spark session for distributed processing
        train_paths (list): List of raw data file paths
    """
    if train_paths is None:
        train_paths = [f'{parent_dir}/raw_data/train_FD001_augmented.txt']
    
    print("\n🟤 Processing Bronze Layer...")
    for train in train_paths:
        # Process each raw file: read → structure → save to Bronze
        bp.process_bronze_table(train, "datamart/bronze/", spark)


# ==============================================================
# SILVER LAYER: Data Consolidation and Cleaning
# ==============================================================


def process_silver_layer(spark, train_paths=None):
    """
    Silver Layer: Data Consolidation and Cleaning
    
    Purpose: Combine Bronze files, ensure unique IDs, standardize schema
    Input: Bronze CSV files (may be multiple datasets)
    Output: Single unified Parquet file with consistent schema
    
    This layer ensures data quality and consistency across sources
    
    Args:
        spark (SparkSession): Active Spark session for distributed processing
        train_paths (list): List of Bronze dataset names
    
    Returns:
        pyspark.sql.DataFrame: Unified Silver dataset
    """
    if train_paths is None:
        train_paths = ['bronze_train_FD001_augmented']
    
    print("\n⚪ Processing Silver Layer...")
    # Combine all Bronze datasets into unified Silver table
    df = sp.process_silver_table(train_paths, "datamart/bronze/", "datamart/silver", spark)
    return df


# ==============================================================
# SILVER LAYER: Feature Selection and Cleanup
# ==============================================================


def clean_silver_features(spark, silver_path="datamart/silver/silver_feature.parquet"):
    """
    Silver Layer: Feature Selection and Cleanup
    
    Purpose: Remove redundant and highly correlated features to improve model efficiency
    
    Feature removal rationale:
    - Constant features: Zero variance, provide no predictive value
    - Highly correlated features: Redundant information, can cause multicollinearity
    
    This step reduces dimensionality while preserving predictive power
    
    Args:
        spark (SparkSession): Active Spark session for distributed processing
        silver_path (str): Path to the silver feature parquet file
    
    Returns:
        str: Path to the cleaned Silver dataset
    """
    print("\n⚪ Cleaning Silver Output...")
    
    # Load the initial Silver dataset
    df = spark.read.parquet(silver_path)
    
    # -------------------- Remove Constant/Redundant Features --------------------
    # These features show little to no variation across the dataset
    # Constant features don't help models distinguish between different engine states
    cols_to_drop = ['PCNfR_dmd', 'farB', 'Nf_dmd', 'epr', 'P2', 'T2']
    
    # Only drop columns that actually exist in the dataframe
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(*existing_cols_to_drop)
    print("Dropped constant or redundant columns:", existing_cols_to_drop)
    
    # -------------------- Remove Highly Correlated Features --------------------
    # These features are strongly correlated with other sensors (correlation > 0.95)
    # Keeping redundant features can lead to:
    # - Increased computational cost without accuracy benefit
    # - Multicollinearity issues in linear models
    # - Overfitting in complex models
    cols_to_drop_corr = ["NRc", "NRf", "Ps30"]
    
    # Only drop columns that exist
    existing_corr_drop = [c for c in cols_to_drop_corr if c in df.columns]
    df = df.drop(*existing_corr_drop)
    print("Dropped highly correlated columns:", existing_corr_drop)
    
    # -------------------- Save Cleaned Silver Dataset --------------------
    cleaned_path = "datamart/silver/silver_feature_cleaned.parquet"
    df.write.mode("overwrite").parquet(cleaned_path)
    print(f"✅ Cleaned Silver saved to: {cleaned_path}")
    
    # -------------------- Data Integrity Verification --------------------
    # Verify that cleaning process hasn't corrupted the dataset
    engine_count = df.select("unit").distinct().count()
    row_count = df.count()
    print(f"📊 Silver integrity check → {engine_count} unique engines, {row_count:,} rows")
    
    return cleaned_path


# ==============================================================
# GOLD LAYER: Feature Engineering and ML Preparation
# ==============================================================


def process_gold_layer(spark, silver_filepath, gold_directory="datamart/gold"):
    """
    Gold Layer: Feature Engineering and ML Preparation
    
    Purpose: Transform cleaned data into ML-ready features
    Operations:
    - Calculate Remaining Useful Life (RUL) - prediction target
    - Engineer time-series features (rolling means, deltas)
    - Create domain-specific ratios and health indicators
    - Normalize features using Z-score standardization
    - Separate features and labels for ML workflows
    
    Output:
    - gold_full.parquet: Complete dataset with all features and RUL
    - feature_store.parquet: All engineered features (model inputs)
    - label_store.parquet: Target variable RUL (model outputs)
    - feature_metadata.csv: Normalization parameters (for inference)
    
    Args:
        spark (SparkSession): Active Spark session for distributed processing
        silver_filepath (str): Path to cleaned Silver dataset
        gold_directory (str): Output directory for Gold layer files
    
    Returns:
        tuple: (gold_df, feature_store, label_store) as pandas DataFrames
    """
    print("\n🟡 Processing Gold Layer...")
    
    # Process Gold layer - returns 3 pandas DataFrames
    # gold_df: Full dataset with normalized features and RUL
    # feature_store: Features only (for model training input)
    # label_store: Labels only (for model training output)
    gold_df, feature_store, label_store = gp.process_gold_table(
        silver_filepath, 
        gold_directory, 
        spark
    )
    
    # -------------------- Final Data Integrity Check --------------------
    # Verify Gold layer maintains data integrity
    # gold_df is a pandas DataFrame (converted from Spark for scikit-learn compatibility)
    print(f"📊 Gold integrity check → {gold_df['unit'].nunique()} unique engines, {len(gold_df):,} rows")
    
    return gold_df, feature_store, label_store


# ==============================================================
# PIPELINE COMPLETION SUMMARY
# ==============================================================


def print_pipeline_summary(gold_directory="datamart/gold"):
    """
    Display final output locations for downstream ML workflows.
    
    Args:
        gold_directory (str): Directory containing Gold layer outputs
    """
    print("\n✅ FULL ETL PIPELINE COMPLETE!")
    print("=" * 70)
    print("📁 Output Files:")
    print(f"   Gold Table:      {gold_directory}/gold_full.parquet")
    print(f"   Feature Store:   {gold_directory}/feature_store.parquet")
    print(f"   Label Store:     {gold_directory}/label_store.parquet")
    print(f"   Normalization:   {gold_directory}/feature_metadata.csv")
    print("=" * 70)
    print("🏁 Pipeline finished successfully.")
    print("\n📋 Next Steps:")
    print("   1. Use feature_store.parquet for model training inputs")
    print("   2. Use label_store.parquet for model training targets")
    print("   3. Use feature_metadata.csv to normalize inference data")
    print("=" * 70)


# ==============================================================
# MAIN PIPELINE ORCHESTRATION
# ==============================================================


def main():
    """
    Execute the complete ETL pipeline: Bronze → Silver → Gold
    
    This is the main entry point that orchestrates all layers sequentially.
    Creates a Spark session, processes all layers, and ensures cleanup.
    
    Returns:
        tuple: (gold_df, feature_store, label_store) as pandas DataFrames
    """
    # Create SparkSession
    spark = create_spark_session()
    
    try:
        # Execute pipeline layers sequentially
        process_bronze_layer(spark)
        process_silver_layer(spark)
        cleaned_path = clean_silver_features(spark)
        gold_df, feature_store, label_store = process_gold_layer(spark, cleaned_path)
        
        # Display summary
        print_pipeline_summary()
        
        return gold_df, feature_store, label_store
        
    finally:
        # Always stop Spark session to free resources
        spark.stop()


# ==============================================================
# SCRIPT ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    # Only run when executed directly (python main.py), not when imported
    main()