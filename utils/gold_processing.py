import os
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from sklearn.preprocessing import StandardScaler

# ==============================================================
# GOLD LAYER PROCESSING
# ==============================================================
# Purpose: Transform Silver layer (cleaned data) into Gold layer (analytics-ready features)
# 
# Key Operations:
# - Calculate Remaining Useful Life (RUL) - the target variable for predictive maintenance
# - Engineer features from raw sensor data (rolling averages, deltas, ratios)
# - Normalize all features using Z-score standardization
# - Create separate feature and label stores for ML model training
# - Save normalization parameters for inference-time transformation
# ==============================================================

def process_gold_table(silver_input, gold_directory, spark):
    """
    Build Gold layer dataset with advanced feature engineering for predictive maintenance.
    
    This function transforms Silver layer sensor data into ML-ready features by:
    1. Computing Remaining Useful Life (RUL) as the prediction target
    2. Creating time-series features (rolling statistics, deltas)
    3. Engineering domain-specific ratios and health indicators
    4. Normalizing features for model training stability
    5. Separating features and labels for ML workflows
    
    Args:
        silver_input (str or DataFrame): Path to Silver Parquet/CSV or Spark DataFrame
        gold_directory (str): Output directory for Gold layer CSV files
        spark (SparkSession): Active Spark session for distributed processing
    
    Returns:
        tuple: (full_dataset, feature_store, label_store) as pandas DataFrames
            - full_dataset: Complete normalized data with all features and RUL
            - feature_store: All engineered features (without RUL)
            - label_store: Only identifiers (unit, cycle) and target (RUL)
    """

    print("🚀 Processing Gold layer...")

    # ==================== STEP 1: Load Silver Layer Data ====================
    # Flexible input handling: accepts file paths or existing DataFrames
    if isinstance(silver_input, str):
        if silver_input.endswith(".parquet"):
            df = spark.read.parquet(silver_input)
        elif silver_input.endswith(".csv"):
            df = spark.read.option("header", True).csv(silver_input, inferSchema=True)
        else:
            raise ValueError("Unsupported file type for silver_input")
    else:
        # Assume it's already a Spark DataFrame
        df = silver_input

    # ==================== STEP 2: Compute Remaining Useful Life (RUL) ====================
    # RUL = the number of cycles remaining until engine failure
    # Formula: RUL = max_cycle_for_engine - current_cycle
    # Example: If engine fails at cycle 200 and we're at cycle 150, RUL = 50
    
    # Define window partitioned by engine unit for group operations
    w = Window.partitionBy("unit")
    
    # Find the maximum cycle (failure point) for each engine
    max_cycle = df.groupBy("unit").agg(F.max("cycle").alias("max_cycle"))
    
    # Join max_cycle back to main dataframe
    df = df.join(max_cycle, on="unit", how="left")
    
    # Calculate RUL: cycles until failure
    df = df.withColumn("RUL", F.col("max_cycle") - F.col("cycle"))
    
    # Remove temporary max_cycle column
    df = df.drop("max_cycle")

    # ==================== STEP 3: Convert to Pandas for Feature Engineering ====================
    # Switch to Pandas for easier time-series operations and scikit-learn integration
    pdf = df.toPandas()
    print(f"✅ Loaded Silver → {len(pdf):,} rows, {pdf['unit'].nunique()} engines")

    # ==================== STEP 4: Time-Series Feature Engineering ====================
    # Create rolling statistics and delta features for key sensors
    # These capture trends and rate-of-change in sensor readings
    
    sensors = ["T24", "T30", "T50", "P15", "P30", "Nf", "Nc"]
    
    for s in sensors:
        # Rolling mean (5-cycle window): Smooths out noise, captures short-term trends
        # min_periods=1 ensures we have values even at the start of each engine's lifecycle
        pdf[f"{s}_roll5_mean"] = (
            pdf.groupby("unit")[s]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Delta (first-order difference): Captures rate of change between consecutive cycles
        # Positive delta = sensor value increasing, negative = decreasing
        # fillna(0) handles the first cycle where no previous value exists
        pdf[f"{s}_delta1"] = pdf.groupby("unit")[s].diff().fillna(0)

    # ==================== STEP 5: Domain-Specific Ratio Features ====================
    # Engineering ratios based on thermodynamic and mechanical relationships
    # These can reveal subtle degradation patterns not visible in raw sensor values
    
    # Temperature ratios: Relative temperature changes across engine stages
    pdf["T_ratio_24_30"] = pdf["T24"] / (pdf["T30"] + 1e-6)  # +1e-6 prevents division by zero
    pdf["T_ratio_30_50"] = pdf["T30"] / (pdf["T50"] + 1e-6)
    
    # Pressure ratio: Pressure differential indicator
    pdf["P_ratio_15_30"] = pdf["P15"] / (pdf["P30"] + 1e-6)
    
    # Speed ratio: Fan speed relative to core speed
    pdf["N_ratio_f_c"]   = pdf["Nf"] / (pdf["Nc"]  + 1e-6)
    
    # Normalized cycle: Position in engine lifecycle (0 = start, 1 = failure)
    pdf["cycle_norm"]    = pdf.groupby("unit")["cycle"].transform(lambda x: x / x.max())
    
    # Health index: Inverse of normalized cycle (1 = healthy, 0 = failed)
    # Provides intuitive health score that decreases as engine degrades
    pdf["health_index"]  = 1 - pdf["cycle_norm"]

    # ==================== STEP 6: Feature Normalization (Z-score) ====================
    # Standardize features to have mean=0 and std=1
    # Critical for ML models (especially neural networks) to train effectively
    
    # Identify all numeric columns
    numeric_cols = pdf.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude identifiers and target variable from normalization
    exclude_cols = ["unit", "cycle", "RUL"]
    cols_to_scale = [c for c in numeric_cols if c not in exclude_cols]

    # Fit StandardScaler on training data and transform
    scaler = StandardScaler()
    pdf_scaled = pdf.copy()
    pdf_scaled[cols_to_scale] = scaler.fit_transform(pdf[cols_to_scale])

    # ==================== STEP 7: Create Feature and Label Stores ====================
    # Separate features from labels for clean ML pipeline organization
    
    # Feature Store: All engineered features (input to ML models)
    feature_store = pdf_scaled.drop(columns=["RUL"])
    
    # Label Store: Target variable with identifiers (ground truth for training)
    label_store = pdf_scaled[["unit", "cycle", "RUL"]]

    # ==================== STEP 8: Save Gold Layer Outputs ====================
    # Save as CSV for broad compatibility (no Arrow/Parquet dependencies needed)
    os.makedirs(gold_directory, exist_ok=True)
    
    # Define output file paths
    gold_full_path = os.path.join(gold_directory, "gold_full.csv")
    feature_path = os.path.join(gold_directory, "feature_store.csv")
    label_path = os.path.join(gold_directory, "label_store.csv")
    meta_path = os.path.join(gold_directory, "feature_metadata.csv")

    # Save complete dataset with all features and labels
    pdf_scaled.to_csv(gold_full_path, index=False)
    
    # Save feature store (for model input)
    feature_store.to_csv(feature_path, index=False)
    
    # Save label store (for model training/evaluation)
    label_store.to_csv(label_path, index=False)

    # ==================== STEP 9: Save Normalization Metadata ====================
    # Store mean and std for each feature so we can apply same transformation
    # to new data during inference (critical for production deployment)
    pd.DataFrame({
        "feature": cols_to_scale,
        "mean": scaler.mean_,      # Mean used for centering
        "std": scaler.scale_       # Standard deviation used for scaling
    }).to_csv(meta_path, index=False)

    # ==================== Verification and Summary ====================
    print(f"💾 Saved Gold full table      → {gold_full_path}")
    print(f"💾 Saved Feature Store        → {feature_path}")
    print(f"💾 Saved Label Store          → {label_path}")
    print(f"💾 Saved normalization config → {meta_path}")

    return pdf_scaled, feature_store, label_store