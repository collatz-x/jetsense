import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.covariance import EmpiricalCovariance


def population_stability_index(training, current, bins=10):
    """
    Calculate Population Stability Index (PSI).
    
    PSI Interpretation:
    - PSI < 0.1: No significant shift (stable)
    - 0.1 ≤ PSI < 0.25: Moderate shift (monitor)
    - PSI ≥ 0.25: Major shift (investigate/retrain)
    
    Args:
        training: Reference/training data array
        current: Current/production data array
        bins: Number of bins for histogram
    
    Returns:
        PSI value (float)
    """
    # Create quantile-based bins from training data
    qs = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(training, qs)
    
    # Get proportions for each bin
    train_counts = np.histogram(training, bins=bin_edges)[0]
    curr_counts = np.histogram(current, bins=bin_edges)[0]
    
    # Convert to proportions and clip to avoid log(0)
    train_props = np.clip(train_counts / len(training), 1e-6, 1)
    curr_props = np.clip(curr_counts / len(current), 1e-6, 1)
    
    # Calculate PSI
    psi = np.sum((curr_props - train_props) * np.log(curr_props / train_props))
    
    return psi


def ks_test(training, current):
    """
    Perform Kolmogorov-Smirnov test for distribution drift.
    
    Returns:
        p-value (float) - Low p-value (<0.05) indicates significant drift
    """
    statistic, pvalue = ks_2samp(training, current, alternative="two-sided")
    return pvalue


def mahalanobis_scores(train_X, cur_X):
    """
    Calculate Mahalanobis distance for multivariate drift detection.
    
    Args:
        train_X: Reference DataFrame with features
        cur_X: Current DataFrame with features
    
    Returns:
        Array of Mahalanobis distances for current data points
    """
    # Calculate mean and covariance from training data
    mu = train_X.mean(axis=0).values
    cov = EmpiricalCovariance().fit(train_X.values)
    
    # Calculate Mahalanobis distance for current data
    distances = cov.mahalanobis(cur_X.values - mu)
    
    return distances


def input_drift_report(train_df, cur_df, feature_cols):
    """
    Generate comprehensive input drift report for all features.
    
    Args:
        train_df: Reference/training DataFrame
        cur_df: Current/production DataFrame
        feature_cols: List of feature column names to monitor
    
    Returns:
        Dictionary containing:
        - features: List of per-feature drift metrics
        - mahalanobis_mean: Mean Mahalanobis distance
        - mahalanobis_p95: 95th percentile Mahalanobis distance
    """
    feature_reports = []
    
    # Univariate drift detection for each feature
    for col in feature_cols:
        try:
            # Get clean data (drop NaNs)
            train_vals = train_df[col].dropna().values
            cur_vals = cur_df[col].dropna().values
            
            if len(train_vals) == 0 or len(cur_vals) == 0:
                feature_reports.append({
                    "feature": col,
                    "ks_p": None,
                    "psi": None,
                    "error": "Insufficient data after dropping NaNs"
                })
                continue
            
            # Calculate drift metrics
            pval = ks_test(train_vals, cur_vals)
            psi_val = population_stability_index(train_vals, cur_vals)
            
            # Calculate basic statistics
            train_mean = float(np.mean(train_vals))
            cur_mean = float(np.mean(cur_vals))
            mean_shift_pct = abs(cur_mean - train_mean) / (abs(train_mean) + 1e-8)
            
            feature_reports.append({
                "feature": col,
                "ks_p": float(pval),
                "psi": float(psi_val),
                "train_mean": train_mean,
                "current_mean": cur_mean,
                "mean_shift_pct": float(mean_shift_pct),
                "train_size": len(train_vals),
                "current_size": len(cur_vals),
            })
            
        except Exception as e:
            feature_reports.append({
                "feature": col,
                "ks_p": None,
                "psi": None,
                "error": str(e)
            })
    
    # Multivariate drift detection using Mahalanobis distance
    try:
        # Drop rows with any NaN in feature columns
        train_clean = train_df[feature_cols].dropna()
        cur_clean = cur_df[feature_cols].dropna()
        
        if len(train_clean) > 0 and len(cur_clean) > 0:
            md = mahalanobis_scores(train_clean, cur_clean)
            
            mahalanobis_mean = float(np.mean(md))
            mahalanobis_p95 = float(np.percentile(md, 95))
            mahalanobis_max = float(np.max(md))
        else:
            mahalanobis_mean = None
            mahalanobis_p95 = None
            mahalanobis_max = None
            
    except Exception as e:
        print(f"Warning: Mahalanobis calculation failed: {e}")
        mahalanobis_mean = None
        mahalanobis_p95 = None
        mahalanobis_max = None
    
    # Compile full report
    report = {
        "features": feature_reports,
        "mahalanobis_mean": mahalanobis_mean,
        "mahalanobis_p95": mahalanobis_p95,
        "mahalanobis_max": mahalanobis_max,
        "n_features_monitored": len(feature_cols),
        "n_features_computed": sum(1 for f in feature_reports if f.get("psi") is not None),
    }
    
    return report


def interpret_psi(psi_value):
    """Helper function to interpret PSI values."""
    if psi_value < 0.1:
        return "No significant shift"
    elif psi_value < 0.25:
        return "Moderate shift - monitor closely"
    else:
        return "Major shift - investigate/retrain"


def interpret_ks_pvalue(pvalue):
    """Helper function to interpret KS test p-values."""
    if pvalue >= 0.05:
        return "No significant drift"
    elif pvalue >= 0.01:
        return "Moderate drift"
    else:
        return "Significant drift"