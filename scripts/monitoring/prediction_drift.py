import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scripts.monitoring.input_drift import population_stability_index

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("Warning: ruptures library not available. Change point detection will be disabled.")

def pred_distribution_drift(ref_pred, cur_pred):
    """
    Detect drift in prediction distribution.
    
    Args:
        ref_pred: Reference predictions (numpy array)
        cur_pred: Current predictions (numpy array)
    
    Returns:
        Dictionary with drift metrics:
        - psi: Population Stability Index
        - ks_p: Kolmogorov-Smirnov test p-value
        - mean_change_pct: Percentage change in mean predictions
        - std_change_pct: Percentage change in standard deviation
    """
    # Calculate PSI (fixed bug: was calling psi(ref_pred, cur_pred))
    psi_value = population_stability_index(ref_pred, cur_pred)
    
    # KS test
    ks_stat, ks_pval = ks_2samp(ref_pred, cur_pred, alternative='two-sided')
    
    # Statistical changes
    ref_mean = np.mean(ref_pred)
    cur_mean = np.mean(cur_pred)
    mean_change_pct = float(abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-8))
    
    ref_std = np.std(ref_pred)
    cur_std = np.std(cur_pred)
    std_change_pct = float(abs(cur_std - ref_std) / (abs(ref_std) + 1e-8))
    
    return {
        "psi": float(psi_value),
        "ks_statistic": float(ks_stat),
        "ks_p": float(ks_pval),
        "mean_change_pct": mean_change_pct,
        "std_change_pct": std_change_pct,
        "ref_mean": float(ref_mean),
        "cur_mean": float(cur_mean),
        "ref_std": float(ref_std),
        "cur_std": float(cur_std),
    }


def monotonicity_violation_rate(df, unit_col, cycle_col, yhat_col):
    """
    Calculate the rate of monotonicity violations in RUL predictions.
    
    For RUL (Remaining Useful Life) predictions, the predicted RUL should
    monotonically decrease as the cycle number increases. This function
    calculates the rate at which this constraint is violated.
    
    Args:
        df: DataFrame with predictions
        unit_col: Column name for unit/asset identifier
        cycle_col: Column name for time cycle
        yhat_col: Column name for predictions
    
    Returns:
        Float between 0 and 1 representing the violation rate
        (0 = perfect monotonicity, 1 = all predictions violate)
    """
    rates = []
    
    # Group by unit and check monotonicity for each
    for unit_id, group in df.sort_values([unit_col, cycle_col]).groupby(unit_col):
        y = group[yhat_col].values
        
        # Need at least 2 points to check monotonicity
        if len(y) < 2:
            continue
        
        # Count violations (RUL should decrease, so diff should be negative)
        # A positive diff means RUL increased, which is a violation
        diffs = np.diff(y)
        violations = np.sum(diffs > 0)
        
        # Calculate rate for this unit
        rate = violations / len(diffs)
        rates.append(rate)
    
    # Return average violation rate across all units
    return float(np.mean(rates)) if rates else 0.0


def check_prediction_range(predictions, expected_min, expected_max):
    """
    Check if predictions fall within expected range.
    
    Args:
        predictions: Array of predictions
        expected_min: Expected minimum value (e.g., 0 for RUL)
        expected_max: Expected maximum value
    
    Returns:
        Dictionary with range violation statistics
    """
    below_min = np.sum(predictions < expected_min)
    above_max = np.sum(predictions > expected_max)
    total = len(predictions)
    
    return {
        "below_min_count": int(below_min),
        "above_max_count": int(above_max),
        "below_min_pct": float(below_min / total),
        "above_max_pct": float(above_max / total),
        "total_violations": int(below_min + above_max),
        "violation_rate": float((below_min + above_max) / total),
        "min_prediction": float(np.min(predictions)),
        "max_prediction": float(np.max(predictions)),
    }


def change_points(yhat_series, penalty=5):
    """
    Detect change points in prediction time series using ruptures library.
    
    Change points indicate sudden shifts in prediction patterns, which could
    signal model degradation or system behavior changes.
    
    Args:
        yhat_series: Pandas Series or numpy array of predictions over time
        penalty: Penalty value for change point detection (higher = fewer CPs)
    
    Returns:
        List of change point indices, or None if ruptures is not available
    """
    if not RUPTURES_AVAILABLE:
        print("ruptures library not available for change point detection")
        return None
    
    try:
        # Convert to numpy array if needed
        if isinstance(yhat_series, pd.Series):
            signal = yhat_series.values
        else:
            signal = yhat_series
        
        # Ensure float type
        signal = signal.astype(float)
        
        # Detect change points using Pelt algorithm with RBF kernel
        algo = rpt.Pelt(model="rbf").fit(signal)
        change_points = algo.predict(pen=penalty)
        
        return change_points
        
    except Exception as e:
        print(f"Error detecting change points: {e}")
        return None


def prediction_stability_metrics(df, unit_col, cycle_col, yhat_col, window=5):
    """
    Calculate prediction stability metrics over time windows.
    
    Args:
        df: DataFrame with predictions
        unit_col: Column name for unit identifier
        cycle_col: Column name for time cycle
        yhat_col: Column name for predictions
        window: Size of rolling window for stability calculation
    
    Returns:
        Dictionary with stability metrics
    """
    stability_scores = []
    
    for unit_id, group in df.sort_values([unit_col, cycle_col]).groupby(unit_col):
        y = group[yhat_col].values
        
        if len(y) < window:
            continue
        
        # Calculate rolling standard deviation
        rolling_std = pd.Series(y).rolling(window=window).std().values
        
        # Filter out NaN values from the start
        rolling_std = rolling_std[~np.isnan(rolling_std)]
        
        if len(rolling_std) > 0:
            stability_scores.extend(rolling_std)
    
    if not stability_scores:
        return {
            "mean_stability": None,
            "p95_stability": None,
        }
    
    return {
        "mean_stability": float(np.mean(stability_scores)),
        "median_stability": float(np.median(stability_scores)),
        "p95_stability": float(np.percentile(stability_scores, 95)),
        "max_instability": float(np.max(stability_scores)),
    }


def comprehensive_prediction_report(
    ref_predictions, 
    cur_predictions, 
    cur_df=None,
    unit_col=None,
    cycle_col=None,
    yhat_col=None,
    expected_min=0,
    expected_max=None
):
    """
    Generate a comprehensive prediction drift report.
    
    Args:
        ref_predictions: Reference predictions array
        cur_predictions: Current predictions array
        cur_df: Optional DataFrame with current predictions for advanced metrics
        unit_col: Column name for unit identifier (if cur_df provided)
        cycle_col: Column name for cycle (if cur_df provided)
        yhat_col: Column name for predictions (if cur_df provided)
        expected_min: Expected minimum prediction value
        expected_max: Expected maximum prediction value
    
    Returns:
        Dictionary with comprehensive drift metrics
    """
    report = {}
    
    # Distribution drift
    report["distribution_drift"] = pred_distribution_drift(ref_predictions, cur_predictions)
    
    # Range violations
    if expected_max is not None:
        report["range_violations"] = check_prediction_range(
            cur_predictions, expected_min, expected_max
        )
    
    # Advanced metrics if DataFrame provided
    if cur_df is not None and all([unit_col, cycle_col, yhat_col]):
        # Monotonicity violations
        report["monotonicity_violation_rate"] = monotonicity_violation_rate(
            cur_df, unit_col, cycle_col, yhat_col
        )
        
        # Stability metrics
        report["stability"] = prediction_stability_metrics(
            cur_df, unit_col, cycle_col, yhat_col
        )
    
    return report