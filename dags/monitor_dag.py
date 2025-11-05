from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pandas as pd
import json
import glob
import os
import sys

# Add project root to path
sys.path.insert(0, "/opt/airflow")

# ============================================================================
# Configuration - Update these paths for your medallion architecture
# ============================================================================
GOLD_REFERENCE_DATA = "datamart/gold/reference_data.parquet"  # Baseline training data (frozen snapshot)
GOLD_CURRENT_DATA = "datamart/gold/feature_store.parquet"  # Current production data (live, updated by ETL)
GOLD_PREDICTIONS = "datamart/gold/latest_predictions.parquet"  # Latest predictions from inference
REFERENCE_PREDICTIONS = "datamart/gold/reference_predictions.parquet"  # Baseline predictions

# Feature columns to monitor (update with your actual features)
FEATURE_COLS = ['op_setting_1', 'op_setting_2', 'op_setting_3', 'T24', 'T30', 'T50', 'P15', 'P30', 'Nf', 'Nc', 'phi', 'BPR', 'htBleed', 'W31', 'W32', 'T24_roll5_mean', 'T24_delta1', 'T30_roll5_mean', 'T30_delta1', 'T50_roll5_mean', 'T50_delta1', 'P15_roll5_mean', 'P15_delta1', 'P30_roll5_mean', 'P30_delta1', 'Nf_roll5_mean', 'Nf_delta1', 'Nc_roll5_mean', 'Nc_delta1', 'T_ratio_24_30', 'T_ratio_30_50', 'P_ratio_15_30', 'N_ratio_f_c', 'cycle_norm', 'health_index']  
UNIT_COL = "unit"  # Column identifying different units/assets
CYCLE_COL = "cycle"  # Time cycle column
PREDICTION_COL = "RUL_predicted"  # Your RUL prediction column (matches inference.py output)

# Thresholds
PSI_THRESHOLD = 0.25  # Major drift if PSI > 0.25
KS_PVALUE_THRESHOLD = 0.05  # Drift if p-value < 0.05
N_DRIFTED_THRESHOLD = 3  # Alert if more than 3 features drift
MONOTONICITY_THRESHOLD = 0.1  # Alert if >10% violations


# ============================================================================
# Task Functions
# ============================================================================

def initialize_reference_data(**context):
    """
    Initialize reference baseline files if they don't exist.
    This runs once on first execution to establish the baseline for drift detection.
    """
    # Check if reference data already exists
    ref_data_exists = os.path.exists(GOLD_REFERENCE_DATA)
    ref_pred_exists = os.path.exists(REFERENCE_PREDICTIONS)
    
    if ref_data_exists and ref_pred_exists:
        print("✅ Reference data already exists, skipping initialization")
        return "skip_initialization"
    
    print("🔧 Initializing reference baseline data for first-time setup...")
    
    # Create reference feature data from existing feature store
    if not ref_data_exists:
        print(f"Creating {GOLD_REFERENCE_DATA}...")
        feature_df = pd.read_parquet("datamart/gold/feature_store.parquet")
        feature_df.to_parquet(GOLD_REFERENCE_DATA)
        print(f"✅ Created reference data with {len(feature_df)} rows")
    
    # Create reference predictions from existing OOT predictions or feature store
    if not ref_pred_exists:
        print(f"Creating {REFERENCE_PREDICTIONS}...")
        
        # Option 1: Try to use existing OOT predictions if available
        oot_pred_path = "datamart/gold/oot_predictions_ridgeregression.parquet"
        if os.path.exists(oot_pred_path):
            pred_df = pd.read_parquet(oot_pred_path)
            
            # Check if it has the right column name
            if 'RUL_predicted' not in pred_df.columns:
                # Try to find and rename the prediction column
                pred_cols = [c for c in pred_df.columns if 'rul' in c.lower() or 'pred' in c.lower()]
                if pred_cols:
                    print(f"Renaming column {pred_cols[0]} to RUL_predicted")
                    pred_df = pred_df.rename(columns={pred_cols[0]: 'RUL_predicted'})
            
            pred_df.to_parquet(REFERENCE_PREDICTIONS)
            print(f"✅ Created reference predictions from OOT data with {len(pred_df)} rows")
        else:
            # Option 2: Run inference on a sample to generate reference predictions
            print("⚠️  No OOT predictions found. Please run inference once to generate reference predictions.")
            raise FileNotFoundError(
                f"Cannot create {REFERENCE_PREDICTIONS}. "
                "Please run inference.py once to generate baseline predictions."
            )
    
    print("✅ Reference data initialization complete!")
    return "initialization_complete"


def prepare_latest_inference_data(**context):
    """Find the latest inference output and prepare it for monitoring."""
    
    # Find all inference prediction files
    prediction_pattern = "datamart/gold/model_predictions/*/*predictions_*.parquet"
    prediction_files = glob.glob(prediction_pattern)
    
    if not prediction_files:
        raise FileNotFoundError(f"No inference predictions found matching {prediction_pattern}")
    
    # Get the latest file by modification time
    latest_pred_file = max(prediction_files, key=os.path.getmtime)
    print(f"Found latest prediction file: {latest_pred_file}")
    
    # Read and copy to monitoring location
    pred_df = pd.read_parquet(latest_pred_file)
    
    # Validate required columns exist
    required_cols = ['unit', 'cycle', 'RUL_predicted']
    missing = set(required_cols) - set(pred_df.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {missing}")
    
    if len(pred_df) == 0:
        raise ValueError(f"Latest prediction file {latest_pred_file} is empty")
    
    pred_df.to_parquet(GOLD_PREDICTIONS)
    print(f"Prepared {len(pred_df)} predictions for monitoring at {GOLD_PREDICTIONS}")
    
    context['ti'].xcom_push(key='latest_file_used', value=latest_pred_file)
    return latest_pred_file


def check_input_drift(**context):
    """Monitor input feature drift between reference and current data."""
    # Import inside the task to avoid serialization issues
    from scripts.monitoring.input_drift import input_drift_report
    
    print("Loading data from gold layer...")
    
    # Load parquet files
    train_df = pd.read_parquet(GOLD_REFERENCE_DATA)
    current_df = pd.read_parquet(GOLD_CURRENT_DATA)
    
    print(f"Reference data shape: {train_df.shape}")
    print(f"Current data shape: {current_df.shape}")
    
    # Run drift detection
    drift_report = input_drift_report(train_df, current_df, FEATURE_COLS)
    
    # Analyze results
    drifted_features = []
    for feature_report in drift_report["features"]:
        feature_name = feature_report["feature"]
        psi = feature_report.get("psi")
        ks_p = feature_report.get("ks_p")
        
        if psi is not None and psi > PSI_THRESHOLD:
            drifted_features.append(f"{feature_name} (PSI={psi:.3f})")
        elif ks_p is not None and ks_p < KS_PVALUE_THRESHOLD:
            drifted_features.append(f"{feature_name} (KS p={ks_p:.4f})")
    
    result = {
        "n_drifted": len(drifted_features),
        "drifted_features": drifted_features,
        "full_report": drift_report,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"Drift detection complete: {len(drifted_features)} features drifted")
    print(f"Drifted features: {drifted_features}")
    
    # Push to XCom for downstream tasks
    context['ti'].xcom_push(key='input_drift_result', value=result)
    
    return result


def check_prediction_drift(**context):
    """Monitor prediction distribution drift and monotonicity violations."""
    # Import inside the task to avoid serialization issues
    from scripts.monitoring.prediction_drift import pred_distribution_drift, monotonicity_violation_rate
    
    print("Loading prediction data from gold layer...")
    
    # Load predictions
    ref_predictions = pd.read_parquet(REFERENCE_PREDICTIONS)
    current_predictions = pd.read_parquet(GOLD_PREDICTIONS)
    
    ref_pred_values = ref_predictions[PREDICTION_COL].values
    cur_pred_values = current_predictions[PREDICTION_COL].values
    
    # Check distribution drift
    dist_drift = pred_distribution_drift(ref_pred_values, cur_pred_values)
    
    # Check monotonicity violations (RUL should decrease over cycles)
    monotonicity_rate = monotonicity_violation_rate(
        current_predictions, 
        UNIT_COL, 
        CYCLE_COL, 
        PREDICTION_COL
    )
    
    result = {
        "distribution_drift": dist_drift,
        "monotonicity_violation_rate": monotonicity_rate,
        "monotonicity_alert": monotonicity_rate > MONOTONICITY_THRESHOLD,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"Prediction drift PSI: {dist_drift['psi']:.4f}")
    print(f"Mean change: {dist_drift['mean_change_pct']:.2%}")
    print(f"Monotonicity violation rate: {monotonicity_rate:.2%}")
    
    context['ti'].xcom_push(key='prediction_drift_result', value=result)
    
    return result


def decide_alert(**context):
    """Decide whether to send an alert based on drift results."""
    ti = context['ti']
    
    # Pull results from previous tasks
    input_drift = ti.xcom_pull(task_ids='check_input_drift', key='input_drift_result')
    pred_drift = ti.xcom_pull(task_ids='check_prediction_drift', key='prediction_drift_result')
    
    # Decision logic
    should_alert = False
    alert_reasons = []
    
    # Check input drift
    if input_drift['n_drifted'] >= N_DRIFTED_THRESHOLD:
        should_alert = True
        alert_reasons.append(
            f"Input drift detected in {input_drift['n_drifted']} features: "
            f"{', '.join(input_drift['drifted_features'])}"
        )
    
    # Check prediction drift
    if pred_drift['distribution_drift']['psi'] > PSI_THRESHOLD:
        should_alert = True
        alert_reasons.append(
            f"Prediction distribution drift (PSI={pred_drift['distribution_drift']['psi']:.3f})"
        )
    
    # Check monotonicity violations
    if pred_drift['monotonicity_alert']:
        should_alert = True
        alert_reasons.append(
            f"High monotonicity violation rate ({pred_drift['monotonicity_violation_rate']:.2%})"
        )
    
    # Save decision
    alert_data = {
        "should_alert": should_alert,
        "alert_reasons": alert_reasons,
        "input_drift": input_drift,
        "prediction_drift": pred_drift
    }
    
    context['ti'].xcom_push(key='alert_decision', value=alert_data)
    
    # Return task_id for branching
    if should_alert:
        print(f"⚠️ ALERT TRIGGERED: {alert_reasons}")
        return 'send_alert_email'
    else:
        print("✅ No significant drift detected")
        return 'no_alert'


def format_alert_email(**context):
    """Format the alert email content."""
    ti = context['ti']
    alert_data = ti.xcom_pull(task_ids='decide_alert', key='alert_decision')
    
    input_drift = alert_data['input_drift']
    pred_drift = alert_data['prediction_drift']
    
    html_content = f"""
    <html>
    <body>
        <h2>🚨 Model Monitoring Alert - {datetime.now().strftime('%Y-%m-%d %H:%M')}</h2>
        
        <h3>Alert Reasons:</h3>
        <ul>
            {''.join(f'<li>{reason}</li>' for reason in alert_data['alert_reasons'])}
        </ul>
        
        <h3>Input Drift Summary:</h3>
        <ul>
            <li><b>Drifted Features:</b> {input_drift['n_drifted']}</li>
            <li><b>Features:</b> {', '.join(input_drift['drifted_features']) if input_drift['drifted_features'] else 'None'}</li>
            <li><b>Mahalanobis Mean:</b> {input_drift['full_report'].get('mahalanobis_mean', 'N/A'):.2f}</li>
            <li><b>Mahalanobis P95:</b> {input_drift['full_report'].get('mahalanobis_p95', 'N/A'):.2f}</li>
        </ul>
        
        <h3>Prediction Drift Summary:</h3>
        <ul>
            <li><b>PSI:</b> {pred_drift['distribution_drift']['psi']:.4f}</li>
            <li><b>KS p-value:</b> {pred_drift['distribution_drift']['ks_p']:.4f}</li>
            <li><b>Mean Change:</b> {pred_drift['distribution_drift']['mean_change_pct']:.2%}</li>
            <li><b>Monotonicity Violation Rate:</b> {pred_drift['monotonicity_violation_rate']:.2%}</li>
        </ul>
        
        <p><b>Action Required:</b> Please investigate the drift and consider retraining the model if necessary.</p>
        
        <p><i>This is an automated alert from the ML monitoring pipeline.</i></p>
    </body>
    </html>
    """
    
    return html_content


# ============================================================================
# DAG Definition
# ============================================================================

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['liqing.lau.2024@mitb.smu.edu.sg'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='model_drift_monitoring',
    default_args=default_args,
    description='Monitor ML model for data and prediction drift (runs monthly)',
    schedule_interval='0 0 1 * *',  # At 00:00 on day 1 of every month
    start_date=datetime(2025, 1, 1),
    catchup=False,  # Set to False to avoid backfilling
    tags=['monitoring', 'ml', 'drift-detection'],
) as dag:
    
    # Task 0: Initialize reference data (runs once, self-checks if already exists)
    initialize_task = PythonOperator(
        task_id='initialize_reference_data',
        python_callable=initialize_reference_data,
        provide_context=True,
    )
    
    # Task 1: Prepare latest inference data for monitoring
    prepare_data_task = PythonOperator(
        task_id='prepare_latest_data',
        python_callable=prepare_latest_inference_data,
        provide_context=True,
    )
    
    # Task 2: Check input drift
    input_drift_task = PythonOperator(
        task_id='check_input_drift',
        python_callable=check_input_drift,
        provide_context=True,
    )
    
    # Task 3: Check prediction drift
    prediction_drift_task = PythonOperator(
        task_id='check_prediction_drift',
        python_callable=check_prediction_drift,
        provide_context=True,
    )
    
    # Task 3: Decide whether to alert (branch)
    decide_alert_task = BranchPythonOperator(
        task_id='decide_alert',
        python_callable=decide_alert,
        provide_context=True,
    )
    
    # Task 4a: Send alert email if drift detected
    send_alert_task = EmailOperator(
        task_id='send_alert_email',
        to=['liqing.lau.2020@scis.smu.edu.sg'], 
        subject='🚨 ML Model Drift Alert - {{ ds }}',
        html_content="{{ ti.xcom_pull(task_ids='format_alert') }}",
    )
    
    # Task 4b: No alert needed
    no_alert_task = EmptyOperator(
        task_id='no_alert',
    )
    
    # Format alert email content
    format_alert_task = PythonOperator(
        task_id='format_alert',
        python_callable=format_alert_email,
        provide_context=True,
    )
    
    # Set task dependencies
    initialize_task >> prepare_data_task >> [input_drift_task, prediction_drift_task]
    [input_drift_task, prediction_drift_task] >> decide_alert_task
    decide_alert_task >> [send_alert_task, no_alert_task]
    decide_alert_task >> format_alert_task >> send_alert_task