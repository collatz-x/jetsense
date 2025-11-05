from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pandas as pd
import json
import glob
import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project root to path
sys.path.insert(0, "/opt/airflow")

SMTP_CONFIG = {
    'host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
    'port': int(os.getenv('SMTP_PORT', '587')),
    'user': os.getenv('SMTP_USER', ''),
    'password': os.getenv('SMTP_PASSWORD', ''),
    'from_email': os.getenv('SMTP_USER', ''),
    'enabled': os.getenv('SMTP_ENABLED', 'false').lower() == 'true'
}

ALERT_EMAIL = os.getenv('ALERT_EMAIL', 'liqing.lau.2024@mitb.smu.edu.sg')

# ============================================================================
# Configuration
# ============================================================================
GOLD_REFERENCE_DATA = "datamart/gold/reference_data.parquet"
GOLD_CURRENT_DATA = "datamart/gold/feature_store.parquet"
GOLD_PREDICTIONS = "datamart/gold/latest_predictions.parquet"
REFERENCE_PREDICTIONS = "datamart/gold/reference_predictions.parquet"
ALERT_OUTPUT_DIR = "monitoring/alerts"

FEATURE_COLS = ['op_setting_1', 'op_setting_2', 'op_setting_3', 'T24', 'T30', 'T50', 'P15', 'P30', 'Nf', 'Nc', 'phi', 'BPR', 'htBleed', 'W31', 'W32', 'T24_roll5_mean', 'T24_delta1', 'T30_roll5_mean', 'T30_delta1', 'T50_roll5_mean', 'T50_delta1', 'P15_roll5_mean', 'P15_delta1', 'P30_roll5_mean', 'P30_delta1', 'Nf_roll5_mean', 'Nf_delta1', 'Nc_roll5_mean', 'Nc_delta1', 'T_ratio_24_30', 'T_ratio_30_50', 'P_ratio_15_30', 'N_ratio_f_c', 'cycle_norm', 'health_index']
UNIT_COL = "unit"
CYCLE_COL = "cycle"
PREDICTION_COL = "RUL_predicted"

# Thresholds
PSI_THRESHOLD = os.getenv('PSI_THRESHOLD', '0.25')
KS_PVALUE_THRESHOLD = os.getenv('KS_PVALUE_THRESHOLD', '0.05')
N_DRIFTED_THRESHOLD = os.getenv('N_DRIFTED_THRESHOLD', '3')
MONOTONICITY_THRESHOLD = os.getenv('MONOTONICITY_THRESHOLD', '0.1')


# ============================================================================
# Email Helper Functions
# ============================================================================

def send_email_alert(subject, html_content, to_email):
    """
    Send email using SMTP configuration from environment variables.
    Falls back gracefully if SMTP not configured.
    """
    if not SMTP_CONFIG['enabled']:
        print("📧 SMTP not enabled (set SMTP_ENABLED=true to enable)")
        return False
    
    if not SMTP_CONFIG['user'] or not SMTP_CONFIG['password']:
        print("⚠️  SMTP credentials not configured (SMTP_USER and SMTP_PASSWORD needed)")
        return False
    
    try:
        print(f"📧 Sending email to {to_email}...")
        print(f"   SMTP Host: {SMTP_CONFIG['host']}:{SMTP_CONFIG['port']}")
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = SMTP_CONFIG['from_email']
        msg['To'] = to_email
        
        # Attach HTML content
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Send email
        with smtplib.SMTP(SMTP_CONFIG['host'], SMTP_CONFIG['port'], timeout=10) as server:
            server.starttls()
            server.login(SMTP_CONFIG['user'], SMTP_CONFIG['password'])
            server.send_message(msg)
        
        print("✅ Email sent successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {str(e)}")
        print("   Alert will be saved to file instead")
        return False


# ============================================================================
# Task Functions
# ============================================================================

def initialize_reference_data_wrapper(**context):
    """Initialize reference baseline files if they don't exist."""
    print("\n🔧 Starting Reference Data Initialization...")
    os.chdir("/opt/airflow")
    
    try:
        ref_data_exists = os.path.exists(GOLD_REFERENCE_DATA)
        ref_pred_exists = os.path.exists(REFERENCE_PREDICTIONS)
        
        if ref_data_exists and ref_pred_exists:
            print("✅ Reference data already exists, skipping initialization")
            context['ti'].xcom_push(key='init_status', value='skipped')
            return "skip_initialization"
        
        print("🔧 Initializing reference baseline data for first-time setup...")
        
        if not ref_data_exists:
            print(f"Creating {GOLD_REFERENCE_DATA}...")
            feature_df = pd.read_parquet("datamart/gold/feature_store.parquet")
            feature_df.to_parquet(GOLD_REFERENCE_DATA)
            print(f"✅ Created reference data with {len(feature_df)} rows")
        
        if not ref_pred_exists:
            print(f"Creating {REFERENCE_PREDICTIONS}...")
            oot_pred_path = "datamart/gold/oot_predictions_ridgeregression.parquet"
            
            if os.path.exists(oot_pred_path):
                pred_df = pd.read_parquet(oot_pred_path)
                
                if 'RUL_predicted' not in pred_df.columns:
                    pred_cols = [c for c in pred_df.columns if 'rul' in c.lower() or 'pred' in c.lower()]
                    if pred_cols:
                        print(f"Renaming column {pred_cols[0]} to RUL_predicted")
                        pred_df = pred_df.rename(columns={pred_cols[0]: 'RUL_predicted'})
                
                pred_df.to_parquet(REFERENCE_PREDICTIONS)
                print(f"✅ Created reference predictions from OOT data with {len(pred_df)} rows")
            else:
                print("⚠️  No OOT predictions found. Please run inference once to generate reference predictions.")
                raise FileNotFoundError(
                    f"Cannot create {REFERENCE_PREDICTIONS}. "
                    "Please run inference.py once to generate baseline predictions."
                )
        
        print("✅ Reference data initialization complete!")
        context['ti'].xcom_push(key='init_status', value='completed')
        return "initialization_complete"
        
    except Exception as e:
        print(f"❌ Reference initialization failed: {str(e)}")
        raise


def prepare_latest_inference_data_wrapper(**context):
    """Find the latest inference output and prepare it for monitoring."""
    print("\n📂 Preparing Latest Inference Data...")
    os.chdir("/opt/airflow")
    
    try:
        prediction_pattern = "datamart/gold/model_predictions/*/*predictions_*.parquet"
        prediction_files = glob.glob(prediction_pattern)
        
        if not prediction_files:
            raise FileNotFoundError(f"No inference predictions found matching {prediction_pattern}")
        
        latest_pred_file = max(prediction_files, key=os.path.getmtime)
        print(f"📊 Found latest prediction file: {latest_pred_file}")
        
        pred_df = pd.read_parquet(latest_pred_file)
        
        required_cols = ['unit', 'cycle', 'RUL_predicted']
        missing = set(required_cols) - set(pred_df.columns)
        if missing:
            raise ValueError(f"Prediction file missing columns: {missing}")
        
        if len(pred_df) == 0:
            raise ValueError(f"Latest prediction file {latest_pred_file} is empty")
        
        pred_df.to_parquet(GOLD_PREDICTIONS)
        print(f"✅ Prepared {len(pred_df)} predictions for monitoring")
        
        context['ti'].xcom_push(key='latest_file_used', value=latest_pred_file)
        context['ti'].xcom_push(key='prediction_count', value=len(pred_df))
        
        return latest_pred_file
        
    except Exception as e:
        print(f"❌ Data preparation failed: {str(e)}")
        raise


def check_input_drift_wrapper(**context):
    """Monitor input feature drift between reference and current data."""
    print("\n🔍 Starting Input Drift Detection...")
    os.chdir("/opt/airflow")
    
    try:
        from scripts.monitoring.input_drift import input_drift_report
        
        print("📊 Loading data from gold layer...")
        train_df = pd.read_parquet(GOLD_REFERENCE_DATA)
        current_df = pd.read_parquet(GOLD_CURRENT_DATA)
        
        print(f"   Reference data shape: {train_df.shape}")
        print(f"   Current data shape: {current_df.shape}")
        
        print("🔬 Running drift detection analysis...")
        drift_report = input_drift_report(train_df, current_df, FEATURE_COLS)
        
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
        
        print(f"✅ Drift detection complete: {len(drifted_features)} features drifted")
        if drifted_features:
            print(f"   ⚠️  Drifted features: {', '.join(drifted_features[:5])}")
            if len(drifted_features) > 5:
                print(f"   ... and {len(drifted_features) - 5} more")
        else:
            print("   ✓ No significant drift detected")
        
        context['ti'].xcom_push(key='input_drift_result', value=result)
        return result
        
    except Exception as e:
        print(f"❌ Input drift check failed: {str(e)}")
        raise


def check_prediction_drift_wrapper(**context):
    """Monitor prediction distribution drift and monotonicity violations."""
    print("\n📈 Starting Prediction Drift Detection...")
    os.chdir("/opt/airflow")
    
    try:
        from scripts.monitoring.prediction_drift import pred_distribution_drift, monotonicity_violation_rate
        
        print("📊 Loading prediction data from gold layer...")
        ref_predictions = pd.read_parquet(REFERENCE_PREDICTIONS)
        current_predictions = pd.read_parquet(GOLD_PREDICTIONS)
        
        print(f"   Reference predictions: {len(ref_predictions)}")
        print(f"   Current predictions: {len(current_predictions)}")
        
        ref_pred_values = ref_predictions[PREDICTION_COL].values
        cur_pred_values = current_predictions[PREDICTION_COL].values
        
        print("🔬 Analyzing prediction distribution...")
        dist_drift = pred_distribution_drift(ref_pred_values, cur_pred_values)
        
        print("🔬 Checking monotonicity violations...")
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
        
        print(f"✅ Prediction drift analysis complete:")
        print(f"   PSI: {dist_drift['psi']:.4f}")
        print(f"   Mean change: {dist_drift['mean_change_pct']:.2%}")
        print(f"   Monotonicity violation rate: {monotonicity_rate:.2%}")
        
        if monotonicity_rate > MONOTONICITY_THRESHOLD:
            print(f"   ⚠️  High violation rate detected!")
        
        context['ti'].xcom_push(key='prediction_drift_result', value=result)
        return result
        
    except Exception as e:
        print(f"❌ Prediction drift check failed: {str(e)}")
        raise


def decide_alert_wrapper(**context):
    """Decide whether to send an alert based on drift results."""
    print("\n⚖️  Making Alert Decision...")
    os.chdir("/opt/airflow")
    
    try:
        ti = context['ti']
        
        input_drift = ti.xcom_pull(task_ids='check_input_drift', key='input_drift_result')
        pred_drift = ti.xcom_pull(task_ids='check_prediction_drift', key='prediction_drift_result')
        
        should_alert = False
        alert_reasons = []
        
        if input_drift['n_drifted'] >= N_DRIFTED_THRESHOLD:
            should_alert = True
            alert_reasons.append(
                f"Input drift detected in {input_drift['n_drifted']} features: "
                f"{', '.join(input_drift['drifted_features'][:5])}"
            )
        
        if pred_drift['distribution_drift']['psi'] > PSI_THRESHOLD:
            should_alert = True
            alert_reasons.append(
                f"Prediction distribution drift (PSI={pred_drift['distribution_drift']['psi']:.3f})"
            )
        
        if pred_drift['monotonicity_alert']:
            should_alert = True
            alert_reasons.append(
                f"High monotonicity violation rate ({pred_drift['monotonicity_violation_rate']:.2%})"
            )
        
        alert_data = {
            "should_alert": should_alert,
            "alert_reasons": alert_reasons,
            "input_drift": input_drift,
            "prediction_drift": pred_drift
        }
        
        context['ti'].xcom_push(key='alert_decision', value=alert_data)
        
        if should_alert:
            print(f"⚠️  ALERT TRIGGERED:")
            for reason in alert_reasons:
                print(f"   • {reason}")
            return 'send_alert'
        else:
            print("✅ No significant drift detected - no alert needed")
            return 'no_alert'
            
    except Exception as e:
        print(f"❌ Alert decision failed: {str(e)}")
        raise


def send_alert_wrapper(**context):
    """Send alert via email (if configured) and save to file (always)."""
    print("\n📧 Sending Alert...")
    os.chdir("/opt/airflow")
    
    try:
        ti = context['ti']
        alert_data = ti.xcom_pull(task_ids='decide_alert', key='alert_decision')
        
        input_drift = alert_data['input_drift']
        pred_drift = alert_data['prediction_drift']
        
        # Format Mahalanobis values safely
        maha_mean = input_drift['full_report'].get('mahalanobis_mean')
        maha_p95 = input_drift['full_report'].get('mahalanobis_p95')
        maha_mean_str = f"{maha_mean:.2f}" if maha_mean is not None else "N/A"
        maha_p95_str = f"{maha_p95:.2f}" if maha_p95 is not None else "N/A"
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Model Drift Alert</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 20px auto; padding: 20px; }}
                h2 {{ color: #d9534f; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                td, th {{ padding: 10px; border: 1px solid #dee2e6; text-align: left; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .alert-box {{ background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; padding: 15px; margin: 20px 0; }}
                .warning-box {{ background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin: 20px 0; }}
                .metric {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h2>🚨 Model Monitoring Alert</h2>
            <p><b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            
            <div class="alert-box">
                <h3 style="margin-top: 0; color: #721c24;">Alert Reasons:</h3>
                <ul>
                    {''.join(f'<li>{reason}</li>' for reason in alert_data['alert_reasons'])}
                </ul>
            </div>
            
            <h3>📊 Input Drift Summary</h3>
            <table>
                <tr><td class="metric">Drifted Features</td><td>{input_drift['n_drifted']}</td></tr>
                <tr><td class="metric">Features List</td><td>{', '.join(input_drift['drifted_features']) if input_drift['drifted_features'] else 'None'}</td></tr>
                <tr><td class="metric">Mahalanobis Mean</td><td>{maha_mean_str}</td></tr>
                <tr><td class="metric">Mahalanobis P95</td><td>{maha_p95_str}</td></tr>
            </table>
            
            <h3>📈 Prediction Drift Summary</h3>
            <table>
                <tr><td class="metric">PSI</td><td>{pred_drift['distribution_drift']['psi']:.4f}</td></tr>
                <tr><td class="metric">KS p-value</td><td>{pred_drift['distribution_drift']['ks_p']:.4f}</td></tr>
                <tr><td class="metric">Mean Change</td><td>{pred_drift['distribution_drift']['mean_change_pct']:.2%}</td></tr>
                <tr><td class="metric">Monotonicity Violations</td><td>{pred_drift['monotonicity_violation_rate']:.2%}</td></tr>
            </table>
            
            <div class="warning-box">
                <h4 style="margin-top: 0;">⚠️  Action Required</h4>
                <ul>
                    <li>Review feature distributions for drifted features</li>
                    <li>Check data quality and ingestion pipeline</li>
                    <li>Evaluate model performance on recent data</li>
                    <li>Consider retraining the model if drift persists</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save to file (always)
        os.makedirs(ALERT_OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        json_path = os.path.join(ALERT_OUTPUT_DIR, f"alert_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(alert_data, f, indent=2, default=str)
        print(f"✅ JSON report saved: {json_path}")
        
        # Save HTML
        html_path = os.path.join(ALERT_OUTPUT_DIR, f"alert_{timestamp}.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        print(f"✅ HTML report saved: {html_path}")
        
        # Try to send email
        email_sent = send_email_alert(
            subject=f'🚨 ML Model Drift Alert - {datetime.now().strftime("%Y-%m-%d")}',
            html_content=html_content,
            to_email=ALERT_EMAIL
        )
        
        # Summary
        print("\n" + "="*70)
        print("🚨 DRIFT ALERT SUMMARY")
        print("="*70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nAlert Reasons:")
        for reason in alert_data['alert_reasons']:
            print(f"  • {reason}")
        print(f"\nEmail sent: {'✅ Yes' if email_sent else '❌ No (saved to file)'}")
        print(f"Reports saved to: {ALERT_OUTPUT_DIR}/")
        print("="*70 + "\n")
        
        context['ti'].xcom_push(key='email_sent', value=email_sent)
        context['ti'].xcom_push(key='report_paths', value={'json': json_path, 'html': html_path})
        
        return html_path
        
    except Exception as e:
        print(f"❌ Alert sending failed: {str(e)}")
        raise


# ============================================================================
# DAG Definition
# ============================================================================

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': [ALERT_EMAIL],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='model_drift_monitoring',
    default_args=default_args,
    description='Monitor ML model for data and prediction drift (Email + File alerts)',
    schedule_interval='0 0 1 * *',
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=['monitoring', 'ml', 'drift-detection', 'medallion-architecture'],
) as dag:
    
    initialize_task = PythonOperator(
        task_id='initialize_reference_data',
        python_callable=initialize_reference_data_wrapper,
        provide_context=True,
    )
    
    prepare_data_task = PythonOperator(
        task_id='prepare_latest_data',
        python_callable=prepare_latest_inference_data_wrapper,
        provide_context=True,
    )
    
    input_drift_task = PythonOperator(
        task_id='check_input_drift',
        python_callable=check_input_drift_wrapper,
        provide_context=True,
    )
    
    prediction_drift_task = PythonOperator(
        task_id='check_prediction_drift',
        python_callable=check_prediction_drift_wrapper,
        provide_context=True,
    )
    
    decide_alert_task = BranchPythonOperator(
        task_id='decide_alert',
        python_callable=decide_alert_wrapper,
        provide_context=True,
    )
    
    send_alert_task = PythonOperator(
        task_id='send_alert',
        python_callable=send_alert_wrapper,
        provide_context=True,
    )
    
    no_alert_task = EmptyOperator(
        task_id='no_alert',
    )
    
    # Task dependencies
    initialize_task >> prepare_data_task >> [input_drift_task, prediction_drift_task]
    [input_drift_task, prediction_drift_task] >> decide_alert_task
    decide_alert_task >> send_alert_task
    decide_alert_task >> no_alert_task