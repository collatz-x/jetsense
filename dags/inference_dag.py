from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add scripts to path
sys.path.insert(0, "/opt/airflow")

from scripts.inference.inference import main as inference_main

# ============================================================================
# Configuration
# ============================================================================
DEFAULT_MODEL_NAME = "engine_rul_prediction_2025-11-02.pkl"
DEFAULT_UNITS = "all"  # Or specify units like "1,5,10"

# ============================================================================
# Task Functions
# ============================================================================


def run_inference(**context):
    """Run inference using the inference script."""
    model_name = context["dag_run"].conf.get("modelname", DEFAULT_MODEL_NAME)
    units = context["dag_run"].conf.get("units", DEFAULT_UNITS)

    print(f"Running inference with model: {model_name}, units: {units}")

    # Change to the Airflow working directory to handle relative paths
    os.chdir("/opt/airflow")

    # Call the inference main function
    inference_main(modelname=model_name, units=units)

    print("Inference completed successfully!")


# ============================================================================
# DAG Definition
# ============================================================================

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["liqing.lau.2024@mitb.smu.edu.sg"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="model_inference_batch",
    default_args=default_args,
    description="Run batch inference for RUL prediction",
    schedule_interval="@daily",  # Runs daily at midnight
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=["inference", "ml", "rul-prediction"],
    params={"modelname": DEFAULT_MODEL_NAME, "units": DEFAULT_UNITS},
) as dag:

    inference_task = PythonOperator(
        task_id="run_inference",
        python_callable=run_inference,
        provide_context=True,
    )

