from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, "/opt/airflow")


# ============================================================================
# Task Functions (wrap main.py functions with Airflow context)
# ============================================================================

def bronze_task_wrapper(**context):
    """Bronze Layer: Raw data ingestion from text files to structured CSVs."""
    # Import inside the task to avoid serialization issues
    from main import create_spark_session, process_bronze_layer
    
    print("\n🟤 Starting Bronze Layer Processing...")
    os.chdir("/opt/airflow")
    
    spark = create_spark_session()
    try:
        process_bronze_layer(spark)
        context['ti'].xcom_push(key='bronze_status', value='success')
        print("✅ Bronze layer complete!")
    except Exception as e:
        print(f"❌ Bronze layer failed: {str(e)}")
        raise
    finally:
        spark.stop()


def silver_task_wrapper(**context):
    """Silver Layer: Data consolidation and cleaning."""
    # Import inside the task to avoid serialization issues
    from main import create_spark_session, process_silver_layer
    
    print("\n⚪ Starting Silver Layer Processing...")
    os.chdir("/opt/airflow")
    
    spark = create_spark_session()
    try:
        df = process_silver_layer(spark)
        context['ti'].xcom_push(key='silver_row_count', value=df.count())
        context['ti'].xcom_push(key='silver_engine_count', value=df.select("unit").distinct().count())
        print("✅ Silver layer complete!")
    except Exception as e:
        print(f"❌ Silver layer failed: {str(e)}")
        raise
    finally:
        spark.stop()


def clean_silver_task_wrapper(**context):
    """Silver Layer: Feature selection and cleanup."""
    # Import inside the task to avoid serialization issues
    from main import create_spark_session, clean_silver_features
    
    print("\n⚪ Starting Silver Feature Cleaning...")
    os.chdir("/opt/airflow")
    
    spark = create_spark_session()
    try:
        cleaned_path = clean_silver_features(spark)
        context['ti'].xcom_push(key='cleaned_silver_path', value=cleaned_path)
        print("✅ Silver cleaning complete!")
    except Exception as e:
        print(f"❌ Silver cleaning failed: {str(e)}")
        raise
    finally:
        spark.stop()


def gold_task_wrapper(**context):
    """Gold Layer: Feature engineering and ML preparation."""
    # Import inside the task to avoid serialization issues
    from main import create_spark_session, process_gold_layer
    
    print("\n🟡 Starting Gold Layer Processing...")
    os.chdir("/opt/airflow")
    
    spark = create_spark_session()
    try:
        ti = context['ti']
        cleaned_path = ti.xcom_pull(task_ids='clean_silver_features', key='cleaned_silver_path')
        
        if not cleaned_path:
            cleaned_path = "datamart/silver/silver_feature_cleaned.parquet"
        
        gold_df, feature_store, label_store = process_gold_layer(spark, cleaned_path)
        
        context['ti'].xcom_push(key='gold_row_count', value=len(gold_df))
        context['ti'].xcom_push(key='gold_engine_count', value=gold_df['unit'].nunique())
        print("✅ Gold layer complete!")
    except Exception as e:
        print(f"❌ Gold layer failed: {str(e)}")
        raise
    finally:
        spark.stop()


def pipeline_summary_task(**context):
    """Display pipeline completion summary with statistics."""
    ti = context['ti']
    
    gold_engines = ti.xcom_pull(task_ids='process_gold_layer', key='gold_engine_count')
    gold_rows = ti.xcom_pull(task_ids='process_gold_layer', key='gold_row_count')
    
    print("\n✅ FULL ETL PIPELINE COMPLETE!")
    print("=" * 70)
    print("📁 Output Files:")
    print(f"   Gold Table:      datamart/gold/gold_full.parquet")
    print(f"   Feature Store:   datamart/gold/feature_store.parquet")
    print(f"   Label Store:     datamart/gold/label_store.parquet")
    print(f"   Normalization:   datamart/gold/feature_metadata.csv")
    print("=" * 70)
    print(f"📊 Final Statistics:")
    print(f"   Engines:         {gold_engines}")
    print(f"   Total Rows:      {gold_rows:,}")
    print("=" * 70)


# ============================================================================
# DAG Definition
# ============================================================================

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["liqing.lau.2024@mitb.smu.edu.sg"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="etl_bronze_silver_gold",
    default_args=default_args,
    description="ETL pipeline for Bronze → Silver → Gold data processing (multi-task)",
    schedule_interval="0 10 * * *",  # Daily at 10:00 AM
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=["etl", "data-processing", "medallion-architecture"],
) as dag:

    bronze_task = PythonOperator(
        task_id="process_bronze_layer",
        python_callable=bronze_task_wrapper,
        provide_context=True,
    )

    silver_task = PythonOperator(
        task_id="process_silver_layer",
        python_callable=silver_task_wrapper,
        provide_context=True,
    )

    clean_silver_task = PythonOperator(
        task_id="clean_silver_features",
        python_callable=clean_silver_task_wrapper,
        provide_context=True,
    )

    gold_task = PythonOperator(
        task_id="process_gold_layer",
        python_callable=gold_task_wrapper,
        provide_context=True,
    )

    summary_task = PythonOperator(
        task_id="pipeline_summary",
        python_callable=pipeline_summary_task,
        provide_context=True,
    )

    # Define task dependencies
    bronze_task >> silver_task >> clean_silver_task >> gold_task >> summary_task

