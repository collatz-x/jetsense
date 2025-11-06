# jetsense

## 📖 Overview

This project implements an **end-to-end ML pipeline for predictive maintenance of turbofan jet engines**. Using sensor data from aircraft engines, the system predicts **Remaining Useful Life (RUL)** and monitors for data drift to ensure model reliability.

### What It Does:
- 🔧 **Predicts engine failures** before they happen
- 📊 **Processes sensor data** through Bronze → Silver → Gold layers (Medallion Architecture)
- 🤖 **Runs ML inference** on new data daily
- 📈 **Monitors data & prediction drift** to detect model degradation
- 🚨 **Sends alerts** when drift is detected via email or file

### Business Value:
- ✈️ Reduce unplanned maintenance downtime
- 💰 Optimize maintenance scheduling
- 🛡️ Improve flight safety
- 📉 Lower maintenance costs

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose installed
- Gmail account (for email alerts)

### 1. Clone and Setup
```bash
# Clone the repository
git clone 
cd turbofan-predictive-maintenance

# Copy .env.example and rename
cp .env.example .env

# Start Airflow & Jupyter
docker-compose up -build
```

### 2. Access Airflow UI
- Open browser: http://localhost:8080
- Default credentials: `admin` / `airflow`

### 3. Trigger Your First Run
```bash
# Run ETL pipeline
docker-compose exec airflow-scheduler airflow dags trigger etl_bronze_silver_gold

# Check the UI to see progress
# DAGs → etl_bronze_silver_gold → Graph View
```

---

## 📧 Email Alerts Setup (Gmail)

The monitoring system sends email alerts when drift is detected. Here's how to enable it:

1. **Go to Gmail App Passwords:**
   - Visit: https://myaccount.google.com/apppasswords
   - (Or: Google Account → Security → 2-Step Verification → App passwords)

2. **Enable 2-Factor Authentication** (if not already enabled)
   - Required for app passwords
   - Settings → Security → 2-Step Verification

3. **Create App Password:**
   - Click "Select app" → Choose "Mail"
   - Click "Select device" → Choose "Other (Custom name)"
   - Enter name: `Airflow Alerts`
   - Click **Generate**
   
4. **Copy the 16-character password**
   - Example: `abcd efgh ijkl mnop`
   - ⚠️ **Remove spaces when copying!** → `abcdefghijklmnop`

---

## 🚨 Alert System

### How Alerts Work

The system monitors two types of drift:

1. **Input Drift** (Feature Distribution Changes)
   - Detects when sensor data patterns change
   - Uses PSI (Population Stability Index) and KS tests
   - Alerts if >3 features drift significantly

2. **Prediction Drift** (Model Output Changes)
   - Monitors RUL prediction distributions
   - Checks for monotonicity violations (RUL should decrease over time)
   - Alerts if prediction patterns are unusual

---

## 📊 Pipeline Details

### 1. ETL Pipeline (`etl_bronze_silver_gold`)
**Schedule:** Daily at 10:00 AM

**What it does:**
- **Bronze Layer:** Ingests raw sensor data from text files
- **Silver Layer:** Cleans and consolidates data
- **Gold Layer:** Engineers 50+ features for ML

**Output:** Feature-ready data in `datamart/gold_inference/`

### 2. ML Inference (`model_inference_batch`)
**Schedule:** Daily at 12:00 AM (midnight)

**What it does:**
- Loads latest trained model
- Runs predictions on new engine data
- Saves RUL predictions as Parquet files

**Output:** Predictions in `datamart/gold_inference/model_predictions/`

### 3. Drift Monitoring (`model_drift_monitoring`)
**Schedule:** Monthly on 1st at 12:00 AM

**What it does:**
- Compares current data vs. baseline (reference) data
- Detects feature drift using statistical tests
- Checks prediction distribution changes
- Sends alerts if drift exceeds thresholds

**Output:** Alert files in email

---

## 📈 Monitoring Dashboard

### View in Airflow UI

1. **DAG Status:**
   - http://localhost:8080
   - Green = Success, Red = Failed, Yellow = Running

## Dataset

- NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS)