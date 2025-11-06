# ==============================================================
# MODEL TRAINING PIPELINE (Ridge + Random Forest - Realistic Version)
# ==============================================================

import os
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------------------
# LOAD GOLD DATA (from Parquet)
# --------------------------------------------------------------
feature_path = "datamart/gold/feature_store.parquet"
label_path   = "datamart/gold/label_store.parquet"

df_feat = pd.read_parquet(feature_path)
df_label = pd.read_parquet(label_path)

# Ensure alignment for merge
if not all(col in df_feat.columns for col in ["unit", "cycle"]):
    raise KeyError("Feature store missing required columns ['unit', 'cycle']")
if not all(col in df_label.columns for col in ["unit", "cycle"]):
    raise KeyError("Label store missing required columns ['unit', 'cycle']")

df = df_feat.merge(df_label, on=["unit", "cycle"], how="inner")

print(f"✅ Loaded {len(df):,} rows across {df['unit'].nunique()} engines")
print(f"🧩 Features: {df_feat.shape[1]-2:,} | Target: 'RUL'")

# --------------------------------------------------------------
# PREPARE FEATURES / LABELS
# --------------------------------------------------------------
drop_cols = ["unit", "cycle", "RUL"]
X = df.drop(columns=drop_cols)
y = df["RUL"]

# --------------------------------------------------------------
# SPLIT ENGINE-WISE (avoid leakage)
# --------------------------------------------------------------
np.random.seed(42)
engine_ids = df["unit"].unique()
np.random.shuffle(engine_ids)

n = len(engine_ids)
train_cut = int(n * 0.7)
val_cut   = int(n * 0.85)

train_units = engine_ids[:train_cut]
val_units   = engine_ids[train_cut:val_cut]
test_units  = engine_ids[val_cut:]

def subset(units_subset):
    mask = df["unit"].isin(units_subset)
    return df.loc[mask, X.columns], df.loc[mask, "RUL"]

X_train, y_train = subset(train_units)
X_val, y_val     = subset(val_units)
X_test, y_test   = subset(test_units)

print(f"🧩 Split by engines → Train: {len(train_units)}, Val: {len(val_units)}, Test: {len(test_units)}")

# --------------------------------------------------------------
# METRIC HELPER
# --------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"📊 {name} → MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
    return mae, rmse, r2

# --------------------------------------------------------------
# MODEL 1 — RIDGE REGRESSION (Regularized Linear)
# --------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("\n🔹 MODEL 1: RIDGE REGRESSION (Regularized Linear)")

ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

param_grid_ridge = {"ridge__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]}
grid_ridge = GridSearchCV(
    ridge_pipe,
    param_grid_ridge,
    scoring="r2",
    cv=3,
    n_jobs=-1
)
grid_ridge.fit(X_train, y_train)

best_ridge = grid_ridge.best_estimator_
print(f"🏆 Best Ridge Alpha: {grid_ridge.best_params_['ridge__alpha']}")

pred_val_ridge = best_ridge.predict(X_val)
mae_ridge, rmse_ridge, r2_ridge = evaluate("Ridge Regression (Val)", y_val, pred_val_ridge)

# --------------------------------------------------------------
# MODEL 2 — RANDOM FOREST (Limited Depth for Realistic Performance)
# --------------------------------------------------------------
print("\n🔹 MODEL 2: RANDOM FOREST (Limited Depth for Realistic Performance)")

rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,            
    max_features="sqrt",
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

pred_val_rf = rf.predict(X_val)
mae_rf, rmse_rf, r2_rf = evaluate("Random Forest (Val)", y_val, pred_val_rf)

pred_test_rf = rf.predict(X_test)
mae_test, rmse_test, r2_test = evaluate("Random Forest (Test)", y_test, pred_test_rf)

# --------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------
print("\n============================================================")
print("🏁 SUMMARY")
print(f"Ridge Regression Val R²: {r2_ridge:.3f}")
print(f"Random Forest Val R²:     {r2_rf:.3f}")
print(f"Random Forest Test R²:    {r2_test:.3f}")
print("============================================================")
