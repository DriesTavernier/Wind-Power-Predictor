
# -*- coding: utf-8 -*-
"""
Wind Turbine Power Prediction
Artificial Neural Network (ANN)
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
# =====================================================================
# 1. LOAD DATA
# =====================================================================

data_train = pd.read_csv(ROOT / "data" / "clean" / "wind_turbine_scada_train.csv")
data_test = pd.read_csv(ROOT / "data" / "clean" / "wind_turbine_scada_test.csv")


# =====================================================================
# 2. SELECT INPUT AND OUTPUT VARIABLES
# =====================================================================

# IMPORTANT:
# Power Deviation is excluded because it is calculated using
# actual power, which is also our target variable.
# Including it would cause target leakage.

X_cols = [
    "Wind Speed (m/s)",
    "Cp"
]

y_col = "LV ActivePower (kW)"

X_train = data_train[X_cols]
y_train = data_train[y_col]

X_test = data_test[X_cols]
y_test = data_test[y_col]


# =====================================================================
# 3. DEFINE NEURAL NETWORK
# =====================================================================

# Feature scaling + neural network
nn_pipeline = Pipeline([
    ("scaler", StandardScaler()),

    ("neural_network", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42
    ))
])

# Also scale target variable during training
model = TransformedTargetRegressor(
    regressor=nn_pipeline,
    transformer=StandardScaler()
)


# =====================================================================
# 4. TRAIN MODEL
# =====================================================================

model.fit(X_train, y_train)

print("\n--- MODEL TRAINING ---")
print("Training completed.")

# Training information
nn = model.regressor_.named_steps["neural_network"]

print("Iterations:", nn.n_iter_)
print("Final training loss:", nn.loss_)


# =====================================================================
# 5. PREDICT TEST SET
# =====================================================================

y_pred = model.predict(X_test)

data_test['predicted'] = y_pred

data_test.to_csv(
    ROOT / "data" / "clean" / "wind_turbine_scada_test_nn.csv",
    index=False
)





