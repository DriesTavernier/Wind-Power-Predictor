# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:32:35 2026

@author: dries
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from evaluate_performance import evaluate_model



data_train = pd.read_csv("../data/clean/wind_turbine_scada_train.csv")
data_test = pd.read_csv("../data/clean/wind_turbine_scada_test.csv")
# =====================================================================
# 1. Select columns to predicts and to predict on
# =====================================================================
# Controleer of jouw exacte kolomnamen hieronder overeenkomen.
# Als ze anders heten (bijv. met hoofdletters of een spatie), pas ze hier aan:
X_cols = ['Wind Speed (m/s)', 'Cp', "Power Deviation (kW)"] 
y_col = 'LV ActivePower (kW)'


X = data_train[X_cols]
y = data_train[y_col]

data_test_x = data_test[X_cols]
data_test_y = data_test[y_col]

# =====================================================================
# 2. Train model
# =====================================================================
model = LinearRegression()
model.fit(X, y)

# =====================================================================
# 4. Predict the test set
# =====================================================================
y_pred = model.predict(data_test_x)
# =====================================================================
# 4. Test accuracy
# =====================================================================
data_test['lm_predicted'] = y_pred


# 3.1 Actual vs theoretical power curve
plt.figure(figsize=(10, 6))
plt.scatter(
    data_test["Wind Speed (m/s)"],
    data_test["LV ActivePower (kW)"],
    s=5,
    alpha=0.25,
    label="Actual power"
)

plt.scatter(
    data_test["Wind Speed (m/s)"],
    data_test["lm_predicted"],
    s=5,
    alpha=0.15,
    label="predicted"
)
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Power (kW)")
plt.title("Actual vs Theoretical Wind Turbine Power Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Coëfficiënten:", model.coef_)
print("Intercept:", model.intercept_)

# =====================================================================
# MODEL PERFORMANCE
# =====================================================================
evaluate_model(data_test["LV ActivePower (kW)"], data_test["lm_predicted"],'Linear Regression')


