# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:32:35 2026

@author: dries
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import pandas as pd
from sklearn.linear_model import LinearRegression


data_train = pd.read_csv(ROOT / "data" / "clean" / "wind_turbine_scada_train.csv")
data_test = pd.read_csv(ROOT / "data" / "clean" / "wind_turbine_scada_test.csv")
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
print("Coëfficiënten:", model.coef_)
print("Intercept:", model.intercept_)

# =====================================================================
# 4. Predict the test set
# =====================================================================
y_pred = model.predict(data_test_x)

data_test['predicted'] = y_pred

data_test.to_csv(
    ROOT / "data" / "clean" / "wind_turbine_scada_test_lm.csv",
    index=False
)
