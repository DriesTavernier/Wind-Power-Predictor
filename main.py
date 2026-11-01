# main.py
import pandas as pd

# Pipeline: elke import voert dat script uit, in deze volgorde
import src.DataExplorationAndCleaning
import src.FeatureEngineering
import src.SplitTrainTestData
import src.LinearRegression
import src.NeuralNetwork

from src.evaluate_performance import plot_power_curve, evaluate_model

# Evaluatie
Y = "LV ActivePower (kW)"

data_theoretical = pd.read_csv("data/clean/wind_turbine_scada_test.csv")
data_lm = pd.read_csv("data/clean/wind_turbine_scada_test_lm.csv")
data_nn = pd.read_csv("data/clean/wind_turbine_scada_test_nn.csv")

plot_power_curve(
    data_theoretical,
    "Theoretical_Power_Curve (KWh)",
    title="Wind Turbine Power Curve - Theoretical",
    label="Theoretical power prediction",
)
evaluate_model(data_theoretical[Y], data_theoretical["Theoretical_Power_Curve (KWh)"])

plot_power_curve(
    data_lm,
    "predicted",
    title="Wind Turbine Power Curve - Linear Regression",
    label="Linear Regression power prediction",
)
evaluate_model(data_lm[Y], data_lm["predicted"])

plot_power_curve(
    data_nn,
    "predicted",
    title="Wind Turbine Power Curve - Neural Network",
    label="Neural Network power prediction",
)
evaluate_model(data_nn[Y], data_nn["predicted"])