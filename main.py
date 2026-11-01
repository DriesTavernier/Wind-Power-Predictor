import pandas as pd
from src.evaluate_performance import plot_power_curve, evaluate_model

data_theoretical = pd.read_csv("data/clean/wind_turbine_scada_test.csv")
data_lm = pd.read_csv("data/clean/wind_turbine_scada_test_lm.csv")
data_nn = pd.read_csv("data/clean/wind_turbine_scada_test_nn.csv")

plot_power_curve(
    data_theoretical,
    "Theoretical_Power_Curve (KWh)",
    title="Wind Turbine Power Curve - Theoretical",
    label="Theoretical power prediction"
)

evaluate_model(data_theoretical["LV ActivePower (kW)"], data_theoretical["Theoretical_Power_Curve (KWh)"])



plot_power_curve(
    data_lm,
    "predicted",
    title="Wind Turbine Power Curve - Linear Regression",
    label="Linear Regression power prediction"
)
evaluate_model(data_lm["LV ActivePower (kW)"], data_lm["predicted"])



plot_power_curve(
    data_nn,
    "predicted",
    title="Wind Turbine Power Curve - Neural Network",
    label="Neural Network power prediction"
)
evaluate_model(data_nn["LV ActivePower (kW)"], data_nn["predicted"])
