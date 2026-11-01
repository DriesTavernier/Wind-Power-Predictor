import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("../data/raw/wind_turbine_scada.csv")
# Display basic information
print("\n--- DATASET OVERVIEW ---")
print(data.head())

print("\n--- DATASET INFORMATION ---")
print(data.info())

print("\n--- STATISTICAL SUMMARY ---")
print(data.describe())


# ==========================================
# EXPLORATORY DATA ANALYSIS
# ==========================================

# 1. Actual power curve
plt.figure(figsize=(10, 6))

plt.scatter(
    data["Wind Speed (m/s)"],
    data["LV ActivePower (kW)"],
    s=5,
    alpha=0.3
)

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Active Power (kW)")
plt.title("Actual Wind Turbine Power Curve")
plt.grid(True)
plt.tight_layout()
plt.show()


# 2. Compare actual and theoretical power
plt.figure(figsize=(10, 6))

plt.scatter(
    data["Wind Speed (m/s)"],
    data["LV ActivePower (kW)"],
    s=5,
    alpha=0.2,
    label="Actual power"
)

plt.scatter(
    data["Wind Speed (m/s)"],
    data["Theoretical_Power_Curve (KWh)"],
    s=5,
    alpha=0.2,
    label="Theoretical power"
)

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Power")
plt.title("Actual vs Theoretical Power Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 3. Wind speed distribution
plt.figure(figsize=(10, 6))

plt.hist(
    data["Wind Speed (m/s)"],
    bins=50,
    edgecolor="black"
)

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Frequency")
plt.title("Wind Speed Distribution")
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 4. ANOMALY INVESTIGATION
# ==========================================

# Define potentially abnormal operating conditions
anomalies = data[
    (data["Wind Speed (m/s)"] > 5) &
    (data["LV ActivePower (kW)"] < 10)
].copy()

# Calculate statistics
n_total = len(data)
n_anomalies = len(anomalies)
percentage = n_anomalies / n_total * 100

print("\n--- ANOMALY INVESTIGATION ---")
print(f"Total observations: {n_total}")
print(f"Potential anomalies: {n_anomalies}")
print(f"Percentage: {percentage:.2f}%")

print("\nWind speed statistics of anomalies:")
print(anomalies["Wind Speed (m/s)"].describe())


# ==========================================
# 5. VISUALIZE ANOMALIES
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    data["Wind Speed (m/s)"],
    data["LV ActivePower (kW)"],
    s=5,
    alpha=0.15,
    label="All observations"
)

plt.scatter(
    anomalies["Wind Speed (m/s)"],
    anomalies["LV ActivePower (kW)"],
    s=12,
    alpha=0.6,
    label="Potential anomalies"
)

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Active Power (kW)")
plt.title("Wind Turbine: Potential Abnormal Operating Conditions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


