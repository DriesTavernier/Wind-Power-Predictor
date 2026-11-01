from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# 1. DATA LOADING & QUALITY CHECK
# ============================================

data = pd.read_csv(ROOT / "data" / "raw" / "wind_turbine_scada.csv")

print("\n--- DATASET OVERVIEW ---")
print(f"Dataset shape: {data.shape}")

print("\n--- DATA TYPES ---")
print(data.dtypes)

print("\n--- MISSING VALUES ---")
print(data.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(data.duplicated().sum())

print("\n--- DESCRIPTIVE STATISTICS ---")
print(data.describe())

print("\n--- SHOW FIRST ROWS ---")
print(data.head(10))


# Convert timestamp
data["Date/Time"] = pd.to_datetime(
    data["Date/Time"],
    format="%d %m %Y %H:%M",
    errors="coerce"
)


# ============================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================

# 3.1 Actual vs theoretical power curve
plt.figure(figsize=(10, 6))
plt.scatter(
    data["Wind Speed (m/s)"],
    data["LV ActivePower (kW)"],
    s=5,
    alpha=0.25,
    label="Actual power"
)
plt.scatter(
    data["Wind Speed (m/s)"],
    data["Theoretical_Power_Curve (KWh)"],
    s=5,
    alpha=0.25,
    label="Theoretical power"
)
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Power (kW)")
plt.title("Actual vs Theoretical Wind Turbine Power Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# 3.2 Wind speed distribution
plt.figure(figsize=(10, 6))

plt.hist(
    data["Wind Speed (m/s)"],
    bins=50,
    edgecolor="black",
    alpha=0.75
)
plt.xlabel("Wind speed (m/s)")
plt.ylabel("Frequency")
plt.title("Wind Speed Distribution")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

#3.3 Wind direction 
plt.figure(figsize=(10, 6))

plt.hist(
    data["Wind Direction (°)"],
    bins=50,
    edgecolor="black",
    alpha=0.75
)
plt.xlabel("Wind Direction (°)")
plt.ylabel("Frequency")
plt.title("Wind Direction Distribution")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================
# 4. ANOMALY INVESTIGATION (INACTIVE TURBINES)
# ============================================
cut_in = 2.5 #m/s
threshold_power = 10 #kW
anomalies = data[
    (data["Wind Speed (m/s)"] > cut_in) &
    (data["LV ActivePower (kW)"] < threshold_power)
].copy()


# Actual power curve with identified anomalies
plt.figure(figsize=(10, 6))

plt.scatter(
    data["Wind Speed (m/s)"],
    data["LV ActivePower (kW)"],
    s=5,
    alpha=0.15,
    label="Normal / remaining observations"
)

plt.scatter(
    anomalies["Wind Speed (m/s)"],
    anomalies["LV ActivePower (kW)"],
    s=15,
    alpha=0.7,
    label="Potential anomalies"
)

plt.xlabel("Wind speed (m/s)")
plt.ylabel("Actual active power (kW)")
plt.title("Potential Anomalies in Wind Turbine Power Production")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================
# 6. DATA CLEANING & EXPORT
# ============================================

# Remove identified potential anomalies
data_clean = data.drop(index=anomalies.index).copy()

# Reset index
data_clean = data_clean.reset_index(drop=True)

# Export cleaned dataset
data_clean.to_csv(
    ROOT / "data" / "clean" / "wind_turbine_scada_clean.csv",
    index=False
)



















