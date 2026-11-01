# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 13:34:48 2026

@author: dries
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# ============================================
# 1. Load data
# ============================================
data = pd.read_csv("../data/clean/wind_turbine_scada_clean.csv")
# Convert timestamp
data["Date/Time"] = pd.to_datetime(
    data["Date/Time"],
    format="%d %m %Y %H:%M",
    errors="coerce"
)
# ============================================
# 2.Power Deviation
# ============================================

# Difference between actual and theoretical power
data["Power Deviation (kW)"] = (
    data["LV ActivePower (kW)"]
    - data["Theoretical_Power_Curve (KWh)"]
)


# 2.1 Power deviation distribution
plt.figure(figsize=(10, 6))
plt.hist(
    data["Power Deviation (kW)"],
    bins=60,
    edgecolor="black",
    alpha=0.75
)
plt.axvline(0, linestyle="--", linewidth=1.5, label="Zero deviation", color="orange")
plt.xlabel("Power deviation (kW)")
plt.ylabel("Frequency")
plt.title("Distribution of Power Deviation from Theoretical Curve")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# 3.4 Power deviation vs wind speed
plt.figure(figsize=(10, 6))
plt.axhline(0, linestyle="--", linewidth=1.5, color="orange")

plt.scatter(
    data["Wind Speed (m/s)"],
    data["Power Deviation (kW)"],
    s=5,
    alpha=0.2
)

plt.scatter(
    data["Wind Speed (m/s)"],
    -data["Theoretical_Power_Curve (KWh)"],
    s=5,
    alpha=0.2,
    label= "Theoretical Power * -1"
)

plt.xlabel("Wind speed (m/s)")
plt.ylabel("Power deviation (kW)")
plt.title("Power Deviation vs Wind Speed")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()


# 3.5 Power deviation vs wind direction
plt.figure(figsize=(10, 6))

plt.scatter(
    data["Wind Direction (°)"],
    data["Power Deviation (kW)"],
    s=5,
    alpha=0.2
)

plt.axhline(0, linestyle="--", linewidth=1.5, color="orange")

plt.xlabel("Wind direction (°)")
plt.ylabel("Power deviation (kW)")
plt.title("Power Deviation vs Wind Direction")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()




# ============================================
# 6. Calculate Cp 
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================
# 6. Calculate Cp & Feature Engineering
# ============================================

# Filter deellastgebied voor de schatting van de straal
Partial_load_region = data[(data["Wind Speed (m/s)"] > 5) & (data["Wind Speed (m/s)"] < 10)].copy()

# 80% van 0.58 (Betz-ish aanname) = 0.464
cp_assumption = 0.8 * 0.58 
air_density = 1.225

# Bereken het rotoroppervlak (A) en daarna de straal (R)
Area_estimated = (2 * Partial_load_region["Theoretical_Power_Curve (KWh)"] * 1000) / (cp_assumption * air_density * Partial_load_region["Wind Speed (m/s)"]**3)
R_estimated = np.sqrt(Area_estimated / np.pi)

# Plot 1: Distributie van de geschatte rotorstraal
plt.figure(figsize=(10, 6))
plt.hist(
    R_estimated,
    bins=15,
    edgecolor="black",
    alpha=0.75
)
plt.xlabel("Estimated Rotor Radius / Blade Length (m)")  # Gecorrigeerd label
plt.ylabel("Frequency")
plt.title("Distribution of Estimated Rotor Radius (R)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Vaste waarde instellen op basis van je analyse (57.5m is heel accuraat voor een 117m rotor!)
R = 57.5
A_actual = np.pi * (R**2)

# Bereken Cp voor de gehele dataset (voorkom delen door 0 door wind speed > 0 te vereisen)
# We maken een kopie om waarschuwingen te voorkomen en filteren sensorruis bij 0 wind
data_clean_wind = data[data["Wind Speed (m/s)"] > 0.5].copy()

data_clean_wind['Cp'] = (2 * data_clean_wind["LV ActivePower (kW)"] * 1000) / (air_density * A_actual * data_clean_wind["Wind Speed (m/s)"]**3)

# Bereken ook de theoretische Cp ter referentie in je plot!
data_clean_wind['Cp_Theoretical'] = (2 * data_clean_wind["Theoretical_Power_Curve (KWh)"] * 1000) / (air_density * A_actual * data_clean_wind["Wind Speed (m/s)"]**3)

# Plot 2: Actual vs Theoretical Cp Curve
plt.figure(figsize=(10, 6))
# Werkelijke Cp (Scatter met ruis)
plt.scatter(
    data_clean_wind["Wind Speed (m/s)"],
    data_clean_wind['Cp'],
    s=5,
    alpha=0.25,
    label="Actual Cp (from Data)",
    color='blue'
)
# Theoretische Cp (Lijn zonder ruis)
plt.scroll_df = data_clean_wind.sort_values(by="Wind Speed (m/s)")
plt.plot(
    plt.scroll_df["Wind Speed (m/s)"],
    plt.scroll_df['Cp_Theoretical'],
    color='red',
    linewidth=2,
    label="Theoretical Cp (Manufacturer)"
)

plt.xlabel("Wind speed (m/s)")
plt.ylabel("Cp (-)")
plt.ylim(-0.1, 0.65) # Begrens de Y-as omdat Cp door ruis soms extreem hoog uitschiet bij lage wind
plt.title("Wind Turbine Efficiency Coefficient (Cp) vs Wind Speed")
plt.legend() # Werkt nu correct omdat labels zijn toegevoegd
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================
# 8. Export feature
# ============================================

# Reset index
data_feature = data.reset_index(drop=True)

# Export cleaned dataset
data_clean_wind.to_csv(
    "../data/clean/wind_turbine_scada_feature.csv",
    index=False
)
