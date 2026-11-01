# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:28:31 2026

@author: dries
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Splits de dataframe willekeurig (80% train, 20% test)
# random_state zorgt ervoor dat de splitsing elke keer exact hetzelfde is als je de code opnieuw runt

data = pd.read_csv("../data/clean/wind_turbine_scada_feature.csv")
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

# Controleer de grootte van de nieuwe datasets
print(f"Totale dataset vorm: {data.shape}")
print(f"Train dataset vorm:  {train_data.shape} ({len(train_data)/len(data)*100:.0f}%)")
print(f"Test dataset vorm:   {test_data.shape} ({len(test_data)/len(data)*100:.0f}%)")



# Export train and test data
train_data.to_csv(
    "../data/clean/wind_turbine_scada_train.csv",
    index=False
)
test_data.to_csv(
    "../data/clean/wind_turbine_scada_test.csv",
    index=False
)