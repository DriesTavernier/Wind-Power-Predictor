# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:57:27 2026

@author: dries
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import numpy as np
import matplotlib.pyplot as plt


def evaluate_model(y_actual, y_predicted, model_name="Model", bins=60):

    # Calculate prediction errors
    errors = np.asarray(y_actual) - np.asarray(y_predicted)

    # Calculate metrics
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)

    # Print results
    print(f"\n--- {model_name} PERFORMANCE ---")
    print(f"RMSE: {rmse:.2f} kW")
    print(f"MAE:  {mae:.2f} kW")
    print(f"Bias: {bias:.2f} kW")

    # Plot error distribution
    plt.figure(figsize=(10, 6))

    plt.hist(
        errors,
        bins=bins,
        density=True,
        edgecolor="black",
        alpha=0.7,
        label="Prediction error distribution"
    )

    # Zero error reference
    plt.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Zero error"
    )


    plt.xlabel("Prediction error (kW)")
    plt.ylabel("Probability density")
    plt.title(f"Prediction Error Distribution - {model_name}")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
    }


def plot_power_curve(
    data,
    col_name,
    title,
    label = 'model'
):
    plt.figure(figsize=(10, 6))

    plt.scatter(data["Wind Speed (m/s)"], data["LV ActivePower (kW)"],
                s=5, alpha=0.25, label="Actual power")
    plt.scatter(data["Wind Speed (m/s)"], data[col_name],
                s=5, alpha=0.05, label=label)

    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Power (kW)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()