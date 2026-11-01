# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:57:27 2026

@author: dries
"""
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
