# src/plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from src.plot_utils import (
    plot_2d_heatmap, plot_2d_gp_surrogate, plot_gp_regression, plot_acquisition,
    plot_1d_gp_surrogate, plot_1d_acquisition
)

class MockOptimizeResult:
    """Wrapper class to make a dictionary into an OptimizeResult-like object."""
    def __init__(self, x_iters, func_vals, space, models):
        self.x_iters = x_iters
        self.func_vals = func_vals
        self.space = space
        self.models = models

def load_results(filename):
    """Loads results from a .pkl file in the results folder.

    Args:
        filename (str): The name of the .pkl file (e.g., 'PI_result.pkl').

    Returns:
        MockOptimizeResult: An object with attributes x_iters, func_vals, space, models.
    """
    with open(f"results/{filename}", "rb") as f:
        data = pickle.load(f)
    return MockOptimizeResult(
        x_iters=data["x_iters"],
        func_vals=data["func_vals"],
        space=data["space"],
        models=data["models"]
    )

if __name__ == "__main__":
    # Define acquisition functions and their corresponding .pkl files
    acq_functions = {
        "EI": "EI_result.pkl",
        "PI": "PI_result.pkl",
        "LCB": "LCB_result.pkl"
    }

    # Load CSV results for BO and Random Search
    ei_df = pd.read_csv("results/EI_results.csv")
    pi_df = pd.read_csv("results/PI_results.csv")
    lcb_df = pd.read_csv("results/LCB_results.csv")
    random_df = pd.read_csv("results/random_search_results.csv")

    # Calculate the best accuracy over iterations
    bo_ei_accuracies = np.maximum.accumulate(ei_df["accuracy"])
    bo_pi_accuracies = np.maximum.accumulate(pi_df["accuracy"])
    bo_lcb_accuracies = np.maximum.accumulate(lcb_df["accuracy"])
    random_accuracies = np.maximum.accumulate(random_df["accuracy"])

    # Plot 1: Convergence plot
    plt.figure(figsize=(10, 6))
    plt.plot(bo_ei_accuracies, label="BO EI", marker="o", color="blue")
    plt.plot(bo_pi_accuracies, label="BO PI", marker="s", color="orange")
    plt.plot(bo_lcb_accuracies, label="BO LCB", marker="^", color="green")
    plt.plot(random_accuracies, label="Random Search", marker="d", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Best Validation Accuracy (%)")
    plt.title("Convergence of Best Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Define ranges for batch_size and learning_rate
    batch_size_range = np.arange(64, 257, 8)  # From 64 to 256 with steps of 8
    lr_range = np.logspace(-5, -4, 50)  # 50 points between 1e-5 and 1e-4

    # Load BO results for EI, PI, and LCB
    result_ei = load_results("EI_result.pkl")
    result_pi = load_results("PI_result.pkl")
    result_lcb = load_results("LCB_result.pkl")

    # Plot 2: 2D heatmap for GP surrogate (mean)
    print("Plotting 2D heatmap for GP surrogate (mean)")
    plot_2d_heatmap(result_pi, batch_size_range, lr_range)

    # Plot 3: 2D heatmaps for each acquisition function
    for acq_name, result in zip(["EI", "PI", "LCB"], [result_ei, result_pi, result_lcb]):
        print(f"Plotting 2D heatmap for {acq_name} acquisition function")
        plot_2d_heatmap(result, batch_size_range, lr_range, acq_func=acq_name)

    # Plot 4: 2D GP surrogate (mean and uncertainty)
    print("Plotting 2D GP surrogate (mean and uncertainty)")
    plot_2d_gp_surrogate(result_pi, batch_size_range, lr_range)

    # Calculate median batch size for use in 1D plots
    median_batch_size = np.median([x[0] for x in result_pi.x_iters])

    # Plot 5: 1D GP regression for learning_rate
    print("Plotting 1D GP regression for learning_rate")
    plot_gp_regression(result_pi, lr_range, "learning_rate", fixed_param_value=median_batch_size, fixed_param_name="batch_size")

    # Plot 6: 1D acquisition function for PI
    print("Plotting 1D PI acquisition function for learning_rate")
    plot_acquisition(result_pi, "PI", lr_range, "learning_rate", fixed_param_value=median_batch_size, fixed_param_name="batch_size")

    # New plots
    # Plot 7: 1D GP surrogate for learning_rate
    print("Plotting new 1D GP surrogate for learning_rate")
    plot_1d_gp_surrogate(result_pi, lr_range, "learning_rate", median_batch_size, "batch_size")

    # Plot 8: 1D EI acquisition function for learning_rate
    print("Plotting new 1D EI acquisition function for learning_rate")
    plot_1d_acquisition(result_pi, "EI", lr_range, "learning_rate", median_batch_size, "batch_size")