# src/plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from .plot_utils import plot_convergence, plot_gp_regression, plot_acquisition

def load_results(filename):
    """Load results from a .pkl file in the results folder."""
    with open(f"results/{filename}", "rb") as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    # Load BO results for EI, PI, LCB
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
    
    # Calculate best accuracy over iterations
    bo_ei_accuracies = np.maximum.accumulate(ei_df["accuracy"])
    bo_pi_accuracies = np.maximum.accumulate(pi_df["accuracy"])
    bo_lcb_accuracies = np.maximum.accumulate(lcb_df["accuracy"])
    random_accuracies = np.maximum.accumulate(random_df["accuracy"])
    
    # Plot convergence
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
    
    # Load BO result for PI (best performing)
    result_pi = load_results("PI_result.pkl")
    
    # Create a mock OptimizeResult for plotting with only necessary data
    class MockOptimizeResult:
        def __init__(self, x_iters, func_vals, space, models):
            self.x_iters = x_iters
            self.func_vals = func_vals
            self.space = space
            self.models = models
    
    mock_pi = MockOptimizeResult(
        x_iters=result_pi["x_iters"],
        func_vals=result_pi["func_vals"],
        space=result_pi["space"],
        models=result_pi["models"]
    )
    
    # Define parameter range for learning rate
    lr_range = np.logspace(np.log10(1e-5), np.log10(1e-4), 100)
    
    # Plot GP regression for learning rate with fixed batch size 64
    plot_gp_regression(
        mock_pi,
        param_range=lr_range,
        param_name="learning_rate",
        fixed_param_value=64,
        fixed_param_name="batch_size"
    )
    
    # Plot acquisition function for PI
    plot_acquisition(
        mock_pi,
        acq_func="PI",
        param_range=lr_range,
        param_name="learning_rate",
        fixed_param_value=64,
        fixed_param_name="batch_size"
    )