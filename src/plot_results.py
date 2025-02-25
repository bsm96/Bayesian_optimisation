import pickle
import numpy as np
from hydra import compose, initialize
from .plot_utils import (
    plot_convergence,
    plot_evaluations,
    plot_gp_regression,
    plot_acquisition,
    plot_random_search_convergence,
    generate_batch_size_range
)

def load_bo_results(acq_name):
    """Load Bayesian Optimization results for a specific acquisition function."""
    with open(f"results/{acq_name}_result.pkl", "rb") as f:
        return pickle.load(f)

def load_random_search_results():
    """Load random search results."""
    with open("results/random_search_result.pkl", "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Load configuration to access ranges
    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name="config")
    
    # Acquisition functions to plot
    acq_functions = ["EI", "PI", "LCB"]
    
    # Plot Bayesian Optimization results
    for acq_name in acq_functions:
        bo_result = load_bo_results(acq_name)
        result = bo_result["optimizer_result"]
        
        # Plot convergence
        plot_convergence(result, acq_name)
        
        # Plot evaluations
        plot_evaluations(result, param_names=["batch_size", "learning_rate"])
        
        # Plot GP regression and acquisition for learning rate
        lr_range = np.logspace(np.log10(cfg.bo.lr_min), np.log10(cfg.bo.lr_max), 100)
        median_batch_size = (cfg.bo.batch_size_min + cfg.bo.batch_size_max) / 2
        plot_gp_regression(result, lr_range, "learning_rate", fixed_param_value=median_batch_size, fixed_param_name="batch_size")
        plot_acquisition(result, acq_name, lr_range, "learning_rate", fixed_param_value=median_batch_size, fixed_param_name="batch_size")
        
        # Plot GP regression and acquisition for batch size
        batch_size_range = generate_batch_size_range(cfg.bo.batch_size_min, cfg.bo.batch_size_max)
        median_lr = 10 ** ((np.log10(cfg.bo.lr_min) + np.log10(cfg.bo.lr_max)) / 2)
        plot_gp_regression(result, batch_size_range, "batch_size", fixed_param_value=median_lr, fixed_param_name="learning_rate")
        plot_acquisition(result, acq_name, batch_size_range, "batch_size", fixed_param_value=median_lr, fixed_param_name="learning_rate")
    
    # Plot random search results
    random_result = load_random_search_results()
    plot_random_search_convergence(random_result["accuracies"])