# src/bayesian_opt.py

from skopt import gp_minimize
import numpy as np
from sklearn.model_selection import ParameterSampler
import matplotlib.pyplot as plt
from .train import train_model
from hydra import compose, initialize
import random
import pickle
import os
from .plot_utils import (
    plot_convergence,
    plot_evaluations,
    plot_gp_regression,
    plot_acquisition,
    plot_random_search_convergence,
    generate_batch_size_range
)

def set_seed(seed):
    """Set seeds for reproducibility across NumPy and random (PyTorch handled in train.py).
    
    Args:
        seed (int): Seed value for random number generators.
    """
    np.random.seed(seed)
    random.seed(seed)

def objective(params, iteration):
    """Objective function for Bayesian Optimization.
    
    Args:
        params (list): Hyperparameters [batch_size, lr].
        iteration (int): Iteration number for saving unique models.
    
    Returns:
        float: Negative validation accuracy (for minimization).
    """
    overrides = [
        f"training.batch_size={int(params[0])}",
        f"optimizer.lr={params[1]:.5f}"
    ]
    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name="config", overrides=overrides)
        accuracy = train_model(cfg, save_model_path=f"models/model_iter_{iteration}.pth")
        return -accuracy  # Minimize negative accuracy

def run_random_search(cfg, n_iterations=30):
    """Run random search for hyperparameter tuning.
    
    Args:
        cfg: Hydra configuration object.
        n_iterations (int): Number of random search iterations.
    
    Returns:
        float: Best validation accuracy.
        dict: Best hyperparameters.
        list: List of accuracies for each iteration (for plotting).
    """
    # Define hyperparameter space
    space = {
        "training.batch_size": range(cfg.bo.batch_size_min, cfg.bo.batch_size_max + 1, 32),
        "optimizer.lr": np.logspace(np.log10(cfg.bo.lr_min), np.log10(cfg.bo.lr_max), 100)
    }
    
    # Sample random hyperparameters
    param_list = list(ParameterSampler(space, n_iter=n_iterations, random_state=cfg.training.seed))
    
    best_accuracy = -float('inf')
    best_params = None
    accuracies = []  # Store accuracies for plotting
    
    for i, params in enumerate(param_list):
        print(f"Random Search Iteration {i+1}/{n_iterations}")
        
        # Set overrides for Hydra configuration
        overrides = [
            f"training.batch_size={int(params['training.batch_size'])}",
            f"optimizer.lr={params['optimizer.lr']:.5f}"
        ]
        
        # Rebuild configuration with overrides
        with initialize(config_path="../config", version_base="1.1"):
            cfg_iter = compose(config_name="config", overrides=overrides)
        
        # Train the model and get accuracy
        accuracy = train_model(cfg_iter, save_model_path=f"models/random_search_iter_{i}.pth")
        accuracies.append(accuracy)
        
        # Check if this is the best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
    
    return best_accuracy, best_params, accuracies

if __name__ == "__main__":
    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name="config")
    
    set_seed(cfg.training.seed)
    
    # Define the search space for batch size and learning rate
    space = [
        (cfg.bo.batch_size_min, cfg.bo.batch_size_max),  # Batch size range
        (cfg.bo.lr_min, cfg.bo.lr_max)                  # Learning rate range
    ]
    
    # Acquisition functions to test
    acq_functions = [
        ("EI", "EI"),    # Expected Improvement
        ("PI", "PI"),    # Probability of Improvement
        ("LCB", 2.576)   # UCB approximation with kappa=2.576
    ]
    
    # Create directories for results and models
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    bo_results = []
    best_bo_accuracy = -float('inf')
    best_bo_acq = None
    
    # Run Bayesian Optimization for each acquisition function
    for acq_name, acq_func in acq_functions:
        print(f"Running BO with {acq_name}:")
        
        def wrapped_objective(params):
            global iteration_counter
            result = objective(params, iteration_counter)
            iteration_counter += 1
            return result
        
        iteration_counter = 0  # Reset iteration counter for each acquisition function
        
        if acq_name == "LCB":
            result = gp_minimize(
                wrapped_objective,
                space,
                n_calls=cfg.bo.n_calls,              # Use value from YAML
                n_initial_points=cfg.bo.n_initial_points,  # Use value from YAML
                acq_func=acq_name,
                kappa=acq_func,                      # kappa=2.576 for UCB
                random_state=cfg.training.seed
            )
        else:
            result = gp_minimize(
                wrapped_objective,
                space,
                n_calls=cfg.bo.n_calls,              # Use value from YAML
                n_initial_points=cfg.bo.n_initial_points,  # Use value from YAML
                acq_func=acq_func,
                random_state=cfg.training.seed
            )
        
        bo_results.append((result, acq_name))
        
        # Save BO result
        with open(f"results/{acq_name}_result.pkl", "wb") as f:
            pickle.dump(result, f)
        
        # Check best BO result
        current_best = -result.fun
        print(f"Best parameters with {acq_name}: {result.x}")
        print(f"Best accuracy with {acq_name}: {current_best:.2f}%")
        if current_best > best_bo_accuracy:
            best_bo_accuracy = current_best
            best_bo_acq = acq_name
        
        # Generate plots for each acquisition function
        plot_convergence(result, acq_name)
        plot_evaluations(result, param_names=["batch_size", "learning_rate"])
        
        # Plot for learning rate
        lr_range = np.logspace(np.log10(cfg.bo.lr_min), np.log10(cfg.bo.lr_max), 100)
        plot_gp_regression(result, lr_range, "learning_rate", fixed_param_value=np.median(space[0]), fixed_param_name="batch_size")
        plot_acquisition(result, acq_name, lr_range, "learning_rate")
        
        # Plot for batch size
        batch_size_range = generate_batch_size_range(cfg.bo.batch_size_min, cfg.bo.batch_size_max)
        plot_gp_regression(result, batch_size_range, "batch_size", fixed_param_value=np.median(space[1]), fixed_param_name="learning_rate")
        plot_acquisition(result, acq_name, batch_size_range, "batch_size")
    
    # Run random search for comparison
    print("\nRunning Random Search for comparison...")
    random_search_iterations = cfg.bo.n_calls  # Same number of iterations as BO
    best_random_accuracy, best_random_params, random_accuracies = run_random_search(
        cfg, n_iterations=random_search_iterations
    )
    
    # Print random search results
    print(f"\nBest Random Search Accuracy: {best_random_accuracy:.2f}%")
    print(f"Best Random Search Parameters: {best_random_params}")
    
    # Plot convergence for random search
    plot_random_search_convergence(random_accuracies, title="Random Search Convergence")
    
    # Print the best performing method
    print(f"\nBest Bayesian Optimization Accuracy: {best_bo_accuracy:.2f}% (with {best_bo_acq})")
    if best_random_accuracy > best_bo_accuracy:
        print(f"Random Search outperformed Bayesian Optimization with {best_random_accuracy:.2f}% vs {best_bo_accuracy:.2f}%")
    else:
        print(f"Bayesian Optimization outperformed Random Search with {best_bo_accuracy:.2f}% vs {best_random_accuracy:.2f}%")