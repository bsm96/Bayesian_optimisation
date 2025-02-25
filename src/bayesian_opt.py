# src/bayesian_opt.py

from skopt import gp_minimize
import numpy as np
from sklearn.model_selection import ParameterSampler
import pickle
import os
from .train import train_model
from hydra import compose, initialize
import random
import torch

def set_seed(seed):
    """Set seeds for reproducibility across NumPy, random, and PyTorch.
    
    Args:
        seed (int): Seed value for random number generators.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def objective(params, iteration, cfg):
    """Objective function for Bayesian Optimization.
    
    Args:
        params (list): Hyperparameters [batch_size, lr].
        iteration (int): Iteration number for saving unique models.
        cfg: Hydra configuration object.
    
    Returns:
        float: Negative validation accuracy (for minimization).
    """
    overrides = [
        f"training.batch_size={int(params[0])}",
        f"optimizer.lr={params[1]:.5f}"
    ]
    with initialize(config_path="../config", version_base="1.1"):
        cfg_iter = compose(config_name="config", overrides=overrides)
        accuracy = train_model(cfg_iter, save_model_path=f"models/model_iter_{iteration}.pth")
        return -accuracy  # Minimize negative accuracy

def run_random_search(cfg, n_iterations=30):
    """Run random search for hyperparameter tuning.
    
    Args:
        cfg: Hydra configuration object.
        n_iterations (int): Number of random search iterations.
    
    Returns:
        dict: Results including best accuracy, best params, accuracies, and model paths.
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
    accuracies = []
    model_paths = []
    
    for i, params in enumerate(param_list):
        print(f"Random Search Iteration {i+1}/{n_iterations}")
        overrides = [
            f"training.batch_size={int(params['training.batch_size'])}",
            f"optimizer.lr={params['optimizer.lr']:.5f}"
        ]
        with initialize(config_path="../config", version_base="1.1"):
            cfg_iter = compose(config_name="config", overrides=overrides)
        model_path = f"models/random_search_iter_{i}.pth"
        accuracy = train_model(cfg_iter, save_model_path=model_path)
        accuracies.append(accuracy)
        model_paths.append(model_path)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
    
    return {
        "best_accuracy": best_accuracy,
        "best_params": best_params,
        "accuracies": accuracies,
        "model_paths": model_paths
    }

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
            result = objective(params, iteration_counter, cfg)
            iteration_counter += 1
            return result
        
        iteration_counter = 0
        
        if acq_name == "LCB":
            result = gp_minimize(
                wrapped_objective,
                space,
                n_calls=cfg.bo.n_calls,
                n_initial_points=cfg.bo.n_initial_points,
                acq_func=acq_name,
                kappa=acq_func,
                random_state=cfg.training.seed
            )
        else:
            result = gp_minimize(
                wrapped_objective,
                space,
                n_calls=cfg.bo.n_calls,
                n_initial_points=cfg.bo.n_initial_points,
                acq_func=acq_func,
                random_state=cfg.training.seed
            )
        
        # Store results with model paths
        bo_result = {
            "optimizer_result": result,
            "acquisition_function": acq_name,
            "model_paths": [f"models/model_iter_{i}.pth" for i in range(cfg.bo.n_calls)]
        }
        bo_results.append(bo_result)
        
        # Save BO result
        with open(f"results/{acq_name}_result.pkl", "wb") as f:
            pickle.dump(bo_result, f)
        
        # Track best BO result
        current_best = -result.fun
        print(f"Best parameters with {acq_name}: {result.x}")
        print(f"Best accuracy with {acq_name}: {current_best:.2f}%")
        if current_best > best_bo_accuracy:
            best_bo_accuracy = current_best
            best_bo_acq = acq_name
    
    # Run random search
    print("\nRunning Random Search for comparison...")
    random_search_iterations = cfg.bo.n_calls
    random_search_result = run_random_search(cfg, n_iterations=random_search_iterations)
    
    # Save random search results
    with open("results/random_search_result.pkl", "wb") as f:
        pickle.dump(random_search_result, f)
    
    # Print results
    print(f"\nBest Random Search Accuracy: {random_search_result['best_accuracy']:.2f}%")
    print(f"Best Random Search Parameters: {random_search_result['best_params']}")
    print(f"Best Bayesian Optimization Accuracy: {best_bo_accuracy:.2f}% (with {best_bo_acq})")
    if random_search_result['best_accuracy'] > best_bo_accuracy:
        print(f"Random Search outperformed Bayesian Optimization with {random_search_result['best_accuracy']:.2f}% vs {best_bo_accuracy:.2f}%")
    else:
        print(f"Bayesian Optimization outperformed Random Search with {best_bo_accuracy:.2f}% vs {random_search_result['best_accuracy']:.2f}%")