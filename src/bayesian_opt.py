# src/bayesian_opt.py
from skopt import gp_minimize
import numpy as np
from sklearn.model_selection import ParameterSampler
import pandas as pd
import os
import pickle
from .train import train_model
from hydra import compose, initialize
import random
import torch

def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def objective(params, iteration, cfg, acq_name):
    """Objective function for Bayesian Optimization."""
    overrides = [
        f"training.batch_size={int(params[0])}",
        f"optimizer.lr={params[1]:.5f}"
    ]
    with initialize(config_path="../config", version_base="1.1"):
        cfg_iter = compose(config_name="config", overrides=overrides)
    accuracy = train_model(cfg_iter, save_model_path=f"models/model_{acq_name}_iter_{iteration}.pth")
    return -accuracy  # Minimize negative accuracy

def run_random_search(cfg, n_iterations=15):
    """Run Random Search over the parameter space and return used parameters."""
    space = {
        "training.batch_size": range(cfg.bo.batch_size_min, cfg.bo.batch_size_max + 1, 32),
        "optimizer.lr": np.logspace(np.log10(cfg.bo.lr_min), np.log10(cfg.bo.lr_max), 100)
    }
    
    param_list = list(ParameterSampler(space, n_iter=n_iterations, random_state=cfg.training.seed))
    best_accuracy = -float('inf')
    best_params = None
    accuracies = []
    model_paths = []
    
    for i, params in enumerate(param_list):
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
    
    return best_accuracy, best_params, accuracies, model_paths, param_list

def wrapped_objective(params, iteration, cfg, acq_name):
    """Wrapper for objective function with iteration tracking."""
    result = objective(params, iteration[0], cfg, acq_name)
    iteration[0] += 1
    return result

if __name__ == "__main__":
    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name="config")
    
    set_seed(cfg.training.seed)
    space = [(cfg.bo.batch_size_min, cfg.bo.batch_size_max), (cfg.bo.lr_min, cfg.bo.lr_max)]
    acq_functions = [("EI", "EI"), ("PI", "PI"), ("LCB", 2.576)]
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    best_bo_accuracy = -float('inf')
    best_bo_acq = None
    
    for acq_name, acq_func in acq_functions:
        print(f"Running BO with {acq_name}...")
        iteration = [0]  # Use list to modify counter without global
        if acq_name == "LCB":
            result = gp_minimize(
                lambda params: wrapped_objective(params, iteration, cfg, acq_name),
                space,
                n_calls=cfg.bo.n_calls,
                n_initial_points=cfg.bo.n_initial_points,
                acq_func=acq_name,
                kappa=acq_func,
                random_state=cfg.training.seed,
                n_jobs=-1  # Use all available cores
            )
        else:
            result = gp_minimize(
                lambda params: wrapped_objective(params, iteration, cfg, acq_name),
                space,
                n_calls=cfg.bo.n_calls,
                n_initial_points=cfg.bo.n_initial_points,
                acq_func=acq_func,
                random_state=cfg.training.seed,
                n_jobs=-1  # Use all available cores
            )
        
        # Save results as CSV
        df = pd.DataFrame({
            "iteration": range(len(result.x_iters)),
            "batch_size": [x[0] for x in result.x_iters],
            "learning_rate": [x[1] for x in result.x_iters],
            "accuracy": [-val for val in result.func_vals]  # Positive accuracy
        })
        df.to_csv(f"results/{acq_name}_results.csv", index=False)
        
        # Save simplified result object as pickle
        with open(f"results/{acq_name}_result.pkl", "wb") as f:
            pickle.dump({
                "x_iters": result.x_iters,
                "func_vals": result.func_vals,
                "space": result.space,
                "models": [model for model in result.models]  # Only if needed for plotting
            }, f)
        
        current_best = -result.fun  # Convert to positive accuracy
        print(f"Best accuracy with {acq_name}: {current_best:.2f}%")
        if current_best > best_bo_accuracy:
            best_bo_accuracy = current_best
            best_bo_acq = acq_name
    
    # Run and save random search
    print("Running Random Search...")
    best_random_accuracy, best_random_params, random_accuracies, random_model_paths, random_params_list = run_random_search(cfg)
    
    df_random = pd.DataFrame({
        "iteration": range(len(random_accuracies)),
        "batch_size": [params["training.batch_size"] for params in random_params_list],
        "learning_rate": [params["optimizer.lr"] for params in random_params_list],
        "accuracy": random_accuracies
    })
    df_random.to_csv("results/random_search_results.csv", index=False)
    
    print(f"Best Random Search Accuracy: {best_random_accuracy:.2f}%")
    print(f"Best Bayesian Optimization Accuracy: {best_bo_accuracy:.2f}% (with {best_bo_acq})")