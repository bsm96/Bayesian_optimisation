# src/plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
from skopt.plots import plot_evaluations

def generate_batch_size_range(min_size, max_size, step=32):
    """Generate a range of batch sizes from min_size to max_size with a given step.
    
    Args:
        min_size (int): Minimum batch size.
        max_size (int): Maximum batch size.
        step (int): Step size between batch sizes.
    
    Returns:
        np.ndarray: Array of batch sizes.
    """
    return np.arange(min_size, max_size + 1, step)

def plot_convergence(result, acq_name):
    """Plot the convergence of the best observed validation accuracy over iterations for BO.
    
    Args:
        result: skopt OptimizeResult object.
        acq_name: Name of the acquisition function (e.g., 'EI', 'PI', 'LCB').
    """
    plt.figure(figsize=(8, 5))
    best_accuracy = np.maximum.accumulate(-result.func_vals)  # Convert negative loss to accuracy
    plt.plot(best_accuracy, marker='o', label=f'{acq_name} Convergence', color='blue')
    plt.title(f'Convergence Plot for {acq_name}')
    plt.xlabel('Iterations')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_evaluations(result, param_names):
    """Plot the evaluated points in the parameter space.
    
    Args:
        result: skopt OptimizeResult object.
        param_names: List of parameter names (e.g., ['batch_size', 'learning_rate']).
    """
    plot_evaluations(result, bins=20)
    plt.suptitle('Evaluated Points in Parameter Space')
    plt.show()

def plot_gp_regression(result, param_range, param_name, fixed_param_value=None, fixed_param_name=None):
    """Plot the GP prediction (mean and uncertainty) over a parameter range.
    
    Args:
        result: skopt OptimizeResult object.
        param_range: Array of parameter values to predict over.
        param_name: Name of the parameter being plotted ('learning_rate' or 'batch_size').
        fixed_param_value: Value to fix the other parameter at.
        fixed_param_name: Name of the fixed parameter.
    """
    gp_model = result.models[-1]
    if param_name == "learning_rate":
        X_plot = np.atleast_2d(param_range).T
        if fixed_param_value is not None:
            # Fix batch_size (first parameter)
            X_plot = np.hstack((np.full((X_plot.shape[0], 1), fixed_param_value), X_plot))
        else:
            # Default to median batch size
            median_batch_size = np.median(result.space[0])
            X_plot = np.hstack((np.full((X_plot.shape[0], 1), median_batch_size), X_plot))
    elif param_name == "batch_size":
        X_plot = np.atleast_2d(param_range).T
        if fixed_param_value is not None:
            # Fix learning_rate (second parameter)
            X_plot = np.hstack((X_plot, np.full((X_plot.shape[0], 1), fixed_param_value)))
        else:
            # Default to median learning rate
            median_lr = np.median(result.space[1])
            X_plot = np.hstack((X_plot, np.full((X_plot.shape[0], 1), median_lr)))
    else:
        raise ValueError("Invalid parameter name")
    
    y_pred, y_std = gp_model.predict(X_plot, return_std=True)
    y_pred = -y_pred  # Convert negative accuracy back to positive
    
    # Convert result.x_iters to NumPy array for advanced indexing
    x_iters = np.array(result.x_iters)
    
    plt.figure(figsize=(8, 5))
    plt.plot(param_range, y_pred, label='GP Mean')
    plt.fill_between(param_range, y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, alpha=0.3, label='95% Confidence Interval')
    if param_name == "learning_rate":
        plt.scatter(x_iters[:, 1], -result.func_vals, c='red', label='Evaluated Points')
    elif param_name == "batch_size":
        plt.scatter(x_iters[:, 0], -result.func_vals, c='red', label='Evaluated Points')
    plt.title(f'GP Regression for {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Validation Accuracy (%)')
    plt.legend()
    plt.show()

def plot_acquisition(result, acq_func, param_range, param_name, fixed_param_value=None, fixed_param_name=None):
    """Plot the GP prediction and acquisition function values.
    
    Args:
        result: skopt OptimizeResult object.
        acq_func: Acquisition function name ('EI', 'PI', 'LCB').
        param_range: Array of parameter values to evaluate.
        param_name: Name of the parameter being plotted ('learning_rate' or 'batch_size').
        fixed_param_value: Value to fix the other parameter at.
        fixed_param_name: Name of the fixed parameter.
    """
    from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
    gp_model = result.models[-1]
    if param_name == "learning_rate":
        X_plot = np.atleast_2d(param_range).T
        if fixed_param_value is not None:
            X_plot = np.hstack((np.full((X_plot.shape[0], 1), fixed_param_value), X_plot))
        else:
            median_batch_size = np.median(result.space[0])
            X_plot = np.hstack((np.full((X_plot.shape[0], 1), median_batch_size), X_plot))
    elif param_name == "batch_size":
        X_plot = np.atleast_2d(param_range).T
        if fixed_param_value is not None:
            X_plot = np.hstack((X_plot, np.full((X_plot.shape[0], 1), fixed_param_value)))
        else:
            median_lr = np.median(result.space[1])
            X_plot = np.hstack((X_plot, np.full((X_plot.shape[0], 1), median_lr)))
    else:
        raise ValueError("Invalid parameter name")
    
    y_pred, y_std = gp_model.predict(X_plot, return_std=True)
    y_pred = -y_pred  # Convert to accuracy
    y_opt = np.max(-result.func_vals)  # Best observed accuracy
    
    if acq_func == 'EI':
        acq_values = gaussian_ei(X_plot, gp_model, y_opt=-y_opt, xi=0.01)
    elif acq_func == 'PI':
        acq_values = gaussian_pi(X_plot, gp_model, y_opt=-y_opt, xi=0.01)
    elif acq_func == 'LCB':
        acq_values = gaussian_lcb(X_plot, gp_model, kappa=2.576)
    else:
        raise ValueError("Invalid acquisition function")
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(param_range, y_pred, label='GP Mean')
    ax1.fill_between(param_range, y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, alpha=0.3, label='95% CI')
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(param_range, acq_values, 'r--', label=f'{acq_func} Acquisition')
    ax2.set_ylabel(f'{acq_func} Value')
    ax2.legend(loc='upper right')
    
    plt.title(f'GP and {acq_func} Acquisition Function for {param_name}')
    plt.show()

def plot_random_search_convergence(accuracies, title="Random Search Convergence"):
    """Plot convergence of the best validation accuracy over random search iterations.
    
    Args:
        accuracies (list): List of validation accuracies from random search.
        title (str): Title of the plot.
    """
    best_accuracies = np.maximum.accumulate(accuracies)  # Best accuracy up to each iteration
    plt.figure(figsize=(8, 5))
    plt.plot(best_accuracies, marker='o', label='Random Search Convergence', color='orange')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.show()