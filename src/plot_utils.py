# src/plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb

def plot_2d_heatmap(result, batch_size_range, lr_range, acq_func=None):
    """Generates a 2D heatmap for the GP surrogate or the acquisition function.

    Args:
        result: skopt OptimizeResult object.
        batch_size_range: Array of batch sizes.
        lr_range: Array of learning rates.
        acq_func: Name of the acquisition function ('EI', 'PI', 'LCB') or None for the GP surrogate.
    """
    gp_model = result.models[-1]  # Uses the last trained GP model
    
    # Create a grid for batch_size and learning_rate
    X_plot = np.array(np.meshgrid(batch_size_range, lr_range)).T.reshape(-1, 2)
    
    if acq_func:
        y_opt = np.max(-result.func_vals)  # Best observed value (converted to accuracy)
        if acq_func == 'EI':
            values = gaussian_ei(X_plot, gp_model, y_opt=-y_opt, xi=0.01)
        elif acq_func == 'PI':
            values = gaussian_pi(X_plot, gp_model, y_opt=-y_opt, xi=0.01)
        elif acq_func == 'LCB':
            values = gaussian_lcb(X_plot, gp_model, kappa=2.576)
        else:
            raise ValueError("Invalid acquisition function")
        values = values.reshape(len(batch_size_range), len(lr_range))
        title = f"{acq_func} Acquisition Function"
    else:
        y_pred, _ = gp_model.predict(X_plot, return_std=True)
        values = -y_pred.reshape(len(batch_size_range), len(lr_range))  # Convert to accuracy
        title = "GP Surrogate Mean"

    # Plot heatmap
    plt.figure(figsize=(10, 5))
    plt.contourf(batch_size_range, lr_range, values.T, cmap='viridis')
    plt.colorbar(label=title)
    plt.xlabel('Batch Size')
    plt.ylabel('Learning Rate')
    plt.title(title)
    plt.show()

def plot_2d_gp_surrogate(result, batch_size_range, lr_range):
    """Plot the GP surrogate mean and uncertainty in 2D.

    Args:
        result: skopt OptimizeResult object.
        batch_size_range: Array of batch sizes.
        lr_range: Array of learning rates.
    """
    gp_model = result.models[-1]  # Uses the last trained GP model
    X_plot = np.array(np.meshgrid(batch_size_range, lr_range)).T.reshape(-1, 2)
    y_pred, y_std = gp_model.predict(X_plot, return_std=True)
    y_pred = -y_pred.reshape(len(batch_size_range), len(lr_range))  # Converts to accuracy
    y_std = y_std.reshape(len(batch_size_range), len(lr_range))

    # Plot mean
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.contourf(batch_size_range, lr_range, y_pred.T, cmap='viridis')
    plt.colorbar(label='Validation Accuracy (%)')
    plt.xlabel('Batch Size')
    plt.ylabel('Learning Rate')
    plt.title('GP Surrogate Mean', fontsize=10)

    # Plot uncertainty
    plt.subplot(1, 2, 2)
    plt.contourf(batch_size_range, lr_range, y_std.T, cmap='viridis')
    plt.colorbar(label='Standard Deviation')
    plt.xlabel('Batch Size')
    plt.ylabel('Learning Rate')
    plt.title('GP Surrogate Uncertainty', fontsize=10)
    plt.tight_layout()
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
            X_plot = np.hstack((np.full((X_plot.shape[0], 1), fixed_param_value), X_plot))
        else:
            median_batch_size = np.median([x[0] for x in result.x_iters])
            X_plot = np.hstack((np.full((X_plot.shape[0], 1), median_batch_size), X_plot))
    elif param_name == "batch_size":
        X_plot = np.atleast_2d(param_range).T
        if fixed_param_value is not None:
            X_plot = np.hstack((X_plot, np.full((X_plot.shape[0], 1), fixed_param_value)))
        else:
            median_lr = np.median([x[1] for x in result.x_iters])
            X_plot = np.hstack((X_plot, np.full((X_plot.shape[0], 1), median_lr)))
    else:
        raise ValueError("Invalid parameter name")
    
    y_pred, y_std = gp_model.predict(X_plot, return_std=True)
    y_pred = -y_pred  # Convert negative accuracy back to positive
    
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
    plt.grid(True)
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
    gp_model = result.models[-1]
    if param_name == "learning_rate":
        X_plot = np.atleast_2d(param_range).T
        if fixed_param_value is not None:
            X_plot = np.hstack((np.full((X_plot.shape[0], 1), fixed_param_value), X_plot))
        else:
            median_batch_size = np.median([x[0] for x in result.x_iters])
            X_plot = np.hstack((np.full((X_plot.shape[0], 1), median_batch_size), X_plot))
    elif param_name == "batch_size":
        X_plot = np.atleast_2d(param_range).T
        if fixed_param_value is not None:
            X_plot = np.hstack((X_plot, np.full((X_plot.shape[0], 1), fixed_param_value)))
        else:
            median_lr = np.median([x[1] for x in result.x_iters])
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

def plot_1d_gp_surrogate(result, param_range, param_name, fixed_param_value, fixed_param_name):
    """Plot the GP surrogate mean and uncertainty as a 1D slice.
    
    Args:
        result: skopt OptimizeResult object.
        param_range: Array of values for the varying parameter.
        param_name: Name of the varying parameter ('learning_rate' or 'batch_size').
        fixed_param_value: Value for the fixed parameter.
        fixed_param_name: Name of the fixed parameter.
    """
    gp_model = result.models[-1]
    if param_name == "learning_rate":
        X_plot = np.array([[fixed_param_value, v] for v in param_range])
    elif param_name == "batch_size":
        X_plot = np.array([[v, fixed_param_value] for v in param_range])
    else:
        raise ValueError("Invalid parameter name")

    y_pred, y_std = gp_model.predict(X_plot, return_std=True)
    y_pred = -y_pred  # Convert to accuracy

    plt.figure(figsize=(8, 5))
    plt.plot(param_range, y_pred, label='GP Mean')
    plt.fill_between(param_range, y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, alpha=0.3, label='95% CI')
    plt.xlabel(param_name)
    plt.ylabel('Validation Accuracy (%)')
    plt.title(f'GP Surrogate Mean for {param_name} (fixed {fixed_param_name}={fixed_param_value})')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_1d_acquisition(result, acq_func, param_range, param_name, fixed_param_value, fixed_param_name, xi=0.01, kappa=2.576):
    """Plot the acquisition function as a 1D slice.
    
    Args:
        result: skopt OptimizeResult object.
        acq_func: Name of the acquisition function ('EI', 'PI', 'LCB').
        param_range: Array of values for the varying parameter.
        param_name: Name of the varying parameter ('learning_rate' or 'batch_size').
        fixed_param_value: Value for the fixed parameter.
        fixed_param_name: Name of the fixed parameter.
        xi: Exploration parameter for EI and PI.
        kappa: Parameter for LCB.
    """
    gp_model = result.models[-1]
    y_opt = np.max(-result.func_vals)  # Best observed accuracy
    if param_name == "learning_rate":
        X_plot = np.array([[fixed_param_value, v] for v in param_range])
    elif param_name == "batch_size":
        X_plot = np.array([[v, fixed_param_value] for v in param_range])
    else:
        raise ValueError("Invalid parameter name")

    if acq_func == 'EI':
        acq_values = gaussian_ei(X_plot, gp_model, y_opt=-y_opt, xi=xi)
    elif acq_func == 'PI':
        acq_values = gaussian_pi(X_plot, gp_model, y_opt=-y_opt, xi=xi)
    elif acq_func == 'LCB':
        acq_values = gaussian_lcb(X_plot, gp_model, kappa=kappa)
    else:
        raise ValueError("Invalid acquisition function")

    plt.figure(figsize=(8, 5))
    plt.plot(param_range, acq_values, label=f'{acq_func} Acquisition')
    plt.xlabel(param_name)
    plt.ylabel(f'{acq_func} Value')
    plt.title(f'{acq_func} Acquisition Function for {param_name} (fixed {fixed_param_name}={fixed_param_value})')
    plt.legend()
    plt.grid(True)
    plt.show()