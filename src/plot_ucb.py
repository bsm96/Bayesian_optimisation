import numpy as np
import matplotlib.pyplot as plt
import pickle

# Funktion til beregning af GP-UCB (Upper Confidence Bound)
def GP_UCB(mean, std, t, dim=2.0, v=1.0, delta=0.1):
    """
    Beregner UCB Acquisition Function vha.:
        UCB(x) = μ(x) + sqrt(v * β_t) * σ(x)
    hvor β_t = 2 * log( t^(d/2 + 2) * π²/(3*δ) ).
    
    Args:
        mean (ndarray): Middelværdier fra GP'en (i samme skala som accuracies).
        std (ndarray): Standardafvigelser fra GP'en.
        t (float): Iterationsnummer (bruges til beregning af β_t).
        dim (float): Dimensionen af inputrummet (her 2, da vi har to hyperparametre).
        v (float): Vægtparameter (default 1.0).
        delta (float): Sandsynlighedsparameter (default 0.1).
    
    Returns:
        ndarray: UCB-værdier for hvert punkt.
    """
    beta = 2 * np.log((t**(dim/2 + 2) * np.pi**2/(3*delta)) + 1e-9)
    return mean + np.sqrt(v * beta) * std

# Indlæs BO-resultatet for UCB (her antaget gemt som "LCB_result.pkl")
result_file = "results/LCB_result.pkl"
with open(result_file, "rb") as f:
    bo_result = pickle.load(f)

# Bo-resultatet gemmer en dictionary med nøgler: x_iters, func_vals, space og models.
# Vi benytter det sidste GP-model fra BO-resultatet.
gp_model = bo_result["models"][-1]

# Definér de to hyperparameter-ranges
batch_size_range = np.arange(64, 257, 8)  # Fx fra 64 til 256 med intervallet 8
lr_range = np.logspace(np.log10(1e-5), np.log10(1e-4), 50)  # 50 værdier mellem 1e-5 og 1e-4

# Opret et grid af punkter (i den skala, som modellen forventer)
# Hver række i grid'et repræsenterer et par: [batch_size, learning_rate]
X_grid = np.array(np.meshgrid(batch_size_range, lr_range)).T.reshape(-1, 2)

# Forudsig middelværdi og standardafvigelse fra GP-modellen for alle punkterne i grid'et
y_pred, y_std = gp_model.predict(X_grid, return_std=True)

# I vores BO-resultater bliver målfunktionen minimeret (dvs. vi har gemt -accuracy),
# så vi konverterer til "positive" accuracies ved at negere forudsigelserne.
mean = -y_pred
std = y_std

# Bestem t som antal BO-iterationer + 1
t = len(bo_result["x_iters"]) + 1

# Beregn UCB-værdierne for hvert punkt
ucb_values = GP_UCB(mean, std, t, dim=2.0, v=1.0, delta=0.1)

# Omform ucb_values til en 2D-matrix med dimensionerne: (len(batch_size_range), len(lr_range))
ucb_heatmap = ucb_values.reshape(len(batch_size_range), len(lr_range))

# Plot heatmappet
plt.figure(figsize=(12, 6))
# Bemærk: Vi transponerer (values.T) så x-aksen svarer til batch size og y-aksen til learning rate
plt.contourf(batch_size_range, lr_range, ucb_heatmap.T, cmap='viridis')
plt.colorbar(label="UCB-værdi")
plt.xlabel("Batch Size")
plt.ylabel("Learning Rate")
plt.title("UCB Acquisition Function Heatmap")
plt.show()
