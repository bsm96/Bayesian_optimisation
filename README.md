# MNIST CNN with Bayesian Optimization

This project trains a Convolutional Neural Network (CNN) on the MNIST dataset using Bayesian Optimization to find the best hyperparameters, such as batch size and learning rate.

Requirements
 - Python 3.x

Navigate to the project directory:
 - cd /path/to/your/project Replace /path/to/your/project with the actual path to your project directory.
 - Install dependencies: pip install -r requirements.txt

Running the Project
 - Run the following Bayesian Optimization script in the terminal: python -m src.bayesian_opt

This will:

 - Start Bayesian Optimization with different acquisition functions (EI, PI, LCB).
 - Train the CNN model for each set of hyperparameters.
 - Save results in the results/ directory as .pkl files.
 - Save trained models in the models/ directory as .pth files.

 Generate plots:
 - Generate and display plots to visualize the optimization process: python src/plot_results.py


 ### Reconstruct a Model: To reconstruct a model from a saved .pth file (e.g., models/model_iter_0.pth), you can use:
import torch
from src.model import CNN
from hydra import compose, initialize

# Load the checkpoint
checkpoint = torch.load("models/model_iter_0.pth")
cfg = checkpoint['cfg']

# Rebuild the model
model = CNN(
    in_channels=cfg.model.in_channels,
    conv_layers=cfg.model.conv_layers,
    dropout=cfg.model.dropout,
    activation=cfg.model.activation
)
model.load_state_dict(checkpoint['model_state_dict'])

# Rebuild the optimizer
optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(f"Model reconstructed with accuracy: {checkpoint['accuracy']:.2f}%")