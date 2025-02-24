# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from .model import CNN
from .data_loader import load_mnist
from tqdm import tqdm  # Import tqdm for progress bars

def train_model(cfg, save_model_path):
    """Train the CNN model and return validation accuracy.
    
    Args:
        cfg: Hydra configuration object containing training and model parameters.
        save_model_path (str): Path to save the trained model.
    
    Returns:
        float: Validation accuracy as a percentage.
    """
    # Set seeds for PyTorch reproducibility
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)
    
    # Load MNIST data with the specified batch size
    train_loader, test_loader = load_mnist(cfg.training.data_dir, cfg.training.batch_size)
    
    # Initialize the CNN model
    model = CNN(
        in_channels=cfg.model.in_channels,
        conv_layers=cfg.model.conv_layers,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with progress bar for epochs
    for epoch in tqdm(range(cfg.training.epochs), desc="Training Epochs"):
        model.train()
        epoch_loss = 0
        # Progress bar for batches within each epoch
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}")):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Print epoch loss for monitoring
        print(f"Epoch {epoch+1}/{cfg.training.epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Evaluate on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100. * correct / total
    
    # Save the trained model
    torch.save(model.state_dict(), save_model_path)
    
    return accuracy