# src/model.py
import torch.nn as nn

class CNN(nn.Module):
    """Convolutional Neural Network for MNIST classification."""
    
    def __init__(self, in_channels, conv_layers, dropout, activation):
        """Initialize the CNN model.
        
        Args:
            in_channels (int): Number of input channels (1 for MNIST).
            conv_layers (list): List of dicts with 'filters' and 'kernel_size'.
            dropout (float): Dropout probability.
            activation (str): Activation function ('relu' supported).
        """
        super().__init__()
        layers = []
        current_channels = in_channels
        for layer in conv_layers:
            layers += [
                nn.Conv2d(current_channels, layer['filters'], layer['kernel_size'], padding=1),
                nn.ReLU() if activation == "relu" else nn.Tanh(),
                nn.MaxPool2d(2),
                nn.Dropout(dropout)
            ]
            current_channels = layer['filters']
        
        self.features = nn.Sequential(*layers)
        # After two MaxPool2d(2), 28x28 becomes 7x7
        self.classifier = nn.Linear(current_channels * 7 * 7, 10)  # 10 classes for MNIST
    
    def forward(self, x):
        """Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, 28, 28).
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x