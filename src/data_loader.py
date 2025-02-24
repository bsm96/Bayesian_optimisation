# src/data_loader.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist(data_dir, batch_size):
    """Loads MNIST dataset with specified batch size.
    
    Args:
        data_dir (str): Directory to store/download MNIST data.
        batch_size (int): Batch size for data loaders.
    
    Returns:
        tuple: (train_loader, test_loader) DataLoader objects.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    train_data = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    test_data = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader