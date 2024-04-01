import torch
from torch import nn
from torch.utils.data import DataLoader

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Trains a PyTorch neural network model.

    Args:
    - model (torch.nn.Module): The neural network to train.
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
    - criterion (torch.nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer.
    - num_epochs (int): Number of epochs to train.
    - device (str): Device to use for training ('cuda' or 'cpu').

    Returns:
    - model: Trained model.
    """
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for data in train_loader:
            inputs = data.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return model
