import torch
from torch import nn
from torch.utils.data import DataLoader

def train_model(model, train_loader, criterion, optimizer, val_loader=None, num_epochs=10, 
                return_train_loss=False, return_val_loss=False, scheduler=None,
                device='cuda'):
    """
    Trains a PyTorch neural network model.

    Args:
    - model (torch.nn.Module): The neural network to train.
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
    - criterion (torch.nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
    - num_epochs (int): Number of epochs to train.
    - return_train_loss (bool): Whether to return the training loss.
    - return_val_loss (bool): Whether to return the validation loss.
    - device (str): Device to use for training ('cuda' or 'cpu').

    Returns:
    - model: Trained model.
    """
    model.to(device)
    train_loss = []
    val_loss = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data in train_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_loss.append(epoch_loss)
        
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for data in val_loader:
                    inputs, targets = data
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_running_loss += loss.item()
            
            val_running_loss /= len(val_loader)
            val_loss.append(val_running_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_running_loss:.4f}")

        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        if scheduler is not None:
            print(f"Epoch {epoch+1}/{num_epochs}, Current Learning Rate: {scheduler.get_last_lr()[0]}")
            scheduler.step()

    
    if return_train_loss and return_val_loss:
        return model, train_loss, val_loss
    elif return_train_loss: 
        return model, train_loss
    elif return_val_loss:    
        return model, val_loss  
    else:
        return model
