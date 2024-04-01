import torch
import numpy as np

def normalize_data(input_data, output_shape='original'):
    """
    Normalizes the data for the model. Handles both 1D and 3D arrays.
    For 1D array: Normalizes the data.
    For 3D array: Normalizes per pixel.
    Args:
    - input_data (numpy.ndarray): Input data array, either 1D [length] or 3D [length, nx, ny].
    - output_shape (str): Shape of the output data. Use 'original' to have it in 3D arrays of the datacube; 
            otherwise output stacked 1D arrays that get fed into the model.

    Returns:
    - numpy.ndarray: Normalized data.
    """
    if input_data.ndim == 1:
        # Normalize 1D data
        normalized_data = (input_data - np.mean(input_data)) / np.std(input_data)
        
    elif input_data.ndim == 3:
        # Normalize per pixel for 3D data
        length, nx, ny = input_data.shape
        reshaped_data = input_data.transpose(1, 2, 0).reshape(-1, length)
        normalized_data = (reshaped_data - np.mean(reshaped_data, axis=1, keepdims=True)) / np.std(reshaped_data, axis=1, keepdims=True)

        # Reshape back to 3D array [length, ny, nx]
        if output_shape == 'original':
            normalized_data = normalized_data.reshape(ny, nx, length).transpose(2, 0, 1)
        else:
            normalized_data = normalized_data.reshape(nx, ny, length).transpose(1, 2, 0).reshape(-1, 1, length)

    else:
        raise ValueError("Input data must be either 1D or 3D array.")
    
    return normalized_data



def array_to_tensor(input_data):
    """
    Prepares and normalizes the data for the model. Handles both 1D and 3D arrays.
    For 1D array: Normalizes the data.
    For 3D array: Normalizes per pixel.
    Args:
    - input_data (numpy.ndarray): Input data array, either 1D [length] or 3D [length, nx, ny].
    
    Returns:
    - torch.Tensor: Normalized data as a PyTorch tensor.
    """
    if input_data.ndim == 1:
        # Normalize 1D data
        normalized_data = (input_data - np.mean(input_data)) / np.std(input_data)
        # Reshape to [1, 1, length]
        tensor_data = torch.tensor(normalized_data, dtype=torch.float).unsqueeze(0).unsqueeze(0)
    
    elif input_data.ndim == 3:
        # Normalize per pixel for 3D data
        length, nx, ny = input_data.shape
        reshaped_data = input_data.transpose(1, 2, 0).reshape(-1, length)
        normalized_data = (reshaped_data - np.mean(reshaped_data, axis=1, keepdims=True)) / np.std(reshaped_data, axis=1, keepdims=True)
        model_input = normalized_data.reshape(nx, ny, length).transpose(1, 2, 0).reshape(-1, 1, length)
        tensor_data = torch.tensor(model_input, dtype=torch.float)
    
    else:
        raise ValueError("Input data must be either 1D or 3D array.")
    
    return tensor_data

class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for PyTorch.
    """
    def __init__(self, input_data):
        self.input_data = array_to_tensor(input_data)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = self.input_data[idx]
        return x