import torch
import numpy as np

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

