import torch
import numpy as np

def apply_mask(cube, mask, extract_value=1):
    """
    Applies a 2D mask to a 3D cube and stacks the masked pixels into a 2D array.
    Args:
    - cube (numpy.ndarray): 3D cube with dimensions [nz, ny, nx].
    - mask (numpy.ndarray): 2D mask with dimensions [ny, nx].
    
    Returns:
    - numpy.ndarray: 2D array of masked pixels with dimensions [nz, n_pixels],
                    where n_pixels is number of valid pixels after masking.
    """
    # Flatten the mask to 1D
    flattened_mask = mask.flatten()

    # Flatten the cube to 2D
    cube = cube.reshape(cube.shape[0], cube.shape[1]*cube.shape[2])
    
    # Stack the masked pixels from the cube into a 2D array
    masked_pixels = cube[:, flattened_mask == extract_value]
    
    return masked_pixels
    

def replace_outliers(data, lower=3, upper=10, fill_value_lower='median',
                    fill_value_upper='median'):
    """
    Replaces the lower outliers in the data with the fill value.
    Args:
    - data (numpy.ndarray): Input data array.
    - lower (float): Lower sigma cutoff below 0.
    - upper (float): upper sigma cutoff above mean.
    - fill_value_lower (float or string): Value to replace the lower outliers with.
    - fill_value_upper (float or string): Value to replace the lower outliers with.
    
    Returns:
    - numpy.ndarray: 2D array with the lower outliers replaced.
    """
    nz = data.shape[0]

    for i in range(nz):
        wave_slice = data[i, :]
        bad_pix = wave_slice < (0 - lower*np.nanstd(wave_slice))
        if fill_value_lower == 'median':
            data[i, bad_pix] = np.nanmedian(wave_slice)
        else: 
            data[i, bad_pix] = fill_value_lower

        bad_pix = wave_slice > (np.nanmean(wave_slice) + upper*np.nanstd(wave_slice))
        if fill_value_upper == 'median':
            data[i, bad_pix] = np.nanmedian(wave_slice)
        else:    
            data[i, bad_pix] = fill_value_upper

    return data

def rescale_data_w_prior_bounds(input_data, log=True, new_min=0, new_max=1, 
                                input_min=None, input_max=None):
    """
    Rescales the data to a new range.
    Args:
    - input_data (numpy.ndarray): Input data array. 
    - log (bool): Whether to take the logarithm of the data.
    - new_min (float): New minimum value.
    - new_max (float): New maximum value.
    - input_min (float): Minimum value of the input data.
    - input_max (float): Maximum value of the input data.
    
    Returns:
    - numpy.ndarray: Rescaled data.
    """
    # Take the logarithm of the data
    if log:
        data = np.log10(input_data)
    else:
        data = input_data
    
    # Perform min-max scaling
    if input_min is None:
        input_min = np.min(data)
    if input_max is None:
        input_max = np.max(data)
    scaled_data = (data - input_min) / (input_max - input_min)
    
    # Rescale to the desired range
    rescaled_data = scaled_data * (new_max - new_min) + new_min

    return rescaled_data


def rescale_data(input_data, log=True, new_min=0, new_max=1, output_minmax=True):
    """
    Rescales the data to a new range.
    Args:
    - input_data (numpy.ndarray): Input data array. 
    - log (bool): Whether to take the logarithm of the data.
    - new_min (float): New minimum value.
    - new_max (float): New maximum value.
    - output_minmax (bool): Whether to output the min and max values of the input data.
    """

    # Take the logarithm of the data
    if log:
        data = np.log10(input_data)
    else:
        data = input_data
    
    # Perform min-max scaling
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Rescale to the desired range
    rescaled_data = scaled_data * (new_max - new_min) + new_min

    if output_minmax:
        return rescaled_data, np.min(data), np.max(data)    
    else:
        return rescaled_data


def normalize_data(input_data, output_shape='original', mode='global'):
    """
    Normalizes the data for the model. Handles both 1D and 3D arrays.
    For 1D array: Normalizes the data.
    For 3D array: Normalizes per pixel or per whole cube.
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



def normalize_array_to_tensor(input_data):
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

def array_to_tensor(input_data):
    """
    Converts the input data to a PyTorch tensor.
    Args:
    - input_data (numpy.ndarray): Input 2D data array, with shape [length_seq, N].
    
    Returns:
    - torch.Tensor: PyTorch tensor.
    """
    if len(input_data.shape) == 1:
        input_data = input_data.reshape(-1, 1) # reshape to [length_seq, 1]
    length_seq, N = input_data.shape[0], input_data.shape[1]
    input_data = input_data.transpose(1, 0).reshape(-1, 1, length_seq)
    tensor_data = torch.tensor(input_data, dtype=torch.float)
    return tensor_data

    

class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for PyTorch.
    """
    def __init__(self, input_data):
        self.input_data = input_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = self.input_data[idx]
        return x



