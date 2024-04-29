import sys
sys.path.append('/Users/mandychen/DeepNoise/')

from astropy.io import fits
from mpdaf.obj import Cube
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data_processing import array_to_tensor, CustomDataset, replace_outliers, apply_mask, rescale_data
from src.model import SpectrumTransformer
from src.train import train_model
from torchsummary import summary
import time

# Load the data
# mask = fits.getdata('/Users/mandychen/sky_subtraction/PKS0405-123_OB1/skymask_badpixel_removed.fits')
# cube = Cube('/Users/mandychen/sky_subtraction/PKS0405-123_OB1/DATACUBE_FINAL_EXP1.fits')
# data = cube.data.data
# data = data[10:-10, :, :] # get rid of the first 10 and last 10 wavelength pixels because they are quite noisy

# # Preprocess the data, clipping outliers and normalizing etc.
# # could also mask nan with data = np.nan_to_num(data) if needed
# sky_spec = apply_mask(data, mask, extract_value=0)
# sky_spec = replace_outliers(sky_spec, lower=3, upper=10, fill_value_lower='median', fill_value_upper='median')
# dc_offset = - np.min(sky_spec) + 10
# sky_spec = sky_spec + dc_offset
# sky_spec, min_val, max_val = rescale_data(sky_spec, log=True, new_min=0, new_max=1, output_minmax=True)
# np.savetxt('/Users/mandychen/sky_subtraction/PKS0405-123_OB1/EXP1_dataprocessing_pars.txt', 
#             [min_val, max_val, dc_offset], delimiter=' ', comments='# min_val, max_val, dc_offset')

# # # Convert the input array to a tensor
# sky_spec = sky_spec[:,:10] # only use the first 10 spectra for testing
# np.savetxt('/Users/mandychen/sky_subtraction/PKS0405-123_OB1/testdata.txt', sky_spec)
sky_spec = np.loadtxt('/Users/mandychen/sky_subtraction/PKS0405-123_OB1/testdata.txt')
input_tensor = array_to_tensor(sky_spec)
print(input_tensor.size())

t0 = time.time()
# Create an instance of the Autoencoder model
num_specpixels, embedding_dim = input_tensor.size(2), 128
output_dim = num_specpixels
print(num_specpixels, embedding_dim, output_dim)
autoencoder = SpectrumTransformer(num_specpixels, embedding_dim, output_dim)

summary(autoencoder, (10, num_specpixels))

# # Pass the input tensor through the autoencoder
output_tensor = autoencoder(input_tensor)
print('time takes', time.time() - t0)
# Print the output tensor
print(output_tensor)
print(output_tensor.size())


# # test the model with 5 epochs
# # Define a loss function
# criterion = nn.MSELoss()  # Mean Squared Error

# # Choose an optimizer
# optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# # Load your data
# dataset = CustomDataset(testdata)
# train_loader = DataLoader(dataset, batch_size=40, shuffle=True)

# # Train the model
# trained_model = train_model(autoencoder, train_loader, criterion, optimizer, num_epochs=5, device='cpu')

# torch.save(autoencoder.state_dict(), '../models/model_state_dict.pth')
