import sys
sys.path.append('/Users/mandychen/DeepNoise/')

from astropy.io import fits
from mpdaf.obj import Cube
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data_processing import normalize_array_to_tensor, CustomDataset 
from src.model import ConvAutoencoder
from src.train import train_model
from torchsummary import summary
import time

# # Load the data
# cube = Cube('/Users/mandychen/sky_subtraction/PKS0405-123_OB1/DATACUBE_FINAL_EXP1_subcube.fits')
# data = cube.data.data

# length = data.shape[0]
# testdata = np.nan_to_num(data) # fill in nan with 0

# # # Convert the input array to a tensor
# input_tensor = normalize_array_to_tensor(testdata[:,0,0])
# print(input_tensor.size())

# t0 = time.time()
# # Create an instance of the Autoencoder model
enc_in_channels, enc_out_channels = 1, 32
enc_kernel_size, enc_stride, enc_padding = 8, 1, 'same'
dec_in_channels, dec_out_channels = enc_out_channels, enc_out_channels
dec_kernel_size, dec_stride, dec_padding = 6, 2, 0
autoencoder = ConvAutoencoder(enc_in_channels, enc_out_channels, enc_kernel_size, enc_stride, enc_padding, 
                 dec_in_channels, dec_out_channels, dec_kernel_size, dec_stride, dec_padding)

summary(autoencoder, (1, 3661))

# # # Pass the input tensor through the autoencoder
# # output_tensor = autoencoder(input_tensor)
# # print('time takes', time.time() - t0)
# # # Print the output tensor
# # print(output_tensor)
# # print(output_tensor.size())


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
