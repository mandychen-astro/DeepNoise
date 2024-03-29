import sys
sys.path.append('/Users/mandychen/DeepNoise/')

from astropy.io import fits
from mpdaf.obj import Cube
import numpy as np

import torch
from src.data_processing import array_to_tensor 
from src.model import AutoEncoder
from torchsummary import summary
import time

# Load the data
cube = Cube('/Users/mandychen/sky_subtraction/PKS0405-123_OB1/DATACUBE_FINAL_EXP1_subcube.fits')
data = cube.data.data

length = data.shape[0]
testdata = data # [:, 0, 0]

# Convert the input array to a tensor
input_tensor = array_to_tensor(testdata)
print(input_tensor.size())

t0 = time.time()
# Create an instance of the Autoencoder model
enc_in_channels, enc_out_channels = 1, 32
enc_kernel_size, enc_stride, enc_padding = 8, 1, 'same'
dec_in_channels, dec_out_channels = enc_out_channels, enc_out_channels
dec_kernel_size, dec_stride, dec_padding = 6, 2, 0
autoencoder = AutoEncoder(enc_in_channels, enc_out_channels, enc_kernel_size, enc_stride, enc_padding, 
                 dec_in_channels, dec_out_channels, dec_kernel_size, dec_stride, dec_padding)

summary(autoencoder, (1, length))
# Pass the input tensor through the autoencoder
output_tensor = autoencoder(input_tensor)
print('time takes', time.time() - t0)
# Print the output tensor
print(output_tensor)
print(output_tensor.size())