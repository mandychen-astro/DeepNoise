import sys
import os

home_directory = os.path.expanduser('~')
sys.path.append(home_directory + '/DeepNoise/')

from astropy.io import fits
from mpdaf.obj import Cube
import numpy as np
import torch

from src.data_processing import array_to_tensor, replace_outliers, apply_mask, rescale_data

# Load the data
mask = fits.getdata('/Users/mandychen/sky_subtraction/PKS0405-123_OB1/skymask_badpixel_removed.fits')
cube = Cube('/Users/mandychen/sky_subtraction/PKS0405-123_OB1/DATACUBE_FINAL_EXP1.fits')
data = cube.data.data
data = data[10:-10, :, :] # get rid of the first 10 and last 10 wavelength pixels because they are quite noisy

# Preprocess the data, clipping outliers and normalizing etc.
# could also mask nan with data = np.nan_to_num(data) if needed
sky_spec = apply_mask(data, mask, extract_value=0)
sky_spec = replace_outliers(sky_spec, lower=3, upper=10, fill_value_lower='median', fill_value_upper='median')
dc_offset = - np.min(sky_spec) + 10

sky_spec = sky_spec + dc_offset
sky_spec, min_val, max_val = rescale_data(sky_spec, log=True, new_min=0, new_max=1, output_minmax=True)
np.savetxt('./PKS0405-123_OB1EXP1_dataprocessing_pars.txt', 
            [min_val, max_val, dc_offset], delimiter=' ', comments='# min_val, max_val, dc_offset')

# Convert the input array to a tensor
input_tensor = array_to_tensor(sky_spec)
print(input_tensor.size())
torch.save(input_tensor, '../data/processed/PKS0405-123_OB1EXP1_input_tensor.pt')
