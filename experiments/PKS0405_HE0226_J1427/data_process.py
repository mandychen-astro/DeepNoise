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

def get_train_and_val(mask, cube, train_split=0.8):
    data = cube.data.data
    data = data[10:-10, :, :] # get rid of the first 10 and last 10 wavelength pixels because they are quite noisy

    # Preprocess the data, clipping outliers and normalizing etc.
    # could also mask nan with data = np.nan_to_num(data) if needed
    sky_spec = apply_mask(data, mask, extract_value=0)
    sky_spec = np.arcsinh(sky_spec) # arcsinh transform
    sky_spec_sigclipped = replace_outliers(sky_spec, lower=3, upper=10, 
                        fill_value_lower='median', fill_value_upper='median')

    # Generate random indices for splitting the data
    num_samples = sky_spec.shape[0]
    train_indices = np.random.choice(num_samples, int(train_split * num_samples), replace=False)
    test_indices = np.setdiff1d(np.arange(num_samples), train_indices)

    # Split the data into training and validation sets
    sky_spec_train = sky_spec[train_indices]
    sky_spec_test = sky_spec[test_indices]

    sky_spec_train_sigclipped = sky_spec_sigclipped[train_indices]
    sky_spec_test_sigclipped = sky_spec_sigclipped[test_indices]
    
    return sky_spec_train, sky_spec_test, sky_spec_train_sigclipped, sky_spec_test_sigclipped

mask = fits.getdata(home_directory + '/sky_subtraction/PKS0405-123_OB1/skymask_badpixel_removed_OB1EXP1.fits')
cube = Cube(home_directory + '/sky_subtraction/PKS0405-123_OB1/DATACUBE_FINAL_EXP1.fits')
t1, v1, tc1, vc1 = get_train_and_val(mask, cube)

mask = fits.getdata(home_directory + '/sky_subtraction/PKS0405-123_OB1/skymask_badpixel_removed_OB1EXP2.fits')
cube = Cube(home_directory + '/sky_subtraction/PKS0405-123_OB1/DATACUBE_FINAL_EXP2.fits')
t2, v2, tc2, vc2 = get_train_and_val(mask, cube)

mask = fits.getdata(home_directory + '/sky_subtraction/HE0226-4110_OB1/skymask_badpixel_removed_OB1EXP1.fits')
cube = Cube(home_directory + '/sky_subtraction/HE0226-4110_OB1/DATACUBE_FINAL_EXP1.fits')
t3, v3, tc3, vc3 = get_train_and_val(mask, cube)

mask = fits.getdata(home_directory + '/sky_subtraction/SDSSJ1427-0121_OB1/skymask_badpixel_removed_OB1EXP1.fits')
cube = Cube(home_directory + '/sky_subtraction/SDSSJ1427-0121_OB1/DATACUBE_FINAL_EXP1.fits')
t4, v4, tc4, vc4 = get_train_and_val(mask, cube)

# Combine the training and validation data
sky_spec_train = np.concatenate((t1, t2, t3, t4), axis=1)
sky_spec_val = np.concatenate((v1, v2, v3, v4), axis=1)

sky_spec_train_sigclipped = np.concatenate((tc1, tc2, tc3, tc4), axis=1)
sky_spec_val_sigclipped = np.concatenate((vc1, vc2, vc3, vc4), axis=1)

num_samples = sky_spec_train.shape[0]
test_indices = np.random.choice(num_samples, 100, replace=False)# choose 100 for testing here
train_test = sky_spec_train[test_indices]
train_test_clipped = sky_spec_train_sigclipped[test_indices]

num_samples = sky_spec_val.shape[0]
test_indices = np.random.choice(num_samples, 100, replace=False)# choose 100 for testing here
val_test = sky_spec_val[test_indices]
val_test_clipped = sky_spec_val_sigclipped[test_indices]

# Convert the input array to a tensor
input_tensor_train = array_to_tensor(sky_spec_train)
input_tensor_val = array_to_tensor(sky_spec_val)
input_tensor_train_clipped = array_to_tensor(sky_spec_train_sigclipped)
input_tensor_val_clipped = array_to_tensor(sky_spec_val_sigclipped) 


print(input_tensor_train.size())
print(input_tensor_val.size())

torch.save(input_tensor_train, '../../data/PKS0405_HE0226_J1427_input_tensor_train.pt')
torch.save(input_tensor_val, '../../data/PKS0405_HE0226_J1427_input_tensor_val.pt')
torch.save(input_tensor_train_clipped, '../../data/PKS0405_HE0226_J1427_input_tensor_train_clipped.pt')
torch.save(input_tensor_val_clipped, '../../data/PKS0405_HE0226_J1427_input_tensor_val_clipped.pt')

np.savetxt('../../data/PKS0405_HE0226_J1427_train_test.txt', train_test)
np.savetxt('../../data/PKS0405_HE0226_J1427_val_test.txt', val_test)
np.savetxt('../../data/PKS0405_HE0226_J1427_train_test_clipped.txt', train_test_clipped)
np.savetxt('../../data/PKS0405_HE0226_J1427_val_test_clipped.txt', val_test_clipped)
