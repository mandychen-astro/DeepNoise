import sys
import os

home_directory = os.path.expanduser('~')
sys.path.append(home_directory + '/DeepNoise/')
data_path = '/project/hwchen/data_mandy/'

from astropy.io import fits
from mpdaf.obj import Cube
import numpy as np
import torch
import gc

from src.data_processing import array_to_tensor, replace_outliers, apply_mask

# Load the data

def get_train_and_val(mask, cube, train_split=0.8):
    data = cube.data.data
    var = cube.var.data
    wave = cube.wave.coord()
    data = data[10:-10, :, :] # get rid of the first 10 and last 10 wavelength pixels because they are quite noisy
    var = var[10:-10, :, :]
    wave = wave[10:-10]

    # Preprocess the data, clipping outliers and normalizing etc.
    # could also mask nan with data = np.nan_to_num(data) if needed
    sky_spec = apply_mask(data, mask, extract_value=0)
    sky_var = apply_mask(var, mask, extract_value=0)
    # sky_spec = np.arcsinh(sky_spec) # arcsinh transform
    # sky_spec = sky_spec**3 / 3000**3 # cube transform to put more weight on sky lines
    # high_value_mask = sky_spec > 50
    # sky_spec[high_value_mask] = sky_spec[high_value_mask]*10 # enhance the high values
    sky_spec = sky_spec / 3000 # normalize the data
    sky_var = sky_var / 3000**2 # normalize the variance
    sky_spec_sigclipped = sky_spec.copy()
    sky_spec_sigclipped = replace_outliers(sky_spec_sigclipped, lower=3, upper=10, 
                        fill_value_lower='median', fill_value_upper='median')
    
    # sky_spec = rescale_data_w_prior_bounds(sky_spec, log=False, input_min=3, input_max=10)
    # sky_spec_sigclipped = rescale_data_w_prior_bounds(sky_spec_sigclipped, log=False, input_min=3, input_max=10)

    # Generate random indices for splitting the data
    num_samples = sky_spec.shape[1]
    print(num_samples, np.sum(mask==0))
    train_indices = np.random.choice(num_samples, int(train_split * num_samples), replace=False)
    test_indices = np.setdiff1d(np.arange(num_samples), train_indices)

    # Split the data into training and validation sets
    sky_spec_train = sky_spec[:, train_indices]
    sky_spec_test = sky_spec[:, test_indices]

    sky_spec_train_sigclipped = sky_spec_sigclipped[:, train_indices]
    sky_spec_test_sigclipped = sky_spec_sigclipped[:, test_indices]

    sky_var_train = sky_var[:, train_indices]
    sky_var_test = sky_var[:, test_indices]
    
    return wave, sky_spec_train, sky_spec_test, sky_spec_train_sigclipped, sky_spec_test_sigclipped, sky_var_train, sky_var_test

mask = fits.getdata(home_directory + '/sky_subtraction/PKS0405-123_OB1/skymask_badpixel_removed_OB1EXP1.fits')
cube = Cube(home_directory + '/sky_subtraction/PKS0405-123_OB1/DATACUBE_FINAL_EXP1.fits')
w1, t1, v1, tc1, vc1, var_t1, var_v1 = get_train_and_val(mask, cube)
print('t1:', t1.shape)
n = t1.shape[1]
w1_cube_train = np.tile(w1, (n, 1)).T
print(w1_cube_train.shape)
n = v1.shape[1]
w1_cube_val = np.tile(w1, (n, 1)).T
print(w1_cube_val.shape)

mask = fits.getdata(home_directory + '/sky_subtraction/PKS0405-123_OB1/skymask_badpixel_removed_OB1EXP2.fits')
cube = Cube(home_directory + '/sky_subtraction/PKS0405-123_OB1/DATACUBE_FINAL_EXP2.fits')
w2, t2, v2, tc2, vc2, var_t2, var_v2 = get_train_and_val(mask, cube)
print('t2:', t2.shape)
n = t2.shape[1]
w2_cube_train = np.tile(w2, (n, 1)).T
print(w2_cube_train.shape)
n = v2.shape[1]
w2_cube_val = np.tile(w2, (n, 1)).T
print(w2_cube_val.shape)

mask = fits.getdata(home_directory + '/sky_subtraction/HE0226-4110_OB1/skymask_badpixel_removed_OB1EXP1.fits')
cube = Cube(home_directory + '/sky_subtraction/HE0226-4110_OB1/DATACUBE_FINAL_EXP1.fits')
w3, t3, v3, tc3, vc3, var_t3, var_v3 = get_train_and_val(mask, cube)
print('t3:', t3.shape)
n = t3.shape[1]
w3_cube_train = np.tile(w3, (n, 1)).T
print(w3_cube_train.shape)
n = v3.shape[1]
w3_cube_val = np.tile(w3, (n, 1)).T
print(w3_cube_val.shape)

mask = fits.getdata(home_directory + '/sky_subtraction/SDSSJ1427-0121_OB1/skymask_badpixel_removed_OB1EXP1.fits')
cube = Cube(home_directory + '/sky_subtraction/SDSSJ1427-0121_OB1/DATACUBE_FINAL_EXP1.fits')
w4, t4, v4, tc4, vc4, var_t4, var_v4 = get_train_and_val(mask, cube)
print('t4:', t4.shape)
n = t4.shape[1]
w4_cube_train = np.tile(w4, (n, 1)).T
print(w4_cube_train.shape)
n = v4.shape[1]
w4_cube_val = np.tile(w4, (n, 1)).T
print(w4_cube_val.shape)

# Combine the training and validation data
sky_spec_train = np.concatenate((t1, t2, t3, t4), axis=1)
sky_spec_val = np.concatenate((v1, v2, v3, v4), axis=1)
print(sky_spec_train.shape)
print(sky_spec_val.shape)

sky_spec_train_sigclipped = np.concatenate((tc1, tc2, tc3, tc4), axis=1)
sky_spec_val_sigclipped = np.concatenate((vc1, vc2, vc3, vc4), axis=1)

sky_var_train = np.concatenate((var_t1, var_t2, var_t3, var_t4), axis=1)
sky_var_val = np.concatenate((var_v1, var_v2, var_v3, var_v4), axis=1)

wave_train = np.concatenate((w1_cube_train, w2_cube_train, w3_cube_train, w4_cube_train), axis=1)
wave_val = np.concatenate((w1_cube_val, w2_cube_val, w3_cube_val, w4_cube_val), axis=1)

num_samples = sky_spec_train.shape[1]
test_indices = np.random.choice(num_samples, 100, replace=False)# choose 100 for testing here
train_test = sky_spec_train[:, test_indices]
train_test_clipped = sky_spec_train_sigclipped[:, test_indices]
train_test_var = sky_var_train[:, test_indices]
train_test_wave = wave_train[:, test_indices]

num_samples = sky_spec_val.shape[1]
test_indices = np.random.choice(num_samples, 100, replace=False)# choose 100 for testing here
val_test = sky_spec_val[:, test_indices]
val_test_clipped = sky_spec_val_sigclipped[:, test_indices]
val_test_var = sky_var_val[:, test_indices] 
val_test_wave = wave_val[:, test_indices]


# Convert the input array to a tensor
input_tensor_train = array_to_tensor(sky_spec_train)
input_tensor_val = array_to_tensor(sky_spec_val)
print(input_tensor_train.size())
print(input_tensor_val.size())
torch.save(input_tensor_train, data_path + 'PKS0405_HE0226_J1427_input_tensor_train_v3.pt')
torch.save(input_tensor_val, data_path + 'PKS0405_HE0226_J1427_input_tensor_val_v3.pt')
del input_tensor_train
del input_tensor_val
gc.collect() 

input_tensor_train_clipped = array_to_tensor(sky_spec_train_sigclipped)
input_tensor_val_clipped = array_to_tensor(sky_spec_val_sigclipped) 
torch.save(input_tensor_train_clipped, data_path + 'PKS0405_HE0226_J1427_input_tensor_train_clipped_v3.pt')
torch.save(input_tensor_val_clipped, data_path + 'PKS0405_HE0226_J1427_input_tensor_val_clipped_v3.pt')
del input_tensor_train_clipped
del input_tensor_val_clipped
gc.collect()

input_tensor_train_var = array_to_tensor(sky_var_train)
input_tensor_val_var = array_to_tensor(sky_var_val)
torch.save(input_tensor_train_var, data_path + 'PKS0405_HE0226_J1427_input_tensor_train_var_v3.pt')
torch.save(input_tensor_val_var, data_path + 'PKS0405_HE0226_J1427_input_tensor_val_var_v3.pt')
del input_tensor_train_var
del input_tensor_val_var
gc.collect()

input_tensor_train_wave = array_to_tensor(wave_train)
input_tensor_val_wave = array_to_tensor(wave_val)
torch.save(input_tensor_train_wave, data_path + 'PKS0405_HE0226_J1427_input_tensor_train_wave_v3.pt')
torch.save(input_tensor_val_wave, data_path + 'PKS0405_HE0226_J1427_input_tensor_val_wave_v3.pt')
del input_tensor_train_wave
del input_tensor_val_wave
gc.collect()

np.savetxt(data_path + 'PKS0405_HE0226_J1427_train_test_v3.txt', train_test)
np.savetxt(data_path + 'PKS0405_HE0226_J1427_val_test_v3.txt', val_test)
np.savetxt(data_path + 'PKS0405_HE0226_J1427_train_test_clipped_v3.txt', train_test_clipped)
np.savetxt(data_path + 'PKS0405_HE0226_J1427_val_test_clipped_v3.txt', val_test_clipped)
np.savetxt(data_path + 'PKS0405_HE0226_J1427_train_test_var_v3.txt', train_test_var)
np.savetxt(data_path + 'PKS0405_HE0226_J1427_val_test_var_v3.txt', val_test_var)
np.savetxt(data_path + 'PKS0405_HE0226_J1427_train_test_wave_v3.txt', train_test_wave)
np.savetxt(data_path + 'PKS0405_HE0226_J1427_val_test_wave_v3.txt', val_test_wave)
