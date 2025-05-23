import sys
import os

home_directory = os.path.expanduser('~')
sys.path.append(home_directory + '/DeepNoise/')
data_path = '/project/hwchen/data_mandy/'

import torch
from torch import nn
from torch.utils.data import DataLoader
from src.model import SpectrumTransformer
from src.train import train_model
from src.data_processing import CustomDataset
# from torchinfo import summary

import numpy as np
import time

print(torch.cuda.device_count())
print(torch.cuda.get_device_name())

# Load the data
train_tensor = torch.load(data_path + 'PKS0405_HE0226_J1427_input_tensor_train_v6.pt')
val_tensor = torch.load(data_path + 'PKS0405_HE0226_J1427_input_tensor_val_v6.pt')
train_tensor_clipped = torch.load(data_path + 'PKS0405_HE0226_J1427_input_tensor_train_clipped_v6.pt')
val_tensor_clipped = torch.load(data_path + 'PKS0405_HE0226_J1427_input_tensor_val_clipped_v6.pt')
print(train_tensor.size())
print(val_tensor.size())

num_specpixels, embedding_dim = train_tensor.size(2), 64
print(num_specpixels, embedding_dim)
autoencoder = SpectrumTransformer(num_specpixels, embedding_dim)

# summary(autoencoder, (1, 1, num_specpixels))

# test the model with 20 epochs
# Define a loss function
criterion = nn.MSELoss()  

# Choose an optimizer
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)#, weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Load your data
train_data = CustomDataset(train_tensor, train_tensor_clipped)
val_data = CustomDataset(val_tensor, val_tensor_clipped)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# Train the model
trained_model, train_loss, val_loss = train_model(model=autoencoder, train_loader=train_loader, 
                criterion=criterion, optimizer=optimizer, val_loader=val_loader,
                return_train_loss=True, return_val_loss=True, 
                num_epochs=15, device='cuda')

torch.save(autoencoder.state_dict(), '../../models/model_PKS0405_HE0226_J1427_state_dict_v9.pth')
np.savetxt('../../models/model_PKS0405_HE0226_J1427_train_loss_v9.txt', train_loss)
np.savetxt('../../models/model_PKS0405_HE0226_J1427_val_loss_v9.txt', val_loss)