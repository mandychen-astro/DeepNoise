import sys
import os

home_directory = os.path.expanduser('~')
sys.path.append(home_directory + '/DeepNoise/')

import torch
from torch import nn
from torch.utils.data import DataLoader
from src.model import SpectrumTransformer
from src.train import train_model
from src.data_processing import CustomDataset
from torchinfo import summary
import time

# Load the data
input_tensor = torch.load('../data/PKS0405-123_OB1EXP1_input_tensor.pt')
print(input_tensor.size())

num_specpixels, embedding_dim = input_tensor.size(2), 64
print(num_specpixels, embedding_dim)
autoencoder = SpectrumTransformer(num_specpixels, embedding_dim)

summary(autoencoder, (1, 1, num_specpixels))

# test the model with 5 epochs
# Define a loss function
criterion = nn.MSELoss()  # Mean Squared Error

# Choose an optimizer
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# Load your data
dataset = CustomDataset(input_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Train the model
trained_model = train_model(autoencoder, train_loader, criterion, optimizer, 
                num_epochs=5, device='cpu')

torch.save(autoencoder.state_dict(), '../models/model_PKS0405-123_OB1EXP1_state_dict.pth')
