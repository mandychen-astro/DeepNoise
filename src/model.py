import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self):
        ####
        pass

    def forward(self, x):
        return 


class Conv1dDecoder(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return