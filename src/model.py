import torch
import torch.nn as nn

class Trim(nn.Module):
    def __init__(self, trim = 3661):
        super(Trim, self).__init__()
        self.trim = trim

    def forward(self, x):
        return x[:, :, :self.trim]

class EncoderBlockConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncoderBlockConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class DecoderBlockConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DecoderBlockConv, self).__init__()    
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, 1, kernel_size=1, stride=1, padding=0),   
            Trim(3681)
        )

    def forward(self, x):
        return self.block(x)

class AutoEncoderConv(nn.Module):   
    def __init__(self, enc_in_channels, enc_out_channels, enc_kernel_size, enc_stride, enc_padding, 
                 dec_in_channels, dec_out_channels, dec_kernel_size, dec_stride, dec_padding):
        super(AutoEncoderConv, self).__init__()
        self.encoder = EncoderBlockConv(enc_in_channels, enc_out_channels, enc_kernel_size, enc_stride, enc_padding)
        self.decoder = DecoderBlockConv(dec_in_channels, dec_out_channels, dec_kernel_size, dec_stride, dec_padding)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

