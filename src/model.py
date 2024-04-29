import torch
import torch.nn as nn

class Trim(nn.Module):
    def __init__(self, trim = 3661):
        super(Trim, self).__init__()
        self.trim = trim

    def forward(self, x):
        return x[:, :, :self.trim]

class ConvEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvEncoderBlock, self).__init__()
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

class ConvDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvDecoderBlock, self).__init__()    
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

class ConvAutoencoder(nn.Module):   
    def __init__(self, enc_in_channels, enc_out_channels, enc_kernel_size, enc_stride, enc_padding, 
                 dec_in_channels, dec_out_channels, dec_kernel_size, dec_stride, dec_padding):
        super(ConvAutoencoder, self).__init__()
        self.encoder = ConvEncoderBlock(enc_in_channels, enc_out_channels, enc_kernel_size, enc_stride, enc_padding)
        self.decoder = ConvDecoderBlock(dec_in_channels, dec_out_channels, dec_kernel_size, dec_stride, dec_padding)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SpectrumTransformerEncoder(nn.Module):
    def __init__(self, num_specpixles, embedding_dim):
        super(SpectrumTransformerEncoder, self).__init__()
        self.spec_embedding = nn.Linear(num_specpixles, embedding_dim)  
        # self.positional_encodings = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4),
            num_layers=4
        )

    def forward(self, x):
        # x is [batch_size, 1, num_specpixels] 
        x = self.spec_embedding(x)  # Convert each spectrum to an embedding
        # x += self.positional_encodings  # Add positional encodings
        x = self.transformer_encoder(x)  # Pass through the transformer encoder
        return x

class SpectrumTransformerDecoder(nn.Module):
    def __init__(self, num_specpixels, embedding_dim, output_dim):
        super(SpectrumTransformerDecoder, self).__init__()
        # Assuming the same embedding dimension for simplicity and symmetry
        self.decoder_embedding = nn.Linear(num_specpixels, embedding_dim)
        # self.positional_encodings = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        # Decoder Layers: Same configuration as the encoder but using TransformerDecoderLayer
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=4),
            num_layers=4
        )

        # Final layer to transform back to spectrum dimensionality
        self.output_layer = nn.Linear(embedding_dim, num_specpixels)

    def forward(self, target, encoded_output):
        # Embed and add positional encodings to the target patches
        memory = self.decoder_embedding(encoded_output)
        # memory += self.positional_encodings
        
        # Decoding the patches with attention to the memory from the encoder
        decoded_patches = self.transformer_decoder(target, memory)
        decoded_patches = self.output_layer(decoded_patches.squeeze(1))
        
        # Transform to original patch dimensionality
        return self.output_layer(decoded_patches)

class SpectrumTransformer(nn.Module):
    def __init__(self, num_specpixels, embedding_dim, output_dim):
        super(SpectrumTransformer, self).__init__()
        self.encoder = SpectrumTransformerEncoder(num_specpixels, embedding_dim)
        self.decoder = SpectrumTransformerDecoder(num_specpixels, embedding_dim, output_dim)

    def forward(self, x):
        encoded_output = self.encoder(x)
        decoded_output = self.decoder(x, encoded_output)
        return decoded_output