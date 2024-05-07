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
        self.dropout1 = nn.Dropout(0.2)
        self.positional_encodings = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.dropout2 = nn.Dropout(0.2)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=1024,
            dropout=0.2),
            num_layers=4
        )

    def forward(self, x):
        # x is [batch_size, 1, num_specpixels] 
        x = self.spec_embedding(x)  # Convert each spectrum to an embedding
        x = self.dropout1(x)
        x += self.positional_encodings  # Add positional encodings
        x = self.dropout2(x)
        x = self.transformer_encoder(x)  # Pass through the transformer encoder
        return x

class SpectrumTransformerDecoder(nn.Module):
    def __init__(self, num_specpixels, embedding_dim):
        super(SpectrumTransformerDecoder, self).__init__()
        # Assuming the same embedding dimension for simplicity and symmetry
        self.decoder_embedding = nn.Linear(num_specpixels, embedding_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.positional_encodings = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.dropout2 = nn.Dropout(0.2)
        
        # Decoder Layers: Same configuration as the encoder but using TransformerDecoderLayer
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=1024,
            dropout=0.2),
            num_layers=4
        )

        # Final layer to transform back to spectrum dimensionality
        self.output_layer = nn.Linear(embedding_dim, num_specpixels)

    def forward(self, x, encoded_output):
        # Embed and add positional encodings to the target patches
        target = self.decoder_embedding(x)
        target = self.dropout1(target)
        target += self.positional_encodings
        target = self.dropout2(target)
        
        # Decoding the patches with attention to the memory from the encoder
        y = self.transformer_decoder(target, encoded_output)
        y = self.output_layer(y)
        
        return y

class SpectrumTransformer(nn.Module):
    def __init__(self, num_specpixels, embedding_dim):
        super(SpectrumTransformer, self).__init__()
        self.encoder = SpectrumTransformerEncoder(num_specpixels, embedding_dim)
        self.decoder = SpectrumTransformerDecoder(num_specpixels, embedding_dim)

    def forward(self, x):
        encoded_output = self.encoder(x)
        decoded_output = self.decoder(x, encoded_output)
        return decoded_output


class InputWeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target):
        """
        Calculate the weighted mean squared error loss.

        Args:
            input (torch.Tensor): Predictions from the model.
            target (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: The computed weighted MSE loss.
        """
        # Compute weights: square of input values, normalized
        squared_inputs = input ** 2
        weight = squared_inputs / squared_inputs.sum()

        # Ensure the weights are the same shape as the input and target
        if weight.shape != input.shape:
            raise ValueError("Weight shape must match input shape")

        # Calculate the squared differences
        diff = input - target
        squared_diff = diff ** 2
        
        # Apply weights
        weighted_squared_diff = weight * squared_diff
        
        # Return the mean of the weighted squared differences
        loss = weighted_squared_diff.mean()
        return loss
