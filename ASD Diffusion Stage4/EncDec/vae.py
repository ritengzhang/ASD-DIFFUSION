import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_channels=16):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            self.conv_block(in_channels, 32, kernel_size=3, stride=2, padding=1),
            self.conv_block(32, 64, kernel_size=3, stride=2, padding=1),
            self.conv_block(64, 128, kernel_size=3, stride=2, padding=1),
            self.conv_block(128, 256, kernel_size=3, stride=2, padding=1),
            nn.Conv3d(256, latent_channels * 2, kernel_size=3, stride=1, padding=1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.deconv_block(256, 128, kernel_size=4, stride=2, padding=1),
            self.deconv_block(128, 64, kernel_size=4, stride=2, padding=1),
            self.deconv_block(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose3d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout3d(0.2)
        )
    
    def deconv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout3d(0.2)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, kld_weight=0.5):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + kld_weight * KLD