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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class EnhancedVAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_channels=128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            self.conv_block(in_channels, 32, kernel_size=3, stride=2, padding=1),
            ResidualBlock(32, 32),
            self.conv_block(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, 64),
            self.conv_block(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, 128),
            self.conv_block(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, 256),
            nn.Conv3d(256, latent_channels * 2, kernel_size=3, stride=1, padding=1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(256, 256),
            self.deconv_block(256, 128, kernel_size=4, stride=2, padding=1),
            ResidualBlock(128, 128),
            self.deconv_block(128, 64, kernel_size=4, stride=2, padding=1),
            ResidualBlock(64, 64),
            self.deconv_block(64, 32, kernel_size=4, stride=2, padding=1),
            ResidualBlock(32, 32),
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

def vae_loss(recon_x, x, mu, logvar, loss_type='mse', kld_weight=1.0):
    """
    Calculate the loss for VAE with options for MSE or BCE reconstruction loss,
    and always compute MSE (L2) reconstruction loss as an additional metric.
    
    Parameters:
    recon_x: The reconstructed output from the VAE.
    x: The original input.
    mu: The mean from the VAE encoder.
    logvar: The log variance from the VAE encoder.
    loss_type: 'mse' or 'bce' for the reconstruction loss used during training.
    kld_weight: The weight for the KLD loss term.
    
    Returns:
    loss: The total loss combining reconstruction and KLD.
    recon_loss: The primary reconstruction loss (BCE or MSE).
    kld_loss: The unweighted KLD loss.
    weighted_kld_loss: The weighted KLD loss.
    l2_recon_loss: The MSE (L2) reconstruction loss (always computed for comparison).
    """
    
    # Primary Reconstruction Loss (either MSE or BCE based on training setup)
    if loss_type == 'mse':
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.numel()
        l2_recon_loss = recon_loss  # MSE is being used as the primary loss
    elif loss_type == 'bce':
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.numel()
        # Additionally compute MSE (L2) loss as a comparison metric
        l2_recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.numel()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    # KLD Loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(1)
    weighted_kld_loss = kld_weight * kld_loss
    
    # Total loss (reconstruction + weighted KLD)
    loss = recon_loss + weighted_kld_loss
    
    return loss, recon_loss, kld_loss, weighted_kld_loss, l2_recon_loss
