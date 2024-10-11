# unet.py
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, cross_attention=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.cross_attention = cross_attention
        if up:
            self.conv1 = nn.Conv3d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose3d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv3d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm3d(out_ch)
        self.bnorm2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t, context=None):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 3]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        h = self.transform(h)
        
        if self.cross_attention is not None and context is not None:
            # Reshape for cross-attention
            b, c, d, height, width = h.shape
            h_flat = h.permute(0, 2, 3, 4, 1).reshape(b, d*height*width, c)
            h_flat = self.cross_attention(h_flat, context)
            h = h_flat.reshape(b, d, height, width, c).permute(0, 4, 1, 2, 3)
        
        return h

class UNet(nn.Module):
    def __init__(self, image_size=64, in_channels=4, out_channels=4, cross_attention=None):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        self.conv0 = nn.Conv3d(in_channels, 32, 3, padding=1)
        self.downs = nn.ModuleList([
            Block(32, 64, time_emb_dim, cross_attention=cross_attention),
            Block(64, 128, time_emb_dim, cross_attention=cross_attention)
        ])
        self.ups = nn.ModuleList([
            Block(128, 64, time_emb_dim, up=True, cross_attention=cross_attention),
            Block(64, 32, time_emb_dim, up=True, cross_attention=cross_attention)
        ])
        self.output = nn.Conv3d(32, out_channels, 1)

    def forward(self, x, timestep, context=None):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t, context)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t, context)
        return self.output(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings