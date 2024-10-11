import torch
import torch.nn as nn

class StableDiffusionModel(nn.Module):
    def __init__(self, unet, vae, text_encoder, use_text_conditioning=False, num_diffusion_steps=1000):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder if use_text_conditioning else None
        self.use_text_conditioning = use_text_conditioning
        self.num_diffusion_steps = num_diffusion_steps
        
        # Define beta schedule
        self.beta = torch.linspace(0.0001, 0.02, num_diffusion_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Register beta schedule as buffers so they're moved to the correct device
        self.register_buffer('beta_buffer', self.beta)
        self.register_buffer('alpha_buffer', self.alpha)
        self.register_buffer('alpha_bar_buffer', self.alpha_bar)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = torch.sqrt(self.alpha_bar_buffer[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar_buffer[t])
        
        return sqrt_alpha_bar.view(-1, 1, 1, 1, 1) * x_start + sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1, 1) * noise

    def p_sample(self, x, t, text_embedding):
        t_tensor = t * torch.ones(x.shape[0], dtype=torch.long, device=x.device)
        noise_pred = self.unet(x, t_tensor, text_embedding)
        
        alpha = self.alpha_buffer[t]
        alpha_bar = self.alpha_bar_buffer[t]
        beta = self.beta_buffer[t]
        
        c1 = torch.sqrt(1 / alpha)
        c2 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        
        return c1.view(-1, 1, 1, 1, 1) * (x - c2.view(-1, 1, 1, 1, 1) * noise_pred) + torch.sqrt(beta).view(-1, 1, 1, 1, 1) * torch.randn_like(x)

    def forward(self, x, text, t):
        with torch.no_grad():
            latent, _, _ = self.vae.encode(x)
        
        # Prepare text embedding
        if self.use_text_conditioning and text is not None:
            text_embedding = self.text_encoder(text)
        else:
            text_embedding = None

        # Forward diffusion
        noise = torch.randn_like(latent)
        noisy_latent = self.q_sample(latent, t, noise)
        
        # Predict noise
        pred_noise = self.unet(noisy_latent, t, text_embedding)
        
        return pred_noise, noise

    def sample(self, batch_size, text=None):
        latent_channels = self.unet.in_channels
        latent_size = self.unet.image_size // 4  # Assuming the VAE downsamples by a factor of 4
        shape = (batch_size, latent_channels, latent_size, latent_size, latent_size)
        device = next(self.parameters()).device
        
        # Prepare text embedding
        if self.use_text_conditioning and text is not None:
            text_embedding = self.text_encoder(text)
        else:
            text_embedding = None

        # Start from pure noise in the latent space
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_diffusion_steps)):
            x = self.p_sample(x, torch.full((batch_size,), t, device=device, dtype=torch.long), text_embedding)
        
        with torch.no_grad():
            decoded = self.vae.decode(x)
            # Ensure the output is squeezed to remove any singleton dimensions
            return decoded.squeeze()