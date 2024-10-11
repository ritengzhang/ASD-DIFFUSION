from EncDec.vae import SimpleVAE
from .unet import UNet
from .text_encoder import SimpleTextEncoder
from .cross_attention import CrossAttention
from diffusion import StableDiffusionModel
import torch

def get_models(config, device):
    vae = SimpleVAE(
        in_channels=1, 
        out_channels=1, 
        latent_channels=config.latent_channels
    ).to(device)

    # Load pre-trained VAE weights
    vae.load_state_dict(torch.load(config.vae_path))
    vae.eval()  # Set VAE to evaluation mode

    cross_attention = CrossAttention(config.latent_channels, 768) if config.use_text_conditioning else None

    unet = UNet(
        image_size=config.image_size // 4,  # Adjust for VAE downsampling
        in_channels=config.latent_channels,
        out_channels=config.latent_channels,
        cross_attention=cross_attention
    ).to(device)

    text_encoder = SimpleTextEncoder().to(device) if config.use_text_conditioning else None

    stable_diffusion_model = StableDiffusionModel(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        use_text_conditioning=config.use_text_conditioning,
        num_diffusion_steps=config.num_diffusion_steps
    ).to(device)

    # Set which modules to train
    modules_to_train = [vae, text_encoder, unet, cross_attention]
    for module, train in zip(modules_to_train, config.train_modules):
        if module is not None:
            for param in module.parameters():
                param.requires_grad = train

    return stable_diffusion_model