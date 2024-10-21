import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from my_datasets.toy_dataset_processed import MRIToyDataset
import os
import numpy as np
from PIL import Image
import ants
from tqdm import tqdm

class ConceptEncoder(nn.Module):
    def __init__(self, input_channels=1, num_tokens=20, token_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_tokens * token_dim)
        )
        self.num_tokens = num_tokens
        self.token_dim = token_dim

    def forward(self, x):
        features = self.encoder(x)
        return features.view(-1, self.num_tokens, self.token_dim) #the output is 20*256, this step force them to be seperate tokens

class UNetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.residual_conv = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x + residual

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x, context):
        q = self.query(x)
        k = self.key(context)
        v = self.value(context)

        attention = F.softmax(torch.bmm(q, k.transpose(1, 2)) / (x.size(-1) ** 0.5), dim=-1)
        return torch.bmm(attention, v)
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        self.conv_in = nn.Conv3d(in_channels, 64, 3, padding=1)
        
        self.down1 = UNetBlock3D(64, 128)
        self.down2 = UNetBlock3D(128, 256)
        self.down3 = UNetBlock3D(256, 256)
        
        self.cross_attn1 = CrossAttention(256)
        self.cross_attn2 = CrossAttention(256)
        
        self.up1 = UNetBlock3D(512, 128)
        self.up2 = UNetBlock3D(256, 64)
        self.up3 = UNetBlock3D(128, 64)
        
        self.conv_out = nn.Conv3d(64, out_channels, 3, padding=1)

    def forward(self, x, t, context):
        t = self.time_mlp(t.unsqueeze(-1).float())
        
        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        b, c, d, h, w = x4.shape
        x4_flat = x4.view(b, c, d*h*w).transpose(1, 2)
        
        x4_flat = self.cross_attn1(x4_flat, context)
        x4_flat = self.cross_attn2(x4_flat, context)
        
        x4 = x4_flat.transpose(1, 2).view(b, c, d, h, w)
        
        x = self.up1(torch.cat([x4, x3], dim=1))
        x = self.up2(torch.cat([x, x2], dim=1))
        x = self.up3(torch.cat([x, x1], dim=1))
        
        return self.conv_out(x)
class EncDiff(nn.Module):
    def __init__(self, unet, concept_encoder):
        super().__init__()
        self.unet = unet
        self.concept_encoder = concept_encoder

    def forward(self, x, timesteps):
        concept_tokens = self.concept_encoder(x)
        return self.unet(x, timesteps, concept_tokens)

    def visualize(self, x):
        concept_tokens = self.concept_encoder(x)
        return concept_tokens

def create_encdiff_model(in_channels=1, out_channels=1):
    unet = UNet3D(in_channels, out_channels)

    concept_encoder = ConceptEncoder(input_channels=in_channels)

    return EncDiff(unet, concept_encoder)



class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_train_timesteps = num_train_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[timesteps])
        noisy_samples = sqrt_alphas_cumprod.view(-1, 1, 1, 1, 1) * original_samples + \
                        sqrt_one_minus_alphas_cumprod.view(-1, 1, 1, 1, 1) * noise
        return noisy_samples




def display_image(image_tensor, title, save_path):
    image_np = image_tensor.cpu().numpy()
    if image_np.ndim == 5:
        image_np = image_np[0, 0]
    elif image_np.ndim == 4:
        image_np = image_np[0]
    elif image_np.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected image shape: {image_np.shape}")
    
    print(f"{title} shape: {image_np.shape}")
    if image_np.shape != (64, 64, 64):
        raise ValueError(f"Expected image shape (64, 64, 64), but got {image_np.shape}")
    
    image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-8)
    ants_image = ants.from_numpy(image_np)
    ants_image.plot_ortho(flat=True, title=title, filename=save_path, xyz_lines=False, orient_labels=False)


def train_encdiff(num_epochs=100, batch_size=4, learning_rate=1e-4, save_interval=10, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up folders for logging and checkpoints
    base_dir = "../log/encdiff"
    tensorboard_dir = os.path.join(base_dir, "tensorboard_logs")
    os.makedirs(tensorboard_dir, exist_ok=True)

    run_number = 1
    while os.path.exists(os.path.join(base_dir, f"run_{run_number}")):
        run_number += 1
    run_dir = os.path.join(base_dir, f"run_{run_number}")
    os.makedirs(run_dir)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    vis_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(checkpoint_dir)
    os.makedirs(vis_dir)

    # Set up TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(tensorboard_dir, f"run_{run_number}"))

    # Prepare dataset and split into train/test
    dataset = MRIToyDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = create_encdiff_model(in_channels=1, out_channels=1)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    noise_scheduler = DDPMScheduler(device=device)

    # Early stopping variables
    best_test_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                clean_images = batch[0].view(-1, 1, 64, 64, 64).float().to(device)
                noise = torch.randn_like(clean_images)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (clean_images.shape[0],), device=device)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                noise_pred = model(noisy_images, timesteps)
                loss = F.mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Calculate reconstruction loss (without gradient)
                with torch.no_grad():
                    reconstructed = model(clean_images, torch.zeros(clean_images.shape[0], device=device).long())
                    recon_loss = F.mse_loss(reconstructed, clean_images)
                    epoch_recon_loss += recon_loss.item()

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Denoise Loss": f"{loss.item():.4f}", "Recon Loss": f"{recon_loss.item():.4f}"})

                # Log to TensorBoard
                writer.add_scalar('Loss/train_denoise', loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Loss/train_recon', recon_loss.item(), epoch * len(train_loader) + batch_idx)

        # Compute and log epoch metrics
        epoch_loss /= len(train_loader)
        epoch_recon_loss /= len(train_loader)
        writer.add_scalar('Loss/epoch_denoise', epoch_loss, epoch)
        writer.add_scalar('Loss/epoch_recon', epoch_recon_loss, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Denoise Loss: {epoch_loss:.4f}, Recon Loss: {epoch_recon_loss:.4f}")

        # Compute and log test loss (without gradient)
        model.eval()
        test_loss = 0
        test_recon_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                clean_images = batch[0].view(-1, 1, 64, 64, 64).float().to(device)
                noise = torch.randn_like(clean_images)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (clean_images.shape[0],), device=device)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                noise_pred = model(noisy_images, timesteps)
                loss = F.mse_loss(noise_pred, noise)
                test_loss += loss.item()

                # Calculate reconstruction loss for test set
                reconstructed = model(clean_images, torch.zeros(clean_images.shape[0], device=device).long())
                recon_loss = F.mse_loss(reconstructed, clean_images)
                test_recon_loss += recon_loss.item()

        test_loss /= len(test_loader)
        test_recon_loss /= len(test_loader)
        writer.add_scalar('Loss/test_denoise', test_loss, epoch)
        writer.add_scalar('Loss/test_recon', test_recon_loss, epoch)
        print(f"Test Denoise Loss: {test_loss:.4f}, Test Recon Loss: {test_recon_loss:.4f}")

        # Early stopping check (using denoising loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1

        # Check if we should stop early
        if epochs_no_improve == patience:
            print(f"Early stopping triggered. Best epoch: {best_epoch+1}")
            break

        # Save visualizations and checkpoint every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            model.eval()
            with torch.no_grad():
                # Visualize from training set
                train_sample = next(iter(train_loader))[0].to(device)
                train_recon = model(train_sample, torch.zeros(train_sample.shape[0], device=device).long())

                # Display original and reconstructed images from training set
                train_original_path = os.path.join(vis_dir, f'train_original_epoch_{epoch+1}.png')
                train_recon_path = os.path.join(vis_dir, f'train_reconstructed_epoch_{epoch+1}.png')
                display_image(train_sample, f"Train Original - Epoch {epoch+1}", train_original_path)
                display_image(train_recon, f"Train Reconstructed - Epoch {epoch+1}", train_recon_path)

                # Log training images to TensorBoard
                train_original_img = Image.open(train_original_path)
                train_recon_img = Image.open(train_recon_path)
                writer.add_image('Train/Original', np.array(train_original_img), epoch, dataformats='HWC')
                writer.add_image('Train/Reconstructed', np.array(train_recon_img), epoch, dataformats='HWC')

                # Visualize from test set
                test_sample = next(iter(test_loader))[0].to(device)
                test_recon = model(test_sample, torch.zeros(test_sample.shape[0], device=device).long())

                # Display original and reconstructed images from test set
                test_original_path = os.path.join(vis_dir, f'test_original_epoch_{epoch+1}.png')
                test_recon_path = os.path.join(vis_dir, f'test_reconstructed_epoch_{epoch+1}.png')
                display_image(test_sample, f"Test Original - Epoch {epoch+1}", test_original_path)
                display_image(test_recon, f"Test Reconstructed - Epoch {epoch+1}", test_recon_path)

                # Log test images to TensorBoard
                test_original_img = Image.open(test_original_path)
                test_recon_img = Image.open(test_recon_path)
                writer.add_image('Test/Original', np.array(test_original_img), epoch, dataformats='HWC')
                writer.add_image('Test/Reconstructed', np.array(test_recon_img), epoch, dataformats='HWC')

            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)

    writer.close()
    print(f"Training completed. Logs and checkpoints saved in {run_dir}")
    print(f"TensorBoard logs saved in {tensorboard_dir}")
    print(f"Best model saved at epoch {best_epoch+1} with test denoising loss: {best_test_loss:.4f}")

if __name__ == "__main__":
    train_encdiff()