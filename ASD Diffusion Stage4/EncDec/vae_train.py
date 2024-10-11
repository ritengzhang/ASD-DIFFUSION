import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split
import ants
import numpy as np
import sys
import os
import json
import uuid
from datetime import datetime
import matplotlib.pyplot as plt

# Add the parent directory and current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

from vae import SimpleVAE, vae_loss
from datasets import get_dataset
from vae_config import get_vae_config, print_vae_config

def display_image(image_tensor, title, save_path):
    # Convert PyTorch tensor to numpy array
    image_np = image_tensor.cpu().numpy()
    
    # Remove batch and channel dimensions if they exist
    if image_np.ndim == 5:  # (B, C, D, H, W)
        image_np = image_np[0, 0]
    elif image_np.ndim == 4:  # (C, D, H, W) or (B, D, H, W)
        image_np = image_np[0]
    elif image_np.ndim == 3:  # (D, H, W)
        pass  # Already in correct format
    else:
        raise ValueError(f"Unexpected image shape: {image_np.shape}")
    
    print(f"{title} shape: {image_np.shape}")
    
    # Ensure the image is of shape (64, 64, 64)
    if image_np.shape != (64, 64, 64):
        raise ValueError(f"Expected image shape (64, 64, 64), but got {image_np.shape}")
    
    # Normalize to [0, 1] range
    image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-8)
    
    # Create ANTs image from numpy array
    ants_image = ants.from_numpy(image_np)
    
    # Use the plot_ortho method from ANTs to save the figure
    ants_image.plot_ortho(flat=True, title=title, filename=save_path, xyz_lines=False, orient_labels=False)

def normalize_data(x):
    return (x - x.min()) / (x.max() - x.min())

def train_vae(config):
    # Create the vae_runs folder in the current directory
    vae_runs_folder = os.path.join(current_dir, 'vae_runs')
    os.makedirs(vae_runs_folder, exist_ok=True)
    
    # Create subfolders for models, results, and learning curves
    models_folder = os.path.join(vae_runs_folder, 'models')
    results_folder = os.path.join(vae_runs_folder, 'results_visualized')
    learning_curves_folder = os.path.join(vae_runs_folder, 'learning_curves')
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(learning_curves_folder, exist_ok=True)
    
    # Create a unique ID for this training run
    run_id = str(uuid.uuid4())[:8]
    
    # Create a folder for this run's results
    run_results_folder = os.path.join(results_folder, f'run_{run_id}')
    os.makedirs(run_results_folder, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get the dataset
    dataset = get_dataset(config.dataset)
    
    if config.train_only:
        train_dataset = dataset
        val_dataset = None
    else:
        train_size = int(config.train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    vae = SimpleVAE(
        in_channels=1, 
        out_channels=1, 
        latent_channels=config.latent_channels
    ).to(device)
    
    optimizer = optim.Adam(vae.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler_T0, T_mult=config.scheduler_T_mult, eta_min=config.scheduler_eta_min)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config.num_epochs):
        vae.train()
        train_loss = 0
        for batch in train_loader:
            if isinstance(batch, list):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            
            x = normalize_data(x)  # Normalize input data
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(x)
            loss = vae_loss(recon_batch, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        scheduler.step()
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        if val_dataset:
            vae.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, list):
                        x = batch[0].to(device)
                    else:
                        x = batch.to(device)
                    x = normalize_data(x)  # Normalize input data
                    recon_batch, mu, logvar = vae(x)
                    val_loss += vae_loss(recon_batch, x, mu, logvar).item()
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        else:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        if (epoch + 1) % 1000 == 0 and config.visualize:
            with torch.no_grad():
                sample = next(iter(train_loader))
                if isinstance(sample, list):
                    sample = sample[0]
                sample = sample.to(device)
                sample = normalize_data(sample)
                recon, _, _ = vae(sample)
                
                # Display original and reconstructed images
                display_image(sample, f"Original - Epoch {epoch+1}", 
                              os.path.join(run_results_folder, f'original_epoch_{epoch+1}.png'))
                display_image(recon, f"Reconstructed - Epoch {epoch+1}", 
                              os.path.join(run_results_folder, f'reconstructed_epoch_{epoch+1}.png'))
    
    # Save the model using an absolute path
    model_path = os.path.join(models_folder, f'vae_model_{run_id}.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(vae.state_dict(), model_path)
    print(f"VAE model saved to {model_path}")

    # Save learning curves
    learning_curve_path = None
    if not config.train_only:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VAE Learning Curves')
        plt.legend()
        learning_curve_path = os.path.join(learning_curves_folder, f'learning_curve_{run_id}.png')
        os.makedirs(os.path.dirname(learning_curve_path), exist_ok=True)
        plt.savefig(learning_curve_path)
        plt.close()
        print(f"Learning curves saved to {learning_curve_path}")

    # Prepare metadata
    metadata = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'learning_rate': config.learning_rate,
            'train_split': config.train_split,
            'visualize': config.visualize,
            'train_only': config.train_only,
            'latent_channels': config.latent_channels,
            'dataset': {
                'name': config.dataset.name
            }
        },
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1] if val_losses else None,
        'model_path': model_path,
        'learning_curve_path': learning_curve_path
    }
    
    # Load existing metadata or create new file
    metadata_path = os.path.join(vae_runs_folder, 'vae_runs_metadata.json')
    try:
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
        if not isinstance(all_metadata, list):
            all_metadata = [all_metadata]
    except (FileNotFoundError, json.JSONDecodeError):
        all_metadata = []
    
    # Append new metadata and save
    all_metadata.append(metadata)
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=4)
    print(f"Metadata appended to {metadata_path}")

if __name__ == "__main__":
    config = get_vae_config()
    print_vae_config(config)
    train_vae(config)