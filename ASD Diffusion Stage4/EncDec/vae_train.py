import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import sys
import os
import json
import uuid
from datetime import datetime
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import ants

# Add the parent directory and current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

from vae import  vae_loss
from datasets import get_dataset
from vae_config import get_vae_config, print_vae_config


def get_relative_path(path):
    return os.path.relpath(path, current_dir)

def normalize_data(x):
    return (x - x.min()) / (x.max() - x.min())

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

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filename):
    return torch.load(filename)

def update_run_info(run_folder, run_info):
    with open(os.path.join(run_folder, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=4)


def get_device_info():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        device_info["cuda_version"] = torch.version.cuda
        device_info["gpu_name"] = torch.cuda.get_device_name(0)
    return device_info

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
def update_all_runs(vae_runs_folder, run_info):
    all_runs_file = os.path.join(vae_runs_folder, 'all_runs.json')
    try:
        if os.path.exists(all_runs_file) and os.path.getsize(all_runs_file) > 0:
            with open(all_runs_file, 'r') as f:
                all_runs = json.load(f)
        else:
            all_runs = []
    except json.JSONDecodeError:
        print(f"Error reading {all_runs_file}. Starting with empty list.")
        all_runs = []
    
    # Prepare the condensed run info
    condensed_run_info = {
        'run_id': run_info['run_id'],
        'config': run_info['config'],
        'start_time': run_info['start_time'],
        'best_loss': run_info['best_loss'],
        'best_epoch': run_info['best_epoch'],
        'latest_epoch': run_info['latest_epoch'],
        'device_info': run_info['device_info']
    }
    
    # Add metrics for best and latest epochs
    best_checkpoint = next((cp for cp in run_info['checkpoints'] if cp['epoch'] == run_info['best_epoch']), None)
    latest_checkpoint = next((cp for cp in run_info['checkpoints'] if cp['epoch'] == run_info['latest_epoch']), None)
    
    if best_checkpoint:
        condensed_run_info['best_metrics'] = {
            'train_loss': best_checkpoint['train_loss'],
            'val_loss': best_checkpoint['val_loss'],
            'learning_rate': best_checkpoint['learning_rate']
        }
    
    if latest_checkpoint:
        condensed_run_info['latest_metrics'] = {
            'train_loss': latest_checkpoint['train_loss'],
            'val_loss': latest_checkpoint['val_loss'],
            'learning_rate': latest_checkpoint['learning_rate']
        }
    
    # Update or append the condensed run info
    for run in all_runs:
        if run['run_id'] == condensed_run_info['run_id']:
            run.update(condensed_run_info)
            break
    else:
        all_runs.append(condensed_run_info)
    
    with open(all_runs_file, 'w') as f:
        json.dump(all_runs, f, indent=4)

def update_all_finished_runs(vae_runs_folder, run_info):
    all_finished_runs_file = os.path.join(vae_runs_folder, 'all_finished_runs.json')
    try:
        if os.path.exists(all_finished_runs_file) and os.path.getsize(all_finished_runs_file) > 0:
            with open(all_finished_runs_file, 'r') as f:
                all_finished_runs = json.load(f)
        else:
            all_finished_runs = []
    except json.JSONDecodeError:
        print(f"Error reading {all_finished_runs_file}. Starting with empty list.")
        all_finished_runs = []
    
    # Only add the run if it has finished (check for 'end_time' key)
    if 'end_time' in run_info:
        condensed_run_info = {
            'run_id': run_info['run_id'],
            'config': run_info['config'],
            'start_time': run_info['start_time'],
            'end_time': run_info['end_time'],
            'best_loss': run_info['best_loss'],
            'best_epoch': run_info['best_epoch'],
            'total_epochs': run_info['latest_epoch'],
            'total_time': run_info['total_time'],
            'device_info': run_info['device_info'],
            'final_train_loss': run_info['final_train_loss'],
            'final_val_loss': run_info['final_val_loss']
        }
        
        # Add or update the run info
        for run in all_finished_runs:
            if run['run_id'] == condensed_run_info['run_id']:
                run.update(condensed_run_info)
                break
        else:
            all_finished_runs.append(condensed_run_info)
        
        with open(all_finished_runs_file, 'w') as f:
            json.dump(all_finished_runs, f, indent=4)
def train_vae(config):
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    vae_runs_folder = os.path.join(current_dir, 'vae_runs')
    os.makedirs(vae_runs_folder, exist_ok=True)
    
    log_dir = os.path.join(current_dir, 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    if config.resume_from:
        run_id = config.resume_from
        run_folder = os.path.join(vae_runs_folder, run_id)
        if not os.path.exists(run_folder):
            raise ValueError(f"No run found with ID: {run_id}")
        print(f"Resuming run: {run_id}")
    else:
        run_id = f"{config.run_name_prefix}{config.dataset.name}_{config.vae_class.__name__}_{str(config.latent_channels)}_{config.run_name_suffix}_{str(uuid.uuid4())[:8]}"
        run_folder = os.path.join(vae_runs_folder, run_id)
        os.makedirs(run_folder, exist_ok=True)
        print(f"Starting new run: {run_id}")
    
    models_folder = os.path.join(run_folder, 'checkpoints')
    results_folder = os.path.join(run_folder, 'results_visualized')
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_id))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(config.dataset)
    
    if config.dataset.max_samples == "max":
        max_samples = len(dataset)
    else:
        max_samples = min(int(config.dataset.max_samples), len(dataset))
    
    if max_samples < len(dataset):
        dataset = Subset(dataset, range(max_samples))
    
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
    
    vae = config.vae_class(
        in_channels=1, 
        out_channels=1, 
        latent_channels=config.latent_channels
    ).to(device)
    
    optimizer = optim.Adam(vae.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler_T0, T_mult=config.scheduler_T_mult, eta_min=config.scheduler_eta_min)
    
    start_epoch = 0
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    if config.resume_from:
        latest_checkpoint = max(
            [f for f in os.listdir(models_folder) if f.startswith('checkpoint_epoch_')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        checkpoint = load_checkpoint(os.path.join(models_folder, latest_checkpoint))
        vae.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        print(f"Resuming training from epoch {start_epoch}")
        
        with open(os.path.join(run_folder, 'run_info.json'), 'r') as f:
            run_info = json.load(f)
    else:
        run_info = {
            'run_id': run_id,
            'config': config.to_dict(),
            'start_time': datetime.now().isoformat(),
            'best_loss': best_loss,
            'best_epoch': start_epoch,
            'latest_epoch': start_epoch,
            'checkpoints': [],
            'device_info': get_device_info()
        }
    
    if config.early_stop:
        early_stopping = EarlyStopping(patience=config.early_stop_patience, delta=config.early_stop_delta)
    
    start_time = time.time()
    val_loss = None
    
    try:
        for epoch in range(start_epoch, config.num_epochs):
            epoch_start_time = time.time()
            vae.train()
            train_loss = 0
            train_recon_loss = 0
            train_kld_loss = 0
            train_l2_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                if isinstance(batch, list):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)

                x = normalize_data(x)

                optimizer.zero_grad()
                recon_batch, mu, logvar = vae(x)
                loss, recon_loss, kld_loss, weighted_kld_loss, l2_recon_loss = vae_loss(
                    recon_batch, x, mu, logvar, loss_type=config.loss_type, kld_weight=config.kld_weight
                )

                loss.backward()
                optimizer.step()

                # Accumulate losses
                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kld_loss += kld_loss.item()
                train_l2_loss += l2_recon_loss.item()

            scheduler.step()

            # Normalize losses
            if config.normalize_loss:
                train_loss /= len(train_loader.dataset)
                train_recon_loss /= len(train_loader.dataset)
                train_kld_loss /= len(train_loader.dataset)
                train_l2_loss /= len(train_loader.dataset)
            else:
                train_loss /= len(train_loader)
                train_recon_loss /= len(train_loader)
                train_kld_loss /= len(train_loader)
                train_l2_loss /= len(train_loader)

            train_losses.append(train_loss)

            # Log losses to TensorBoard
            writer.add_scalar('Loss/train_total', train_loss, epoch)
            writer.add_scalar('Loss/train_recon', train_recon_loss, epoch)
            writer.add_scalar('Loss/train_kld', train_kld_loss, epoch)
            writer.add_scalar('Loss/train_l2', train_l2_loss, epoch)
            
            if val_dataset:
                vae.eval()
                val_loss = 0
                val_recon_loss = 0
                val_kld_loss = 0
                val_l2_loss = 0

                with torch.no_grad():
                    for batch in val_loader:
                        if isinstance(batch, list):
                            x = batch[0].to(device)
                        else:
                            x = batch.to(device)
                        x = normalize_data(x)
                        recon_batch, mu, logvar = vae(x)
                        loss, recon_loss, kld_loss, weighted_kld_loss, l2_recon_loss = vae_loss(
                            recon_batch, x, mu, logvar, loss_type=config.loss_type, kld_weight=config.kld_weight
                        )

                        val_loss += loss.item()
                        val_recon_loss += recon_loss.item()
                        val_kld_loss += kld_loss.item()
                        val_l2_loss += l2_recon_loss.item()

                if config.normalize_loss:
                    val_loss /= len(val_loader.dataset)
                    val_recon_loss /= len(val_loader.dataset)
                    val_kld_loss /= len(val_loader.dataset)
                    val_l2_loss /= len(val_loader.dataset)
                else:
                    val_loss /= len(val_loader)
                    val_recon_loss /= len(val_loader)
                    val_kld_loss /= len(val_loader)
                    val_l2_loss /= len(val_loader)

                val_losses.append(val_loss)

                # Log normalized and unweighted losses to TensorBoard for validation
                writer.add_scalar('Loss/val_total', val_loss, epoch)
                writer.add_scalar('Loss/val_recon', val_recon_loss, epoch)
                writer.add_scalar('Loss/val_kld', val_kld_loss, epoch)
                writer.add_scalar('Loss/val_l2', val_l2_loss, epoch)

                # Print normalized, unweighted losses
                print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f} '
                      f'(Recon: {train_recon_loss:.4f}, '
                      f'KLD: {train_kld_loss:.4f}, '
                      f'L2: {train_l2_loss:.4f}), '
                      f'Val Loss: {val_loss:.4f} '
                      f'(Recon: {val_recon_loss:.4f}, '
                      f'KLD: {val_kld_loss:.4f}, '
                      f'L2: {val_l2_loss:.4f}), '
                      f'LR: {scheduler.get_last_lr()[0]:.6f}')
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    run_info['best_loss'] = best_loss
                    run_info['best_epoch'] = epoch + 1

                if config.early_stop:
                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
            else:
                # Print normalized, unweighted losses (for training only, no validation)
                print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f} '
                      f'(Recon: {train_recon_loss:.4f}, '
                      f'KLD: {train_kld_loss:.4f}, '
                      f'L2: {train_l2_loss:.4f}), '
                      f'LR: {scheduler.get_last_lr()[0]:.6f}')
                
                if train_loss < best_loss:
                    best_loss = train_loss
                    run_info['best_loss'] = best_loss
                    run_info['best_epoch'] = epoch + 1

            run_info['latest_epoch'] = epoch + 1
            
            # Visualization and saving results at intervals
            if (epoch + 1) % config.visualization_steps == 0 and config.visualize:
                vae.eval()
                with torch.no_grad():
                    # Visualize from training set
                    train_sample = next(iter(train_loader))
                    if isinstance(train_sample, list):
                        train_sample = train_sample[0]
                    train_sample = train_sample.to(device)
                    train_sample = normalize_data(train_sample)
                    train_recon, _, _ = vae(train_sample)

                    # Display original and reconstructed images from training set
                    train_original_path = os.path.join(results_folder, f'train_original_epoch_{epoch+1}.png')
                    train_recon_path = os.path.join(results_folder, f'train_reconstructed_epoch_{epoch+1}.png')
                    display_image(train_sample, f"Train Original - Epoch {epoch+1}", train_original_path)
                    display_image(train_recon, f"Train Reconstructed - Epoch {epoch+1}", train_recon_path)

                    # Log training images to TensorBoard
                    train_original_img = Image.open(train_original_path)
                    train_recon_img = Image.open(train_recon_path)
                    writer.add_image('Train/Original', np.array(train_original_img), epoch, dataformats='HWC')
                    writer.add_image('Train/Reconstructed', np.array(train_recon_img), epoch, dataformats='HWC')

                    # Visualize from validation set if available
                    if val_dataset:
                        val_sample = next(iter(val_loader))
                        if isinstance(val_sample, list):
                            val_sample = val_sample[0]
                        val_sample = val_sample.to(device)
                        val_sample = normalize_data(val_sample)
                        val_recon, _, _ = vae(val_sample)

                        # Display original and reconstructed images from validation set
                        val_original_path = os.path.join(results_folder, f'val_original_epoch_{epoch+1}.png')
                        val_recon_path = os.path.join(results_folder, f'val_reconstructed_epoch_{epoch+1}.png')
                        display_image(val_sample, f"Val Original - Epoch {epoch+1}", val_original_path)
                        display_image(val_recon, f"Val Reconstructed - Epoch {epoch+1}", val_recon_path)

                        # Log validation images to TensorBoard
                        val_original_img = Image.open(val_original_path)
                        val_recon_img = Image.open(val_recon_path)
                        writer.add_image('Val/Original', np.array(val_original_img), epoch, dataformats='HWC')
                        writer.add_image('Val/Reconstructed', np.array(val_recon_img), epoch, dataformats='HWC')


            # Save checkpoint based on checkpoint_steps
            if (epoch + 1) % config.checkpoint_steps == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': vae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_loss': best_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_loss': train_loss,
                    'train_recon_loss': train_recon_loss,
                    'train_kld_loss': train_kld_loss,
                    'train_l2_loss': train_l2_loss,
                    'val_loss': val_loss if val_dataset else None,
                    'val_recon_loss': val_recon_loss if val_dataset else None,
                    'val_kld_loss': val_kld_loss if val_dataset else None,
                    'val_l2_loss': val_l2_loss if val_dataset else None,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                
                checkpoint_filename = os.path.join(models_folder, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(checkpoint, checkpoint_filename)
                
                run_info['checkpoints'].append({
                    'epoch': epoch + 1,
                    'filename': get_relative_path(checkpoint_filename),
                    'train_loss': train_loss,
                    'train_recon_loss': train_recon_loss,
                    'train_kld_loss': train_kld_loss,
                    'train_l2_loss': train_l2_loss,
                    'val_loss': val_loss if val_dataset else None,
                    'val_recon_loss': val_recon_loss if val_dataset else None,
                    'val_kld_loss': val_kld_loss if val_dataset else None,
                    'val_l2_loss': val_l2_loss if val_dataset else None,
                    'learning_rate': scheduler.get_last_lr()[0]
                })
            
            update_run_info(run_folder, run_info)
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            run_info['time_per_epoch'] = epoch_duration
    
    except Exception as e:
        error_info = {
            'error_message': str(e),
            'epoch': epoch + 1 if 'epoch' in locals() else start_epoch,
            'batch_idx': batch_idx if 'batch_idx' in locals() else None,
            'sample_idx': batch_idx * config.batch_size if 'batch_idx' in locals() else None
        }
        run_info['error'] = error_info
        print(f"Error occurred: {str(e)}")
    
    finally:
        end_time = time.time()
        run_info['total_time'] = end_time - start_time
        
        # Save learning curves
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            if val_losses:
                plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('VAE Learning Curves')
            plt.legend()
            learning_curve_path = os.path.join(run_folder, 'learning_curve.png')
            plt.savefig(learning_curve_path)
            plt.close()
            print(f"Learning curves saved to {get_relative_path(learning_curve_path)}")
        except Exception as e:
            print(f"Error saving learning curves: {str(e)}")
        
        # Update and save final run info
        run_info['end_time'] = datetime.now().isoformat()
        run_info['final_train_loss'] = train_loss if 'train_loss' in locals() else None
        run_info['final_val_loss'] = val_loss if 'val_loss' in locals() else None
        update_run_info(run_folder, run_info)
        
        # Update all_runs.json and all_finished_runs.json
        try:
            update_all_runs(vae_runs_folder, run_info)
            update_all_finished_runs(vae_runs_folder, run_info)
        except Exception as e:
            print(f"Error updating run information: {str(e)}")
        
        writer.close()
        print(f"Training completed. Run information saved to {get_relative_path(os.path.join(run_folder, 'run_info.json'))}")

if __name__ == "__main__":
    config = get_vae_config()
    print_vae_config(config)
    train_vae(config)
