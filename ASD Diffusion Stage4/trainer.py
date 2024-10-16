import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import ants
import numpy as np
import os
import json
import uuid
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import traceback

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

class Trainer:
    def __init__(self, config, models, dataset, device):
        self.config = config
        self.models = models
        self.device = device
        
        if isinstance(config.dataset.max_samples, int):
            self.dataset = torch.utils.data.Subset(dataset, range(min(config.dataset.max_samples, len(dataset))))
        else:
            self.dataset = dataset
        self.run_id = self.config.get_run_id()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.diffusion_runs_folder = os.path.join(current_dir, 'diffusion_runs')
        self.run_folder = os.path.join(self.diffusion_runs_folder, self.run_id)
        
        self.tensorboard_folder = os.path.join(self.diffusion_runs_folder, 'tensorboard_logs', self.run_id)
        self.checkpoints_folder = os.path.join(self.run_folder, 'checkpoints')
        self.results_visualized_folder = os.path.join(self.run_folder, 'results_visualized')
        
        os.makedirs(self.tensorboard_folder, exist_ok=True)
        os.makedirs(self.checkpoints_folder, exist_ok=True)
        os.makedirs(self.results_visualized_folder, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.tensorboard_folder)
        
        if self.config.trainer.train_only:
            self.train_dataset = dataset
            self.val_dataset = None
        else:
            train_size = int(self.config.trainer.train_split * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.trainer.batch_size, shuffle=True)
        if not self.config.trainer.train_only:
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.config.trainer.batch_size, shuffle=False)
        
        # Set optimizer
        if self.config.trainer.optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.models.parameters()), lr=self.config.trainer.learning_rate)
        elif self.config.trainer.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.models.parameters()), lr=self.config.trainer.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.trainer.optimizer}")
        
        # Set scheduler
        if self.config.trainer.scheduler == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        elif self.config.trainer.scheduler == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.trainer.num_epochs)
        else:
            self.scheduler = None
        
        self.noise_loss_fn = nn.MSELoss()
        self.recon_loss_fn = nn.MSELoss()
        
        self.start_epoch = 0
        if self.config.trainer.resume_from:
            self.load_checkpoint(self.config.trainer.resume_from)

        self.best_loss = float('inf')
        self.run_info = {
            "config": {
                "dataset": self.config.dataset.__dict__,
                "model": self.config.model.__dict__,
                "trainer": self.config.trainer.__dict__
            },
            "start_time": datetime.now().isoformat(),
            "checkpoints": [],
            "best_checkpoint": None,
            "latest_checkpoint": None
        }

        if self.config.trainer.early_stop:
            self.early_stopping = EarlyStopping(
                patience=self.config.trainer.early_stop_patience,
                delta=self.config.trainer.early_stop_delta
            )
        else:
            self.early_stopping = None
    
    def train_step(self, x, text, t):
        self.optimizer.zero_grad()
        noise_pred, target_noise = self.models(x, text, t)
        noise_loss = self.noise_loss_fn(noise_pred, target_noise)
        
        loss = noise_loss
        loss_breakdown = {'noise_loss': noise_loss.item()}
        
        if self.config.trainer.use_reconstruction_loss:
            with torch.no_grad():
                latent, mu, logvar = self.models.vae.encode(x)
            recon = self.models.vae.decode(latent)
            recon_loss = self.recon_loss_fn(recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss += self.config.trainer.reconstruction_loss_weight * recon_loss + self.config.trainer.kl_loss_weight * kl_loss
            loss_breakdown.update({
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item()
            })
        
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_breakdown

    def evaluate(self, loader):
        self.models.eval()
        total_loss = 0
        loss_breakdown = {'noise_loss': 0, 'recon_loss': 0, 'kl_loss': 0}
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, list):
                    x = batch[0].to(self.device)
                    text = batch[1] if len(batch) > 1 else None
                else:
                    x = batch.to(self.device)
                    text = None
                t = torch.randint(0, self.models.num_diffusion_steps, (x.shape[0],), device=self.device)
                noise_pred, target_noise = self.models(x, text, t)
                noise_loss = self.noise_loss_fn(noise_pred, target_noise)
                
                loss = noise_loss
                loss_breakdown['noise_loss'] += noise_loss.item()
                
                if self.config.trainer.use_reconstruction_loss:
                    latent, mu, logvar = self.models.vae.encode(x)
                    recon = self.models.vae.decode(latent)
                    recon_loss = self.recon_loss_fn(recon, x)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    
                    loss += self.config.trainer.reconstruction_loss_weight * recon_loss + self.config.trainer.kl_loss_weight * kl_loss
                    loss_breakdown['recon_loss'] += recon_loss.item()
                    loss_breakdown['kl_loss'] += kl_loss.item()
                
                total_loss += loss.item()
        
        num_batches = len(loader)
        return total_loss / num_batches, {k: v / num_batches for k, v in loss_breakdown.items()}

    def save_learning_curve(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        if val_losses:
            plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Diffusion Model Learning Curves')
        plt.legend()
        learning_curve_path = os.path.join(self.run_folder, 'learning_curve.png')
        plt.savefig(learning_curve_path)
        plt.close()
        return learning_curve_path

    def save_checkpoint(self, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.models.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics
        }
        checkpoint_path = os.path.join(self.checkpoints_folder, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        checkpoint_info = {
            'epoch': epoch,
            'loss': metrics['val_loss'] if 'val_loss' in metrics else metrics['train_loss'],
            'path': os.path.relpath(checkpoint_path, self.diffusion_runs_folder),
            'metrics': metrics
        }
        self.run_info["checkpoints"].append(checkpoint_info)
        
        if checkpoint_info['loss'] < self.best_loss:
            self.best_loss = checkpoint_info['loss']
            self.run_info["best_checkpoint"] = checkpoint_info
        
        self.run_info["latest_checkpoint"] = checkpoint_info

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.models.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        
        # Set the learning rate from the checkpoint
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = checkpoint['metrics']['learning_rate']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}. "
              f"Learning rate set to {checkpoint['metrics']['learning_rate']:.6f}")
        
        # Print loss breakdown
        train_loss_breakdown = checkpoint['metrics']['train_loss_breakdown']
        print(f"Train Loss Breakdown - Noise: {train_loss_breakdown['noise_loss']:.4f}, "
              f"Recon: {train_loss_breakdown['recon_loss']:.4f}, "
              f"KL: {train_loss_breakdown['kl_loss']:.4f}")
        
        if 'val_loss_breakdown' in checkpoint['metrics']:
            val_loss_breakdown = checkpoint['metrics']['val_loss_breakdown']
            print(f"Val Loss Breakdown - Noise: {val_loss_breakdown['noise_loss']:.4f}, "
                  f"Recon: {val_loss_breakdown['recon_loss']:.4f}, "
                  f"KL: {val_loss_breakdown['kl_loss']:.4f}")

    def update_json_files(self):
        # Update run_info.json
        with open(os.path.join(self.run_folder, 'run_info.json'), 'w') as f:
            json.dump(self.run_info, f, indent=4)

        # Update all_runs.json
        all_runs_path = os.path.join(self.diffusion_runs_folder, 'all_runs.json')
        if os.path.exists(all_runs_path):
            with open(all_runs_path, 'r') as f:
                all_runs = json.load(f)
        else:
            all_runs = {}
        
        all_runs[self.run_id] = {
            "config": self.run_info["config"],
            "start_time": self.run_info["start_time"],
            "best_checkpoint": self.run_info["best_checkpoint"],
            "latest_checkpoint": self.run_info["latest_checkpoint"]
        }
        
        with open(all_runs_path, 'w') as f:
            json.dump(all_runs, f, indent=4)

        # Update finished_runs.json if training is complete
        if self.run_info.get("end_time"):
            finished_runs_path = os.path.join(self.diffusion_runs_folder, 'finished_runs.json')
            if os.path.exists(finished_runs_path):
                with open(finished_runs_path, 'r') as f:
                    finished_runs = json.load(f)
            else:
                finished_runs = {}
            
            finished_runs[self.run_id] = all_runs[self.run_id]
            finished_runs[self.run_id].update({
                "end_time": self.run_info["end_time"],
                "total_time": self.run_info["total_time"],
                "final_train_loss": self.run_info["final_train_loss"],
                "final_val_loss": self.run_info.get("final_val_loss"),
                "learning_curve_path": self.run_info["learning_curve_path"],
                "device_info": self.run_info["device_info"]
            })
            
            with open(finished_runs_path, 'w') as f:
                json.dump(finished_runs, f, indent=4)

    def visualize(self, epoch):
        self.models.eval()
        with torch.no_grad():
            # Visualize training sample
            train_batch = next(iter(self.train_loader))
            if isinstance(train_batch, list):
                train_image = train_batch[0].to(self.device)
                train_text = train_batch[1] if len(train_batch) > 1 else None
            else:
                train_image = train_batch.to(self.device)
                train_text = None
            self.display_image(train_image[0], "Original Train Image", epoch)
            
            train_latent, _, _ = self.models.vae.encode(train_image)
            train_reconstructed = self.models.vae.decode(train_latent)
            self.display_image(train_reconstructed[0], "Reconstructed Train Image", epoch)

            if not self.config.trainer.train_only:
                # Visualize validation sample
                val_batch = next(iter(self.val_loader))
                if isinstance(val_batch, list):
                    val_image = val_batch[0].to(self.device)
                    val_text = val_batch[1] if len(val_batch) > 1 else None
                else:
                    val_image = val_batch.to(self.device)
                    val_text = None

                self.display_image(val_image[0], "Original Val Image", epoch, is_validation=True)
                
                val_latent, _, _ = self.models.vae.encode(val_image)
                val_reconstructed = self.models.vae.decode(val_latent)
                self.display_image(val_reconstructed[0], "Reconstructed Val Image", epoch, is_validation=True)

    def display_image(self, image_tensor, title, epoch, is_validation=False):
        image_np = image_tensor.detach().cpu().numpy()
        
        if image_np.ndim == 4:
            image_np = image_np[0]
        
        if image_np.shape != (64, 64, 64):
            raise ValueError(f"Expected image shape (64, 64, 64), but got {image_np.shape} for {title}")
        
        image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-8)
        
        ants_image = ants.from_numpy(image_np)
        
        save_path = os.path.join(self.results_visualized_folder, f'{title.lower().replace(" ", "_")}_epoch_{epoch}.png')
        
        ants_image.plot_ortho(flat=True, title=title, filename=save_path, xyz_lines=False, orient_labels=False)
        
        self.writer.add_image(f'{"Test" if is_validation else "Train"}/{title}', plt.imread(save_path), epoch, dataformats='HWC')

    def train(self):
        train_losses = []
        val_losses = []
        start_time = datetime.now()

        try:
            for epoch in range(self.start_epoch, self.config.trainer.num_epochs):
                epoch_start_time = datetime.now()
                self.models.train()
                train_loss = 0
                train_loss_breakdown = {'noise_loss': 0, 'recon_loss': 0, 'kl_loss': 0}
                for batch_idx, batch in enumerate(self.train_loader):
                    if isinstance(batch, list):
                        x = batch[0].to(self.device)
                        text = batch[1] if len(batch) > 1 else None
                    else:
                        x = batch.to(self.device)
                        text = None
                    t = torch.randint(0, self.models.num_diffusion_steps, (x.shape[0],), device=self.device)
                    loss, loss_breakdown = self.train_step(x, text, t)
                    train_loss += loss
                    for key in loss_breakdown:
                        train_loss_breakdown[key] += loss_breakdown[key]
                
                train_loss /= len(self.train_loader)
                train_losses.append(train_loss)
                for key in train_loss_breakdown:
                    train_loss_breakdown[key] /= len(self.train_loader)
                    self.writer.add_scalar(f'Loss/train_{key}', train_loss_breakdown[key], epoch)

                self.writer.add_scalar('Loss/train', train_loss, epoch)
                
                metrics = {
                    "train_loss": train_loss,
                    "train_loss_breakdown": {
                        "noise_loss": train_loss_breakdown['noise_loss'],
                        "recon_loss": train_loss_breakdown['recon_loss'],
                        "kl_loss": train_loss_breakdown['kl_loss']
                    },
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                }
                
                if not self.config.trainer.train_only:
                    val_loss, val_loss_breakdown = self.evaluate(self.val_loader)
                    val_losses.append(val_loss)
                    self.writer.add_scalar('Loss/val', val_loss, epoch)
                    for key in val_loss_breakdown:
                        self.writer.add_scalar(f'Loss/val_{key}', val_loss_breakdown[key], epoch)
                    
                    metrics.update({
                        "val_loss": val_loss,
                        "val_loss_breakdown": {
                            "noise_loss": val_loss_breakdown['noise_loss'],
                            "recon_loss": val_loss_breakdown['recon_loss'],
                            "kl_loss": val_loss_breakdown['kl_loss']
                        }
                    })
                    
                    print(f"Epoch {epoch+1}/{self.config.trainer.num_epochs}")
                    print(f"Train Loss: {train_loss:.4f} (Noise: {train_loss_breakdown['noise_loss']:.4f}, "
                        f"Recon: {train_loss_breakdown['recon_loss']:.4f}, KL: {train_loss_breakdown['kl_loss']:.4f})")
                    print(f"Val Loss: {val_loss:.4f} (Noise: {val_loss_breakdown['noise_loss']:.4f}, "
                        f"Recon: {val_loss_breakdown['recon_loss']:.4f}, KL: {val_loss_breakdown['kl_loss']:.4f})")
                    print(f"LR: {metrics['learning_rate']:.6f}")
                    
                    if self.scheduler:
                        self.scheduler.step(val_loss)
                    
                    # Early stopping check
                    if self.early_stopping:
                        self.early_stopping(val_loss)
                        if self.early_stopping.early_stop:
                            print("Early stopping triggered")
                            break
                else:
                    print(f"Epoch {epoch+1}/{self.config.trainer.num_epochs}")
                    print(f"Train Loss: {train_loss:.4f} (Noise: {train_loss_breakdown['noise_loss']:.4f}, "
                        f"Recon: {train_loss_breakdown['recon_loss']:.4f}, KL: {train_loss_breakdown['kl_loss']:.4f})")
                    print(f"LR: {metrics['learning_rate']:.6f}")
                    
                    if self.scheduler:
                        self.scheduler.step(train_loss)
                
                if epoch % self.config.trainer.visualization_steps == 0 and self.config.trainer.visualize:
                    self.visualize(epoch)

                if epoch  % self.config.trainer.checkpoint_steps == 0:
                    self.save_checkpoint(epoch, metrics)
                    self.update_json_files()

                epoch_end_time = datetime.now()
                epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
                print(f"Epoch duration: {epoch_duration:.2f} seconds")

        except Exception as e:
            end_time = datetime.now()
            error_msg = f"Error occurred during training: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.run_info.update({
                "error": error_msg,
                "error_epoch": epoch,
                "error_batch": batch_idx if 'batch_idx' in locals() else None,
                "end_time": end_time.isoformat(),
                "total_time": (end_time - start_time).total_seconds(),
            })
            self.update_json_files()
            self.save_learning_curve(train_losses, val_losses)
        
        else:
            end_time = datetime.now()
            print("Training complete!")
            
            final_model_path = os.path.join(self.checkpoints_folder, f'final_model.pth')
            torch.save(self.models.state_dict(), final_model_path)
            print(f"Final model saved to {final_model_path}")
            
            learning_curve_path = self.save_learning_curve(train_losses, val_losses)
            
            self.run_info.update({
                "end_time": end_time.isoformat(),
                "total_time": (end_time - start_time).total_seconds(),
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1] if val_losses else None,
                "learning_curve_path": os.path.relpath(learning_curve_path, self.diffusion_runs_folder),
                "device_info": {
                    "device": str(self.device),
                    "cuda_available": torch.cuda.is_available(),
                    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                }
            })
            
            self.update_json_files()
            
            print(f"Run information updated in {os.path.join(self.diffusion_runs_folder, 'all_runs.json')} and {os.path.join(self.diffusion_runs_folder, 'finished_runs.json')}")
        
        finally:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()
        if exc_type is not None:
            print(f"An error occurred: {exc_type.__name__}: {exc_val}")
            print(f"Traceback:\n{''.join(traceback.format_tb(exc_tb))}")
        return False
