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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

class Trainer:
    def __init__(self, config, models, dataset, device):
        self.config = config
        self.models = models
        self.device = device
        self.dataset = dataset
        
        self.run_id = str(uuid.uuid4())[:8]
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.diffusion_runs_folder = os.path.join(current_dir, 'diffusion_runs')
        os.makedirs(self.diffusion_runs_folder, exist_ok=True)
        
        self.reconstructions_folder = os.path.join(self.diffusion_runs_folder, 'reconstructions_visualized', f'run_{self.run_id}')
        self.models_folder = os.path.join(self.diffusion_runs_folder, 'models')
        self.learning_curves_folder = os.path.join(self.diffusion_runs_folder, 'learning_curves')
        os.makedirs(self.reconstructions_folder, exist_ok=True)
        os.makedirs(self.models_folder, exist_ok=True)
        os.makedirs(self.learning_curves_folder, exist_ok=True)
        
        if self.config.train_only:
            self.train_dataset = dataset
            self.test_dataset = None
        else:
            train_size = int(self.config.train_split * len(dataset))
            test_size = len(dataset) - train_size
            self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        if not self.config.train_only:
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Set optimizer
        if self.config.optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.models.parameters()), lr=self.config.learning_rate)
        elif self.config.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.models.parameters()), lr=self.config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        # Set scheduler
        if self.config.scheduler == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        elif self.config.scheduler == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs)
        else:
            self.scheduler = None
        
        self.loss_fn = nn.MSELoss()

    def train_step(self, x, text, t):
        self.optimizer.zero_grad()
        noise_pred, target_noise = self.models(x, text, t)
        loss = self.loss_fn(noise_pred, target_noise)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, loader):
        self.models.eval()
        total_loss = 0
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
                loss = self.loss_fn(noise_pred, target_noise)
                total_loss += loss.item()
        return total_loss / len(loader)

    def display_image(self, image, title, epoch):
        image_np = image.detach().cpu().numpy()
        
        if image_np.ndim == 2:
            image_np = image_np[..., np.newaxis]
        elif image_np.ndim == 4:
            image_np = image_np[0]
        
        if image_np.shape[0] == 1 or image_np.shape[0] == 3:
            image_np = np.moveaxis(image_np, 0, -1)
        
        if image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)
        
        print(f"{title} shape: {image_np.shape}")
        
        ants_image = ants.from_numpy(image_np)
        
        save_path = os.path.join(self.reconstructions_folder, f'{title.lower().replace(" ", "_")}_epoch_{epoch}.png')
        ants_image.plot_ortho(flat=True, title=title, filename=save_path, xyz_lines=False, orient_labels=False)
        print(f"Image saved to {save_path}")

    def update_metadata(self, run_data):
        metadata_path = os.path.join(self.diffusion_runs_folder, 'diffusion_runs_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        metadata[self.run_id] = run_data
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def save_learning_curves(self, train_losses, test_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        if test_losses:
            plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Diffusion Model Learning Curves')
        plt.legend()
        learning_curve_path = os.path.join(self.learning_curves_folder, f'learning_curve_{self.run_id}.png')
        plt.savefig(learning_curve_path)
        plt.close()
        print(f"Learning curves saved to {learning_curve_path}")
        return learning_curve_path

    def train(self):
        train_losses = []
        test_losses = []
        
        for epoch in range(self.config.num_epochs):
            self.models.train()
            train_loss = 0
            for batch in self.train_loader:
                if isinstance(batch, list):
                    x = batch[0].to(self.device)
                    text = batch[1] if len(batch) > 1 else None
                else:
                    x = batch.to(self.device)
                    text = None
                t = torch.randint(0, self.models.num_diffusion_steps, (x.shape[0],), device=self.device)
                loss = self.train_step(x, text, t)
                train_loss += loss
            
            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)
            
            if not self.config.train_only:
                test_loss = self.evaluate(self.test_loader)
                test_losses.append(test_loss)
                print(f"Epoch {epoch+1}/{self.config.num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                if self.scheduler:
                    self.scheduler.step(test_loss)
            else:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}, Train Loss: {train_loss:.4f}")
                if self.scheduler:
                    self.scheduler.step()
            
            if (epoch + 1) % 10 == 0 and self.config.visualize:
                sample_batch = next(iter(self.train_loader))
                if isinstance(sample_batch, list):
                    sample_image = sample_batch[0].to(self.device)
                    sample_text = sample_batch[1] if len(sample_batch) > 1 else None
                else:
                    sample_image = sample_batch.to(self.device)
                    sample_text = None

                self.display_image(sample_image[0], "Original Image", epoch + 1)

                with torch.no_grad():
                    latent, _, _ = self.models.vae.encode(sample_image)
                    reconstructed = self.models.vae.decode(latent)
                    reconstructed_image = reconstructed[0]

                self.display_image(reconstructed_image, "Reconstructed Image", epoch + 1)

        print("Training complete!")
        
        model_path = os.path.join(self.models_folder, f'diffusion_model_{self.run_id}.pth')
        torch.save(self.models.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        learning_curve_path = self.save_learning_curves(train_losses, test_losses)
        
        run_data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1] if test_losses else None,
            'model_path': model_path,
            'reconstructions_folder': self.reconstructions_folder,
            'learning_curve_path': learning_curve_path
        }
        
        self.update_metadata(run_data)
        print(f"Metadata updated in {os.path.join(self.diffusion_runs_folder, 'diffusion_runs_metadata.json')}")