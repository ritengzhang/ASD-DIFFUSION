from dataclasses import dataclass, field
import os
import json
from typing import List, Optional

@dataclass
class DatasetConfig:
    name: str = "MRIToyDataset"
    num_samples: int = 1000
    image_size: int = 64
    text_length: int = 10

@dataclass
class ModelConfig:
    vae_type: str = "SimpleVAE"
    unet_type: str = "UNet"
    text_encoder_type: str = "SimpleTextEncoder"
    cross_attention_type: str = "CrossAttention"
    latent_channels: int = 16
    image_size: int = 64
    use_text_conditioning: bool = False
    num_diffusion_steps: int = 1000
    min_beta: float = 0.0001
    max_beta: float = 0.02
    vae_run_id: str = None
    vae_path: str = field(init=False)
    train_modules: List[bool] = field(default_factory=lambda: [True, True, True, True])

    def __post_init__(self):
        self.vae_path = find_vae_model_by_id(self.vae_run_id)

@dataclass
class TrainerConfig:
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    train_split: float = 0.8
    visualize: bool = True
    train_only: bool = False
    optimizer: str = "Adam"
    scheduler: str = "ReduceLROnPlateau"

@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    trainer: TrainerConfig

def find_vae_model_by_id(run_id):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vae_runs_folder = os.path.join(current_dir, 'EncDec', 'vae_runs')
    metadata_path = os.path.join(vae_runs_folder, 'vae_runs_metadata.json')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"VAE metadata file not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)
    
    for metadata in all_metadata:
        if metadata.get('run_id') == run_id:
            vae_path = metadata.get('model_path')
            if not vae_path or not os.path.exists(vae_path):
                raise FileNotFoundError(f"VAE model for run {run_id} not found at {vae_path}")
            return vae_path
    
    raise ValueError(f"No VAE run found with ID {run_id}")

def find_latest_vae_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vae_runs_folder = os.path.join(current_dir, 'EncDec', 'vae_runs')
    metadata_path = os.path.join(vae_runs_folder, 'vae_runs_metadata.json')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"VAE metadata file not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)
    
    if not all_metadata:
        raise ValueError("No VAE runs found in the metadata file")
    
    latest_run = sorted(all_metadata, key=lambda x: x['timestamp'], reverse=True)[0]
    
    vae_path = latest_run.get('model_path')
    if not vae_path or not os.path.exists(vae_path):
        raise FileNotFoundError(f"Latest VAE model not found at {vae_path}")
    
    return vae_path

def get_config(vae_run_id: Optional[str] = None):
    return Config(
        dataset=DatasetConfig(),
        model=ModelConfig(vae_run_id=vae_run_id or "43f41f2d"),
        trainer=TrainerConfig()
    )

def print_config(config):
    print("Configuration:")
    print(f"Dataset: {config.dataset.name}")
    print(f"Number of samples: {config.dataset.num_samples}")
    print(f"Image size: {config.dataset.image_size}")
    print(f"Text length: {config.dataset.text_length}")
    print(f"VAE type: {config.model.vae_type}")
    print(f"UNet type: {config.model.unet_type}")
    print(f"Text encoder type: {config.model.text_encoder_type}")
    print(f"Cross attention type: {config.model.cross_attention_type}")
    print(f"Latent channels: {config.model.latent_channels}")
    print(f"Use text conditioning: {config.model.use_text_conditioning}")
    print(f"Number of diffusion steps: {config.model.num_diffusion_steps}")
    print(f"Beta range: [{config.model.min_beta}, {config.model.max_beta}]")
    print(f"VAE run ID: {config.model.vae_run_id}")
    print(f"VAE weights path: {config.model.vae_path}")
    print(f"Train modules: {config.model.train_modules}")
    print(f"Batch size: {config.trainer.batch_size}")
    print(f"Number of epochs: {config.trainer.num_epochs}")
    print(f"Learning rate: {config.trainer.learning_rate}")
    print(f"Train split: {config.trainer.train_split}")
    print(f"Visualize: {config.trainer.visualize}")
    print(f"Train only: {config.trainer.train_only}")
    print(f"Optimizer: {config.trainer.optimizer}")
    print(f"Scheduler: {config.trainer.scheduler}")