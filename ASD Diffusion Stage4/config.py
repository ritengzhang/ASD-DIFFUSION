import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import Union, List, Optional

@dataclass
class DatasetConfig:
    name: str = "MRIToyDataset"
    max_samples: Union[int, str] = "max"
    image_size: int = 64
    text_length: int = 10

    def to_dict(self):
        return asdict(self)

@dataclass
class ModelConfig:
    run_name_prefix: str = ""
    run_name_suffix: str = ""
    vae_type: str = "EnhancedVAE"
    unet_type: str = "UNet"
    text_encoder_type: str = "SimpleTextEncoder"
    cross_attention_type: str = "CrossAttention"
    latent_channels: int = 256
    image_size: int = 64
    use_text_conditioning: bool = False
    num_diffusion_steps: int = 1000
    min_beta: float = 0.0001
    max_beta: float = 0.02
    vae_run_id: Optional[str] = None
    train_modules: List[bool] = field(default_factory=lambda: [True, True, True, True])
    vae_in_channels: int = 1
    vae_out_channels: int = 1
    attention_note: str = "NoAttention"

    def __post_init__(self):
        if self.vae_run_id:
            self.vae_path = self.find_latest_vae_checkpoint(self.vae_run_id)
        else:
            self.vae_path = None

    @staticmethod
    def find_latest_vae_checkpoint(run_id):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vae_runs_folder = os.path.join(current_dir, 'EncDec', 'vae_runs')
        run_folder = os.path.join(vae_runs_folder, run_id)
        checkpoints_folder = os.path.join(run_folder, 'checkpoints')
        
        if not os.path.exists(checkpoints_folder):
            raise FileNotFoundError(f"Checkpoints folder not found for VAE run {run_id}. Path: {checkpoints_folder}")
        
        checkpoints = [f for f in os.listdir(checkpoints_folder) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found for VAE run {run_id} in {checkpoints_folder}")
        
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return os.path.join('EncDec', 'vae_runs', run_id, 'checkpoints', latest_checkpoint)

@dataclass
class TrainerConfig:
    batch_size: int = 16
    num_epochs: int = 1000
    learning_rate: float = 1e-4
    train_split: float = 0.8
    visualize: bool = True
    train_only: bool = True
    optimizer: str = "Adam"
    scheduler: str = "ReduceLROnPlateau"
    use_reconstruction_loss: bool = True
    reconstruction_loss_weight: float = 1.0
    kl_loss_weight: float = 1.0
    visualization_steps: int = 100
    checkpoint_steps: int = 100
    resume_from: Optional[str] = None
    early_stop: bool = False
    early_stop_patience: int = 100
    early_stop_delta: float = 0.001


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    trainer: TrainerConfig

    def get_run_id(self):
        components = []
        if self.model.run_name_prefix:
            components.append(self.model.run_name_prefix)
        components.extend([
            self.dataset.name,
            self.model.vae_type,
            f"{self.model.latent_channels}ch",
            self.model.attention_note
        ])
        if self.model.run_name_suffix:
            components.append(self.model.run_name_suffix)
        base_name = "_".join(components)
        unique_id = str(uuid.uuid4())[:8]
        return f"{base_name}_{unique_id}"

def get_config(vae_run_id: Optional[str] = "MRIToyDataset_EnhancedVAE_0a89a8df_modified"):
    return Config(
        dataset=DatasetConfig(),
        model=ModelConfig(vae_run_id=vae_run_id),
        trainer=TrainerConfig()
    )

def print_config(config):
    print("Configuration:")
    print(f"Run ID: {config.get_run_id()}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Max samples: {config.dataset.max_samples}")
    print(f"Image size: {config.dataset.image_size}")
    print(f"Text length: {config.dataset.text_length}")
    print(f"VAE type: {config.model.vae_type}")
    print(f"UNet type: {config.model.unet_type}")
    print(f"Text encoder type: {config.model.text_encoder_type}")
    print(f"Cross attention type: {config.model.cross_attention_type}")
    print(f"Latent channels: {config.model.latent_channels}")
    print(f"VAE input channels: {config.model.vae_in_channels}")
    print(f"VAE output channels: {config.model.vae_out_channels}")
    print(f"Use text conditioning: {config.model.use_text_conditioning}")
    print(f"Number of diffusion steps: {config.model.num_diffusion_steps}")
    print(f"Beta range: [{config.model.min_beta}, {config.model.max_beta}]")
    print(f"VAE run ID: {config.model.vae_run_id}")
    print(f"VAE path: {config.model.vae_path}")
    print(f"Train modules: {config.model.train_modules}")
    print(f"Run name prefix: {config.model.run_name_prefix}")
    print(f"Run name suffix: {config.model.run_name_suffix}")
    print(f"Attention note: {config.model.attention_note}")
    print(f"Batch size: {config.trainer.batch_size}")
    print(f"Number of epochs: {config.trainer.num_epochs}")
    print(f"Learning rate: {config.trainer.learning_rate}")
    print(f"Train split: {config.trainer.train_split}")
    print(f"Visualize: {config.trainer.visualize}")
    print(f"Train only: {config.trainer.train_only}")
    print(f"Optimizer: {config.trainer.optimizer}")
    print(f"Scheduler: {config.trainer.scheduler}")
    print(f"Use reconstruction loss: {config.trainer.use_reconstruction_loss}")
    print(f"Reconstruction loss weight: {config.trainer.reconstruction_loss_weight}")
    print(f"KL loss weight: {config.trainer.kl_loss_weight}")
    print(f"Visualization steps: {config.trainer.visualization_steps}")
    print(f"Checkpoint steps: {config.trainer.checkpoint_steps}")
    print(f"Resume from: {config.trainer.resume_from}")
