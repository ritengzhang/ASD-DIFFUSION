import random
from dataclasses import dataclass, field, asdict
from typing import Union, Type
from vae import SimpleVAE, EnhancedVAE, PowerfulVAE  # Import VAE classes

@dataclass
class DatasetConfig:
    name: str = "MRIToyDataset"
    max_samples: Union[int, str] = "max"

    def to_dict(self):
        return asdict(self)

@dataclass
class VAEConfig:
    run_name_prefix: str = ""
    run_name_suffix: str = ""
    batch_size: int = 16
    num_epochs: int = 10000
    learning_rate: float = 1e-4
    train_split: float = 0.8
    visualize: bool = True
    train_only: bool = False
    latent_channels: int = 256
    vae_path: str = "vae_weights.pth"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    scheduler_T0: int = 10
    scheduler_T_mult: int = 2
    scheduler_eta_min: float = 1e-7
    loss_type: str = "mse"
    kld_weight: float = 0.01
    l2_weight: float = 1
    visualization_steps: int = 1000
    checkpoint_steps: int = 1000
    random_seed: int = 42
    resume_from: Union[str, None] = None
    normalize_loss: bool = True
    early_stop: bool = False
    early_stop_patience: int = 1000
    early_stop_delta: float = 0.001
    vae_class: Type[Union[SimpleVAE, EnhancedVAE, PowerfulVAE]] = EnhancedVAE  # Use the actual class

    def to_dict(self):
        config_dict = asdict(self)
        config_dict['dataset'] = self.dataset.to_dict()
        config_dict['vae_class'] = self.vae_class.__name__  # Convert class to string for serialization
        return config_dict

def get_vae_config():
    config = VAEConfig()
    random.seed(config.random_seed)
    return config

def print_vae_config(config):
    print("VAE Configuration:")
    for key, value in config.to_dict().items():
        if key == 'dataset':
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        elif key == 'vae_class':
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")
