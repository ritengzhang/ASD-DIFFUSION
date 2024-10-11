from dataclasses import dataclass, field

@dataclass
class DatasetConfig:
    name: str = "MRIToyDataset"

@dataclass
class VAEConfig:
    batch_size: int = 4
    num_epochs: int = 50000
    learning_rate: float = 1e-7
    train_split: float = 0.8
    visualize: bool = True
    train_only: bool = False
    latent_channels: int = 16
    vae_path: str = "vae_weights.pth"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    scheduler_T0: int = 10
    scheduler_T_mult: int = 2
    scheduler_eta_min: float = 1e-6

def get_vae_config():
    return VAEConfig()

def print_vae_config(config):
    print("VAE Configuration:")
    for key, value in config.__dict__.items():
        if key == 'dataset':
            print(f"{key}:")
            for k, v in value.__dict__.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")