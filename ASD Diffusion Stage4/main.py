from config import get_config
from models import get_models
from trainer import Trainer
from datasets import get_dataset
import torch

def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = get_dataset(config.dataset)
    
    # Get models
    models = get_models(config.model, device)
    
    # Initialize trainer
    trainer = Trainer(config.trainer, models, dataset, device)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()