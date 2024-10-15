import torch
from config import get_config, print_config
from datasets import get_dataset
from models import get_models
from trainer import Trainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = get_config()
    print_config(config)
    
    dataset = get_dataset(config.dataset)
    models = get_models(config.model, device)

    with Trainer(config, models, dataset, device) as trainer:
        trainer.train()

if __name__ == "__main__":
    main()
