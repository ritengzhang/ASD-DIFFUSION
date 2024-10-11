# from .mri_dataset import MRIDataset
# from .mri_text_dataset import MRIWithTextDataset
from .toy_dataset_processed import MRIToyDataset  # MRI image only

def get_dataset(config):
    if config.name == "MRIDataset":
        return #MRIDataset(num_samples=config.num_samples, image_size=config.image_size)
    elif config.name == "MRIWithTextDataset":
        return #MRIWithTextDataset(num_samples=config.num_samples, image_size=config.image_size, text_length=config.text_length)
    elif config.name == "MRIToyDataset":
        # Load the dataset using the provided or default path to the existing dataset
        return MRIToyDataset()#load existing
    else:
        raise ValueError(f"Unknown dataset: {config.name}")
