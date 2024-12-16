import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import random_split
import warnings
import sys
import os

warnings.filterwarnings("ignore", category=FutureWarning)

# Add the parent directory (assuming `my_datasets` is in the same directory as `visualization.py`)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])
from my_datasets.toy_dataset_processed import MRIToyDataset
from EncDec.vae import SimpleVAE, EnhancedVAE, FlattenedVAE, EncDiff, create_encdiff_model, vae_loss, DeepVAE

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MRIToyDataset()
# convert to torch tensor
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Extract data and indices from the train and validation datasets
train_data = torch.stack([item[0] for item in train_dataset]).to(device)
val_data = torch.stack([item[0] for item in val_dataset]).to(device)
dataset = torch.from_numpy(dataset.data).float().to(device)
    
def extract_latent_features(dataset, device, model_type='EnhancedVAE', checkpoint_path=None, 
                           latent_channels=256):
    """
    Extract latent features for each sample in the dataset using the VAE encoder
    
    Args:
        model: Trained VAE model
        dataset: Dataset tensor of shape [n_samples, 1, 64, 64, 64]
    
    Returns:
        latent_features: Numpy array of shape [n_samples, latent_dim]
    """

    if model_type == 'SimpleVAE':
        model = SimpleVAE(latent_channels=latent_channels)
    elif model_type == 'EnhancedVAE':
        model = EnhancedVAE(latent_channels=latent_channels)
    elif model_type == 'FlattenedVAE':
        model = FlattenedVAE(latent_channels=latent_channels)
    elif model_type == 'DeepVAE':
        model = DeepVAE(latent_channels=latent_channels)
    elif model_type == 'EncDiff':
        model = create_encdiff_model(in_channels=1, out_channels=1)
    else:
        raise ValueError("Invalid model_type. Choose either 'SimpleVAE' or 'EnhancedVAE'.")
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # print(checkpoint.keys())
        if model_type == 'EncDiff':
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    latent_features = []
    print(dataset.shape)
    
    with torch.no_grad():
        for i in range(dataset.shape[0]):
            # Extract single sample and add batch dimension
            sample = dataset[i:i+1]
            
            # Get latent representation (assuming your model has an encode method)
            latent, _ = model.visualize(sample) #latent shape (1, channel size, feature_dim)
            
            # Convert to numpy and flatten if needed
            latent_np = latent.cpu().numpy()
            latent_features.append(latent_np.reshape(1, -1))
    
    return np.vstack(latent_features)

def compute_rsa_matrix(features, metric='euclidean'):
    """
    Compute RSA similarity matrix using specified distance metric
    
    Args:
        features: Numpy array of shape [n_samples, feature_dim]
        metric: Distance metric to use (default: 'euclidean')
    
    Returns:
        similarity_matrix: Numpy array of shape [n_samples, n_samples]
    """
    distances = pdist(features, metric=metric)
    print(distances.max(), distances.min())
    print(distances[:20])
    # Convert distance matrix to square form
    max_distance = distances.max()
    normalized_distances = distances / max_distance
    print("Normalized distances:", normalized_distances[:20])
    similarity_matrix = 1 - squareform(normalized_distances)

    
    return similarity_matrix

def plot_similarity_matrix(similarity_matrix, title="RSA Similarity Matrix", output_dir=None):
    """
    Plot the similarity matrix as a heatmap
    
    Args:
        similarity_matrix: Square numpy array
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
                cmap='viridis_r',
                xticklabels=5, 
                yticklabels=5,
                vmin=0,
                vmax=1)
    plt.title(title)

    file_name = f"{title}.png"
    file_path = os.path.join(output_dir, file_name)

    plt.savefig(file_path, format='png', dpi=300)
    print(f"Plot saved as {file_path}")
    plt.show()

# Main execution
def run_rsa_analysis(dataset, device, model_type, checkpoint_path, latent_channels):
    """
    Run complete RSA analysis pipeline
    
    Args:
        model: Trained VAE model
        dataset: Dataset tensor
    
    Returns:
        latent_features: Extracted features
        similarity_matrix: Computed similarity matrix
    """
    # get model

    # Extract latent features
    print("Extracting latent features...")
    latent_features = extract_latent_features(dataset, device, model_type=model_type, checkpoint_path=checkpoint_path, 
                           latent_channels=latent_channels)
    
    # Compute RSA matrix
    print("Computing similarity matrix...")
    similarity_matrix = compute_rsa_matrix(latent_features)
    
    return latent_features, similarity_matrix

# Usage example
if __name__ == "__main__":
    latent_features, similarity_matrix = run_rsa_analysis(dataset, device,
                        model_type='DeepVAE', 
                        checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
                        latent_channels=16)
    # Plot results
    print("Plotting similarity matrix...")
    plot_similarity_matrix(similarity_matrix, title="D16 Euc Similarity Matrix", output_dir='../Visualization')

    # Run RSA analysis for training data
    latent_features_train, similarity_matrix_train = run_rsa_analysis(train_data, device,
                        model_type='DeepVAE', 
                        checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
                        latent_channels=16)
    # Plot results for training data
    print("Plotting similarity matrix for training data...")
    plot_similarity_matrix(similarity_matrix_train, title="D16_train_Euc_Similarity_Matrix", output_dir='../Visualization')

    # Run RSA analysis for validation data
    latent_features_val, similarity_matrix_val = run_rsa_analysis(val_data, device,
                        model_type='DeepVAE', 
                        checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
                        latent_channels=16)
    # Plot results for validation data
    print("Plotting similarity matrix for validation data...")
    plot_similarity_matrix(similarity_matrix_val, title="D16_val_Euc_Similarity_Matrix", output_dir='../Visualization')

# print(f"Latent features shape: {latent_features.shape}")
# print(f"Similarity matrix shape: {similarity_matrix.shape}")