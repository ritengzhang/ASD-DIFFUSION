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
from EncDec.vae import SimpleVAE, EnhancedVAE, FlattenedVAE, EncDiff, create_encdiff_model, DeepVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MRIToyDataset()
# convert to torch tensor
# dataset = torch.from_numpy(dataset.data).float().to(device)
# print(dataset.data.shape)
# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Extract data
train_data = torch.stack([dataset[i][0] for i in train_dataset.indices]).to(device)
val_data = torch.stack([dataset[i][0] for i in val_dataset.indices]).to(device)

def extract__channel_latent_features(dataset, device, model_type='EnhancedVAE', checkpoint_path=None, 
                           latent_channels=256):
    """
    Extract latent features for each sample in the dataset using the VAE encoder,
    preserving channel-wise information
    
    Args:
        model: Trained VAE model
        dataset: Dataset tensor of shape [n_samples, 1, 64, 64, 64]
        latent_channels: Number of channels in latent space
    
    Returns:
        latent_features: List of numpy arrays, one per channel
                        Each array has shape [n_samples, feature_dim_per_channel]
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
        if model_type == 'EncDiff':
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Initialize list to store features for each channel
    channel_features = [[] for _ in range(latent_channels)]
    
    with torch.no_grad():
        for i in range(dataset.shape[0]):
            sample = dataset[i:i+1]
            _, latent = model.visualize(sample)  # shape: (1, channel_size, feature_dim)
            
            # Separate channels and store
            for ch in range(latent_channels):
                channel_np = latent[ch].cpu().numpy()
                channel_features[ch].append(channel_np.reshape(1, -1))
    
    # Stack features for each channel
    channel_features = [np.vstack(features) for features in channel_features]
    
    return channel_features

def compute_channel_similarity_matrices(channel_features, metric='euclidean'):
    """
    Compute RSA similarity matrix for each channel
    
    Args:
        channel_features: List of numpy arrays, one per channel
        metric: Distance metric to use
    
    Returns:
        similarity_matrices: List of numpy arrays, one per channel
    """
    similarity_matrices = []
    
    for features in channel_features:
        distances = pdist(features, metric=metric)
        max_distance = distances.max()
        normalized_distances = distances / max_distance
        similarity_matrix = 1 - squareform(normalized_distances)
        similarity_matrices.append(similarity_matrix)
    
    return similarity_matrices

# def plot_channel_similarity_matrices(similarity_matrices, output_dir=None, ncols=5):
#     """
#     Plot similarity matrices for each channel in a grid
    
#     Args:
#         similarity_matrices: List of similarity matrices
#         output_dir: Directory to save plots
#         ncols: Number of columns in the subplot grid
#     """
#     n_channels = len(similarity_matrices)
#     nrows = (n_channels + ncols - 1) // ncols
    
#     fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
#     axes = axes.flatten()
    
#     for i, (matrix, ax) in enumerate(zip(similarity_matrices, axes)):
#         sns.heatmap(matrix,
#                     cmap='viridis_r',
#                     xticklabels=5,
#                     yticklabels=5,
#                     vmin=0,
#                     vmax=1,
#                     ax=ax)
#         ax.set_title(f'Channel {i}')
    
#     # Remove empty subplots
#     for j in range(i+1, len(axes)):
#         fig.delaxes(axes[j])
    
#     plt.tight_layout()
    
#     if output_dir:
#         file_path = os.path.join(output_dir, "EncDiff_channel_similarity_matrices.png")
#         plt.savefig(file_path, format='png', dpi=300, bbox_inches='tight')
#         print(f"Plot saved as {file_path}")
    
#     plt.show()

def analyze_channel_similarities(dataset, device, model_type='DeepVAE', checkpoint_path=None, latent_channels=256):

    # Extract channel-wise latent features
    channel_features = extract__channel_latent_features(
        dataset, device,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        latent_channels=latent_channels
    )
    
    # Compute similarity matrices for each channel
    similarity_matrices = compute_channel_similarity_matrices(channel_features)
    
    return channel_features, similarity_matrices

def plot_channel_similarity_matrices(similarity_matrices, split_name, model_type, checkpoint_path=None, output_dir='../Visualization'):
    """
    Plot similarity matrix for each channel
    """
    if checkpoint_path:
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_path = os.path.join(output_dir, f"{model_type}_{checkpoint_name}_{split_name}_similarities")
        os.makedirs(output_path, exist_ok=True)
    
    for ch_idx, similarity_matrix in enumerate(similarity_matrices):
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                    cmap='viridis_r',
                    xticklabels=5, 
                    yticklabels=5,
                    vmin=0,
                    vmax=1)
        plt.title(f"{split_name} Channel {ch_idx} Similarity Matrix")
        
        file_name = f"{model_type}_{checkpoint_name}_{split_name}_channel_{ch_idx}_similarity.png"
        file_path = os.path.join(output_path, file_name)
        plt.savefig(file_path, format='png', dpi=300)
        print(f"Channel {ch_idx} plot saved as {file_path}")
        plt.close()


if __name__ == "__main__":
    # Process training data
    train_channel_features = extract__channel_latent_features(
        train_data, device,
        model_type='DeepVAE',
        checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
        latent_channels=16
    )
    train_similarity_matrices = compute_channel_similarity_matrices(train_channel_features)
    plot_channel_similarity_matrices(
        train_similarity_matrices, 'train',
        model_type='DeepVAE',
        checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth'
    )

    # Process validation data 
    val_channel_features = extract__channel_latent_features(
        val_data, device, 
        model_type='DeepVAE', 
        checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
        latent_channels=16
    )
    val_similarity_matrices = compute_channel_similarity_matrices(val_channel_features)
    plot_channel_similarity_matrices(
        val_similarity_matrices, 'val',
        model_type='DeepVAE',
        checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth'
    )
