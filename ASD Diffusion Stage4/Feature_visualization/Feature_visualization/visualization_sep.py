import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MRIToyDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# convert to torch tensor
# dataset = torch.from_numpy(dataset.data).float().to(device)
# print(dataset.data.shape)
# Extract data and indices
train_data = torch.stack([dataset[i][0] for i in train_dataset.indices]).to(device)
train_indices = train_dataset.indices

val_data = torch.stack([dataset[i][0] for i in val_dataset.indices]).to(device)
val_indices = val_dataset.indices

df = pd.read_csv('../ToyDataProcessedDataset/subject_info.csv')
possible_col = ["subID", "SEX", "AGE_AT_SCAN", "DX_GROUP", "HANDEDNESS_CATEGORY", "FIQ"]
# Create separate class dictionaries
train_classes = {col: df[col].values[train_indices] for col in possible_col}
val_classes = {col: df[col].values[val_indices] for col in possible_col}

def visualize_mri_features(dataset, classes, device, split_name, model_type, checkpoint_path=None, 
                           latent_channels=256, random_state=42, perplexity=15, n_iter=1000, output_dir='../Visualization'):
    """
    
    Parameters:
    - dataset: torch.Tensor
      The MRI dataset to visualize.
    - classes: dict of numpy arrays
      The class labels for each sample, where keys are class names 
      and values are numpy arrays of class labels.
    - model_type: str, optional (default='EnhancedVAE')
      The type of model to use. Either 'SimpleVAE' or 'EnhancedVAE'.
    - checkpoint_path: str, optional (default=None)
      Path to the model checkpoint file. If None, a new model will be initialized.
    - latent_channels: int, optional (default=256)
      Number of latent channels for the VAE model.
    - random_state: int, optional (default=42)
      Random state for reproducibility.
    - perplexity: float, optional (default=30)
      The perplexity parameter for t-SNE.
    - n_iter: int, optional (default=1000)
      Number of iterations for t-SNE optimization.
    
    Returns:
    - None (saves the plot as an HTML file)
    """
    
    # Initialize the model
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
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    
    model = model.to(device)
    model.eval()

    # Create the output directory with model name and checkpoint
    output_path = os.path.join(output_dir, f"{model_type}_{checkpoint_name}_tsne_visualizations")
    os.makedirs(output_path, exist_ok=True)  # Create the folder if it doesn't exist
    
    # Extract features
    with torch.no_grad():
        _, features = model.visualize(dataset.data) # (105, channel, feature)
    # Iterate through each channel and apply t-SNE
    for channel_idx in range(16):  # Loop over channels 
        # Extract features for the current channel (shape: [n_samples, flattened_dim])
        channel_features = features[channel_idx].cpu().numpy()
        print(channel_features.shape)
        if len(channel_features.shape) > 2:
            n_samples = channel_features.shape[0]
            features_reshaped = channel_features.reshape(n_samples, -1)
        else:
            features_reshaped = channel_features
        print(features_reshaped.shape)
        # Apply t-SNE (converting flattened_dim-dimensional features to 2D for visualization)
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, n_iter=n_iter)
        tsne_result = tsne.fit_transform(features_reshaped)
        
        # Convert t-SNE results into a DataFrame for easier handling with Plotly Express
        df_tsne = pd.DataFrame({
            'Component 1': tsne_result[:, 0],
            'Component 2': tsne_result[:, 1],
        })
        
        for class_name, class_values in classes.items():
            df_tsne[class_name] = class_values

        # Explicitly convert 'SEX' and 'DX_GROUP' to categorical for Plotly
        df_tsne['SEX'] = df_tsne['SEX'].astype(str)
        df_tsne['DX_GROUP'] = df_tsne['DX_GROUP'].astype(str)

        # Create scatter plot
        fig = px.scatter(df_tsne, 
                         x='Component 1', y='Component 2', 
                         color='AGE_AT_SCAN',
                         color_continuous_scale='Viridis',
                         symbol='HANDEDNESS_CATEGORY',
                         symbol_map={'R': 'circle', 'L': 'star-diamond', 'Ambi': 'cross'},
                         title= f"{model_type}_{checkpoint_name}_{split_name}_Channel_{channel_idx}_t-SNE", 
                         hover_data=classes.keys()
                         )
        
        fig.update_traces(marker=dict(size=10))
        fig.update_layout(
            legend=dict(
                yanchor="top", 
                y=1.15, 
                xanchor="left", 
                x=0.7, # Position the shape legend 
                bgcolor="rgba(255, 255, 255, 0.8)",  # Set a semi-transparent white background
                bordercolor="black",  # Optional: Add a border color
                borderwidth=1  # Optional: Add a border width
            ),
            coloraxis_colorbar=dict(
                title="AGE_AT_SCAN",
                yanchor="top",
                y=0.99, 
                xanchor="left",
                x=1.02  # position the color legend 
            )
        )
        
        file_name = f"{model_type}_{checkpoint_name}_{split_name}_channel_{channel_idx}_tsne.html"
        file_path = os.path.join(output_path, file_name)
        fig.write_html(file_path)
        print(f"Channel {channel_idx} visualization saved as '{file_path}'")


# Visualize training set
visualize_mri_features(train_data, train_classes, device,
                       split_name='train',
                       model_type='DeepVAE', 
                       checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
                       latent_channels=16)

# Visualize validation set  
visualize_mri_features(val_data, val_classes, device,
                       split_name='val',
                       model_type='DeepVAE', 
                       checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
                       latent_channels=16)