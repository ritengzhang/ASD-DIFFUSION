import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import plotly.express as px
from torch.utils.data import random_split

import sys
import os

# Add the parent directory (assuming `my_datasets` is in the same directory as `visualization.py`)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])
from my_datasets.toy_dataset_processed import MRIToyDataset
from EncDec.vae import SimpleVAE, EnhancedVAE, FlattenedVAE, EncDiff, create_encdiff_model, vae_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MRIToyDataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# convert to torch tensor
dataset = torch.from_numpy(dataset.data).float().to(device)
print(dataset.data.shape)

df = pd.read_csv('../ToyDataProcessedDataset/subject_info.csv')
possible_col = ["subID", "SEX", "AGE_AT_SCAN", "DX_GROUP", "HANDEDNESS_CATEGORY", "FIQ"]
#print(df.head)
# sex_column = df[possible_col[3]].values 
# classes = sex_column
# Create a dictionary for the classes
classes = {
    'subID': df[possible_col[0]].values,  # Add subID
    'SEX': df[possible_col[1]].values,
    'DX_GROUP': df[possible_col[3]].values,
    'AGE_AT_SCAN': df[possible_col[2]].values,
    'HANDEDNESS_CATEGORY': df[possible_col[4]].values,
    'FIQ': df[possible_col[5]].values
}


def visualize_mri_features(dataset, classes, device, model_type='EnhancedVAE', checkpoint_path=None, 
                           latent_channels=256, random_state=42, perplexity=30, n_iter=1000, output_dir='Feature visualization', pca_components=30):
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
    - pca_components: int, optional (default=30)
      Number of principal components to retain before applying t-SNE.
    
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
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    
    model = model.to(device)
    model.eval()
    
    # Extract features
    with torch.no_grad():
        features = model.visualize(dataset.data)
    
    # Reshape the features if necessary
    if len(features.shape) > 2:
        n_samples = features.shape[0]
        features_reshaped = features.reshape(n_samples, -1)
    else:
        features_reshaped = features
    
    # Ensure the features are detached from the computation graph and converted to numpy
    features_reshaped = features_reshaped.cpu().numpy()
    
    # Step 1: Apply PCA to reduce dimensionality to a manageable level (e.g., 30 components)
    pca = PCA(n_components=pca_components, random_state=random_state)
    features_pca = pca.fit_transform(features_reshaped)
    
    # Step 2: Apply t-SNE to reduce PCA-reduced features to 2 dimensions
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, n_iter=n_iter)
    features_tsne = tsne.fit_transform(features_pca)
    
    # Convert t-SNE results into a DataFrame for easier handling with Plotly Express
    df_tsne = pd.DataFrame({
        'Component 1': features_tsne[:, 0],
        'Component 2': features_tsne[:, 1],
    })
    
    for class_name, class_values in classes.items():
        df_tsne[class_name] = class_values

     # Explicitly convert 'SEX' and 'DX_GROUP' to categorical for Plotly
    df_tsne['SEX'] = df_tsne['SEX'].astype(str)  # Convert to string or categorical
    df_tsne['DX_GROUP'] = df_tsne['DX_GROUP'].astype(str)
    
    # Create the scatter plot with Plotly Express
    fig = px.scatter(df_tsne, 
                     x='Component 1', y='Component 2', 
                     color='DX_GROUP',  # Use one class for color
                     color_continuous_scale='bluered',
                     marginal_x="box", 
                     symbol='SEX',  # Use another class for shape
                     title= f"{model_type}_{checkpoint_name}_PCA_t-SNE visualization of MRI features", 
                     labels={'DX_GROUP': 'Diagnosis Group'},
                     hover_data=classes.keys()  # Show all class attributes on hover
                     )
    
    fig.update_layout(
        legend=dict(
            yanchor="top", 
            y=0.99, 
            xanchor="left", 
            x=0.01  # Position the shape legend 
        ),
        coloraxis_colorbar=dict(
            title="Diagnosis Group",
            yanchor="top",
            y=0.99, 
            xanchor="left",
            x=1.02  # position the color legend 
        )
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate file name
    file_name = f"PCA_{model_type}_{checkpoint_name}_mri_features.html"
    file_path = os.path.join(output_dir, file_name)
    
    # Save the HTML representation of the figure
    fig.write_html(file_path)
    print(f"Visualization saved as '{file_path}'")

visualize_mri_features(dataset, classes, device,
                       model_type='EncDiff', 
                       checkpoint_path='Feature visualization/encdiff_best_model.pth',
                       latent_channels=1)