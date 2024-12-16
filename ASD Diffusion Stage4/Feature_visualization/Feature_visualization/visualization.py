import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from torch.utils.data import random_split
import warnings
import gc
import sys
import os
import json 

warnings.filterwarnings("ignore", category=FutureWarning)

# Add the parent directory (assuming `my_datasets` is in the same directory as `visualization.py`)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])
from my_datasets.toy_dataset_processed import MRIToyDataset, load_abcd_dataset, BrainDataset
from EncDec.vae import SimpleVAE, EnhancedVAE, FlattenedVAE, vae_loss, DeepVAE
from EncDec.encdiff import EncDiff, create_encdiff_model

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# dataset = MRIToyDataset()
dataset = load_abcd_dataset()
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(len(train_dataset), len(val_dataset))
# print(train_dataset)
# convert to torch tensor
# dataset = torch.from_numpy(dataset.data).float().to(device)
# print(dataset.data.shape)
# Extract data and indices from the train and validation datasets
if dataset.data.shape[0] == 105:
    train_data = torch.stack([item[0] for item in train_dataset]).to(device)
    val_data = torch.stack([item[0] for item in val_dataset]).to(device)

    train_indices = [item[1] for item in train_dataset]
    val_indices = [item[1] for item in val_dataset]

    df = pd.read_csv('../ToyDataProcessedDataset/subject_info.csv')
    possible_col = ["subID", "SEX", "AGE_AT_SCAN", "DX_GROUP", "HANDEDNESS_CATEGORY", "FIQ"]

    train_classes = {col: df[col].values[train_indices] for col in possible_col}
    val_classes = {col: df[col].values[val_indices] for col in possible_col}
else:
    train_data = torch.stack([item[0] for item in train_dataset]).to(device)
    val_data = torch.stack([item[0] for item in val_dataset]).to(device)

    # Get subject IDs using for loops
    train_subject_ids = []
    val_subject_ids = []

    for i in range(len(train_dataset)):
        _, _, subject_id = train_dataset[i]
        # Remove 'sub-' prefix
        clean_id = subject_id[4:] if subject_id.startswith('sub-') else subject_id
        train_subject_ids.append(clean_id)

    for i in range(len(val_dataset)):
        _, _, subject_id = val_dataset[i]
        clean_id = subject_id[4:] if subject_id.startswith('sub-') else subject_id
        val_subject_ids.append(clean_id)

    df = pd.read_csv('../ToyDataProcessedDataset/processed_abcd_ksad01.csv', low_memory=False)
    possible_col = ["subjectkey","sex", "interview_age", "label", "ksads_14_402_p", "ksads_14_405_p", "ksads_14_396_p", "ksads_14_400_p", "ksads_14_398_p", "ksads_14_397_p", "ksads_14_404_p", "ksads_14_406_p", "ksads_14_401_p", "ksads_14_403_p", "ksads_10_869_p", "ksads_1_840_p", "ksads_15_901_p"]  

    # Match with CSV using subject IDs
    train_df = pd.DataFrame([df[df['subjectkey'] == id].iloc[0] for id in train_subject_ids])
    val_df = pd.DataFrame([df[df['subjectkey'] == id].iloc[0] for id in val_subject_ids])

    # Create class dictionaries
    train_classes = {col: train_df[col].values for col in possible_col}
    val_classes = {col: val_df[col].values for col in possible_col}
    all_classes = {col: df[col].values for col in possible_col}
    print(train_classes["subjectkey"][:5])
    print(val_classes["subjectkey"][:5])
    
    count = 0
    for item in train_dataset:
        print(f"Subject ID: {item[2]}")
        count += 1
        if count >= 5:
            break

    count = 0
    for item in val_dataset:
        print(f"Subject ID: {item[2]}")
        count += 1
        if count >= 5:
            break

def visualize_mri_features(dataset, classes, device, title, model_type='EnhancedVAE', checkpoint_path=None, 
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
    
    # Extract features
    with torch.no_grad():
        features, _ = model.visualize(dataset.data) #(105, channels, 256)
    
    # Reshape the features if necessary
    if len(features.shape) > 2:
        n_samples = features.shape[0]
        features_reshaped = features.reshape(n_samples, -1)
    else:
        features_reshaped = features
    print(features_reshaped.shape)
    # Ensure the features are detached from the computation graph and converted to numpy
    features_reshaped = features_reshaped.cpu().numpy()
    # features_reshaped = features_reshaped.detach().numpy()  
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, n_iter=n_iter)
    features_tsne = tsne.fit_transform(features_reshaped)
    
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
                     color='AGE_AT_SCAN',  # Use one class for color
                     color_continuous_scale='Viridis',
                     symbol='HANDEDNESS_CATEGORY',  # Use another class for shape
                     symbol_map={'R': 'circle', 'L': 'star-diamond', 'Ambi': 'cross'},  # Custom shape mapping
                     title= f"{title}_{model_type}_{checkpoint_name}_t-SNE visualization of MRI features", 
                     hover_data=classes.keys()  # Show all class attributes on hover
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate file name
    file_name = f"{model_type}_{checkpoint_name}_mri_features_{title}.html"
    file_path = os.path.join(output_dir, file_name)
    
    # Save the HTML representation of the figure
    fig.write_html(file_path)
    print(f"Visualization saved as '{file_path}'")

def visualize_abcd_features(dataset, classes, device, title, model_type='EnhancedVAE', 
                             checkpoint_path=None, config_path=None, latent_channels=256, 
                             random_state=42, perplexity=30, n_iter=1000, 
                             output_dir='../Visualization'):
    """
    Visualize features from the ABCD MRI dataset using dimensionality reduction.
    
    Parameters:
    - dataset: torch.Tensor
      The ABCD MRI dataset to visualize.
    - classes: dict of numpy arrays
      The class labels for each sample, where keys are class names 
      and values are numpy arrays of class labels.
    - device: torch.device
      The device to run computations on (cuda/cpu).
    - title: str
      Title for the visualization.
    - model_type: str, optional (default='EnhancedVAE')
      The type of model to use for feature extraction.
    - checkpoint_path: str, optional (default=None)
      Path to the model checkpoint file.
    - latent_channels: int, optional (default=256)
      Number of latent channels for the VAE model.
    - random_state: int, optional (default=42)
      Random state for reproducibility.
    - perplexity: float, optional (default=30)
      The perplexity parameter for t-SNE.
    - n_iter: int, optional (default=1000)
      Number of iterations for t-SNE optimization.
    - output_dir: str, optional (default='../Visualization')
      Directory to save the output visualization.
    
    Returns:
    - None (saves the plot as an HTML file)
    """
     # Memory management
    torch.cuda.empty_cache()
    gc.collect()
    
    # Select and initialize the model
    model_classes = {
        'SimpleVAE': SimpleVAE,
        'EnhancedVAE': EnhancedVAE,
        'FlattenedVAE': FlattenedVAE,
        'DeepVAE': DeepVAE,
        'EncDiff': create_encdiff_model
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Invalid model_type. Choose from {list(model_classes.keys())}")
    
    # Handle EncDiff model differently
    if model_type == 'EncDiff':
        with open(config_path, 'r') as f:
            config = json.load(f)
        model, _ = create_encdiff_model(
            config=config,
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            device=device
        )
    else:
        model = model_classes[model_type](latent_channels=latent_channels)
    
    # Load checkpoint if provided
    checkpoint_name = 'no_checkpoint'
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if model_type == 'EncDiff':
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    
    model = model.to(device)
    model.eval()
    print("done")
    
    # Extract features 
    features_list = []
    with torch.no_grad():
        for i in range(0, len(dataset), 32):
            batch = dataset[i:i + 32].to(device)
            
            if model_type == 'EncDiff':
                batch_features, _ = model.encode_image(batch)
            else:
                batch_features, _ = model.visualize(batch)
                
            # Move batch results to CPU immediately
            features_list.append(batch_features.cpu())
            
            if (i // 32) % 3 == 0:
                print(f"Processed {i} samples")
            # Clear cache after each batch
            gc.collect()

    print("done2")
    # Concatenate all batches
    features = torch.cat(features_list, dim=0)
    print(features.shape)
    
    # Reshape features to 2D
    if len(features.shape) > 2:
        n_samples = features.shape[0]
        features_reshaped = features.reshape(n_samples, -1)
    else:
        features_reshaped = features
    
    features_reshaped = features_reshaped.to(device).numpy()
    print(features_reshaped.shape)
    print("Mean of each feature:", np.mean(features_reshaped, axis=0))
    print("Variance of each feature:", np.var(features_reshaped, axis=0))

    save_path = '../Visualization/paths/Braindataset/encdiff/64_train_features_reshaped.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists

    # Save the features
    torch.save(features_reshaped, save_path)
    print(f"Features saved to {save_path}")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, 
                perplexity=min(perplexity, len(features_reshaped)-1), 
                n_iter=n_iter)
    features_tsne = tsne.fit_transform(features_reshaped)
    print("done3")
    print(features_tsne.shape)
    
    # Prepare DataFrame for visualization
    df_tsne = pd.DataFrame({
        'Component 1': features_tsne[:, 0],
        'Component 2': features_tsne[:, 1],
    })
    
    for class_name, class_values in classes.items():
        df_tsne[class_name] = class_values
    
    # Create the scatter plot with Plotly Express
    fig = px.scatter(
        df_tsne, 
        x='Component 1', 
        y='Component 2', 
        color='interview_age',  # Use interview_age for color
        color_continuous_scale='Viridis',
        symbol='ksads_14_396_p',  # Use ksads_14_396_p for symbol
        symbol_map={0: 'circle', 1: 'star-diamond'},  # Custom symbol mapping
        title=f"{title}_{model_type}_{checkpoint_name}_t-SNE visualization of MRI features",
        hover_data=list(classes.keys())
    )
    
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(
        legend=dict(
            yanchor="top", 
            y=1.15, 
            xanchor="left", 
            x=0.7,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        coloraxis_colorbar=dict(
            title="Interview Age",
            yanchor="top",
            y=0.99, 
            xanchor="left",
            x=1.02
        )
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate file name and save
    file_name = f"{model_type}_{checkpoint_name}_abcd_features_{title}.html"
    file_path = os.path.join(output_dir, file_name)
    
    fig.write_html(file_path)
    print(f"Visualization saved as '{file_path}'")

if dataset.data.shape[0] == 105:
    visualize_mri_features(train_data, train_classes, device,
                        title='train',
                        model_type='EnhancedVAE', 
                        checkpoint_path='../Visualization/paths/MRIToy/Echeckpoint_epoch_10000.pth',
                        latent_channels=256)

    visualize_mri_features(val_data, val_classes, device,
                        title='test',
                        model_type='EnhancedVAE', 
                        perplexity=5,
                        checkpoint_path='../Visualization/paths/MRIToy/Echeckpoint_epoch_10000.pth',
                        latent_channels=256)
else:
    visualize_abcd_features(train_data, train_classes, device,
                        title='train',
                        model_type='EncDiff', 
                        checkpoint_path='../Visualization/paths/Braindataset/encdiff/64_encdiff_best_model.pth',
                        config_path = "../Visualization/paths/Braindataset/encdiff/config.json",
                        latent_channels=32)
    
    #  Clear memory before validation
    torch.cuda.empty_cache()
    gc.collect()

    visualize_abcd_features(val_data, val_classes, device,
                        title='test',
                        model_type='EncDiff', 
                        perplexity=15,
                        checkpoint_path='../Visualization/paths/Braindataset/encdiff/64_encdiff_best_model.pth',
                        config_path = "../Visualization/paths/Braindataset/encdiff/config.json",
                        latent_channels=32)