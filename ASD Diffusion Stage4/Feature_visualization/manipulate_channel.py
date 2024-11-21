import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.spatial.distance import pdist, squareform
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import random_split
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])
from Feature_visualization.feature_matrix import run_rsa_analysis
from Feature_visualization.characteristic_matrix import run_analysis
from feature_matrix_sep import analyze_channel_similarities
from my_datasets.toy_dataset_processed import MRIToyDataset
from EncDec.vae import SimpleVAE, EnhancedVAE, FlattenedVAE, EncDiff, create_encdiff_model, vae_loss, DeepVAE

torch.manual_seed(42) 
np.random.seed(42) 

def extract_channel_features(dataset, device, model_type='DeepVAE', checkpoint_path=None, latent_channels=256):
    """Extract channel-wise latent features using model's visualize method"""
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
    
    with torch.no_grad():
        # Process entire dataset at once
        _, channel_features = model.visualize(dataset)
        # channel_features is list of [batch_size, feature_dim] tensors
        # Convert to numpy arrays
        # reshape each channel feature to [n_samples, feature_dim] instead of [n_sample, feature_dim, dim, dim]
        channel_features = [cf.cpu().numpy().reshape(cf.shape[0], -1) for cf in channel_features]
        
    return channel_features

def analyze_channel_modifications(device, model_type='DeepVAE', checkpoint_path=None, 
                                latent_channels=256, modification_steps=11):
    """Main analysis function with separated train/val handling"""
    # Setup datasets
    dataset = MRIToyDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_data = torch.stack([item[0] for item in train_dataset]).to(device)
    val_data = torch.stack([item[0] for item in val_dataset]).to(device)
    
    # Get characteristic matrices
    df = pd.read_csv('../ToyDataProcessedDataset/subject_info.csv')
    possible_col = ["SEX", "AGE_AT_SCAN", "DX_GROUP", "HANDEDNESS_SCORES", "FIQ"]
    matrices_train = run_analysis(df, possible_col, train_dataset.indices)
    matrices_val = run_analysis(df, possible_col, val_dataset.indices)
    
    # Extract channel features
    train_channel_features = extract_channel_features(train_data, device, model_type, 
                                                    checkpoint_path, latent_channels)
    val_channel_features = extract_channel_features(val_data, device, model_type, 
                                                  checkpoint_path, latent_channels)
    
    # Analysis parameters
    modification_percentages = np.linspace(0, 1, modification_steps)
    output_dir = os.path.join('../Visualization', f"{model_type}_{latent_channels}_channel_modifications")
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze each channel
    for channel_idx in range(16):
        train_results = {}
        val_results = {}
        
        for percentage in modification_percentages:
            # Create modified copies of all channel features
            modified_train_channels = train_channel_features.copy()
            # print(f"original: {train_channel_features[channel_idx][:1]}")
            # print(modified_train_channels[channel_idx][:1])
            # print(len(train_channel_features))
            # print(train_channel_features[channel_idx].shape)
            modified_val_channels = val_channel_features.copy()
            # print(f"original: {val_channel_features[channel_idx][:1]}")
            # print(modified_val_channels[channel_idx][:1])
            
            # Modify only the target channel
            modified_train_channels[channel_idx] = train_channel_features[channel_idx] * (1 - percentage)
            # print(f"modified: {modified_train_channels[channel_idx][:1]}")
            modified_val_channels[channel_idx] = val_channel_features[channel_idx] * (1 - percentage)
            # print(f"modified_val: {modified_val_channels[channel_idx][:1]}")
            
            # Concatenate all channels for similarity computation
            train_features = np.hstack(modified_train_channels)
            # print(train_features.shape)
            val_features = np.hstack(modified_val_channels)
            
            # Compute similarity matrices
            train_distances = pdist(train_features, metric='euclidean')
            # print(train_distances[:10])
            val_distances = pdist(val_features, metric='euclidean')

            # Normalize distances to [0,1] range
            if train_distances.max() > 0:
                train_distances = train_distances / train_distances.max()
                print(f"normalized: {train_distances[:10]}")
                val_distances = val_distances / val_distances.max()

            # Convert to similarity matrices
            train_sim = 1 - squareform(train_distances)
            val_sim = 1 - squareform(val_distances)
            
            # Calculate tau for each characteristic
            for char_name in matrices_train.keys():
                if char_name not in train_results:
                    train_results[char_name] = []
                    val_results[char_name] = []
                
                train_tau, _ = kendalltau(train_sim.flatten(), matrices_train[char_name].flatten())
                val_tau, _ = kendalltau(val_sim.flatten(), matrices_val[char_name].flatten())
                
                train_results[char_name].append(train_tau)
                val_results[char_name].append(val_tau)
        
        # Plot results for this channel
        plot_channel_results(train_results, val_results, modification_percentages, 
                           channel_idx, output_dir)

def plot_channel_results(train_results, val_results, mod_percentages, channel_idx, output_dir):
    """Plot training and validation results for a single channel"""
    plt.figure(figsize=(12, 6))
    markers = ['o', 's', '^', 'D', 'v']
    
    for (char_name, train_taus), marker in zip(train_results.items(), markers):
        plt.plot(mod_percentages * 100, train_taus, 
                label=f'Train {char_name}', marker=marker, linestyle='-')
        plt.plot(mod_percentages * 100, val_results[char_name], 
                label=f'Val {char_name}', marker=marker, linestyle='--')
    
    plt.xlabel('Feature Modification (%)')
    plt.ylabel("Kendall's Tau")
    plt.title(f'Channel {channel_idx} Modification Effects')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    filename = f'channel_{channel_idx}_modification_effects.png'
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyze_channel_modifications(
        device,
        model_type='DeepVAE',
        checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
        latent_channels=16
    )