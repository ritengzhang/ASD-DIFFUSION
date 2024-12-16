import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
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

torch.manual_seed(42) 
np.random.seed(42) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MRIToyDataset()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# convert to torch tensor
train_data = torch.stack([item[0] for item in train_dataset]).to(device)
val_data = torch.stack([item[0] for item in val_dataset]).to(device)
dataset = torch.from_numpy(dataset.data).float().to(device)
print(dataset.data.shape)
print(train_data.shape)
print(val_data.shape)

train_indices = [item[1] for item in train_dataset]
val_indices = [item[1] for item in val_dataset]

def plot_channel_kendall_tau(similarity_matrices, characteristic_matrices, split_name=None, output_dir='../Visualization', model_name='Unknown', checkpoint_path=None):
    """
    Calculate and optionally plot Kendall's tau values for each channel's similarity matrix
    compared against each characteristic matrix.
    
    Parameters:
    -----------
    similarity_matrices : list
        List of similarity matrices, one per channel
    characteristic_matrices : dict
        Dictionary containing characteristic matrices
    output_dir : str
        Directory to save individual plots if plotting is enabled
    """
    if checkpoint_path:
    # Create the output directory with model name and checkpoint
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_path = os.path.join(output_dir, f"{model_name}_{split_name}_{checkpoint_name}_Kendall_Tau")
        os.makedirs(output_path, exist_ok=True)  # Create the folder if it doesn't exist
    
    # Process each channel
    for channel_idx, feature_matrix in enumerate(similarity_matrices):          
        # Create individual plot for this channel
        fig, ax = plot_kendall_tau(
            feature_matrix, 
            characteristic_matrices,
            output_dir=output_path,
            title=f'{model_name} {split_name} Channel {channel_idx}'
        )
        plt.close(fig)  # Close the figure to free memory

    print(f"All plots saved in directory: {output_path}")
    

def plot_kendall_tau(feature_matrix, characteristic_matrices, figsize=(10, 6), output_dir='../Visualization', title=None, model_type='Unknown'):
    """
    Plot Kendall's tau values for each characteristic matrix comparison.
    
    Parameters:
    -----------
    feature_matrix : numpy.ndarray
        The reference feature matrix of shape (105, 105)
    characteristic_matrices : dict
        Dictionary containing characteristic matrices
    figsize : tuple, optional
        Figure size in inches (default=(10, 6))
    """
    # Calculate Kendall's tau for each matrix
    tau_values = []
    labels = []
    p_values = []
    
    # Flatten the feature matrix
    feature_flat = feature_matrix.flatten()
    
    # Calculate tau for each characteristic matrix
    for name, char_matrix in characteristic_matrices.items():
        if char_matrix.shape != feature_matrix.shape:
            raise ValueError(f"Matrix {name} has incorrect shape {char_matrix.shape}")
        
        char_flat = char_matrix.flatten()
        tau, p_value = kendalltau(feature_flat, char_flat)
        tau_values.append(tau)
        labels.append(name)
        p_values.append(p_value)
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    bars = ax.bar(labels, tau_values, color='skyblue', edgecolor='black')
    
    # Customize the plot
    ax.set_title(f"{title} Kendall's Tau Correlation Coefficients", pad=20, fontsize=12)
    ax.set_xlabel("Characteristic Matrices", fontsize=10)
    ax.set_ylabel("Kendall's Tau Value", fontsize=10)
    
    # Set y-axis limits to show the full possible range of tau values
    ax.set_ylim(-1, 1)
    
    # Add tau and p-value labels for each bar
    for bar, p_value in zip(bars, p_values):
        height = bar.get_height()
        # Format p-value with scientific notation if very small
        if p_value < 0.001:
            p_text = f'p = {p_value:.2e}'
        else:
            p_text = f'p = {p_value:.3f}'
            
        # Add tau value above bar
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'τ = {height:.3f}',
                ha='center', va='bottom')
        
        # Add p-value below tau value
        # Adjust vertical position based on whether bar is positive or negative
        if height >= 0:
            va_position = -0.1
        else:
            va_position = 0.1
        ax.text(bar.get_x() + bar.get_width()/2., height + va_position,
                p_text,
                ha='center', va='top' if height >= 0 else 'bottom',
                fontsize=8,
                color='darkred' if p_value < 0.05 else 'black')  # Highlight significant p-values
    
    # Rotate x-axis labels for better readability if needed
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Create filename with model type
    clean_title = title.replace(' ', '_').replace('-', '_')
    file_name = f"{clean_title}_Tau_Correlation.png"
    file_path = os.path.join(output_dir, file_name)

    plt.savefig(file_path, format='png', dpi=300)
    print(f"Plot saved as {file_path}")

    print(f"\nKendall's Tau values for {title}:")
    for name, tau, p in zip(labels, tau_values, p_values):
        print(f"{name}: τ = {tau:.3f} p = {p}")
    
    return fig, ax

df = pd.read_csv('../ToyDataProcessedDataset/subject_info.csv')
possible_col = ["SEX", "AGE_AT_SCAN", "DX_GROUP", "HANDEDNESS_SCORES", "FIQ"]

# Run analysis for the whole dataset
matrices = run_analysis(df, possible_col)
# Run analysis for training data
matrices_train = run_analysis(df, possible_col, train_indices)
# Run analysis for validation data
matrices_val = run_analysis(df, possible_col, val_indices)

# Example usage:
seperate = True
if seperate:
    # latent_features, similarity_matrix = analyze_channel_similarities(dataset, device,
    #                     model_type='DeepVAE', 
    #                     checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
    #                     latent_channels=16)
    # plot_channel_kendall_tau(similarity_matrix, matrices, output_dir='../Visualization', model_name='D16', checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth')

    latent_features_train, similarity_matrix_train = analyze_channel_similarities(train_data, device, 
                        model_type='DeepVAE', 
                        checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
                        latent_channels=16)
    plot_channel_kendall_tau(similarity_matrix_train, matrices_train, split_name='train', output_dir='../Visualization', model_name='D16', checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth')

    latent_features_val, similarity_matrix_val = analyze_channel_similarities(val_data, device, 
                        model_type='DeepVAE', 
                        checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
                        latent_channels=16)
    plot_channel_kendall_tau(similarity_matrix_val, matrices_val, split_name='val', output_dir='../Visualization', model_name='D16', checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth')
else:    
    latent_features, similarity_matrix = run_rsa_analysis(dataset, device,
                        model_type='DeepVAE', 
                        checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
                        latent_channels=16)

    fig, ax = plot_kendall_tau(similarity_matrix, matrices, title= "D16", model_type='DeepVAE')
    plt.show()

    latent_features_train, similarity_matrix_train = run_rsa_analysis(train_data, device,
                        model_type='DeepVAE', 
                        checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
                        latent_channels=16)
    fig_train, ax_train = plot_kendall_tau(similarity_matrix_train, matrices_train, title="D16_train", model_type='DeepVAE')
    plt.show()

    latent_features_val, similarity_matrix_val = run_rsa_analysis(val_data, device,
                        model_type='DeepVAE', 
                        checkpoint_path='../Visualization/paths/D16_checkpoint_epoch_6000.pth',
                        latent_channels=16)
    fig_val, ax_val = plot_kendall_tau(similarity_matrix_val, matrices_val, title="D16_val", model_type='DeepVAE')
    plt.show()

