import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import sys
import pandas as pd
import os
from torch.utils.data import DataLoader, random_split, Subset
import warnings

# Add the parent directory (assuming `my_datasets` is in the same directory as `visualization.py`)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])
from my_datasets.toy_dataset_processed import MRIToyDataset, NPYDataset, CIFARDataset, load_abcd_dataset, BrainDataset
from EncDec.vae import SimpleVAE, EnhancedVAE, FlattenedVAE, vae_loss

warnings.filterwarnings("ignore", category=FutureWarning)
# torch.manual_seed(42)
# np.random.seed(42)

# def visualize_mri_features(features, classes, random_state=42, perplexity=30, n_iter=1000):
#     """
#     Visualize MRI features using t-SNE dimensionality reduction and a scatter plot.
    
#     Parameters:
#     - features: numpy array of shape (n_samples, n_features)
#       The feature representation of the MRI data.
#     - classes: numpy array of shape (n_samples,)
#       The class labels for each sample.
#     - random_state: int, optional (default=42)
#       Random state for reproducibility.
#     - perplexity: float, optional (default=30)
#       The perplexity parameter for t-SNE.
#     - n_iter: int, optional (default=1000)
#       Number of iterations for t-SNE optimization.
    
#     Returns:
#     - None (displays the plot)
#     """
#     # Reshape the features if necessary
#     if len(features.shape) > 2:
#         n_samples = features.shape[0]
#         features_reshaped = features.reshape(n_samples, -1)
#     else:
#         features_reshaped = features
    
#     # Apply t-SNE
#     features_reshaped = features_reshaped.detach().numpy()
#     tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, n_iter=n_iter)
#     features_tsne = tsne.fit_transform(features_reshaped)
    
#     # Create a scatter plot
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=classes, cmap='viridis')
#     plt.colorbar(scatter)
#     plt.title('t-SNE visualization of MRI features')
#     plt.xlabel('t-SNE component 1')
#     plt.ylabel('t-SNE component 2')
#     plt.show()


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset = MRIToyDataset()
# # convert to torch tensor
# dataset = torch.from_numpy(dataset.data).float().to(device)
# print(dataset.shape)
# print(dataset.data.shape)
# df = pd.read_csv('../ToyDataProcessedDataset/subject_info.csv')
# #print(df.head)
# possible_col = ["SEX", "AGE_AT_SCAN", "DX_GROUP", "HANDEDNESS_CATEGORY", "FIQ"]
# sex_column = df[possible_col[0]].values 
# classes = sex_column
# classes = {
#     'SEX': df[possible_col[0]].values,
#     'DX_GROUP': df[possible_col[2]].values,
#     'AGE_AT_SCAN': df[possible_col[1]].values,
#     'HANDEDNESS_CATEGORY': df[possible_col[3]].values,
#     'FIQ': df[possible_col[4]].values
# }

# model = create_encdiff_model(in_channels=1, out_channels=1)
# checkpoint = torch.load('Feature visualization/encdiff_best_model.pth', map_location=device)
# model.load_state_dict(checkpoint)
# model = model.to(device)
# model.eval()
    
#     # Extract features
# with torch.no_grad():
#     features = model.visualize(dataset.data)
# print(features.shape)

# num_channels = features.shape[1]

# # Iterate through each channel
# for channel_idx in range(num_channels):
#     # Extract the features for the current channel (shape: [105, 256])
#     channel_features = features[:, channel_idx, :]

#     # Apply t-SNE (converting 256-dimensional features to 2D for visualization)
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
#     tsne_result = tsne.fit_transform(channel_features)
    
#     # Plot the t-SNE results
#     plt.figure(figsize=(6, 5))
#     plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=classes, cmap='viridis')
#     plt.title(f't-SNE Visualization for Channel {channel_idx + 1}')
#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')
#     plt.show()
# # Path to your model checkpoint file
# model = EnhancedVAE(latent_channels=256)
# checkpoint_path = "../Visualization/paths/checkpoint_epoch_500.pth"

# # Load the checkpoint
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Adjust device if you're using CUDA

# # Load the state dict into the model
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
# feature = model.visualize(dataset)
# print(feature.shape)
# print(feature[0, 0, :, :, :])

# Example usage:
# Assuming you have your features and classes as numpy arrays
# features = np.random.rand(100, 64, 64, 64)  # 100 samples of 64x64x64 features
# classes = np.random.randint(0, 5, 100)  # 100 samples with 5 possible classes
# visualize_mri_features(feature, classes)


# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# print(len(train_dataset))  # Print the number of samples in the training dataset
# print(len(val_dataset)) 

# # To print the shape of an individual sample from the subset (assuming all samples have the same shape)
# sample_train = train_dataset[0][0]  # Access the first sample's data (assuming it’s a tuple with data and label)
# print(sample_train.shape)

# sample_val = val_dataset[0][0]  # Access the first sample's data (assuming it’s a tuple with data and label)
# print(sample_val.shape)

# model = create_encdiff_model(in_channels=1, out_channels=1)

    
# # Load checkpoint if provided
# checkpoint = torch.load("../Visualization/paths/encdiff_best_model.pth", map_location=device)
# model.load_state_dict(checkpoint)


# # model.eval()
    
# #     # Initialize list to store features for each channel
# # channel_features = [[] for _ in range(20)]

# # with torch.no_grad():
# #     for i in range(dataset.shape[0]):
# #         # Extract single sample 
# #         sample = dataset[i:i+1]
        
# #         # Get latent representation
# #         latent = model.visualize(sample)  # shape: (1, channel_size, feature_dim)
        
# #         # If latent is a tuple (mean, logvar) from VAE, just take the mean
# #         if isinstance(latent, tuple):
# #             latent = latent[0]
# #         latent_np = latent.cpu().numpy()
        
# #         # Separate channels and store
# #         for ch in range(20):
# #             channel_features[ch].append(latent_np[:, ch, :].reshape(1, -1))

# # # Stack features for each channel
# # channel_features = [np.vstack(features) for features in channel_features]
# # print(len(channel_features)) # 20
# # for ch in range(20):
# #     print(np.array(channel_features[ch][:10]))  # Print first 10 entries
# #     print(np.array(channel_features[ch]).shape)  # Print shape of each channel (n_sample, feature_dim)

# model = model.to(device)
# model.eval()

# latent_features = []

# #old visualize func
# # with torch.no_grad():
# #     for i in range(dataset.shape[0]):
# #         # Extract single sample and add batch dimension
# #         sample = dataset[i:i+1]
        
# #         # Get latent representation (assuming your model has an encode method)
# #         latent = model.visualize(sample) #latent shape (1, channel size, feature_dim)
        
# #         # If latent is a tuple (mean, logvar) from VAE, just take the mean
# #         if isinstance(latent, tuple):
# #             latent = latent[0]
        
# #         # Convert to numpy and flatten if needed
# #         latent_np = latent.cpu().numpy()
# #         latent_features.append(latent_np.reshape(1, -1))

# # features = np.vstack(latent_features)
# # print(len(features))
# # print(features.shape) #(n_sample, channel*feature_dim)

    
# with torch.no_grad():
#     for i in range(dataset.shape[0]):
#         sample = dataset[i:i+1]
#         latent, _ = model.visualize(sample)  # Only use the first returned value
#         latent_np = latent.cpu().numpy()
#         latent_features.append(latent_np.reshape(1, -1))
    
# features =np.vstack(latent_features)
# print(len(features))
# print(features.shape) #(n_sample, channel*feature_dim)


# def visualize(self, x):
#         """
#         Returns both the full mu tensor and a list of per-channel tensors.
        
#         Args:
#             x: Input tensor
            
#         Returns:
#             tuple: (h, channel_tensors) where:
#                 - h: Tensor of shape [batch_size, latent_channels, feature_dim]
#                 - channel_tensors: List of tensors, each [batch_size, feature_dim]
#         """
#         h = self.concept_encoder(x)
        
#         # Create list of per-channel tensors
#         channel_tensors = [h[:, i, ...] for i in range(h.shape[1])]
        
#         return h, channel_tensors # shape: (n_samples, channel_size, feature_dim) and 
#                                     #(list of len channel_size, each tensor [batch_size, feature_dim])
torch.manual_seed(42)
np.random.seed(42)
import pickle

# Extract raw data from pickle and save as .npy
# with open('../ToyDataProcessedDataset/brain_dataset.pkl', 'rb') as f:
#     data = pickle.load(f)
#     if hasattr(data, 'data'):  # If it's a class instance
#         raw_data = data.data
#     else:
#         raw_data = data
# np.save('../ToyDataProcessedDataset/brain_dataset.npy', raw_data)

# data = np.load('../ToyDataProcessedDataset/brain_dataset.npy')
# print(type(data))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset = MRIToyDataset()
dataset = load_abcd_dataset()
print(type(dataset))
train_size = int(0.8 * len(dataset))
print(f"Training set size: {train_size}")
test_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
# train_data = torch.tensor(train_dataset.data, dtype=torch.float32)
# print(dataset.data.shape)
# for item in dataset:
#     print(item[1])

# print(val_dataset[1])
# convert to torch tensor
# dataset = torch.from_numpy(dataset.data).float().to(device)

print(dataset.data.shape)
# train_indices = [item[1] for item in train_dataset]
# print(train_indices)
# val_indices = [item[1] for item in val_dataset]
# df = pd.read_csv('../ToyDataProcessedDataset/subject_info.csv')
# possible_col = ["subID", "SEX", "AGE_AT_SCAN", "DX_GROUP", "HANDEDNESS_CATEGORY", "FIQ"]
# classes = {
#     'subID': df[possible_col[0]].values,  # Add subID
#     'SEX': df[possible_col[1]].values,
#     'DX_GROUP': df[possible_col[3]].values,
#     'AGE_AT_SCAN': df[possible_col[2]].values,
#     'HANDEDNESS_CATEGORY': df[possible_col[4]].values,
#     'FIQ': df[possible_col[5]].values
# }

# train_classes = {col: df[col].values[train_indices] for col in possible_col}
# val_classes = {col: df[col].values[val_indices] for col in possible_col}
train_data = torch.stack([item[0] for item in train_dataset]).to(device)
print(train_data.shape)
from EncDec.encdiff import EncDiff, create_encdiff_model
import json 

config_path = "../Visualization/paths/encdiff/config.json"
with open(config_path, 'r') as f:
        config = json.load(f)
model, _ = create_encdiff_model(
        config=config,
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        device=device
    )
# print(train_data.shape)
# for item in dataset:
#     print(item[0].shape)  # This should print (1, 64, 64, 64) if the channel dimension is present
#     break

# model = EnhancedVAE(latent_channels=256)
# checkpoint_path = "../Visualization/paths/Echeckpoint_epoch_10000.pth"

# # Load the checkpoint
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Adjust device if you're using CUDA

# # Load the state dict into the model
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
# feature, channels= model.visualize(dataset)
# print(feature.shape)
# print(feature[0, 0, :, :, :])
# print(len(channels))
# print(channels[0].shape)
# for key, values in val_classes.items():
#     print(f"{key}: {values[:10]}")
# import plotly.graph_objects as go
# # Create a plot with multiple data traces
# fig = go.Figure()

# # Add a trace for variable 1
# fig.add_trace(go.Scatter(x=[1, 2, 3], y=[10, 15, 13], mode='lines', name='Variable 1'))

# # Add a trace for variable 2
# fig.add_trace(go.Scatter(x=[1, 2, 3], y=[16, 5, 11], mode='lines', name='Variable 2'))

# # Add a trace for variable 3
# fig.add_trace(go.Scatter(x=[1, 2, 3], y=[9, 12, 8], mode='lines', name='Variable 3'))

# # Show the interactive plot
# fig.show()

# Load checkpoint if provided
# checkpoint_path = "../Visualization/paths/encdiff_best_model.pth"
# checkpoint = torch.load(checkpoint_path, map_location=device)

# model.load_state_dict(checkpoint)
# model = model.to(device)
# model.eval()

# with torch.no_grad():
#     features = model.encode_image(train_data.to(device))

# print(features.shape)

import psutil
import os

def print_memory_usage():
    """Print current memory usage on Mac"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print("\nMemory Usage:")
    print(f"RSS (Resident Set Size): {memory_info.rss / 1024**2:.1f} MB")
    print(f"VMS (Virtual Memory Size): {memory_info.vms / 1024**2:.1f} MB")
    print(f"System Memory Usage: {psutil.virtual_memory().percent}%")
    print(f"Available System Memory: {psutil.virtual_memory().available / 1024**2:.1f} MB")

def estimate_batch_memory(dataset, batch_size):
    """Estimate memory requirements for processing"""
    sample_data = dataset[0][0]
    sample_size = sample_data.numel() * sample_data.element_size()  # bytes
    batch_memory = sample_size * batch_size / 1024**2  # MB
    
    print("\nMemory Estimates:")
    print(f"Single sample size: {sample_size / 1024**2:.1f} MB")
    print(f"Batch size: {batch_size}")
    print(f"Estimated batch memory: {batch_memory:.1f} MB")
    
    return batch_memory

# At start of script
print_memory_usage()

# Before loading dataset
estimate_batch_memory(dataset, batch_size=32)