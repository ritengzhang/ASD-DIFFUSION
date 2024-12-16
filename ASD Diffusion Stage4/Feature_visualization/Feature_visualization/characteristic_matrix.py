import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt
import os
from torch.utils.data import random_split
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])
from my_datasets.toy_dataset_processed import MRIToyDataset

torch.manual_seed(42)
np.random.seed(42)


def calculate_separate_similarity_matrices(data, columns):
    """
    Calculate separate dissimilarity matrices for each numerical characteristic.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing participant data
    columns : list
        List of column names to analyze
        
    Returns:
    --------
    dict
        Dictionary containing dissimilarity matrices for each characteristic
    """
    
    similarity_matrices = {}
    
    for col in columns:
        # Extract and reshape data
        col_data = data[col].values.reshape(-1, 1)
        
        # Handle missing values
        if np.any(np.isnan(col_data)):
            print(f"Warning: Found missing values in {col}. These will be handled by pairwise deletion.")
        
        # # Normalize the data
        # col_mean = np.nanmean(col_data)
        # col_std = np.nanstd(col_data)
        # normalized_data = (col_data - col_mean) / col_std
        
        # Calculate pairwise Euclidean distances
        distances = pdist(col_data, metric='euclidean')
        print(distances[:10])
        # Convert distance matrix to square form
        max_distance = distances.max()
        normalized_distances = distances / max_distance
        print("Normalized distances:", normalized_distances[:10])
        
        # Convert to square matrix
        matrix = 1 - squareform(normalized_distances)
        
        # Normalize to [0,1] range for consistent visualization
        # if np.max(matrix) > 0:  # Avoid division by zero
        #     matrix = matrix / np.max(matrix)
            
        similarity_matrices[col] = matrix
    # print(len(similarity_matrices))
    # print(len(similarity_matrices['SEX']))
    # sex = similarity_matrices['SEX']
    # print(sex.shape)
    
    return similarity_matrices

def plot_similarity_matrix(matrix, title, output_dir=None):
    """
    Plot dissimilarity matrix using seaborn heatmap.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Square dissimilarity matrix
    title : str
        Title for the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, 
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

def plot_all_matrices(matrices, title_prefix = 'Full'):
    """
    Plot all dissimilarity matrices.
    
    Parameters:
    -----------
    matrices : dict
        Dictionary of dissimilarity matrices
    title_prefix : str
        Prefix for the plot titles
    """
    for characteristic, matrix in matrices.items():
        plot_similarity_matrix(matrix, f"{title_prefix} Similarity Matrix - {characteristic}", output_dir='../Visualization')

# Example usage
def run_analysis(df, possible_col, indices=None):
    """
    Run complete analysis and visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    possible_col : list
        List of columns to analyze
    indices : list
        List of indices to filter the DataFrame
    """
    # Filter the DataFrame based on the provided indices
    if indices is not None:
        df_filtered = df.iloc[indices]
    else:
        df_filtered = df
    
    # Filter to only numerical columns
    numerical_cols = df_filtered[possible_col].select_dtypes(include=[np.number]).columns.tolist()
    print(f"Analyzing numerical columns: {numerical_cols}")
    
    # Calculate matrices
    matrices = calculate_separate_similarity_matrices(df_filtered, numerical_cols)
    
    return matrices


if __name__ == "__main__":
    df = pd.read_csv('../ToyDataProcessedDataset/subject_info.csv')
    possible_col = ["SEX", "AGE_AT_SCAN", "DX_GROUP", "HANDEDNESS_SCORES", "FIQ"]
    matrices = run_analysis(df, possible_col)

    # Plot all matrices
    plot_all_matrices(matrices)

    # Split the dataset into training and validation sets
    dataset = MRIToyDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    
    # Run analysis for training data
    matrices_train = run_analysis(df, possible_col, train_indices)
    # Plot results for training data
    plot_all_matrices(matrices_train, title_prefix="Train")
    
    # Run analysis for validation data
    matrices_val = run_analysis(df, possible_col, val_indices)
    # Plot results for validation data
    plot_all_matrices(matrices_val, title_prefix="Validation")
