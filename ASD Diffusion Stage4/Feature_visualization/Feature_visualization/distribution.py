import numpy as np
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import sys
from torch.utils.data import random_split

torch.manual_seed(42) 
np.random.seed(42) 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])
from my_datasets.toy_dataset_processed import MRIToyDataset, load_abcd_dataset, BrainDataset

# df = pd.read_csv('../ToyDataProcessedDataset/subject_info.csv')
# possible_col = ["SEX", "AGE_AT_SCAN", "DX_GROUP", "HANDEDNESS_CATEGORY", "FIQ"]

# Split the dataset into training and validation sets
# dataset = MRIToyDataset()
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# train_indices = train_dataset.indices
# val_indices = val_dataset.indices

# # Create separate dictionaries for the classes in the training and validation sets
# train_classes = {col: df[col].values[train_indices] for col in possible_col}
# val_classes = {col: df[col].values[val_indices] for col in possible_col}

# Load dataset and CSV
dataset = load_abcd_dataset()
df = pd.read_csv('../ToyDataProcessedDataset/processed_abcd_ksad01.csv', low_memory=False)
possible_col = ["sex", "interview_age", "label", "ksads_14_402_p", "ksads_14_405_p", "ksads_14_396_p", "ksads_14_400_p", "ksads_14_398_p", "ksads_14_397_p", "ksads_14_404_p", "ksads_14_406_p", "ksads_14_401_p", "ksads_14_403_p", "ksads_10_869_p", "ksads_1_840_p", "ksads_15_901_p"]  

# Split using random_split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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

# Match with CSV using subject IDs
train_df = pd.DataFrame([df[df['subjectkey'] == id].iloc[0] for id in train_subject_ids])
val_df = pd.DataFrame([df[df['subjectkey'] == id].iloc[0] for id in val_subject_ids])

# Create class dictionaries
train_classes = {col: train_df[col].values for col in possible_col}
val_classes = {col: val_df[col].values for col in possible_col}
all_classes = {col: df[col].values for col in possible_col}


# def plot_class_distributions(classes, title_prefix, output_dir='../Visualization'):
#     os.makedirs(output_dir, exist_ok=True)
#     for class_name, values in classes.items():
#         plt.figure(figsize=(10, 6))
#         # Check if the class needs a specific order, like "HANDEDNESS_CATEGORY"
#         if class_name == "HANDEDNESS_CATEGORY":
#             category_order = ['R', 'L', 'Ambi']  # Define the desired order for HANDEDNESS_CATEGORY
#             plt.hist(values, bins=30, alpha=0.7, color='blue', edgecolor='black')
#             plt.xticks(category_order)  # Set the tick positions and labels
#         else:
#             # For other classes, plot without a specific order
#             plt.hist(values, bins=30, alpha=0.7, color='blue', edgecolor='black')
#             plt.xlabel(class_name)
#         plt.title(f'{title_prefix} Distribution of {class_name}')
#         plt.ylabel('Frequency')
#         # plt.show()

#         # Save the plot
#         file_name = f'{title_prefix}_Distribution_of_{class_name}.png'
#         file_path = os.path.join(output_dir, file_name)
#         plt.savefig(file_path, format='png', dpi=300)
#         plt.close()
#         print(f"Plot saved as {file_path}")

# # Plot class distributions for the training set
# plot_class_distributions(train_classes, title_prefix='Training', output_dir='../Visualization/Training_Distribution')

# # Plot class distributions for the validation set
# plot_class_distributions(val_classes, title_prefix='Validation', output_dir='../Visualization/Validation_Distribution')

def plot_class_distributions(classes, title_prefix, output_dir='../Visualization'):
    os.makedirs(output_dir, exist_ok=True)
    for class_name, values in classes.items():
        plt.figure(figsize=(10, 6))

        if class_name == "interview_age":
            plt.hist(values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        else:
            unique_vals, counts = np.unique(values, return_counts=True)
            plt.bar(unique_vals, counts)
            
        plt.title(f'{title_prefix} Distribution of {class_name}')
        plt.xlabel(class_name)
        plt.ylabel('Frequency')
        
        file_name = f'{title_prefix}_Distribution_of_{class_name}.png'
        file_path = os.path.join(output_dir, file_name)
        plt.savefig(file_path, format='png', dpi=300)
        plt.close()
        print(f"Plot saved as {file_path}")

# Plot distributions
plot_class_distributions(all_classes, "Full", "../Visualization/ABCD_Full_Distribution")
plot_class_distributions(train_classes, "Training", "../Visualization/ABCD_Training_Distribution")
plot_class_distributions(val_classes, "Validation", "../Visualization/ABCD_Validation_Distribution")