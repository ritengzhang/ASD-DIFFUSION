import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import sys
import pandas as pd
import os
from torch.utils.data import DataLoader, random_split, Subset

torch.manual_seed(42)
np.random.seed(42)
# Add the parent directory (assuming `my_datasets` is in the same directory as `visualization.py`)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])
from my_datasets.toy_dataset_processed import MRIToyDataset, NPYDataset, CIFARDataset, load_abcd_dataset, BrainDataset
from EncDec.vae import SimpleVAE, EnhancedVAE, FlattenedVAE, vae_loss, create_encdiff_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_abcd_dataset()
print(dataset.subject_ids[:10])
# df = pd.read_csv('../ToyDataProcessedDataset/abcd_ksad01.csv', skiprows=[1], low_memory=False)
# description_row = pd.read_csv('../ToyDataProcessedDataset/abcd_ksad01.csv', skiprows=[0], nrows=1, low_memory=False)
# print(description_row)
# print(df.dtypes)
# print(df["collection_id"].values[:10])
# print(df["subjectkey"].values[:10])
# print(dataset.labels[:100])
# # Get unique subject IDs

# # Create a mapping from dataset_subjects to labels
# subject_to_label = {
#     subject_id[4:] if subject_id.startswith('sub-') else subject_id: int(label.item())
#     for subject_id, label in zip(dataset.subject_ids, dataset.labels)
# }

# # Standardize subject IDs in the CSV to match the dataset format
# df["standardized_subjectkey"] = df["subjectkey"].apply(lambda x: x.replace('_', '', 1))

# # Map the labels to the DataFrame
# df["label"] = df["standardized_subjectkey"].map(subject_to_label)
# df.dropna(subset=["label"], inplace=True)
# df["subjectkey"] = df["standardized_subjectkey"]
# # Drop the standardized column if no longer needed
# df.drop(columns=["standardized_subjectkey"], inplace=True)

# df = pd.concat([description_row, df], ignore_index=True)

# # Check the resulting DataFrame
# print(df.head())
# print(df["label"].value_counts())
# Save the cleaned DataFrame
# df.to_csv('processed_abcd_ksad01_des.csv', index=False)
# print("Processed DataFrame saved to 'processed_abcd_ksad01_des.csv'.")

# #check for missing ids
# dataset_subjects = set(subject_id[4:] if subject_id.startswith('sub-') else subject_id for subject_id in dataset.subject_ids)
# csv_subjects = set(subject_id.replace('_', '', 1) for subject_id in df['subjectkey'])

# # Find missing subjects
# missing_in_csv = dataset_subjects - csv_subjects
# missing_in_dataset = csv_subjects - dataset_subjects

# # Print statistics
# print(f"Dataset subjects: {len(dataset_subjects)}")
# print(f"CSV subjects: {len(csv_subjects)}")

# # no missing CSV data, 205 missing MRI data

# print(f"Subjects missing from CSV: {len(missing_in_csv)}")
# print(f"Subjects missing from dataset: {len(missing_in_dataset)}")

# if missing_in_csv:
#     print("\nFirst 5 subjects missing from CSV:")
#     print(list(missing_in_csv)[:5])


# df = pd.read_csv('../ToyDataProcessedDataset/processed_abcd_ksad01.csv', low_memory=False)
# print(df.dtypes)

# # Get unique subject IDs
# dataset_subjects = set(subject_id[4:] if subject_id.startswith('sub-') else subject_id for subject_id in dataset.subject_ids)
# csv_subjects = set(df['subjectkey'])

# # Find missing subjects
# missing_in_csv = dataset_subjects - csv_subjects
# missing_in_dataset = csv_subjects - dataset_subjects

# # Print statistics
# print(f"Dataset subjects: {len(dataset_subjects)}")
# print(f"CSV subjects: {len(csv_subjects)}")
# print(f"Subjects missing from CSV: {len(missing_in_csv)}")
# print(f"Subjects missing from dataset: {len(missing_in_dataset)}")

# if missing_in_csv:
#     print("\nFirst 5 subjects missing from CSV:")
#     print(list(missing_in_csv)[:5])

# # Calculate overlap percentage
# overlap = len(dataset_subjects.intersection(csv_subjects))
# overlap_pct = (overlap / len(dataset_subjects)) * 100
# print(f"\nOverlap percentage: {overlap_pct:.2f}%")

# print(df.head())
df = pd.read_csv('../ToyDataProcessedDataset/processed_abcd_ksad01.csv', low_memory=False)
possible_col = ["subjectkey", "sex", "interview_age", "label", "ksads_14_402_p"]  # Replace with your columns

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

print(train_classes["subjectkey"][:5])
print(val_classes["subjectkey"][:5])
print(len(train_df))
print(len(val_df))

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

train_data = torch.stack([item[0] for item in train_dataset]).to(device)
val_data = torch.stack([item[0] for item in val_dataset]).to(device)
print(train_data.shape)
print(val_data.shape)