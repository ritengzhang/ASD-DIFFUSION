import os
import random
import string
import ants
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class MRIToyDataset(Dataset):
    def __init__(self, root_dir='../Data4YuansGroup/Data/', 
                 existing_load_path='../ToyDataProcessedDataset/mri_toy_dataset.pth',
                 ):
        self.root_dir = root_dir
        self.existing_load_path = existing_load_path
        
        if os.path.exists(existing_load_path):
            print(f"Loading existing dataset from {existing_load_path}")
            loaded_data = torch.load(existing_load_path)
            self.data = loaded_data['data']
            self.text = loaded_data['text']
        else:
            print("Creating new dataset")
            self.create_dataset()

    def create_dataset(self, max_display=1, image_size=64):
        self.data = []
        self.text = []
        displayed = 0
        
        # Find one reference image for alignment (MNI aligned image)
        reference_template_path = self.find_reference_template()
        if not reference_template_path:
            raise FileNotFoundError("No reference template found for alignment.")
        
        # Load the reference template
        reference_template = ants.image_read(reference_template_path)

        # Iterate through the data folder
        for subject_folder in os.listdir(self.root_dir):
            subject_path = os.path.join(self.root_dir, subject_folder, 'anat')
            if os.path.isdir(subject_path):
                mni_file = [f for f in os.listdir(subject_path) if f.endswith('_desc-preproc_T1w.nii.gz') and 'space-MNI152NLin2009cAsym' not in f]
                brain_mask_file = [f for f in os.listdir(subject_path) if f.endswith('_desc-brain_mask.nii.gz')]

                if mni_file and brain_mask_file:
                    mri_path = os.path.join(subject_path, mni_file[0])
                    mask_path = os.path.join(subject_path, brain_mask_file[0])
                    
                    # Process the image
                    img_np = self.process_image(mri_path, mask_path, reference_template, image_size, displayed < max_display)
                    
                    # Add to dataset list
                    self.data.append(img_np[np.newaxis, ...])  # Add channel dimension
                    self.text.append(''.join(random.choices(string.ascii_lowercase + ' ', k=10)))

                    displayed += 1

        self.data = np.array(self.data, dtype=np.float32)

    def find_reference_template(self):
        for subject_folder in os.listdir(self.root_dir):
            subject_path = os.path.join(self.root_dir, subject_folder, 'anat')
            if os.path.isdir(subject_path):
                reference_files = [f for f in os.listdir(subject_path) if 'space-MNI152NLin2009cAsym' in f and f.endswith('_desc-preproc_T1w.nii.gz')]
                if reference_files:
                    return os.path.join(subject_path, reference_files[0])
        return None

    def process_image(self, mri_path, mask_path, reference_template, image_size, display=False):
        # Load the MRI image and brain mask
        anat = ants.image_read(mri_path)
        brain_mask = ants.image_read(mask_path)
        
        if display:
            self.display_image(anat, "Original Image (Non-Aligned)")

        # Extract brain using mask
        brain = self.extract_brain(anat, brain_mask)
        
        if display:
            self.display_image(brain, "Brain Extracted")

        # Align the extracted brain to the reference template
        aligned_brain = ants.registration(
            fixed=reference_template,
            moving=brain,
            type_of_transform='SyN' #'Affine' # Q1： which one to use?
        )['warpedmovout'] 

        if display:
            self.display_image(aligned_brain, "Aligned Brain")

        # Resample to a consistent voxel size
        resampled_img = ants.resample_image( 
            image=aligned_brain, 
            resample_params=(image_size, image_size, image_size), 
            use_voxels=True, 
            interp_type=1 # Q2: which one to use: Nearest neighbor interpolation, # interp_type=1,# Interpolation: one of 0 (linear), 1 (nearest neighbor), 2 (gaussian), 3 (windowed sinc), 4 (bspline)
        )                 # Q3: resamping, does 128 128 128 resampling make sense, or is 128 128 image resoluation, and the thrid one is a channel where we can't change?
                        
        
        if display:
            self.display_image(resampled_img, "Resampled Image")

        # Convert to numpy array and normalize
        img_np = resampled_img.numpy()
        img_np = (img_np - np.mean(img_np)) / np.std(img_np)
        
        return img_np

    def extract_brain(self, anat, brain_mask):
        anat_arr = anat.numpy()
        brain_mask_arr = brain_mask.numpy()
        brain_arr = np.zeros(anat_arr.shape, dtype=anat_arr.dtype)
        brain_values = anat_arr[brain_mask_arr == 1]
        brain_arr[brain_mask_arr == 1] = brain_values
        return ants.from_numpy(brain_arr, origin=anat.origin, spacing=anat.spacing, direction=anat.direction)

    def display_image(self, image, title):
        print(f"{title} shape: {image.shape}")
        image.plot_ortho(flat=True, title=title, xyz_lines=False, orient_labels=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.text[idx]

    def save(self, save_path):
        save_data = {
            'data': self.data,
            'text': self.text
        }
        torch.save(save_data, save_path)
        print(f"Dataset saved to {save_path}")

# Usage example
if __name__ == "__main__":
    dataset = MRIToyDataset(root_dir='../Data4YuansGroup/Data/', 
                            existing_load_path='../ToyDataProcessedDataset/mri_toy_dataset.pth')
    
    # Save the dataset if it was newly created
    if not os.path.exists('../ToyDataProcessedDataset/mri_toy_dataset.pth'):
        dataset.save('../ToyDataProcessedDataset/mri_toy_dataset.pth')

    print(f"Dataset size: {len(dataset)}")
    print(f"First item shape: {dataset[0][0].shape}")