import os
import random
import string
import ants
import torch
from torch.utils.data import Dataset
import numpy as np

class MRIToyDataset(Dataset):
    def __init__(self, root_dir='../Data4YuansGroup/Data/', image_size=128, max_display=5): 
        self.data = []
        self.text = []
        displayed = 0
        
        # Find one reference image for alignment (MNI aligned image)
        reference_template_path = None
        for subject_folder in os.listdir(root_dir):
            subject_path = os.path.join(root_dir, subject_folder, 'anat')
            if os.path.isdir(subject_path):
                reference_files = [f for f in os.listdir(subject_path) if 'space-MNI152NLin2009cAsym' in f and f.endswith('_desc-preproc_T1w.nii.gz')]
                if reference_files:
                    reference_template_path = os.path.join(subject_path, reference_files[0])
                    break

        if not reference_template_path:
            raise FileNotFoundError("No reference template found for alignment.")
        
        # Load the reference template
        reference_template = ants.image_read(reference_template_path)

        # Iterate through the data folder
        for subject_folder in os.listdir(root_dir):
            subject_path = os.path.join(root_dir, subject_folder, 'anat')
            if os.path.isdir(subject_path):
                mni_file = [f for f in os.listdir(subject_path) if f.endswith('_desc-preproc_T1w.nii.gz') and 'space-MNI152NLin2009cAsym' not in f]
                brain_mask_file = [f for f in os.listdir(subject_path) if f.endswith('_desc-brain_mask.nii.gz')]

                if mni_file and brain_mask_file:
                    mri_path = os.path.join(subject_path, mni_file[0])
                    mask_path = os.path.join(subject_path, brain_mask_file[0])
                    
                    # Load the MRI image and brain mask
                    anat = ants.image_read(mri_path)
                    brain_mask = ants.image_read(mask_path)
                    
                    if displayed < max_display:
                        # Display original image using plot_ortho
                        print(f"Original image shape: {anat.shape}")
                        anat.plot_ortho(flat=True, title='Original Image (Non-Aligned)', xyz_lines=False, orient_labels=False)

                        # Extract brain using mask (following the slow(er) method from the tutorial)
                        anat_arr = anat.numpy()  # Convert anat to numpy array
                        brain_mask_arr = brain_mask.numpy()  # Convert brain mask to numpy array

                        # Initialize a new array to hold extracted brain values
                        brain_arr = np.zeros(anat_arr.shape, dtype=anat_arr.dtype)

                        # Extract brain values using the mask
                        brain_values = anat_arr[brain_mask_arr == 1]
                        brain_arr[brain_mask_arr == 1] = brain_values

                        # Create a new ANTsImage using the brain array
                        brain = ants.from_numpy(brain_arr, origin=anat.origin, spacing=anat.spacing, direction=anat.direction)

                        # Display brain extracted image using plot_ortho
                        brain.plot_ortho(flat=True, title='Brain Extracted', xyz_lines=False, orient_labels=False)

                        # Align the extracted brain to the reference template
                        aligned_brain = ants.registration(
                            fixed=reference_template,
                            moving=brain,
                            type_of_transform='SyN' #'Affine' # Q1： which one to use?
                        )['warpedmovout']

                        # Display aligned brain using plot_ortho
                        aligned_brain.plot_ortho(flat=True, title='Aligned Brain', xyz_lines=False, orient_labels=False)

                        # Resample to a consistent voxel size (following tutorial approach)
                        resampled_img = ants.resample_image(
                            image=aligned_brain, 
                            resample_params=(image_size, image_size, image_size), 
                            use_voxels=True, 
                            interp_type=1  # Q2: which one to use: Nearest neighbor interpolation, # interp_type=1,# Interpolation: one of 0 (linear), 1 (nearest neighbor), 2 (gaussian), 3 (windowed sinc), 4 (bspline)
                        )                  # Q3: resamping, does 128 128 128 resampling make sense, or is 128 128 image resoluation, and the thrid one is a channel where we can't change?
                        
                        # Display resampled brain using plot_ortho
                        print(f"Resampled image shape: {resampled_img.shape}")
                        resampled_img.plot_ortho(flat=True, title='Resampled Image', xyz_lines=False, orient_labels=False)

                        displayed += 1

                    # Convert to numpy array and normalize
                    img_np = resampled_img.numpy()
                    img_np = (img_np - np.mean(img_np)) / np.std(img_np)
                    
                    # Add to dataset list
                    self.data.append(img_np[np.newaxis, ...])  # Add channel dimension
                    self.text.append(''.join(random.choices(string.ascii_lowercase + ' ', k=10)))

        self.data = np.array(self.data, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.text[idx]

# Create dataset with option to control number of displays
mri_toy_dataset = MRIToyDataset(max_display=1)

# Save the dataset
save_path = '../ToyDataProcessedDataset/mri_toy_dataset.pth'
torch.save(mri_toy_dataset, save_path)

print(f"Dataset saved to {save_path}")
