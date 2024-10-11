import os
import random
import string
import ants
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

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
                        # Display original image dimensions
                        print(f"Original image shape: {anat.shape}")
                        
                        # Display original image slice
                        plt.figure(figsize=(15, 5))
                        plt.subplot(1, 4, 1)
                        plt.imshow(anat.numpy()[:, :, anat.shape[2]//2], cmap='gray')
                        plt.title('Original Image Slice (Non-Aligned)')
                        plt.axis('off')
                        
                        # Extract brain using mask (fast way from tutorial)
                        brain = anat * (brain_mask > 0)  # Multiply the image by the mask to zero out non-brain areas

                        # Display extracted brain slice
                        plt.subplot(1, 4, 2)
                        plt.imshow(brain.numpy()[:, :, brain.shape[2]//2], cmap='gray')
                        plt.title('Brain Extracted Slice')
                        plt.axis('off')
                        
                        # Align the extracted brain to the reference template
                        aligned_brain = ants.registration(
                            fixed=reference_template,
                            moving=brain,
                            type_of_transform='SyN'
                        )['warpedmovout']

                        # Display aligned brain slice
                        plt.subplot(1, 4, 3)
                        plt.imshow(aligned_brain.numpy()[:, :, reference_template.shape[2]//2], cmap='gray')
                        plt.title('Aligned Brain Slice')
                        plt.axis('off')
                        
                        # Resample to a consistent voxel size (following tutorial approach)
                        resampled_img = ants.resample_image(
                            image=aligned_brain, 
                            resample_params=(image_size, image_size, image_size), 
                            use_voxels=True, 
                            interp_type=1  # Nearest neighbor interpolation
                        )
                        
                        # Display resampled image dimensions
                        print(f"Resampled image shape: {resampled_img.shape}")
                        
                        # Display resampled image slice
                        plt.subplot(1, 4, 4)
                        plt.imshow(resampled_img.numpy()[:, :, resampled_img.shape[2]//2], cmap='gray')
                        plt.title('Resampled Image Slice')
                        plt.axis('off')
                        plt.show()

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
mri_toy_dataset = MRIToyDataset(max_display=5)

# Save the dataset
save_path = '../ToyDataProcessedDataset/mri_toy_dataset.pth'
torch.save(mri_toy_dataset, save_path)

print(f"Dataset saved to {save_path}")