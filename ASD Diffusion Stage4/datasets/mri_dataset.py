# import torch
# import numpy as np
# from torch.utils.data import Dataset

# class MRIDataset(Dataset):
#     def __init__(self, num_samples=1000, image_size=16):
#         self.data = np.zeros((num_samples, 1, image_size, image_size, image_size))
#         for i in range(num_samples):
#             base_value = np.random.randint(0, 16)
#             self.data[i, 0] = base_value + np.random.normal(0, 0.1, (image_size, image_size, image_size))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return torch.tensor(self.data[idx], dtype=torch.float32)