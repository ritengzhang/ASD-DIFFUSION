# import torch
# import numpy as np
# from torch.utils.data import Dataset
# import random
# import string

# class MRIWithTextDataset(Dataset):
#     def __init__(self, num_samples=1000, image_size=16, text_length=10):
#         self.data = np.zeros((num_samples, 1, image_size, image_size, image_size))
#         self.text = []
#         for i in range(num_samples):
#             base_value = np.random.randint(0, 16)
#             self.data[i, 0] = base_value + np.random.normal(0, 0.1, (image_size, image_size, image_size))
#             self.text.append(''.join(random.choices(string.ascii_lowercase + ' ', k=text_length)))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return torch.tensor(self.data[idx], dtype=torch.float32), self.text[idx]

