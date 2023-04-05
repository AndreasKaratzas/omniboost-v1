
import os
import torch
import numpy as np

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Custom dataset class for loading data from a npy file.
    """

    def __init__(self, root_dir: str, use_gpu: bool):

        # load data
        mapping = list(
            sorted(os.listdir(os.path.join(root_dir, "mapping"))))
        workload = list(
            sorted(os.listdir(os.path.join(root_dir, "workload"))))

        # convert to torch tensors
        self.X = [torch.from_numpy(
            np.load(os.path.join(root_dir, "mapping", f))) for f in mapping]
        self.Y = [torch.from_numpy(
            np.load(os.path.join(root_dir, "workload", f))) for f in workload]
        
        # load data on the GPU
        if use_gpu:
            self.X = [x.cuda() for x in self.X]
            self.Y = [y.cuda() for y in self.Y]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.Y)
