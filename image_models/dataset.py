#############################################
# dataset class
#############################################

# from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class KneeDataset(Dataset):

    def __init__(self, input_df, root_dir, transform=None):

        self.data_df = input_df
        self.root_dir = root_dir
        self.transform = transform
        self.labels = np.where(self.data_df.Label == 'Normal', 0, 1)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # image
        fnames = self.data_df.FileName
        img_fname = fnames[idx].split('.')[0] + '.png'
        img_name = os.path.join(self.root_dir, img_fname)
        data = Image.open(img_name).convert('RGB')
        # repeat the row to 3D

        if self.transform:
            data = self.transform(data)

        target = torch.tensor(self.labels[idx], dtype=torch.long)

        return data, target


class KneeDatasetMultiClass(Dataset):

    def __init__(self, input_df, root_dir, aug=False, noise=None, transform=None):
        self.data_df = input_df
        self.root_dir = root_dir
        self.transform = transform
        self.aug = aug
        if noise == None:
            self.labels = self.data_df.TriLabels.values
        else:
            self.labels = self.data_df.TriLabels_noise.values

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # image
        if self.aug == True:
            fnames = self.data_df.AugFName
        else:
            fnames = self.data_df.FileName
        img_fname = fnames[idx].split('.')[0] + '.png'
        img_name = os.path.join(self.root_dir, img_fname)
        data = Image.open(img_name).convert('RGB')
        if self.transform:
            data = self.transform(data)

        target = torch.tensor(self.labels[idx], dtype=torch.long)

        return data, target


