"""Dataset for deraining with paired images."""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image

class DerainPairedDataset(Dataset):
    """Paired image dataset for deraining tasks.

    Args:
        data_root (str): Path to the root directory containing rain and clean images.
        split (str, optional): 'train' or 'val'. Defaults to 'train'.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, data_root, split='train', transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        
        # Assume specific folder structure
        self.rain_dir = self.data_root / 'input'
        self.clean_dir = self.data_root / 'target'
        
        if not (self.rain_dir.exists() and self.clean_dir.exists()):
            raise ValueError(f"Missing directories: {self.rain_dir} or {self.clean_dir}")

        # Load image file names
        self.image_files = [f for f in os.listdir(self.rain_dir) if f.endswith(('png', '.jpg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Fetch a single pair of images."""
        img_name = self.image_files[idx]
        rain_path = str(self.rain_dir / img_name)
        clean_path = str(self.clean_dir / img_name)

        # Read images
        rain_img = Image.open(rain_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')

        # Apply transformations
        if self.transform:
            rain_img = self.transform(rain_img)
            clean_img = self.transform(clean_img)
        else:
            rain_img = to_tensor(rain_img)
            clean_img = to_tensor(clean_img)

        return {'lq': rain_img, 'gt': clean_img, 'lq_path': rain_path, 'gt_path': clean_path}