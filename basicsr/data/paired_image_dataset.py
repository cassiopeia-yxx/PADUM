"""Paired image dataset for image restoration tasks."""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image

class PairedImageDataset(Dataset):
    """Paired image dataset for image restoration tasks.

    Args:
        data_root (str): Path to the root directory containing input and target images.
        split (str, optional): 'train' or 'val'. Defaults to 'train'.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, data_root, split='train', transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        
        # Assume specific folder structure
        self.input_dir = self.data_root / 'input'
        self.target_dir = self.data_root / 'target'
        
        if not (self.input_dir.exists() and self.target_dir.exists()):
            raise ValueError(f"Missing directories: {self.input_dir} or {self.target_dir}")

        # Load image file names
        self.image_files = [f for f in os.listdir(self.input_dir) if f.endswith(('png', '.jpg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Fetch a single pair of images."""
        img_name = self.image_files[idx]
        input_path = str(self.input_dir / img_name)
        target_path = str(self.target_dir / img_name)

        # Read images
        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        # Apply transformations
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        else:
            input_img = to_tensor(input_img)
            target_img = to_tensor(target_img)

        return {'input': input_img, 'target': target_img, 'input_path': input_path, 'target_path': target_path}