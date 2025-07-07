"""Miscellaneous utility functions for the PADUM project."""
import os
import random
import numpy as np
import torch

def set_random_seed(seed):
    """Set random seed for reproducibility.
    
    Args:
        seed (int): Seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_exp_dirs(opt):
    """Create experiment directories based on options.
    
    Args:
        opt (dict): Configuration dictionary containing experiment settings.
    """
    path = opt['path']['experiments_root']
    os.makedirs(path, exist_ok=True)
    return path