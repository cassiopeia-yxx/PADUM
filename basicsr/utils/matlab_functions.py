"""MATLAB compatibility functions for the PADUM project."""
import numpy as np
from scipy.io import loadmat, savemat

def imresize(img, size_factor, method='bicubic'):
    """Resize image with MATLAB-like behavior.
    
    Args:
        img (np.ndarray): Input image.
        size_factor (float): Resize factor.
        method (str): Interpolation method.
    
    Returns:
        np.ndarray: Resized image.
    """
    # Implementation details...
    return resized_img