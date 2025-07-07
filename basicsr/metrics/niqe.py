"""Natural Image Quality Evaluator (NIQE) for image quality assessment."""

import numpy as np
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray


def niqe(image, mu_kernel=7, sigma_kernel=7):
    """Calculate the NIQE score for an input image.
    
    Args:
        image (np.ndarray): Input image (RGB or grayscale).
        mu_kernel (int): Kernel size for local mean calculation.
        sigma_kernel (int): Kernel size for local standard deviation calculation.
    
    Returns:
        float: NIQE score indicating image quality (lower is better).
    """
    # Convert to grayscale if necessary
    if image.ndim == 3 and image.shape[2] == 3:
        image = rgb2gray(image)
    
    # Ensure image is in float64 format
    image = image.astype(np.float64)
    
    # Calculate local mean and standard deviation
    mu = gaussian_filter(image, sigma=mu_kernel/6.0, mode='nearest')
    sigma = gaussian_filter((image - mu)**2, sigma=sigma_kernel/6.0, mode='nearest')**0.5
    
    # Compute quality map
    quality_map = 1.0 / (1.0 + sigma)
    
    # Return average quality score
    return float(np.mean(quality_map))