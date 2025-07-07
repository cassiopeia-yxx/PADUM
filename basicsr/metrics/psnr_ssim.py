"""PSNR and SSIM calculation for image quality assessment."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2, crop_border=0):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        crop_border (int): Number of pixels to crop from the borders.
    
    Returns:
        float: PSNR value in dB.
    """
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    
    # Ensure images are in the same format
    if img1.dtype != img2.dtype:
        raise ValueError(f"Images must have the same dtype. Got {img1.dtype} and {img2.dtype}")
    
    # Convert to float64 for calculation
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0**2 / mse)

def calculate_ssim(img1, img2, crop_border=0):
    """Calculate SSIM (Structural Similarity Index) between two images.
    
    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        crop_border (int): Number of pixels to crop from the borders.
    
    Returns:
        float: SSIM value.
    """
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    
    # Ensure images are in the same format
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape. Got {img1.shape} and {img2.shape}")
    
    # Convert to grayscale if necessary
    if img1.ndim == 3 and img1.shape[2] == 3:
        img1 = np.dot(img1[..., :3], [0.2989, 0.5870, 0.1140])
        img2 = np.dot(img2[..., :3], [0.2989, 0.5870, 0.1140])
    
    return ssim(img1, img2, data_range=img2.max() - img2.min())