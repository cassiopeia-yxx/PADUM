"""Image utility functions for the PADUM project."""
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch tensor to numpy image.
    
    Args:
        tensor (torch.Tensor): Input tensor.
        out_type (np.dtype): Output data type.
        min_max (tuple): Min and max values for normalization.
    
    Returns:
        np.ndarray: Converted image.
    """
    # Implementation details...
    return img