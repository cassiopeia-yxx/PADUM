"""Loss function utilities for the PADUM project."""
import torch

def get_loss_function(loss_type, **kwargs):
    """Get loss function based on type.
    
    Args:
        loss_type (str): Type of loss ('l1', 'l2', 'ssim', etc.).
        \**kwargs: Additional arguments for loss initialization.
    
    Returns:
        torch.nn.Module: Configured loss function.
    """
    if loss_type == 'l1':
        return L1Loss(**kwargs)
    elif loss_type == 'l2':
        return L2Loss(**kwargs)
    elif loss_type == 'ssim':
        return SSIMLoss(**kwargs)
    else:
        raise ValueError(f'Unsupported loss type: {loss_type}')