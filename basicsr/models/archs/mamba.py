"""Mamba-based architecture for image deraining."""

import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    """Implements the Mamba block for sequence modeling.
    
    Args:
        dim (int): Feature dimension of the input.
        d_state (int, optional): State dimension in Mamba. Defaults to 16.
        d_conv (int, optional): Convolution kernel size. Defaults to 4.
        expand (int, optional): Expansion factor for the inner dimension. Defaults to 2.
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension
            d_state=d_state,  # SSM state dimension
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Inner dimension scale
        )

    def forward(self, x):
        """Forward pass through Mamba block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D), where B is batch size,
                              L is sequence length, and D is feature dimension.
        Returns:
            torch.Tensor: Output tensor with same shape as input.
        """
        B, L, D = x.shape
        x = self.norm(x)
        # Reshape to fit Mamba's expected input: (B, D, L)
        x = x.permute(0, 2, 1).contiguous()
        x = self.mamba(x)
        # Reshape back to (B, L, D)
        return x.permute(0, 2, 1).contiguous()