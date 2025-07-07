"""Implementation of the Pixel Adaptive Deep Unfolding Network (PADUM) with State Space Model."""

import torch
import torch.nn as nn
from basicsr.models.archs.mamba import MambaBlock

class PADM(nn.Module):
    """Pixel Adaptive Deep Unfolding Network for image deraining.

    Args:
        num_in_ch (int): Number of input channels (default: 3 for RGB).
        num_out_ch (int): Number of output channels (default: 3).
        num_feat (int): Number of base feature maps in the network.
        num_blocks (int): Number of residual blocks.
        mamba_layers (list): List specifying which layers use Mamba blocks.
        upscale (int): Upscaling factor (not used if no upscaling is needed).
        bias (bool): Whether to include bias in convolutional layers.
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_blocks=16, 
                 mamba_layers=[4, 8], upscale=1, bias=False):
        super(PADM, self).__init__()
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1)
        
        # Body of the network - stacked residual blocks
        self.body = nn.ModuleList()
        for i in range(num_blocks):
            if i in mamba_layers:
                # Use Mamba block at specified layers
                self.body.append(MambaResBlock(num_feat))
            else:
                # Use standard residual block
                self.body.append(ResidualBlock(num_feat, bias=bias))
        
        # Final reconstruction layer
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, kernel_size=3, padding=1)
        
        # Initialize network
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights using a normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the entire network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where
                              B = batch size
                              C = number of channels
                              H, W = image height and width
        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        # Extract initial features
        feat = self.conv_first(x)
        
        # Pass through body blocks
        for block in self.body:
            feat = block(feat)
        
        # Final reconstruction
        out = self.conv_last(feat)
        
        # Residual connection
        return x + out

class ResidualBlock(nn.Module):
    """Standard residual block with two convolutional layers."""
    def __init__(self, nf, bias=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.bias = bias
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return identity + out

class MambaResBlock(nn.Module):
    """Residual block containing a Mamba layer for long-range dependency modeling."""
    def __init__(self, nf, d_state=16, d_conv=4, expand=2):
        super(MambaResBlock, self).__init__()
        self.norm = nn.LayerNorm(nf)
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=1)
        self.mamba = MambaBlock(
            dim=nf,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=1)
    
    def forward(self, x):
        """Mamba-enhanced residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        identity = x
        
        # First convolution
        x = self.conv1(x)
        
        # Reshape for Mamba: flatten spatial dimensions
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()  # Shape: (B, H*W, C)
        
        # Mamba layer
        x = self.mamba(x)
        
        # Reshape back to original
        x = x.permute(0, 2, 1).view(B, C, H, W)
        
        # Second convolution
        x = self.conv2(x)
        
        return identity + x