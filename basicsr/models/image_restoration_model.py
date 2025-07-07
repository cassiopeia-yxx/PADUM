"""Base class for image restoration models."""
import torch
import torch.nn as nn

class ImageRestorationModel(nn.Module):
    """Base class for image restoration models.
    
    This class provides a common interface and basic functionality for
    various image restoration tasks such as denoising, deraining, and dehazing.
    """
    
    def __init__(self, opt):
        """Initialize the model.
        
        Args:
            opt (dict): Configuration dictionary containing model settings.
        """
        super(ImageRestorationModel, self).__init__()
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model-specific initialization should be done in subclasses
        
    def feed_data(self, data):
        """Process input data.
        
        Args:
            data (dict): Dictionary containing input data.
        """
        pass
    
    def forward(self):
        """Forward pass through the network."""
        pass
    
    def optimize_parameters(self):
        """Update network weights."""
        pass
    
    def get_current_visuals(self):
        """Return visualization results."""
        pass
    
    def get_current_losses(self):
        """Return training/test losses."""
        pass
    
    def print_network(self):
        """Print network structure."""
        pass
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad for networks.
        
        Args:
            nets (network list): List of networks.
            requires_grad (bool): Whether the networks require gradients.
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad