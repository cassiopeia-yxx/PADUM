"""Option parsing utilities for the PADUM project."""
import yaml
import argparse
from easydict import EasyDict as edict

def parse_options():
    """Parse command line arguments and configuration file."""
    parser = argparse.ArgumentParser(description='Image Deraining with Pixel Adaptive Deep Unfolding Network')
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='Job launcher.')
    parser.add_argument('--auto_resume', action='store_true', help='Auto resume from latest checkpoint.')
    
    args = parser.parse_args()
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    
    opt = edict(opt)
    return opt