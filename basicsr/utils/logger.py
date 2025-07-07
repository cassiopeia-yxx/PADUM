"""Logging utility for the PADUM project."""
import logging
import os
from datetime import datetime

def get_root_logger(logger_name='PADUM', file_name=None):
    """Get the root logger.
    
    Args:
        logger_name (str): Name of the logger.
        file_name (str, optional): If specified, save log to this file.
    
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if file_name:
        log_dir = os.path.join('experiments', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, file_name))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger