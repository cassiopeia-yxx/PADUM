"""Logging utilities for the PADUM project."""
import logging
import os
import sys
from datetime import datetime

def setup_logger(name='PADUM', log_path=None, level=logging.INFO):
    """Set up a logger with specified name and log file path.
    
    Args:
        name (str): Name of the logger.
        log_path (str): Path to the log file. If None, logs will not be written to file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logging in root logger
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    # Create file handler if log_path is provided
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
    
    return logger

def get_root_logger():
    """Get the root logger instance.
    
    Returns:
        logging.Logger: Root logger instance.
    """
    return logging.getLogger('PADUM')