"""
Simple logging setup for F1 model
"""
import logging
from datetime import datetime

def setup_logger(name="f1_model"):
    """
    Creates a logger that writes to both console and file
    
    Why we do this:
    - Console output: See what's happening in real-time
    - File output: Keep permanent record for debugging later
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Capture INFO and above
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Console handler - shows logs in terminal
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    
    # File handler - saves logs to file
    timestamp = datetime.now().strftime('%Y%m%d')
    file_handler = logging.FileHandler(f'logs/f1_model_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)  # Save everything to file
    
    # Format: "2025-12-31 10:23:15 - INFO - Training started"
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    return logger