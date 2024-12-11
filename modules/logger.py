import logging
import datetime
import os

def setup_logger():
    # Get the logger
    logger = logging.getLogger('ImageApp')
    
    # Only setup handlers if they haven't been set up already
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create handlers
        # Console handler
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)

        # File handler with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        f_handler = logging.FileHandler(f'logs/log_{timestamp}.txt')
        f_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        c_formatter = logging.Formatter(log_format, datefmt='%H:%M:%S')
        f_formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        
        c_handler.setFormatter(c_formatter)
        f_handler.setFormatter(f_formatter)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger