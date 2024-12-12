import logging
import datetime
import os

def setup_logger():
    # Get the logger
    logger = logging.getLogger('ImageApp')
    
    # If logger already has handlers, return it (prevent duplicate handlers)
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)

    # Create logs directory in the project root
    root_dir = os.path.dirname(os.path.dirname(__file__))
    logs_dir = os.path.join(root_dir, 'logs')
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Create handlers
    # Console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)

    # File handler with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    f_handler = logging.FileHandler(os.path.join(logs_dir, f'log_{timestamp}.txt'))
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