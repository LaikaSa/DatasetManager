import logging
import datetime
import os

def setup_logger(debug_mode=False):
    # Get the logger
    logger = logging.getLogger('DatasetManager')
    
    # If logger already has handlers, remove them (to allow changing debug mode)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Set base level based on debug mode
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Create logs directory in the project root
    root_dir = os.path.dirname(os.path.dirname(__file__))
    logs_dir = os.path.join(root_dir, 'logs')
    
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    f_handler = logging.FileHandler(os.path.join(logs_dir, f'log_{timestamp}.txt'))
    f_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Create formatters
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    c_formatter = logging.Formatter(log_format, datefmt='%H:%M:%S')
    f_formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    c_handler.setFormatter(c_formatter)
    f_handler.setFormatter(f_formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger