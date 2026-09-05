import logging

def setup_logger(debug_mode=False):
    # Get the logger
    logger = logging.getLogger('DatasetManager')
    
    # If logger already has handlers, remove them (to allow changing debug mode)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Set base level based on debug mode
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Create console handler only (no file logging)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Create formatter
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    c_formatter = logging.Formatter(log_format, datefmt='%H:%M:%S')
    c_handler.setFormatter(c_formatter)

    logger.addHandler(c_handler)

    return logger