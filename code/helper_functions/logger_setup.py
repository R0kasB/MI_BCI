import logging
import os

def setup_logger(name, log_file=None, level=logging.INFO):
    """Function to setup a logger for a specific module or class.
    
    Args:
        name (str): Name of the logger.
        log_file (str, optional): File path to log the messages. If None, logs only to the console.
        level (int): Logging level, e.g., logging.INFO, logging.ERROR.

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    if log_file:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create a file handler to log to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
    return logger
