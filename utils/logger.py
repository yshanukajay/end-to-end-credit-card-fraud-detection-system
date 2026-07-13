import os
import logging
from logging.handlers import RotatingFileHandler

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger that writes logs to both the console
    and a module-specific file under the logs/ directory in the project root.
    
    Structure:
    logs/
      <module_name>/
        <module_name>.log
    """
    # Extract clean module name (e.g., 'src.data_ingestion' -> 'data_ingestion')
    clean_name = name.split('.')[-1]
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if get_logger is called multiple times for the same logger name
    if not logger.handlers:
        # Define formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 1. Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 2. File Handler (with dynamic path creation)
        try:
            # Find the repository root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(current_dir)  # parent of 'src'
            
            log_dir = os.path.join(root_dir, 'logs', clean_name)
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f"{clean_name}.log")
            
            # Use RotatingFileHandler to avoid files growing infinitely
            file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback if logs folder creation or write fails
            print(f"Warning: Failed to set up file logging for {clean_name}: {e}")
            
    return logger
