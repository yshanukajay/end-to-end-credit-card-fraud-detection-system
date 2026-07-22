import os
import sys
import logging
from datetime import datetime
from typing import Optional

# Derive project root for dynamic path resolution
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)  # parent of 'utils'

# Global flag to track if logging has already been configured in the current process
_logging_initialized = False

def setup_logging(process_name: Optional[str] = None, force: bool = False) -> logging.Logger:
    """
    Configures the root logger to output to both console and a timestamped file:
    logs/<process_name>/<timestamp>.txt
    
    If process_name is not provided, it is automatically derived from the script being run.
    """
    global _logging_initialized
    if _logging_initialized and not force:
        return logging.getLogger()

    # Reconfigure stdout and stderr to use UTF-8 to prevent UnicodeEncodeError on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    if process_name is None:
        # Detect the entry point script name
        entry_script = os.path.basename(sys.argv[0])
        process_name = os.path.splitext(entry_script)[0]
        # Check for interactive/temporary run environments (Jupyter, IPython, pytest, command line -c)
        if not process_name or process_name in ['-c', 'ipykernel_launcher', 'pytest']:
            process_name = 'interactive'
            
    # Clean up name if it contains path characters or spaces
    process_name = os.path.basename(process_name).replace(' ', '_')
    
    # Generate timestamp for filename: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid double logging or custom config overrides
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 2. File Handler with permission fallback
    try:
        log_dir = os.path.join(_PROJECT_ROOT, 'logs', process_name)
        try:
            os.makedirs(log_dir, exist_ok=True)
        except PermissionError:
            log_dir = os.path.join('/tmp', 'logs', process_name)
            os.makedirs(log_dir, exist_ok=True)
            
        log_file = os.path.join(log_dir, f"{timestamp}.txt")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"✓ Logging initialized. Process: '{process_name}', Log File: {log_file}")
    except Exception as e:
        print(f"Warning: Failed to set up file logging for {process_name}: {e}", file=sys.stderr)
        
    _logging_initialized = True
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.
    If the root logger has not been configured yet, setup_logging() is automatically called first.
    """
    global _logging_initialized
    if not _logging_initialized:
        setup_logging()
    return logging.getLogger(name)
