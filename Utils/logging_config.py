import logging
import os
from datetime import datetime


def setup_logging(module_name: str) -> str:
    """
    Set up logging configuration for a module.
    Returns the path to the log file.
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a timestamped log file for this module
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{module_name}_{timestamp}.log")

    # Create handlers with different log levels
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)  # Only errors to file

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # All messages to console

    # Configure logging with both handlers
    logging.basicConfig(
        level=logging.INFO,  # Set root logger to lowest level
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[file_handler, console_handler],
    )

    return log_file
