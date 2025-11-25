import logging
import os

def setup_logger(config):
    """Configure file-based logging for the project."""
    log_file = config["logging"]["file"]
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"], logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )

    logging.info("Logger initialized successfully.")
