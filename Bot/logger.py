import logging

def setup_logger() -> None:
    """Set up the logger to log messages to a file."""
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )