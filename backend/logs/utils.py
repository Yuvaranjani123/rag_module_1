import logging
import os

def setup_logging(log_file='app.log', log_level=logging.INFO):
    """
    Set up logging configuration for the project.
    Logs will be written to the specified log_file in the media/logs directory.
    """
    log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'media', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_path}")
