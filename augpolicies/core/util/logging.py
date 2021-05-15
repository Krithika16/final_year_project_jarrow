import logging
import os

def get_logger(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        os.makedirs("logs")
    except FileExistsError:
        pass

    handler = logging.FileHandler(args.config['log_path'])
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
