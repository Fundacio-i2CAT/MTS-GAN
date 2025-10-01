"""Logger file"""
import logging


def setup_logger(debug=False):
    """Logger"""
    logger_deep = logging.getLogger('DataLogger')

    if not logger_deep.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger_deep.addHandler(handler)

    if debug:
        logger_deep.setLevel(logging.DEBUG)
    else:
        logger_deep.setLevel(logging.INFO)

    return logger_deep
