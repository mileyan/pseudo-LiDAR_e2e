import logging
import os


def set_logger(name='exp', filepath=None, level='INFO'):
    formatter = logging.Formatter(
        fmt="[%(asctime)s %(name)s] %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
        style='%')

    logger = logging.getLogger(name)
    logger.setLevel(logging.getLevelName(level))
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if filepath is not None:
        if os.path.dirname(filepath) is not '':
            if not os.path.isdir(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
        file_handle = logging.FileHandler(filename=filepath, mode="a")
        file_handle.set_name("file")
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    return logger

def get_logger(name):
    return logging.getLogger(name)
