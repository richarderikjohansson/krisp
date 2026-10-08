import logging


def get_logger() -> logging.Logger:
    """Logger

    :return: Logger object
    """
    logger = logging.getLogger("runtime_logger")
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_format)

        logger.addHandler(console_handler)
        logger.propagate = False
    return logger
