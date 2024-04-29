import logging
formatter = logging.Formatter('%(name)s: %(asctime)s %(levelname)s %(message)s')


def get_logger(name, log_file=None, level=logging.INFO):
    """Function setup as many loggers as you want"""

    if log_file is None:
        log_file = name + '.log'

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

