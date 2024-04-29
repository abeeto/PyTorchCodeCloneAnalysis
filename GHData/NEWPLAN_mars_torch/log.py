import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s: %(asctime)s %(filename)s:%(lineno)d] \t%(message)s')
#format='[%(levelname)s: %(asctime)s %(thread)d %(funcName)s %(filename)s:%(lineno)d] \t%(message)s'
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logger.info("This is a log info")
    logger.debug("Debugging")
    logger.warning("Warning information")
    logger.error("Fatal error here")
    print("LOG DONE")
