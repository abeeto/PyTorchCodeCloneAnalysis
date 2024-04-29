import logging
import sys
import ctypes
import codecs

def get_logger(name):
  logger = logging.getLogger(name)

  formatter = logging.Formatter(
    "%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
  )

  handler = logging.StreamHandler(sys.stderr)
  handler.setFormatter(formatter)

  if (logger.hasHandlers()):
    logger.handlers.clear()

  logger.setLevel(logging.INFO)
  logger.addHandler(handler)
  logger.propagate = False
  return logger

def get_cudart():
  try:
    _cudart = ctypes.CDLL('libcudart.so')
  except:
    _cudart = None
  return _cudart

def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
      first_tok = f.readline().split(None, 1)[0]
      return len(first_tok.split(u"￨")) - 1

def preprocess_parallel_files(file_paths, train_val_split=0.2, split_train_val=True):
    print(count_features(file_paths[0]))
  

