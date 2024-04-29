import os
import tarfile
import zipfile
import logging
import re
import datetime
from six.moves.urllib import request
from progress.bar import Bar

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string) # replace more than 2 whitespace with 1 whitespace   
    return string.strip().split()

class ProgressBar(Bar):
    message = 'Loading'
    fill = '#'
    suffix = '%(percent).1f%% | ETA: %(eta)ds'

class Logger():
    def __init__(self, logfilepath = None):
        super(Logger, self).__init__()
        if not os.path.exists('./log'):
            os.makedirs('./log')
        if logfilepath is None:
            logfilepath = datetime.datetime.now().strftime("./log/%Y%m%d_%H%M%S"+".log")
        self.logger = logging.getLogger('mylogger')
        fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
        fileHandler = logging.FileHandler(logfilepath)
        streamHandler = logging.StreamHandler()
        fileHandler.setFormatter(fomatter)
        streamHandler.setFormatter(fomatter)
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(streamHandler)
        self.logger.setLevel(logging.DEBUG)
print = Logger().logger.info