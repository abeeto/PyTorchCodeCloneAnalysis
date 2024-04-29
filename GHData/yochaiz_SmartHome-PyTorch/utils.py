import torch
import logging.config
import os
from shutil import copy2, copytree, copyfile
import argparse
from datetime import datetime
import sys
import signal


# parse arguments
def parseArguments():
    parser = argparse.ArgumentParser(description='PyTorch SmartHome Training')
    parser.add_argument('--results_dir', metavar='RESULTS_DIR', type=str, default='./results', help='results dir')
    parser.add_argument("--settings", type=str, default='./settings.json', help="Settings JSON file")
    parser.add_argument('--gpus', default=0, type=int, help='gpus used for training - e.g 0,1,3')
    parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequential", action='store_true', help="Init sequential state for a new game")
    group.add_argument("--random", action='store_true', help="Init random state for a new game")
    # TODO: move some parameters to settings.json
    # TODO: add --resume option

    args = parser.parse_args()
    return args


def initSavePath(results_dir):
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    save_path = os.path.join(results_dir, time_stamp)
    save_path = os.path.abspath(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # TODO: do we need train_path ???
    train_path = '{}/train'.format(save_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    return save_path, train_path, time_stamp


# save source code files to training results folder
# dstDir - the folder to save the source code files to
def saveCode(dstDir, baseDir):
    folderName = 'code'

    fullPath = '{}/{}'.format(dstDir, folderName)
    if not os.path.exists(fullPath):
        os.makedirs(fullPath)

    # TODO: sort files structure here
    # copy baseFolder files
    copyFolderName = 'ddpg'
    copyPath = '{}/{}'.format(fullPath, copyFolderName)
    if not os.path.exists(copyPath):
        os.makedirs(copyPath)
    for fname in os.listdir(baseDir):
        if os.path.isfile(fname):
            copy2('{}/{}'.format(baseDir, fname), copyPath)

    # copy utils folder
    copyFolderName = 'utils'
    copytree('{}/../{}'.format(baseDir, copyFolderName), '{}/{}'.format(fullPath, copyFolderName))

    # copy model
    copyFolderName = 'models'
    copyPath = '{}/{}'.format(fullPath, copyFolderName)
    if not os.path.exists(copyPath):
        os.makedirs(copyPath)
    copy2('{}/../{}/{}.py'.format(baseDir, copyFolderName, modelName), copyPath)
    with open('{}/__init__.py'.format(copyPath), 'w') as f:
        f.write('from .{} import *'.format(modelName))

    # copy general files
    baseFilesList = ['../uniq.py', '../preprocess.py', '../data.py', '../actquant.py', '../quantize.py']
    for fname in baseFilesList:
        copy2('{}/{}'.format(baseDir, fname), fullPath)

    return fullPath


def setup_logging(log_file, logger_name):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    # disable logging to stdout
    logger.propagate = False

    return logger


def initGamesLogger(logger_name, save_path):
    return setup_logging(os.path.join(save_path, 'log_[{}].txt'.format(logger_name)), logger_name)


# attach signals handler to program
def attachSignalsHandler(results, logger):
    # define terminate signal handler
    def SIGTERMHandler(signal, frame):
        if logger is not None:
            logger.info('_ _ _ Program was terminated by user or server _ _ _')

        results.moveToEnded()
        sys.exit(0)

    def SIGSTOPHandler(signal, frame):
        if logger is not None:
            logger.info('_ _ _ Program was paused by user or server _ _ _')

    def SIGCONTHandler(signal, frame):
        if logger is not None:
            logger.info('_ _ _ Program was resumed by user or server _ _ _')

    signal.signal(signal.SIGTERM, SIGTERMHandler)
    signal.signal(signal.SIGSTOP, SIGSTOPHandler)
    signal.signal(signal.SIGCONT, SIGCONTHandler)


def save_checkpoint(state, is_best, path='.', filename='checkpoint'):
    fileType = 'pth.tar'
    default_filename = '{}/{}_checkpoint.{}'.format(path, filename, fileType)
    torch.save(state, default_filename)
    if is_best:
        copyfile(default_filename, '{}/{}_opt.{}'.format(path, filename, fileType))
