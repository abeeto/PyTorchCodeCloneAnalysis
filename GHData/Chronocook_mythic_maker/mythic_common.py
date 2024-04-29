""""
This module contains shared functions that do cool things.

@author : Brad Beechler (brad.e.beechler@gmail.com)
Modified: 20171234 (Brad Beechler)
"""

from uplog import log
import sys
import psutil
import argparse
from string import printable
import numpy as np
import unidecode
import string
import time
import math
import torch


trainable_characters = string.printable
num_characters = len(trainable_characters)


class TrainerSettings:
    # General
    args = None
    debug = False
    cuda = None
    # multicuda = False
    model_file = None
    # For trainer
    text_file = None
    model = None
    epochs = None
    print_every = None
    hidden_size = None
    layers = None
    learning_rate = None
    dropout = None
    chunk_size = None
    batch_size = None

    def __init__(self):
        # Read the command line
        self.__get_command_line()
        # Add the command line info into the config dict
        self.__args_to_config()
        if self.text_file is not None:
            self.text_string, self.text_length = read_file_as_string(self.text_file)

    def report(self):
        log.out.info("Settings:" + "\n" +
                     "text_file    : " + str(self.text_file) + "\n" +
                     "model_file   : " + str(self.model_file) + "\n" +
                     "model        : " + str(self.model) + "\n" +
                     "epochs       : " + str(self.epochs) + "\n" +
                     "chunk_size   : " + str(self.chunk_size) + "\n" +
                     "batch_size   : " + str(self.batch_size) + "\n" +
                     "hidden_size  : " + str(self.hidden_size) + "\n" +
                     "layers       : " + str(self.layers) + "\n" +
                     "learning_rate: " + str(self.learning_rate) + "\n" +
                     "dropout      : " + str(self.dropout) + "\n" +
                     "print_every  : " + str(self.print_every)
                     )

    def __args_to_config(self):
        """
        Takes the argparse object and puts the values into this object
        (there's probably a way better way to do this BTW)
        """
        # General
        self.debug = self.args.debug
        self.cuda = self.args.cuda
        # self.multicuda = self.args.multicuda
        self.model_file = self.args.model_file
        # For trainer
        self.text_file = self.args.text_file
        self.model = self.args.model
        self.epochs = self.args.epochs
        self.print_every = self.args.print_every
        self.hidden_size = self.args.hidden_size
        self.layers = self.args.layers
        self.learning_rate = self.args.learning_rate
        self.dropout = self.args.dropout
        self.chunk_size = self.args.chunk_size
        self.batch_size = self.args.batch_size

    def __get_command_line(self):
        """
        Get command line information using the argparse module
        """
        # General
        ap = argparse.ArgumentParser(description='Trains models in torch framework.')
        ap.add_argument('--debug', dest='debug', action='store_true',
                        help='Switch to activate debug mode.')
        ap.set_defaults(debug=False)
        # ap.add_argument('--cuda', dest='cuda', action='store_true',
        #                 help='Switch to activate CUDA support.')
        # ap.set_defaults(cuda=False)
        ap.add_argument('--cuda', type=int, default=None,
                        help='Switch to activate CUDA support on card n.', required=False)
        # ap.add_argument('--multicuda', dest='multicuda', action='store_true',
        #                 help='Switch to activate distributed CUDA support!')
        # ap.set_defaults(multicuda=False)
        ap.add_argument('--model_file', type=str, default=None,
                        help='Torch model filename (foo.pt)', required=False)
        # For the trainer
        ap.add_argument('--text_file', type=str, default=None,
                        help='Raw data file (ascii text)', required=True)
        ap.add_argument('--model', type=str, default="gru",
                        help='Model type', required=False)
        ap.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to run for', required=False)
        ap.add_argument('--print_every', type=int, default=100,
                        help='Print results every n epochs', required=False)
        ap.add_argument('--hidden_size', type=int, default=100,
                        help='Number of hidden layers', required=False)
        ap.add_argument('--layers', type=int, default=2,
                        help='Number of layers', required=False)
        ap.add_argument('--learning_rate', type=float, default=0.01,
                        help='The learning rate', required=False)
        ap.add_argument('--dropout', type=float, default=0.2,
                        help='The dropout rate', required=False)
        ap.add_argument('--chunk_size', type=int, default=64,
                        help='Chunk size', required=False)
        ap.add_argument('--batch_size', type=int, default=128,
                        help='Batch size', required=False)
        self.args = ap.parse_args()


class WriterSettings:
    # General
    args = None
    debug = False
    cuda = None
    model_file = None
    # For writer
    output_file = None
    seed_string = None
    predict_length = None
    iterations = None
    temperature = None

    def __init__(self):
        # Read the command line
        self.__get_command_line()
        # Add the command line info into the config dict
        self.__args_to_config()

    def __args_to_config(self):
        """
        Takes the argparse object and puts the values into this object
        (there's probably a way better way to do this BTW)
        """
        # General
        self.debug = self.args.debug
        self.cuda = self.args.cuda
        self.model_file = self.args.model_file
        # For writer
        self.output_file = self.args.output_file
        self.seed_string = self.args.seed_string
        self.predict_length = self.args.predict_length
        self.iterations = self.args.iterations
        self.temperature = self.args.temperature


    def __get_command_line(self):
        """
        Get command line information using the argparse module
        """
        # General
        ap = argparse.ArgumentParser(description='Writes outputs from trained models.')
        ap.add_argument('--debug', dest='debug', action='store_true',
                        help='Switch to activate debug mode.')
        ap.set_defaults(debug=False)
        # ap.add_argument('--cuda', dest='cuda', action='store_true',
        #                 help='Switch to activate CUDA support.')
        # ap.set_defaults(cuda=False)
        ap.add_argument('--cuda', type=int, default=None,
                        help='Switch to activate CUDA support on card n.', required=False)
        ap.add_argument('--model_file', type=str, default=None,
                        help='Torch model filename (foo.pt)', required=True)
        # For the writer
        ap.add_argument('--output_file', type=str, default=None,
                        help='If set will write text to this file', required=False)
        ap.add_argument('--seed_string', type=str, default='A',
                        help='Initial seed string', required=False)
        ap.add_argument('--predict_length', type=int, default=200,
                        help='Length of the prediction', required=False)
        ap.add_argument('--iterations', type=int, default=1,
                        help='Times to loop the writing', required=False)
        ap.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature setting (higher is more random)', required=False)
        self.args = ap.parse_args()


class ExtractorSettings:
    # General
    args = None
    debug = False
    # Extractor specific
    type = None

    def __init__(self):
        # Read the command line
        self.__get_command_line()
        # Add the command line info into the config dict
        self.__args_to_config()

    def __args_to_config(self):
        """
        Takes the argparse object and puts the values into this object
        (there's probably a way better way to do this BTW)
        """
        # General
        self.debug = self.args.debug
        # Extractor specific
        self.type = self.args.type
        self.data_file = self.args.data_file
        self.out_file = self.args.out_file
        self.samples = self.args.samples
        self.key = self.args.key
        self.clean = self.args.clean

    def __get_command_line(self):
        """
        Get command line information using the argparse module
        """
        # General
        ap = argparse.ArgumentParser(description='Extracts text files from various data formats.')
        ap.add_argument('--debug', dest='debug', action='store_true',
                        help='Switch to activate debug mode.')
        ap.set_defaults(debug=False)
        # Extractor specific
        ap.add_argument('--type', type=str, default=None,
                        help='Type of extraction (i.e. json, mbox)', required=True)
        ap.add_argument('--data_file', type=str, default=None,
                        help='Filename to extract from.', required=True)
        ap.add_argument('--out_file', type=str, default='./extracted.txt',
                        help='Filename to write text to.', required=False)
        ap.add_argument('--samples', type=int, default=None,
                        help='Number of samples to grab.', required=False)
        ap.add_argument('--key', type=str, default=None,
                        help='Specify a key (for json).', required=False)
        ap.add_argument('--clean', dest='clean', action='store_true',
                        help='Switch to clean data.')
        ap.set_defaults(clean=False)
        self.args = ap.parse_args()


def report_sys_info():
    # Report basic system stats
    log.out.info("Python version  : " + sys.version)
    log.out.info("Number of CPUs  : " + str(psutil.cpu_count()))
    log.out.info("Memory total    : " + str(round(float(psutil.virtual_memory().total) / 2 ** 30, 2)) + "GB")
    log.out.info("Memory useage   : " + str(round(float(psutil.virtual_memory().used) / 2 ** 30, 2)) + "GB")
    log.out.info("Memory available: " + str(round(float(psutil.virtual_memory().available) / 2 ** 30, 2)) + "GB")


def get_median_length(measure_array):
    length_function = lambda x: len(x)
    vector_function = np.vectorize(length_function)
    return int(np.median(vector_function(measure_array)))


def clean_string_to_printable(string_in, lower=True):
    log.out.info("Size in: " + str(len(string_in)))
    if lower:
        char_filter = printable.lower()
    else:
        char_filter = printable
    string_out = "".join(c for c in string_in if c in char_filter)
    log.out.info("Size out: " + str(len(string_out)))
    return string_out


def read_file_as_string(filename):
    """
    Open a file and returns its handle and length
    :param filename:
    :return:
    """
    text_str = unidecode.unidecode(open(filename).read())
    return text_str, len(text_str)


def char_tensor(input_string):
    """
    Transform a string into a tensor
    :param input_string: the string you want to transform
    :return: tensor for torch
    """
    tensor = torch.zeros(len(input_string)).long()
    for c in range(len(input_string)):
        try:
            tensor[c] = trainable_characters.index(input_string[c])
        except:
            continue
    return tensor


def time_since(start_time):
    """
    :param start_time: start time
    :return: A human readable elapsed time
    """
    s = time.time() - start_time
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
