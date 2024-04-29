import argparse
import os


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        self.parser.add_argument('--image_gradient', action='store_true', default=False)

        self.parser.add_argument('--refined', action='store_true', default=True)

        self.parser.add_argument('--batchsize', type=int, default=4, help='input batch size')
        self.parser.add_argument('--input_height', type=int, default=228, help='scale image to this size')
        self.parser.add_argument('--input_width', type=int, default=304, help='scale image to this size')

        self.parser.add_argument('--epochs', type=int, default=1, help='number of epochs')

        self.parser.add_argument('--lr_G', type=float, default=0.0004, help='initial learning rate for adam of G')

        self.parser.add_argument('--data_dir', type=str, default='./data')

        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument('--results_dir', type=str, default='./results', help='results are saved here')

        self.parser.add_argument('--checkpoint_every', type=int, default=100, help='checkpoint every epoch')
        self.parser.add_argument('--results_every', type=int, default=10, help='results every epoch')
        self.parser.add_argument('--session', type=int, default=1, help='session')
        self.parser.add_argument('--log_dir', type=str, default='./log', help='logs are saved here')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt