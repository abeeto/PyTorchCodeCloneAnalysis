import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--test', action='store_true', help='compute metrics only')

        self.parser.add_argument('--exp_id', default='default')

        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')

        self.parser.add_argument('--load_teacher', default='',
                                 help='path to pretrained  teacher model')

        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. ')

        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')

        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet

        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')

        self.parser.add_argument('--num_epochs', type=int, default=100,
                                 help='total training epochs.')

        self.parser.add_argument('--predict', action='store_true',
                                 help='predict to file.')

        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')

        self.parser.add_argument('--val_intervals', type=int, default=1,
                                 help='number of epochs to run validation.')

        self.parser.add_argument('--test_intervals', type=int, default=1,
                                 help='number of validation runs to run metrics computation')

        self.parser.add_argument("--batches_per_update", type=int, default=1,
                                 help="number of processed batches per one weights update")
        self.parser.add_argument("--video", type=str, default="",
                                 help="video file path")

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]

        opt.root_dir = os.path.dirname(__file__)
        opt.data_dir = 'airbus/'
        # opt.data_dir = os.path.join(opt.root_dir, '../')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp')
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')
        return opt
