# author: lgx
# date: 2022/10/13 13:02
# description: 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import random
import numpy as np
import time
from collections import defaultdict
import importlib
import yaml
from enum import Enum


class RandomState(object):
    def __init__(self, seed):
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def get_rng_state():
        rng_dict = dict()
        rng_dict["torch"] = torch.get_rng_state()
        rng_dict["cuda"] = torch.cuda.get_rng_state_all()
        rng_dict["numpy"] = np.random.get_state()
        rng_dict["random"] = random.getstate()
        return rng_dict

    @staticmethod
    def set_rng_state(rng_dict):
        torch.set_rng_state(rng_dict["torch"])
        torch.cuda.set_rng_state_all(rng_dict["cuda"])
        np.random.set_state(rng_dict["numpy"])
        random.setstate(rng_dict["random"])


class GpuDataParallel(object):
    def __init__(self):
        self.gpu_list = []
        self.output_device = None

    def set_device(self, device):
        if device == 'cpu':
            self.output_device == 'cpu'
            return

        if device is not None and torch.cuda.device_count():
            self.gpu_list = list(range(torch.cuda.device_count()))
            output_device = self.gpu_list[0]
        self.output_device = output_device if len(self.gpu_list) > 0 else "cpu"

    def model_to_device(self, model):
        model = model.to(self.output_device)
        if len(self.gpu_list) > 1:
            print('DataParallel', self.gpu_list)
            model = nn.DataParallel(model, device_ids=self.gpu_list, output_device=self.output_device)
        return model

    def data_to_device(self, data):
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.output_device, detype=torch.float32)
        return data.to(self.output_device)

    def fn_to_device(self, fn):
        return fn.to(self.output_device)

    @staticmethod
    def load_weights(model, weights_path, ignore_weights):
        print('Load weights from {}.'.format(weights_path))
        try:
            weights = torch.load(weights_path, map_location=torch.device('cpu'))
            for w in ignore_weights:
                if weights.pop(w, None) is not None:
                    print('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
            model.load_state_dict(weights, strict=True)
        except RuntimeError:
            weights = torch.load(weights_path, map_location=torch.device('cpu'))['model_state_dict']
            for w in ignore_weights:
                if weights.pop(w, None) is not None:
                    print('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
            model.load_state_dict(weights, strict=True)
        return model


class Optimizer(object):
    def __init__(self, model, optim_dict):
        otpim_module = eval(optim_dict['module'])
        self.optim_dict = optim_dict
        self.optimizer = otpim_module(model.parameters(), **optim_dict['args'])
        self.scheduler = self.define_lr_scheduler(self.optimizer)

    @property
    def name(self):
        return self.optim_dict["module"] + '.' + self.optim_dict["scheduler_module"]

    def define_lr_scheduler(self, optimizer):
        scheduler_module = eval(self.optim_dict['scheduler_module'])
        lr_scheduler = scheduler_module(optimizer, **self.optim_dict['scheduler_args'])
        return lr_scheduler

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def step_scheduler(self, loss=None):
        if loss is None:
            self.scheduler.step()
        else:
            self.scheduler.step(loss)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def scheduler_state_dict(self):
        return self.scheduler.state_dict()

    def load_scheduler_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)


class Recorder:
    def __init__(self, work_dir=None, save_log=False, loss_name='loss', metric_name='Acc'):
        self.cur_time = time.time()
        self.save_log = save_log
        self.log_path = '{}/log.txt'.format(work_dir) if save_log else None
        self.timer = defaultdict(float)
        self.loss = AverageMeter(loss_name, ':6f')
        self.metric = AverageMeter(metric_name, ':6f')

    def print_time(self):
        self.print_log("Current time:  " + time.asctime())

    def print_log(self, print_str, print_time=True):
        if print_time:
            print_str = "[ " + time.asctime() + ' ] ' + print_str
        print(print_str)

        if self.save_log:
            with open(self.log_path, 'a') as f:
                f.write(print_str + '\n')

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.cur_time = time.time()
        return split_time

    def timer_reset(self):
        self.cur_time = time.time()
        self.timer = defaultdict(float)

    def record_timer(self, key):
        self.timer[key] += self.split_time()

    def print_time_statistics(self, show_proportion=True):
        if show_proportion:
            proportion = {
                k: '{:02d}%'.format(int(round(v * 100 / sum(self.timer.values()))))
                for k, v in self.timer.items()}
        else:
            proportion = {
                k: '{:02d}ms'.format(int(round(v*1000)))
                for k, v in self.timer.items()}

        output = '\tTime consumption:'
        for k, v in proportion.items():
            output += ' [{}]{}'.format(k, v)

        self.print_log(output)

    def update_metric(self, val, count=1):
        self.metric.update(val, count)

    def update_loss(self, val, count=1):
        self.loss.update(val, count)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


def parse_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        try:
            yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            yaml_arg = yaml.load(f)
    return yaml_arg


def dict_set_val(dict_, key, val):
    assert isinstance(dict_, dict)
    st = [dict_]
    while len(st) > 0:
        cur = st.pop(0)
        for k, v in cur.items():
            if k == key:
                cur[k] = val
            elif isinstance(v, dict):
                st.append(v)
