"""
Author: lgx
Date: 2022/10/18 12:55
Description: A PyTorch-based framework used for construct deep learning baseline quickly.
    You only need to customize your own Dataset, Network, (may have Loss, Metric and Visualization funciton),
    every customization modules' parameters must have default value, and if you do not mention it in config file,
    it will be set to default.
"""

import os
from time import asctime
import shutil
import inspect
import argparse
from tqdm import tqdm
import yaml
from utils import RandomState, GpuDataParallel, Optimizer, Recorder
from utils import import_class, parse_yaml, dict_set_val
import tensorboard_logger

import torch.backends as backends
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import load, save, no_grad

ENABLE_TENSORBOARD = True
backends.cudnn.deterministic = True
backends.cudnn.enabled = True
backends.cudnn.benchmark = True


def config():
    parser = argparse.ArgumentParser(description='Deep Learning baseline template')
    parser.add_argument('--cfg-path', default='configs/template_cfg', help='the directory contain following 3 files')
    parser.add_argument('--hyp-cfg', default='configs/hyp_cfg.yaml', help='hyperparameter config file')
    parser.add_argument('--data-cfg', default='datasets/data_cfg.yaml', help='dataset config file')
    parser.add_argument('--model-cfg', default='models/model_cfg.yaml', help='model config file')
    return parser


class Processor:
    def __init__(self, hyp_cfg, data_cfg, model_cfg):
        self.hyp_cfg = hyp_cfg
        assert hyp_cfg['phase'] in ['train', 'test']
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.work_dir = self.init_path()  # make work directory
        self.recoder = Recorder(self.work_dir, save_log=True,
                                loss_name=hyp_cfg['loss_args']['loss_name'],
                                metric_name=hyp_cfg['metric_args']['metric_name'])
        self.save_cfg()   # save hyp data model config to work directory
        self.checkpoint_path = '{}/last.pt'.format(self.work_dir)
        if hyp_cfg['random_fix']:
            self.rng = RandomState(seed=hyp_cfg['random_seed'])

        self.device = GpuDataParallel()
        self.device.set_device(hyp_cfg['device'])

        self.data_loader = self.load_data()
        self.model = self.load_model()
        self.optimizer = Optimizer(self.model, hyp_cfg['optimizer_args'])
        self.start_epoch = 1
        self.top_metric = 0
        if hyp_cfg['phase'] == 'train':
            self.load_state()

        self.loss_fn = self.criterion()
        self.metric_fn = self.metric()
        self.vis_function = self.visualization()

        global ENABLE_TENSORBOARD
        if hyp_cfg['phase'] == 'test':
            ENABLE_TENSORBOARD = False
        if ENABLE_TENSORBOARD:
            tensorboard_logger.configure(self.work_dir)

    def init_path(self):
        hyp_cfg = self.hyp_cfg
        tmp_path = os.path.join(hyp_cfg['work_dir'], hyp_cfg['phase'])
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        exp_dirs = os.listdir(tmp_path)
        if len(exp_dirs) == 0:
            work_dir = os.path.join(tmp_path, hyp_cfg['name'])
        else:
            final_dir = exp_dirs[-1]
            final_idx = int(''.join(list(filter(str.isdigit, final_dir)))) if len(final_dir) != len(hyp_cfg['name']) else 1
            if hyp_cfg['exist_ok']:
                work_dir = os.path.join(tmp_path, final_dir)
            else:
                work_dir = os.path.join(tmp_path, hyp_cfg['name'] + '{}'.format(final_idx + 1))
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        if not os.path.exists(os.path.join(work_dir, 'vis')):
            os.makedirs(os.path.join(work_dir, 'vis'))
        if not os.path.exists(os.path.join(work_dir, 'configs')):
            os.makedirs(os.path.join(work_dir, 'configs'))
        print('work dir: {}'.format(work_dir))
        return work_dir

    def save_cfg(self):
        def dump_yaml(dict_, name):
            if not isinstance(dict_, dict):
                dict_ = vars(dict_)
            with open(os.path.join(self.work_dir, 'configs', name), 'w') as f:
                yaml.dump(dict_, f)
            self.recoder.print_log('{} has been saved to {}'.format(name, self.work_dir))
        dump_yaml(self.hyp_cfg, 'hyp_cfg.yaml')
        dump_yaml(self.data_cfg, 'data_cfg.yaml')
        dump_yaml(self.model_cfg, 'model_cfg.yaml')

    def load_data(self):
        self.recoder.print_log('Loading data')
        data_loader = dict()
        dataset_module = import_class(self.data_cfg['module'])
        shutil.copy2(inspect.getfile(dataset_module), self.work_dir)

        if self.device.gpu_list:
            self.hyp_cfg['batch_size'] *= len(self.device.gpu_list)

        train_args = []
        valid_args = []
        test_args = []
        for k, v in self.data_cfg.items():
            if k.startswith('train'):
                train_args.append(v)
            elif k.startswith('valid'):
                valid_args.append(v)
            elif k.startswith('test'):
                test_args.append(v)

        if len(train_args) > 0 and self.hyp_cfg['phase'] == 'train':
            dataset = dataset_module(**train_args[0])
            for i in range(1, len(train_args)):
                dataset += dataset_module(**train_args[i])
            dataloader_args = self.hyp_cfg['dataloader_args']['train']
            data_loader['train'] = self.create_dataloader(dataset, dataloader_args)

        if len(valid_args) > 0:
            dataset = dataset_module(**valid_args[0])
            for i in range(1, len(valid_args)):
                if valid_args[i] != {}:
                    dataset += dataset_module(**valid_args[i])
            dataloader_args = self.hyp_cfg['dataloader_args']['valid']
            data_loader['valid'] = self.create_dataloader(dataset, dataloader_args)

        if len(test_args) > 0 and test_args[0] != {}:
            dataset = dataset_module(**test_args[0])
            for i in range(1, len(test_args)):
                if test_args[i] != {}:
                    dataset += dataset_module(**test_args[i])
            dataloader_args = self.hyp_cfg['dataloader_args']['test']
            data_loader['test'] = self.create_dataloader(dataset, dataloader_args)
        self.recoder.print_log('Loading data finished.')
        return data_loader

    @staticmethod
    def create_dataloader(dataset, args, collate_fn=None):
        dl = DataLoader(
            dataset=dataset,
            batch_size=args['batch_size'],
            shuffle=args['shuffle'],
            num_workers=args['num_workers'],
            pin_memory=args['pin_memory'],
            collate_fn=collate_fn
        )
        return dl

    def load_model(self):

        model_module = import_class(self.model_cfg['module'])
        shutil.copy2(inspect.getfile(model_module), self.work_dir)
        model = self.device.model_to_device(model_module(**self.model_cfg['args']))

        if self.hyp_cfg['phase'] == 'train':
            # resume: start from last trained model
            if self.hyp_cfg['resume'] and os.path.exists(self.checkpoint_path):
                self.recoder.print_log('Loading last.pt')
                state_dict = load(self.checkpoint_path)
                model.load_state_dict(state_dict, strict=True)
            # start from pretrained model
            elif self.model_cfg['pretrained'] is not None:
                self.recoder.print_log('Loading pretrained model')
                state_dict = load(self.model_cfg['pretrained'])
                for w in self.model_cfg['ignore_weights']:
                    if state_dict.pop(w, None) is not None:
                        print('Sucessfully Remove Weights: {}.'.format(w))
                    else:
                        print('Can Not Remove Weights: {}.'.format(w))
                model.load_state_dict(state_dict, strict=False)

        elif self.hyp_cfg['phase'] == 'test':
            if self.model_cfg['weights'] is None:
                self.model_cfg['weights'] = os.path.join(self.work_dir, 'best.pt')
            assert os.path.exists(self.model_cfg['weights']), 'no best.pt'
            state_dict = load(self.model_cfg['weights'])
            model.load_state_dict(state_dict, strict=True)

        self.recoder.print_log('Loading model finished.')
        return model

    def save_model(self, last=True):
        if last:
            model_path = self.checkpoint_path
        else:
            model_path = os.path.join(self.work_dir, 'best.pt')
        save(self.model.state_dict(), model_path)

    def criterion(self):
        try:
            criterion_module = import_class(self.hyp_cfg['loss_args']['module'])
        except:
            criterion_module = eval(self.hyp_cfg['loss_args']['module'])
        return criterion_module(**self.hyp_cfg['loss_args']['args'])

    def metric(self):
        try:
            metric_module = import_class(self.hyp_cfg['metric_args']['module'])
        except:
            metric_module = eval(self.hyp_cfg['metric_args']['module'])
        metric_fn = metric_module(**self.hyp_cfg['metric_args']['args'])
        return metric_fn

    def visualization(self):
        try:
            vis_module = import_class(self.hyp_cfg['visualization_args']['module'])
        except:
            vis_module = eval(self.hyp_cfg['visualization_args']['module'])
        vis_function = vis_module(**self.hyp_cfg['visualization_args']['args'])
        return vis_function

    def start(self):
        self.recoder.print_log('Hyperparameters:\n{}\n'.format(str(self.hyp_cfg)))
        self.recoder.print_log('Data Parameters:\n{}\n'.format(str(self.data_cfg)))
        self.recoder.print_log('Model parameters:\n{}\n'.format(str(self.model_cfg)))
        if self.hyp_cfg['phase'] == 'train':
            for epoch in range(self.start_epoch, self.hyp_cfg['num_epoches'] + 1):
                self.train(epoch)
                self.eval(epoch, loader_name=['valid'])

        elif self.hyp_cfg['phase'] == 'test':
            if self.model_cfg['weights'] is None:
                raise ValueError('Please appoint --weights.')
            self.recoder.print_log('Model:   {}.'.format(self.model_cfg['module']))
            self.recoder.print_log('Weights: {}.'.format(self.model_cfg['weights']))

            self.eval(1, loader_name=['valid'])
            self.recoder.print_log('Evaluation Done.\n')

        self.recoder.print_log('Top {}: {}'.format(self.recoder.metric.name, self.top_metric))

    def train(self, epoch):
        self.model.train()

        loader = self.data_loader['train']

        current_learning_rate = [group['lr'] for group in self.optimizer.optimizer.param_groups]
        if ENABLE_TENSORBOARD:
            tensorboard_logger.log_value('learning-rate', current_learning_rate[0], epoch)

        self.recoder.timer_reset()
        self.recoder.loss.reset()
        self.recoder.metric.reset()
        with tqdm(loader) as loop:
            for batch_idx, data in enumerate(loop):
                self.recoder.record_timer('load_data')
                img, target = data
                input = self.device.data_to_device(img)
                target = self.device.data_to_device(target)
                self.recoder.record_timer('device')
                output = self.model(input)
                self.recoder.record_timer('forward')
                loss = self.loss_fn(target, output)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.recoder.record_timer('backward')
                self.recoder.loss.update(loss.item())
                with no_grad():
                    metric = self.metric_fn(target, output)
                self.recoder.metric.update(metric.item())
                loop.set_description('[ ' + asctime() + ' ] ' + f'Train Epoch: {epoch}')

                loss_str = str(self.recoder.loss).split(' ')
                metric_str = str(self.recoder.metric).split(' ')
                info_dict = {loss_str[0]: ''.join(loss_str[1:]),
                             metric_str[0]: ''.join(metric_str[1:]),
                             'lr': '{:.6f}'.format(current_learning_rate[0])}
                loop.set_postfix(info_dict)

        train_loss = self.recoder.loss.avg
        metric = self.recoder.metric.avg
        self.optimizer.step_scheduler(train_loss)

        self.recoder.print_log('Train {}: {:.6f}  {}: {:.6f}.'.format(self.recoder.loss.name, train_loss,
                                                                      self.recoder.metric.name, metric))
        self.recoder.print_time_statistics()
        if ENABLE_TENSORBOARD:
            tensorboard_logger.log_value('Train-' + self.recoder.loss.name, train_loss, epoch)
            tensorboard_logger.log_value('Train-' + self.recoder.metric.name, metric, epoch)
        self.save_model(epoch)
        self.save_state(epoch)

    def eval(self, epoch, loader_name):
        self.model.eval()
        for l_name in loader_name:
            loader = self.data_loader[l_name]

            with no_grad():
                with tqdm(loader) as loop:
                    self.recoder.loss.reset()
                    self.recoder.metric.reset()
                    for batch_idx, data in enumerate(loop):
                        imgs, target = data
                        input = self.device.data_to_device(imgs)
                        target = self.device.data_to_device(target)
                        output = self.model(input)
                        loss = self.loss_fn(target, output)
                        self.recoder.loss.update(loss.item())
                        metric = self.metric_fn(target, output)
                        self.recoder.metric.update(metric.item())
                        # Visualization
                        if self.hyp_cfg['visualization_args']['flag']:
                            prefix = os.path.join(self.work_dir, 'vis', 'batch_{}.jpg'.format(batch_idx))
                            self.vis_function(imgs, output, prefix)
                        loop.set_description('[ ' + asctime() + ' ] ' + f'Test Epoch: {epoch}')

                        loss_str = str(self.recoder.loss).split(' ')
                        metric_str = str(self.recoder.metric).split(' ')
                        info_dict = {loss_str[0]: ''.join(loss_str[1:]),
                                     metric_str[0]: ''.join(metric_str[1:])}
                        loop.set_postfix(info_dict)

        val_loss = self.recoder.loss.avg
        metric = self.recoder.metric.avg

        if metric > self.top_metric:
            self.top_metric = metric
            self.save_model(last=False)

        if ENABLE_TENSORBOARD:
            tensorboard_logger.log_value('Val-' + self.recoder.loss.name, val_loss, epoch)
            tensorboard_logger.log_value('Val-' + self.recoder.metric.name, metric, epoch)

        self.recoder.print_log('Val {}: {:.6f}  {}: {:.6f}  top {}: {:.6f}.'
                               .format(self.recoder.loss.name, val_loss,
                                       self.recoder.metric.name, metric,
                                       self.recoder.metric.name, self.top_metric))

        self.recoder.print_time_statistics()
        if ENABLE_TENSORBOARD:
            tensorboard_logger.log_value(self.recoder.loss.name, val_loss, epoch)
            tensorboard_logger.log_value(self.recoder.metric.name, metric, epoch)

    def save_state(self, epoch, state_path=None):
        if state_path is None:
            state_path = os.path.join(self.work_dir, 'state.pt')
        save({
            'epoch': epoch,
            'top_metric': self.top_metric,
            'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.get_rng_state()
        }, state_path)

    def load_state(self, state_path=None):
        if state_path is None:
            state_path = os.path.join(self.work_dir, 'state.pt')
        if self.hyp_cfg['resume'] and os.path.exists(state_path):
            state_dict = load(state_path)
            self.rng.set_rng_state(state_dict['rng_state'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.optimizer.load_scheduler_state_dict(state_dict['scheduler_state_dict'])
            self.start_epoch = state_dict['epoch'] + 1
            self.top_metric = state_dict['top_metric']
            self.recoder.print_log('Loading Random/Optimizer/Best state finished')


if __name__ == '__main__':
    parser = config()
    args = parser.parse_args()
    # The superiority of config files under 'cfg_path'  greater than 3 independent config files
    if (args.cfg_path is not None) and os.path.exists(args.cfg_path):
        cfg_list = os.listdir(args.cfg_path)
        if 'hyp_cfg.yaml' in cfg_list and 'data_cfg.yaml' in cfg_list and 'model_cfg.yaml' in cfg_list:
            args.hyp_cfg = os.path.join(args.cfg_path, 'hyp_cfg.yaml')
            args.data_cfg = os.path.join(args.cfg_path, 'data_cfg.yaml')
            args.model_cfg = os.path.join(args.cfg_path, 'model_cfg.yaml')

    hyp_cfg, data_cfg, model_cfg = args.hyp_cfg, args.data_cfg, args.model_cfg
    hyp_cfg, data_cfg, model_cfg = parse_yaml(hyp_cfg), parse_yaml(data_cfg), parse_yaml(model_cfg)

    # shared common parameters
    common_paras = hyp_cfg['common']
    for k, v in common_paras.items():
        dict_set_val(hyp_cfg, k, v)
        dict_set_val(data_cfg, k, v)
        dict_set_val(model_cfg, k, v)

    processor = Processor(hyp_cfg, data_cfg, model_cfg)
    processor.start()


