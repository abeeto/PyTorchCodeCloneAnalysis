try:
    from torch_trainer.general_utils import read_yaml, get_class
    from torch_trainer.configuration_objects import TrainingConfiguration
    from torch_trainer.traces import Trace
except ModuleNotFoundError:
    from general_utils import read_yaml, get_class
    from configuration_objects import TrainingConfiguration
    from traces import Trace

import torch
import os
from shutil import rmtree
from datetime import datetime
from time import sleep
from torch.utils.tensorboard import SummaryWriter
from typing import List
import yaml
from tqdm import tqdm, trange
from collections import OrderedDict
import numpy as np
import cv2
import signal
from pathlib import Path


class measurementObj:
    def __init__(self, loaders, writer_path, measurements=None, losses_names=None):
        if measurements is None:
            measurements = {}
        if type(loaders) is list:
            self.loaders = loaders
        elif type(loaders) is torch.utils.data.DataLoader:
            self.loaders = [loaders]
        else:
            print('error - data loader is invalid')
        self.writer = SummaryWriter(writer_path)
        self.aggregated_loss = 0.0
        self.steps = 0
        self.name = os.path.basename(writer_path)
        self.pname = self.name + (' ' * (7 - len(self.name)))
        self.traces: List[Trace] = []
        self.losses = OrderedDict({l: 0.0 for l in losses_names})

        if measurements:
            for meas in measurements.values():
                cls_ = get_class(meas['type'], meas['path'])
                self.traces.append(cls_(self.writer, self.size))

    def step(self, loss, inputs, pred, labels, losses=None):
        for m in self.traces:
            m.add_measurement(inputs, pred, labels)
        self.aggregated_loss += loss
        self.steps += 1
        if self.losses and losses:
            for agg_loss_type, loss_res in zip(self.losses.keys(), losses):
                self.losses[agg_loss_type] += loss_res.item()

    def epoch_step(self, epoch):
        for m in self.traces:
            try:
                m.write_epoch(epoch)
            except Exception:
                pass
        avr = self.aggregated_loss / self.size
        self.writer.add_scalar('Loss', avr, global_step=epoch)
        for loss_name, loss_val in self.losses.items():
            self.writer.add_scalar(f'Losses/{loss_name}', loss_val/self.size, global_step=epoch)
        self.init_loop()

    def init_loop(self):
        self.aggregated_loss = 0.0
        self.steps = 0
        for loss_name in self.losses.keys():
            self.losses[loss_name] = 0.0

    @property
    def size(self):
        return max(self.steps, 1)


class TorchTrainer:
    """wrapper class for torch models training"""

    def __init__(self, cfg: TrainingConfiguration, root, gpu_index: int):
        self.cfg = cfg
        if int(gpu_index) >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(gpu_index))
            print('using device: ', torch.cuda.get_device_name(self.device))
        else:
            self.device = torch.device("cpu")
            print('using cpu')
        self.model: torch.nn.Module = None
        self.optimizer = None
        self.start_epoch: int = 0
        self.root = root
        self.dataset = None
        self.running = False
        self.loss_debug = OrderedDict()
        self.epoch = 0

    @classmethod
    def new_train(cls, out_path, model_cfg, optimizer_cfg, dataset_cfg, gpu_index, exp_name):
        sections = {'model': read_yaml(model_cfg),
                    'optimizer': read_yaml(optimizer_cfg),
                    'data': read_yaml(dataset_cfg)}
        config = TrainingConfiguration(**sections)

        model_dir = config.model.type + datetime.now().strftime("_%d%b%y_%H%M")
        if len(exp_name):
            model_dir += '_' + exp_name
        root = os.path.join(out_path, model_dir)
        if os.path.isdir(root):
            print("root directory is already exist - will delete the previous and create new")
            rmtree(root)
        os.makedirs(root)
        print('writing results to directory: %s' % root)
        os.makedirs(os.path.join(root, 'checkpoints'))
        with open(os.path.join(root, 'cfg.yaml'), 'w') as f:
            yaml.dump(data=sections, stream=f)
        cls = TorchTrainer(cfg=config, root=root, gpu_index=gpu_index)
        cls.init_nn()
        cls.model.to(cls.device)
        return cls

    @classmethod
    def warm_startup(cls, in_path, gpu_index, best=False):
        in_path = Path(in_path)
        if in_path.is_dir():
            config_dict = read_yaml(in_path.joinpath('cfg.yaml').as_posix())
            root = in_path.as_posix()
        else:
            config_dict = read_yaml(in_path.parent.parent.joinpath('cfg.yaml').as_posix())
            root = in_path.parent.parent.as_posix()

        config = TrainingConfiguration(**config_dict)
        cls = TorchTrainer(cfg=config, root=root, gpu_index=gpu_index)
        cls.init_nn()
        if best:
            cp_path = os.path.join(root, 'checkpoints', 'checkpoint.pth')
        else:
            cp_path = os.path.join(root, 'checkpoints', 'last_checkpoint.pth')
            if not os.path.exists(cp_path):
                cp_path = os.path.join(root, 'checkpoints', 'checkpoint.pth')
        cls.load_checkpoint(cp_path)
        return cls

    def init_nn(self):
        model_cls = get_class(self.cfg.model.type, file_path=self.cfg.model.path)
        self.model = model_cls(**self.cfg.model.kargs)
        dataset_cls = get_class(self.cfg.data.type, file_path=self.cfg.data.path)
        self.dataset = dataset_cls(self.root, self.cfg.model.in_channels, self.cfg.model.out_channels,
                                   **self.cfg.data.kargs)
        optim_cls = get_class(self.cfg.optimizer.type, module_path='torch.optim')
        self.optimizer = optim_cls(self.model.parameters(), **self.cfg.optimizer.kargs)

    def load_checkpoint(self, cp_path):
        print('loading checkpoint', cp_path)
        checkpoint = torch.load(cp_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.model.to(self.device)

    def save_checkpoint(self, better, epoch, mid_checkpoint: bool = False):
        if better:
            fpath = os.path.join(self.root, 'checkpoints', 'checkpoint.pth')
        elif not mid_checkpoint:
            fpath = os.path.join(self.root, 'checkpoints', 'last_checkpoint.pth')
        else:
            fpath = os.path.join(self.root, 'checkpoints', f'checkpoint_{epoch}.pth')
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}
                   , fpath)

    def init_measurements_obj(self, loss_names):
        """create helper objects in order to make the train function more clear"""
        train_loaders, test_loaders = self.dataset.get_data_loaders(self.cfg.model.batch_size)
        train = measurementObj(loaders=train_loaders, writer_path=os.path.join(self.root, 'train'), losses_names=loss_names)
        test = measurementObj(loaders=test_loaders, writer_path=os.path.join(self.root, 'test'),
                              measurements=self.cfg.model.test_traces, losses_names=loss_names)
        return train, test

    def init_loss_func(self):
        losses = OrderedDict()
        for loss_name, loss_cfg in self.cfg.model.loss.items():
            module_path = loss_cfg.module_path
            criterion_cls = get_class(loss_name, module_path)
            class_instance = criterion_cls(**loss_cfg.kargs) if len(loss_cfg.kargs) else criterion_cls()
            if issubclass(criterion_cls, torch.nn.Module):
                class_instance = class_instance.to(self.device)
            losses[loss_name] = class_instance

        def crit(inputs, preds, labels):
            criteria = []
            for loss_name, criterion in losses.items():
                loss_cfg = self.cfg.model.loss[loss_name]
                if 'Consistency' in criterion.__repr__(): # TODO fix it
                    criteria.append(loss_cfg.factor * criterion(inputs, preds, labels))
                else:
                    if loss_cfg.im_channels is not None:
                        loss = loss_cfg.factor * criterion(preds[:, loss_cfg.im_channels], labels[:, loss_cfg.im_channels])
                        criteria.append(loss)
                    else:
                        criteria.append(loss_cfg.factor * criterion(preds, labels))
            return criteria
        return crit, list(losses.keys())

    def signal_handler(self, sig, frame):
        self.running = False

    def train(self):
        crit, loss_names = self.init_loss_func()
        train_meas, test_meas = self.init_measurements_obj(loss_names)
        best_loss = 2 ** 16
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        test_freq = -1 # init for warm startup
        ep_prog = trange(self.start_epoch, self.cfg.model.epochs, desc='epochs', ncols=120)
        for epoch in ep_prog:
            if epoch % 100 == 0 or test_freq == -1:
                test_freq = max(int(-7 * np.log((epoch+1)/(self.cfg.model.epochs+1))), 1)
            self.model.train()
            for train_loader in train_meas.loaders:
                prog = tqdm(train_loader, desc='train', leave=False, ncols=100)
                for x, y in prog:
                    if self.running:
                        self.optimizer.zero_grad()
                        inputs, labels = x.to(self.device), y.to(self.device)
                        outputs = self.model(inputs)
                        losses = crit(inputs, outputs, labels)
                        loss = sum(losses)
                        loss.backward()
                        self.optimizer.step()
                        train_meas.step(loss, inputs, outputs, labels, losses)
                        prog.set_description(f'train loss {loss.item():.2}')
            train_meas.epoch_step(epoch)

            if epoch % test_freq == 0 and self.running:
                self.model.eval()
                with torch.no_grad():
                    for test_loader in test_meas.loaders:
                        prog = tqdm(test_loader, desc='validation', leave=False, ncols=100)
                        if self.running:
                            for x, y in prog:
                                inputs, labels = x.to(self.device), y.to(self.device)
                                outputs = self.model(inputs)
                                losses = crit(inputs, outputs, labels)
                                loss = sum(losses)
                                test_meas.step(loss.item(), inputs, outputs, labels, losses)
                                prog.set_description(f'validation loss {loss.item():.2}')
                if test_meas.aggregated_loss < best_loss:
                    best_loss = test_meas.aggregated_loss
                    self.save_checkpoint(better=True, epoch=epoch)
                else:
                    self.save_checkpoint(better=False, epoch=epoch)
                test_meas.epoch_step(epoch)
            # if epoch % 500 == 0:
            #     self.save_checkpoint(better=False, epoch=epoch, mid_checkpoint=True)
            if not self.running:
                print('saving model and exiting...')
                self.save_checkpoint(False, epoch - 1)
                return

    def debug_save(self, inputs, outputs, labels):
        out_path = '/tmp/minibatch_debug'
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        for i in range(outputs.shape[0]):
            inp = np.transpose((inputs[i].cpu().numpy() * 255).astype(np.uint8), (2, 1, 0))
            pred = np.transpose((outputs[i].cpu().numpy() * 255).astype(np.uint8), (2, 1, 0))
            lbl = np.transpose((labels[i].cpu().numpy() * 255).astype(np.uint8), (2, 1, 0))
            cv2.imwrite('{}/{}_input.png'.format(out_path, i), cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))
            cv2.imwrite('{}/{}_predict.png'.format(out_path, i), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
            cv2.imwrite('{}/{}_label.png'.format(out_path, i), cv2.cvtColor(lbl, cv2.COLOR_RGB2BGR))
