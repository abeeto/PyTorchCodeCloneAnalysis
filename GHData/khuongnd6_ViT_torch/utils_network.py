# %%
from types import ClassMethodDescriptorType
from typing import Counter
from numpy import core
import torch
from torch._C import Value
from torch.autograd.grad_mode import no_grad
from torch.jit import Error
import torch.optim as optim
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.modules.activation import GELU
from torch.optim import optimizer
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms

# from functools import partial
# from xcit import XCiT
from adabelief_pytorch import AdaBelief

from utils_stats import *

# %%
import numpy as np
# import scipy
from PIL import Image

import time, os, json, re, string, math, random, datetime
import sys


# %%
class LRSchedule:
    @classmethod
    def get_base_fn(cls):
        def _fn(e):
            return 1.0
        return _fn
    
    @classmethod
    def get_step_fn(cls, step=10, gamma=0.5):
        assert step > 0
        assert 1 >= gamma >= 0
        def _fn(e):
            return gamma ** np.floor(e / step)
        return _fn
    
    @classmethod
    def get_exp_fn(cls, gamma=0.99, step=1):
        assert 1 >= gamma >= 0
        assert step > 0
        def _fn(e):
            return gamma ** float(e / step)
        return _fn
    
    @classmethod
    def get_cosine(cls, step=20, min_scale=0.1):
        assert 1 >= min_scale >= 0
        def _fn(e):
            return (1.0 - min_scale) / 2 * (np.cos(np.mod(e / step, 0.5) * np.pi * 2) + 1) + min_scale
        return _fn
    
    @classmethod
    def get_cosine_exp(cls, step=20, min_scale=0.1, gamma=0.5):
        assert 1 >= min_scale >= 0
        assert 1 >= gamma >= 0
        def _fn(e):
            return (
                (1.0 - min_scale) / 2 * (np.cos(np.mod(e / step, 0.5) * np.pi * 2) + 1) + min_scale
            ) * gamma ** float(e / step)
        return _fn
    
    # @classmethod
    # def get_cosine_annealing(cls, step=20, min_scale=0.1):
    #     assert 1 >= min_scale >= 0
    #     def _fn(e):
    #         return (1.0 - min_scale) / 2 * (np.cos(np.mod(e / step, 0.5) * np.pi * 2) + 1) + min_scale
    #     return _fn



# %%
def classification_count_correct(outputs, labels):
    # "outputs have class dim at the last dim and labels are class indices"
    # if first_run_count_correct:
    #     print(outptus.shape, labels.shape)
    #     first_run_count_correct = False
    with torch.no_grad():
        outputs_class = torch.argmax(outputs, dim=-1)
        correct = (outputs_class == labels)
        # correct = correct.sum().item()
        correct = np.array(correct.cpu().detach().numpy()).reshape(-1)
        return correct


# %%
class EmptyWith:
    def __init__(self, *args, **kwargs):
        pass
    
    def __enter__(self, *args, **kwargs):
        return self
    
    def __exit__(self, *args, **kwargs):
        return self

# EW = EmptyWith()

# with EW:
#     a = 10
#     print(a)


# %%
class Network:
    
    optimizer_fns = {
        'sgd': lambda **kwargs: optim.SGD(**{'momentum': 0.9, **kwargs}),
        'adam': lambda **kwargs: optim.Adam(**kwargs),
        'adadelta': lambda **kwargs: optim.Adadelta(**kwargs),
        'adagrad': lambda **kwargs: optim.Adagrad(**kwargs),
        'adamw': lambda **kwargs: optim.AdamW(**kwargs),
        'adabelief': lambda **kwargs: AdaBelief(**{'eps': 1e-16, 'betas': (0.9,0.999), 'weight_decouple': True, 'rectify': True, **kwargs})
    }
    
    metrics_best_classification = {
        'acc': {
            'value': 0.0,
            'higher_is_better': True,
        }
    }
    
    log_print_info = {
        None: {
            'len': 0,
        }
    }
    
    def __init__(self,
                model='wide_resnet50_2',
                frozen_model_bottom=[],
                frozen_model_top=[],
                opt='sgd',
                loss_fn=lambda *args: 0.0,
                lr=0.001,
                lr_type='step',
                lr_step=10,
                lr_gamma=0.5,
                lr_scale=0.1,
                # pretrained=True,
                device='cuda',
                metrics=[],
                # metrics_best=None,
                # stats={},
                info={},
                telem={},
                stats_fp=None,
                epochs=1,
                earlystop_epoch=0,
                splits=['train', 'val'],
                use_default_acc=True,
                use_default_loss=True,
                ds=None,
                ):
        if isinstance(model, nn.Module):
            self.model = model
        else:
            raise ValueError('model must be of type <torch.nn.Module>')
            # self.model = self.get_model(arch=model, pretrained=pretrained)
        
        self.ds = None
        _samples = None
        if ds is not None:
            self.ds = ds
            _samples = {
                'train': len(self.ds.loaders['train'].dataset),
                'val': len(self.ds.loaders['test'].dataset),
            }
        self.splits = splits
        self.epochs = int(max(epochs, 1))
        self.epoch = 0
        self.S = Stats(
            name='',
            info={**info},
            telem={**telem},
            path=stats_fp,
            splits=self.splits,
            metrics=metrics,
            # metrics_hib=[],
            # metrics_lib=[],
            epoch=self.epochs,
            sample=_samples,
            with_bar=['sample'],
            bar_len=20,
            use_default_acc='use_default_acc',
            use_default_loss='use_default_loss',
            timer_progress='sample',
        )
        
        self.frozen_model_bottom = frozen_model_bottom
        if isinstance(self.frozen_model_bottom, nn.Module):
            self.frozen_model_bottom = [self.frozen_model_bottom]
        if not isinstance(self.frozen_model_bottom, list):
            self.frozen_model_bottom = []
        self.frozen_model_top = frozen_model_top
        if isinstance(self.frozen_model_top, nn.Module):
            self.frozen_model_top = [self.frozen_model_top]
        if not isinstance(self.frozen_model_top, list):
            self.frozen_model_top = []
        
        self.loss_fn = loss_fn
        if not callable(self.loss_fn):
            raise ValueError('loss_fn [{}] must be callable'.format(self.loss_fn))
        self.lr = float(max(0.000000001, lr))
        self.optimizer = self.get_optimizer(name=opt, model=self.model, lr=self.lr)
        self.lr_scheduler = self.get_lr_scheduler(
            type=lr_type,
            optimizer=self.optimizer,
            step=lr_step,
            gamma=lr_gamma,
            scale=lr_scale,
            # **kwargs,
        )
        self.device = device
    
    def fit(self,
                dataloader_train=None,
                dataloader_val=None,
                epochs=None,
                epoch_start=None,
                earlystop_epoch=10,
                ):
        val_splits = ['val']
        dataloaders = {
            'train': dataloader_train,
            'val': dataloader_val,
        }
        if dataloader_train is None and dataloader_val is None:
            dataloaders = {
                'train': self.ds.loaders['train'],
                'val': self.ds.loaders['test'],
            }
        if epochs is None:
            epochs = self.epochs
        if epoch_start is None:
            epoch_start = self.epoch
        # self.stats.update(time_start=time.time())
        epoch_final = epoch_start + epochs
        stopping = False
        for _epoch in range(epoch_start, epoch_final):
            _time_start_epoch = time.time()
            self.epoch = max(self.epoch, _epoch)
            print()
            if stopping:
                print()
                print('Stopped Early after {} epochs! Training Fishished.'.format(_epoch))
                break
            
            

            # for _dataloader, _training in zip([dataloader_train, dataloader_val], [True, False]):
            for _split, _dataloader in dataloaders.items():
                if _split not in self.splits:
                    raise ValueError('split [{}] not found in self.splits {}'.format(_split, self.splits))
                _training = _split not in val_splits
                if _split in ['val', 'test'] and _training:
                    print('please check the splits to make sure val_splits is set correctly')
                if _dataloader is None:
                    continue
                _time_start = time.time()
                # _split = ['val', 'train'][int(_training)]
                self.S.set_split(_split)
                self.S.new_round()
                dls = [_dataloader]
                if isinstance(_dataloader, list):
                    dls = _dataloader
                # _stat_agg = []
                for _dl in dls:
                    _stat = self.run_one_epoch(
                        dataloader=_dl,
                        epoch=_epoch,
                        training=_training,
                        print_fps=30,
                        # metrics_best={
                        #     _type: values[_split]
                        #     for _type, values in metrics_best.items()
                        #     if _split in values
                        # },
                        # metrics_best={
                        #     k: v.bests
                        #     for k, v in metrics[_split].items()
                        # },
                        epoch_final=epoch_final,
                        lr_scheduler=self.lr_scheduler,
                    )
                    # _stat_agg.append(_stat)
                    print()
                _time_current = time.time()
                # _stat = {
                #     'epoch': _epoch + 1,
                #     'time_start': _time_start,
                #     'time_finish': _time_current,
                #     'time_cost': _time_current - _time_start,
                #     'loss': float(np.mean([float(v['loss']) for v in _stat_agg])),
                #     'acc': float(np.mean([float(v['acc']) for v in _stat_agg])),
                #     # 'vram': peak_vram_gb,
                # }
                
                if _training:
                    self.lr_scheduler.step()
                    _lr = self.lr_scheduler.get_last_lr()
                self.S.finish_round(save=True)
                    
                # for k, v in metrics[_split].items():
                #     if k in _stat:
                #         metrics[_split][k].update(_stat[k])
                # _SM = self.SM[_split]
                # _stat_metrics = _SM.get_current_stat()
                # self.stats.update(**{_split: _stat_metrics})
                if _split == 'val':
                    best_acc = self.S.SM['val'].total_metrics['acc'].best
                    val_accs = [v['acc'] for v in self.S.SM['val'].get_stat()]
                    if len(val_accs) >= earlystop_epoch:
                        last_accs = val_accs[-earlystop_epoch:]
                        if max(last_accs) < best_acc:
                            stopping = True
                
                # if earlystop_epoch > 0 and _split == 'val':
                #     for metric_type in metrics_best:
                #         if metric_type not in self.stats.stats[_split]:
                #             continue
                #         d = metrics_best[metric_type]
                #         if d.get('_skip_for_earlystopping', False):
                #             continue
                #         w = d.get('_improve_check_window', 5)
                #         values = [v[metric_type] for v in self.stats.stats[_split]]
                #         if len(values) < w + 1:
                #             continue
                #         _last_value = values[-1]
                #         _prev_values = values[-w-1:-1]
                #         _comps = np.sign([_last_value - v for v in _prev_values])
                #         if d.get('_higher_is_better', True) and np.all(_comps < 0):
                #             stopping = True
                #         if not d.get('_higher_is_better', True) and np.all(_comps > 0):
                #             stopping = True
            _time_elapsed_epoch = time.time() - _time_start_epoch
            # self.stats.save()
            
        self.S.finish()
        
        # self.stats.update(
        #     time_finish=time.time(),
        #     completed=True,
        # )
        # self.stats.save()
    
    def run_one_epoch(self,
                dataloader,
                epoch=0,
                training=True,
                print_fps=60,
                metrics_best={},
                epoch_final=None,
                lr_scheduler=None,
                DEBUG=False,
                ):
        
        # DEBUG = not training
        
        losses = []
        _loss_avg = -1.
        batch_count = len(dataloader)
        time_start = time.time()
        time_last_print = time_start
        correct_count = 0
        sample_count = len(dataloader.dataset)
        acc_percent = 0.0
        time_ett = 1.0
        _outputs = []
        _labels = []
        
        if epoch_final is None:
            epoch_final = epoch
        
        self.model.to(self.device)
        for m in self.frozen_model_bottom:
            m.to(self.device)
        for m in self.frozen_model_top:
            m.to(self.device)
        
        self.S.update(
            # _split,
            sample=0,
            epoch=epoch,
        )
        self.S.print()
        
        
        corrects = []
        all_outputs = []
        all_labels = []
        
        sample_count_current = 0
        for batch_index, data in enumerate(dataloader, 0):
            inputs, labels = data
            sample_count_current += len(labels)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            x = inputs
            with torch.no_grad():
                for m in self.frozen_model_bottom:
                    x = m(x)
            
            if training:
                x = self.model(x)
            else:
                with torch.no_grad():
                    x = self.model(x)
            
            with torch.no_grad():
                for m in self.frozen_model_top:
                    x = m(x)
            
            outputs = x
            loss = None
            if training:
                loss = self.loss_fn(outputs, labels)
            else:
                with torch.no_grad():
                    loss = self.loss_fn(outputs, labels)
            
            # if return_values or debug:
            #     _outputs.append(outputs.cpu().detach().numpy())
            #     _labels.append(labels.cpu().detach().numpy())
            
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            _corrects = [int(v) for v in classification_count_correct(outputs, labels)]
            if DEBUG:
                outputs_class = torch.argmax(outputs, dim=-1)
                o = list(np.array(outputs_class.cpu().detach().numpy()).reshape(-1))
                l = list(np.array(labels.cpu().detach().numpy()).reshape(-1))
                all_outputs.extend(o)
                all_labels.extend(l)
            
            _loss_value = float(loss.item())
            losses.append(_loss_value)
            _loss_avg = sum(losses) / len(losses)
            
            # if i % max(20, int(batch_count // 4)) == 0:
            #     peak_vram_gb = max(peak_vram_gb, get_vram_fn())
            
            _progress_percent = (batch_index + 1) / batch_count * 100
            
            time_current = time.time()
            time_elapsed = time_current - time_start
            time_ett = time_elapsed / max(0.000001, _progress_percent / 100)
            time_eta = time_ett * (1 - _progress_percent / 100)
            
            # metrics_acc.update(_corrects)
            # pl.update(
            #     step=batch_index + 1,
            #     # acc=_corrects,
            #     loss=_loss_value,
            # )
            _lr = None
            if training and lr_scheduler:
                _lr = lr_scheduler.get_last_lr()
            
            # _SM.update(
            #     acc=_corrects,
            #     loss=_loss_value,
            #     lr=_lr,
            #     sample=sample_count_current,
            #     # epoch=epoch,
            # )
            self.S.update(
                # _split,
                acc=[v for v in _corrects],
                loss=_loss_value,
                lr=_lr,
                sample=sample_count_current,
                # epoch=epoch,
            )
            if print_fps > 0 and time_current >= time_last_print + 1 / print_fps:
                time_last_print = max(time_last_print + 1/30, time_current - 1/60)
                self.S.print()
                # _SM.print()
                # pl.print()
            
        time_current = time.time()
        time_elapsed = time_current - time_start
        
        if DEBUG:
            assert training == False
            all_outputs = np.array(all_outputs)
            all_labels = np.array(all_labels)
            print()
            print('got outputs shape {} and labels shape {}'.format(all_outputs.shape, all_labels.shape))
            all_corrects = (all_outputs == all_labels).astype(np.int32)
            print('acc: ', float(np.mean(all_corrects)))
            print('examples:')
            r = 1
            c = 20
            for i in range(1):
                print('output:', all_outputs[i * c: (i+1) * c])
                print('label: ', all_labels[i * c: (i+1) * c])
                
            
        
    
    @classmethod
    def get_optimizer(cls, name='sgd', model=None, lr=0.001, **kwargs):
        _params = model.parameters()
        if name in cls.optimizer_fns:
            return cls.optimizer_fns[name](params=_params, lr=lr, **kwargs)
        else:
            return None
            # raise ValueError('optimizer `{}` is currently not supported! must be one of [ {} ]'.format(arch, ' | '.join([
            #     str(v) for v in cls.optimizer_fns.keys()
            # ])))
    
    @classmethod
    def get_lr_scheduler(cls, optimizer=None, type='step', step=10, gamma=0.5, scale=0.1, **kwargs):
        if type == 'none' or not isinstance(type, str):
            _fn = lambda e: e
        elif type == 'step':
            _fn = LRSchedule.get_step_fn(step=step, gamma=gamma)
        elif type == 'exp':
            _fn = LRSchedule.get_exp_fn(gamma=gamma)
        elif type == 'cos':
            _fn = LRSchedule.get_cosine(step=step, min_scale=scale)
        elif type == 'cos_exp':
            _fn = LRSchedule.get_cosine_exp(step=step, min_scale=scale, gamma=gamma)
        else:
            raise NotImplementedError(f'lr scheduler {type} has not been implemented')
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_fn)
        return lr_scheduler
    
    @classmethod
    def get_output_shape(cls, model, input_shape=[1, 3, 224, 224], device='cuda'):
        # print('getting the output shape of the backbone model with {}'.format(input_shape))
        model.to(device)
        output_shape = model(torch.rand(*input_shape).to(device)).data.shape
        # num_features = int(output_shape[-1])
        # print('found shape: [N, {}]'.format(num_features))
        return output_shape
    
    @classmethod
    def save_stats(cls, _stats={}, fp_json_master='./logs/master2.json', time_stamp='<latest>'):
        raise NotImplementedError('method no longer supported')
        # fp = args['stats_json']
        # fp_master = args['master_stats_json']
        fp_master = fp_json_master
        
        if isinstance(fp_master, str) and fp_master.endswith('.json'):
            _stats_master = {
                time_stamp: _stats,
            }
            if os.path.isfile(fp_master):
                try:
                    _stats_old = json.load(open(fp_master, 'r'))
                    if isinstance(_stats_old, dict):
                        # _stats_old[time_stamp] = _stats
                        _stats_master = {
                            **_stats_old,
                            **_stats_master,
                        }
                except:
                    pass
            _dp = os.path.split(fp_master)[0]
            if not os.path.isdir(_dp):
                os.makedirs(_dp)
            _ = json.dump(_stats_master, open(fp_master, 'w'), indent=4)



# %%
# cnn_archs = [
#     'dino_vits8',
#     'resnext50_32x4d',
#     # 'resnext101_32x8d',
#     'wide_resnet50_2',
#     # 'wide_resnet101_2',
#     # 'densenet201',
#     # 'densenet169',
#     'densenet121',
# ]
# for a in cnn_archs:
#     model = Network.get_model(a)
#     model.to('cuda')
#     r = {}
#     r['arch'] = a
#     b = Network.get_output_shape(model)
#     r['output_full'] = b
#     r['fc'] = None
#     r['output_without_top'] = None
#     try:
#         r['fc'] = str(model.fc)
#         model.fc = nn.Identity()
#         c = Network.get_output_shape(model)
#         r['output_without_top'] = c
#     except:
#         pass
    
#     print(json.dumps(r, indent=4))

# %%

# %%    