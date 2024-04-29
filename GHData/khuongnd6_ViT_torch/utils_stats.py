# %%
from typing import Counter
import numpy as np
# import pandas as pd
import json, os, re, string, time, datetime

from torch._C import Value


# %%
class TimerLog:
    def __init__(self, format='time[{elapsed:.1f}/{total:.1f}{unit}]', format_finish=None, auto_start=False, **kwargs):
        self.format = format
        self.format_finish = format_finish
        self.time_start = time.time()
        self.time_finish = None
        self.progress = 0.000000001
        self.started = False
        self.finished = False
        if auto_start:
            self.start()
        self.update()
    
    def __str__(self) -> str:
        _format = self.format
        if isinstance(self.format_finish, str) and self.finished:
            _format = self.format_finish
        return _format.format(**self.get_dict_str())
    
    def get_dict_str(self):
        self.update()
        secs_limit = max(self.time_elapsed, self.time_total)
        _unit = self.format_time(secs_limit)[-1]
        return {
            'start': self.time_start,
            'current': self.time_current,
            'elapsed': self.format_time(self.time_elapsed, secs_limit)[0],
            'total': self.format_time(self.time_total, secs_limit)[0],
            'remain': self.format_time(self.time_remain, secs_limit)[0],
            'unit': _unit,
        }
    
    def get_dict(self):
        self.update()
        # secs_limit = max(self.time_elapsed, self.time_total)
        # _unit = self.format_time(secs_limit)[-1]
        return {
            'time_start': self.time_start,
            'time_current': self.time_current,
            'time_finish': self.time_finish,
            'time_elapsed': self.time_elapsed,
            'time_cost': self.time_elapsed,
            'time': self.time_elapsed,
            'time_total': self.time_total,
            'time_remain': self.time_remain,
            # 'unit': _unit,
        }
    
    def reset(self):
        self.start()
    
    def start(self):
        self.started = True
        self.finished = False
        self.time_start = time.time()
        self.update()
        
    def update(self, progress=None, **kwargs):
        if self.finished:
            return False
        if not self.started:
            raise ValueError('TimerLog not started')
        self.time_current = time.time()
        self.time_elapsed = self.time_current - self.time_start
        if progress:
            self.progress = float(np.clip(progress, 0.0001, 1.0))
        # self.progress
        self.time_total = self.time_elapsed / self.progress
        self.time_remain = self.time_total - self.time_elapsed
        self.value = self.time_elapsed
    
    def finish(self):
        if not self.started:
            raise ValueError('TimerLog not started')
        self.finished = True
        self.update()
        self.time_finish = self.time_current
    
    @classmethod
    def format_time(self, secs=0, secs_limit=0):
        units = {
            'd': 864000,
            'h': 3600,
            'm': 60,
            's': 1,
        }
        for k, v in units.items():
            if k == 's' or secs >= v or secs_limit >= v:
                return [secs / v, k]


class CounterLog:
    box_chars = ['█', '▉','▊','▋', '▌', '▍', '▎', '▏', ' '][::-1]
    def __init__(self,
                total=10,
                value=0,
                format='[{value}/{total}]',
                format_start=None,
                format_finish=None,
                bar_len=40,
                **kwargs,
                ):
        self.initial_value = value
        self.value = value
        self.total = total
        self.bar_len = bar_len
        self.format = format
        self.format_start = format
        self.format_finish = format
        if format_start:
            self.format_start = format_start
        if format_finish:
            self.format_finish = format_finish
    
    def reset(self):
        self.value = self.initial_value
        self.update()
    
    def finish(self):
        self.value = self.total
        self.update()
    
    def update(self, value=None, total=None, **kwargs):
        if value:
            self.value = value
        if total:
            self.total = total
        self.progress = self.value / self.total
        self.progress_percent = self.progress * 100
        
        self.bar = self.get_box_string(self.progress, self.bar_len)
    
    def get_dict(self):
        self.update()
        return {
            'value': self.value,
            'total': self.total,
            'progress': self.progress,
            'progress_percent': self.progress_percent,
            'bar': self.bar,
        }
    
    def __str__(self) -> str:
        return self.format.format(**self.get_dict())
    
    @classmethod
    def get_box_string(cls, value=0.2, length=10):
        box_count = float(np.clip(value, 0.0, 1.0) * length)
        full_box_count = int(np.floor(box_count))
        partial_box_count = box_count - full_box_count
        partial_box_index = int(np.clip(
            np.round(partial_box_count * (len(cls.box_chars) - 1)),
            0,
            len(cls.box_chars) - 1,
        ))
        s = cls.box_chars[-1] * full_box_count + cls.box_chars[partial_box_index]
        s = s.ljust(length, ' ')[:length]
        return s


class Metrics:
    def __init__(self,
                name='metric',
                higher_is_better=True,
                def_value=None,
                format='{value:6.4f}',
                best_str='(best)',
                prev_best=None,
                **kwargs,
                ):
        self.name = str(name)
        self.best_str = best_str
        self.format = format
        self.prev_best = prev_best
        self.higher_is_better = bool(higher_is_better)
        self.def_value = def_value or 0.0
        self.is_best = False
        if def_value is None:
            def_value = 0.0 if higher_is_better else 999.999
        self.reset(def_value)
    
    def reset(self, def_value=None):
        if def_value:
            self.def_value = def_value
        self.values = []
        self.best = self.def_value
        self.best_index = -1
        self.avg = self.def_value
        self.value = self.def_value
        self.percent = self.avg * 100.
        self.percent_best = self.best * 100.
        self.is_best = False
    
    def update(self, value=None, last_value_only=False):
        if value is not None:
            if last_value_only:
                if len(self.values) >= 1:
                    if isinstance(value, (np.ndarray, list)):
                        value = value[-1]
                    self.values[-1] = value
            else:
                if isinstance(value, np.ndarray):
                    value = [v for v in value]
                if isinstance(value, list):
                    self.values.extend(value)
                else:
                    self.values.append(value)
        if len(self.values) >= 1:
            if self.higher_is_better:
                
                try:
                    self.best = max(self.values)
                except:
                    # print(self.values)
                    pass
                self.best_index = int(np.argmax(self.values))
            else:
                self.best = min(self.values)
                self.best_index = int(np.argmin(self.values))
            self.avg = float(np.mean(self.values))
            self.value = self.avg
            self.percent = self.avg * 100.
            self.percent_best = self.best * 100.
            if self.prev_best:
                self.is_best = self.avg > self.prev_best
    
    @property
    def last_value(self):
        if len(self.values) >= 1:
            return self.values[-1]
        else:
            return self.def_value
    
    def get_dict(self):
        return {
            'avg': self.avg,
            'value': self.value,
            'best': self.best,
            'percent': self.percent,
            'percent_best': self.percent_best,
            'isbest': self.best_str if (self.is_best and self.with_best) else '',
        }
    
    def __str__(self) -> str:
        return self.format.format(**self.get_dict())


class StatMetrics:
    box_chars = ['█', '▉','▊','▋', '▌', '▍', '▎', '▏', ' '][::-1]
    
    def __init__(self,
                name='<unknown>',
                metrics=[],
                metrics_hib=[],
                metrics_lib=[],
                counters={},
                with_bar=[],
                bar_len=20,
                use_default_acc=True,
                use_default_loss=True,
                timer_progress='',
                # **kwargs,
                ):
        self.name = str(name)
        self.metric_names = []
        self.metrics = {}
        self.total_metrics = {}
        self.metrics_kwargs = {}
        self.fields = {}
        
        for k, v in counters.items():
            if isinstance(v, int) and v > 0 and isinstance(k, str):
                if k == with_bar or (isinstance(with_bar, list) and k in with_bar):
                    _kwargs = {
                        'total': v,
                        'value': 0,
                        'format': str(k)[:20] + '[{value}/{total}][{bar}]',
                        'bar_len': bar_len,
                        '_class_fn': CounterLog,
                    }
                else:
                    _kwargs = {
                        'total': v,
                        'value': 0,
                        'format': str(k)[:20] + '[{value}/{total}]',
                        '_class_fn': CounterLog,
                    }
                self.metrics_kwargs[k] = _kwargs
                # self.metrics[k] = self.fields[k]
                # self.total_metrics[k] = self.fields[k]
        for _metrics, _hib in zip([metrics, metrics_hib, metrics_lib], [True, True, False]):
            if isinstance(_metrics, list):
                for m in _metrics:
                    if isinstance(m, str):
                        _name = m
                        _kwargs = {
                            'name': _name,
                            'higher_is_better': _hib,
                            'def_value': 0.0 if _hib else 999.999,
                            'format': str(_name)[:20] + '[{avg:8.6f}]',
                            '_class_fn': Metrics,
                        }
                    elif isinstance(m, dict):
                        _name = str(m.get('name', ''.join(np.random.choice(list(string.digits), 6))))
                        _kwargs = {
                            **m,
                            'name': _name,
                            '_class_fn': Metrics,
                        }
                    else:
                        raise ValueError('metric params must be a list of str')
                    
                    self.metrics_kwargs[_name] = _kwargs
                    # self.metrics[_name] = Metrics(**_kwargs)
                    # self.metrics[_name] = None
                    # self.total_metrics[_name] = Metrics(**_kwargs)
                    # self.fields[_name] = self.metrics[_name]
            else:
                pass
        
        if use_default_loss and 'loss' not in self.metrics:
            _name = 'loss'
            self.metrics_kwargs[_name] = {
                'name': _name,
                'higher_is_better': False,
                'def_value': 999.999,
                'format': 'loss[{avg:8.6f}]',
                'best_str': '(best)',
                'prev_best': 999.999,
                '_class_fn': Metrics,
            }
            # self.metrics[_name] = None
            # self.total_metrics[_name] = Metrics(**self.metrics_kwargs[_name])
            # self.fields[_name] = self.metrics[_name]
        
        if use_default_acc and 'acc' not in self.metrics:
            _name = 'acc'
            self.metrics_kwargs[_name] = {
                'name': _name,
                'higher_is_better': True,
                'def_value': 0.0,
                'format': 'acc[{percent:6.2f}%{isbest}]',
                'best_str': '(best)',
                'prev_best': 0.0,
                '_class_fn': Metrics,
            }
            # self.metrics[_name] = None
            # self.total_metrics[_name] = Metrics(**self.metrics_kwargs[_name])
            # self.fields[_name] = self.metrics[_name]
        
        self.timer_progress = timer_progress
        self.metrics_kwargs['time'] = {
            'format': 'time[{elapsed:.1f}/{total:.1f}{unit}]',
            'format_finish': 'time[{elapsed:.1f}{unit}]',
            'auto_start': True,
            '_class_fn': TimerLog,
        }
        # self.timer = TimerLog(**self.metrics_kwargs['time'])
        # self.metrics['time'] = None
        # self.fields['time'] = None
        self.total_metrics['time'] = Metrics(name='time')
        # if self.timer_progress in self.fields:
        # self.metric_names = [m for m in self.metrics.keys()]
        self.metric_names = [m for m in self.metrics_kwargs.keys()]
        self.reset()
    
    def reset(self):
        for k, m in self.total_metrics.items():
            m.reset()
        for k, m in self.metrics.items():
            if m:
                m.reset()
        for k, v in self.fields.items():
            if v:
                v.reset()
        self.index = -1
        self.all_metrics = []
        # self.new_round()
    
    def new_round(self):
        new_metrics = {}
        self.update_values(last_value_only=True)
        for _name in self.metric_names:
            class_fn = self.metrics_kwargs[_name]['_class_fn']
            if not class_fn:
                if _name in ['time']:
                    class_fn = TimerLog
                else:
                    class_fn = Metrics
            new_metric = class_fn(**self.metrics_kwargs[_name])
            if _name not in self.total_metrics or self.total_metrics[_name] is None:
                if class_fn is CounterLog:
                    total_class_fn = CounterLog
                else:
                    total_class_fn = Metrics
                self.total_metrics[_name] = total_class_fn(**self.metrics_kwargs[_name])
            self.metrics[_name] = new_metric
            new_metrics[_name] = new_metric
            if _name in self.fields:
                self.fields[_name] = new_metric
        
        for _name in self.fields:
            if _name in ['time']:
                continue
            if _name not in self.metrics:
                self.fields[_name].reset()
        
        self.update_values(last_value_only=False)
        
        self.all_metrics.append({k: v for k, v in new_metrics.items()})
        self.update_values()
        self.index += 1
        return len(self.all_metrics)
    
    def finish_round(self):
        # self.timer.update()
        self.metrics['time'].finish()
    
    def update(self, _name='', _value=None, **kwargs):
        self.metrics['time'].update()
        if self.metrics['time']:
            _progress = None
            if self.timer_progress in self.metrics:
                assert isinstance(self.metrics[self.timer_progress], CounterLog)
                _progress = self.metrics[self.timer_progress].progress
            self.metrics['time'].update(progress=_progress)
        for k, v in kwargs.items():
            if k in self.metrics:
                self.metrics[k].update(v)
            elif k in self.fields:
                self.fields[k].update(v)
        
        if _name in self.metric_names and _value is not None:
            self.metrics[_name].update(_value)
        
        self.update_values()
    
    def update_values(self, last_value_only=True):
        for _name in self.metric_names:
            if _name not in self.metrics:
                continue
            if self.metrics[_name] is None:
                continue
            self.metrics[_name].update()
            _avg = float(self.metrics[_name].value)
            self.total_metrics[_name].update(_avg, last_value_only=last_value_only)
            # print(_name, last_value_only, len(self.total_metrics[_name].values))
    
    @property
    def stats(self):
        self.total_metrics()
    
    def __str__(self):
        return self.get_str()
    
    def get_str(self):
        ss = [self.name]
        for k, v in self.fields.items():
            v.update()
            s = str(v)
            ss.append(s)
        for k, v in self.metrics.items():
            if k in self.fields:
                continue
            v.update()
            s = str(v)
            ss.append(s)
        return ' '.join(ss) + ' ' * 10
    
    def print(self, in_place=True):
        _str = self.get_str()
        if in_place:
            print('\r' + _str, end='')
        else:
            print(_str)
    
    def get_stat(self,):
        return [
            self.get_current_stat(m)
            for m in self.all_metrics
        ]
    
    def get_current_stat(self, metrics=None):
        if metrics is None:
            _metrics = self.metrics
        else:
            _metrics = metrics
        return {
            **{
                k: v.value
                for k, v in {**_metrics}.items()
            },
            **{
                k: _metrics['time'].get_dict()[k]
                for k in ['time_start', 'time_finish', 'time_cost']
            }
        }


class Stats:
    def __init__(self,
                name='<unknown>',
                info={},
                telem={},
                path='./stats_temp.json',
                splits=['train', 'val'],
                metrics=[],
                metrics_hib=[],
                metrics_lib=[],
                counters={},
                with_bar=[],
                bar_len=20,
                use_default_acc=True,
                use_default_loss=True,
                epoch=None,
                sample=None,
                timer_progress='',
                **kwargs):
        
        self.path = path
        self.splits = splits
        self.split = splits[0]
        
        self.SM = {
            _split: StatMetrics(
                name=str(_split),
                metrics=metrics,
                metrics_hib=metrics_hib,
                metrics_lib=metrics_lib,
                counters={
                    **counters,
                    **({'epoch': epoch} if epoch else {}),
                    **({'sample': (sample[_split] if isinstance(sample, dict) else sample)} if sample is not None else {}),
                },
                with_bar=with_bar,
                bar_len=bar_len,
                use_default_acc=use_default_acc,
                use_default_loss=use_default_loss,
                timer_progress=timer_progress,
            )
            for _split in splits
        }
                
        self.info = {}
        _time_current = time.time()
        self.telem = {
            'hardware': '<unknown>',
            'sample_count_train': 1,
            'sample_count_val': 1,
            'completed': False,
            'time_stamp': '<unknown>',
            'time_start': _time_current,
            'time_finish': None,
            'time_elapsed': None,
            'time_updated': _time_current,
            'bs': None,
            'mode': '<unknown>',
            # 'mode': str(mode) if mode else '<unknown>',
        }
        self.results = {
            'epochs': 0,
            'epoch.time': 0.0,
            'epoch.sample_time': 0.0,
            
            **{
                '{}.{}'.format(_split, k): v
                for k, v in {
                    # 'acc': 0.0,
                    # 'loss': 999.999,
                    'time': 0.0,
                    'sample_time': 0.0,
                }.items()
                for _split in self.splits
            },
        }
        # for _split in self.splits:
        #     for m in metrics:
        #         if isinstance(m, str):
        #             M = Metrics(name=m, higher_is_better=True, def_value=0.0)
        #         else:
        #             assert isinstance(m, Metrics)
                
        self.epoch = 0
        self.logs = {
            _split: []
            for _split in self.splits
        }
        self.update_log(info=info, telem=telem, **kwargs)
    
    def update(self, split=None, *args, **kwargs):
        split = self.get_split(split)
        self.SM[split].update(**kwargs)
    
    def update_log(self, epoch=None, info=None, telem=None, ds=None, **kwargs):
        if epoch:
            self.epoch = max(self.epoch, int(epoch))
        if ds:
            self.telem['sample_count_train'] = ds.info['sample_count']['train']
            self.telem['sample_count_val'] = ds.info['sample_count']['test']
        if info:
            self.info = {
                **self.info,
                **info,
            }
        if telem:
            self.telem = {
                **self.telem,
                **telem,
            }
        
        for k, v in kwargs.items():
            for _split in self.splits:
                if k == f'{_split}_all':
                    self.logs[_split] = [*v]
                    continue
                if k == _split:
                    self.logs[_split].append(v)
                    continue
            if k in self.telem:
                self.telem[k] = v
                continue
        
        if self.telem['time_start'] and self.telem['time_finish']:
            self.telem['time_elapsed'] = self.telem['time_finish'] - self.telem['time_start']
        self.telem['time_updated'] = time.time()
        
        self.update_results()
    
    def update_results(self):
        
        split_time = {}
        split_sample_time = {}
        split_sample_count = {}
        epoch_time = 0.0
        for _split, _list in self.logs.items():
            split_sample_count[_split] = self.telem.get(f'sample_count_{_split}')
            if split_sample_count[_split] is None:
                split_sample_count[_split] = 1
            split_time[_split] = 0.000001
            if len(_list) > 0:
                try:
                    split_time[_split] = float(np.mean([
                        v['time_finish'] - v['time_start']
                        for v in _list
                    ]))
                except:
                    pass
                try:
                    split_sample_count[_split] = float(np.mean([v['sample'] for v in _list]))
                    # print(split_sample_count[_split], _list)
                except:
                    pass
            split_sample_count[_split] = max(1, int(split_sample_count[_split]))
            split_sample_time[_split] = split_time[_split] / split_sample_count[_split]
                
            epoch_time += split_time[_split]
        
        # print('updating time with:', json.dumps(split_time))
        self.results = {
            **self.results,
            **{
                'epochs': self.epoch,
                **{
                    f'{_split}.{k}': max([v[k] for v in _logs] + [def_value])
                    for _split, _logs in self.logs.items()
                    for k, def_value in {'acc': 0.0, 'loss': 9.999}.items()
                },
                
                # 'train.acc': max([v['acc'] for v in self.logs[_split]] + [0.]),
                # 'val.acc': max([v['acc'] for v in self.val] + [0.]),
                # 'train.loss': min([v['loss'] for v in self.train] + [999.999]),
                # 'val.loss': min([v['loss'] for v in self.val] + [999.999]),
                
                'epoch.time': epoch_time,
                **{
                    f'{_split}.time': split_time[_split]
                    for _split in self.splits
                },
                # 'train.time': train_time,
                # 'val.time': val_time,
                **{
                    f'{_split}.sample_time': split_sample_time[_split]
                    for _split in self.splits
                },
                # 'train.sample_time': train_time / max(1, self.telem['sample_count_train'] or 1),
                # 'val.sample_time': val_time / max(1, self.telem['sample_count_val'] or 1),
            },
        }
    
    @property
    def stats(self):
        return {
            'info': {**self.info},
            'telem': {**self.telem},
            'results': {**self.results},
            **{k: [*v] for k, v in self.logs.items()},
        }
    
    def save(self, path=None):
        _stats = self.stats
        _path = path or self.path
        assert isinstance(_path, str)
        assert _path.endswith('.json')
        _dp = os.path.split(_path)[0]
        if not os.path.isdir(_dp):
            os.makedirs(_dp)
        _ = json.dump(_stats, open(_path, 'w'), indent=4)
        # print('saved at {}'.format(_path))
        return True
    
    def new_round(self, split=None):
        # create and start a new round
        split = self.get_split(split)
        self.SM[split].new_round()
    
    def finish_round(self, split=None, save=True):
        # finishes the current round
        split = self.get_split(split)
        # print('update log [{}]'.format(split))
        # print(S.SM[split].get_stat())
        self.SM[split].finish_round()
        self.update_log(
            epoch=self.SM[split].metrics['epoch'].value,
            # train=None,
            # val=None,
            # train_all=None,
            # val_all=None,
            **{
                '{}_all'.format(split): self.SM[split].get_stat()
            },
            # info=None,
            # telem={
                
            # },
            # ds=None,
            # **kwargs,
        )
        self.save()
    
    def finish(self, save=True):
        # split = self.get_split()
        _epoch = max([self.SM[_split].metrics['epoch'].value for _split in self.splits])
        times = [
            self.SM[_split].get_stat()[0][k]
            for _split in self.splits
            for k in ['time_start']
        ]
        _time_start = min(times)
        _time_current = time.time()
        self.update_log(
            epoch=_epoch,
            # train=None,
            # val=None,
            # train_all=None,
            # val_all=None,
            # info=None,
            telem={
                'time_start': _time_start,
                'time_finish': _time_current,
                'time_elapsed': _time_current - _time_start,
                # 'batch_size': None,
            },
            # ds=None,
            # **kwargs,
        )
        if save:
            self.save()
    
    def print(self, split=None, in_place=True):
        split = self.get_split(split)
        self.SM[split].print(in_place=in_place)
    
    def get_split(self, split=None, check_SM=True):
        if split is None:
            split = self.split
        if check_SM:
            assert split in self.SM
        return split
    
    def set_split(self, split=None):
        if split is None:
            split = self.split
        assert split in self.SM
        self.split = split
        return split
        

# %%
# epochs = 2
# bs = 20
# batch_count = 100
# S = Stats(
#     name='test_master',
#     info={'random_info': 123, 'other_info': 'abc'},
#     telem={
#         'hardware': '1x3090',
#         'bs': bs,
#     },
#     path='./logs/stats_test.json',
#     splits=['train', 'val'],
#     metrics=['lr'],
    
#     # metrics_hib=[],
#     # metrics_lib=[],
#     counters={
#         'epoch': epochs,
#         'sample': batch_count * bs,
#     },
#     with_bar=['sample'],
#     bar_len=20,
#     use_default_acc=True,
#     use_default_loss=True,
#     timer_progress='sample',
# )

# _accs = []
# print()
# for epoch in range(epochs):
#     _accs.append([])
#     for _split in S.splits:
#         S.set_split(_split)
#         if epoch > 0 or True:
#             S.new_round()
#         for step in range(batch_count):
#             _lr = 0.1 * (0.9 ** epoch)
#             _acc = 1.0 - np.random.sample() / (2 - epoch / 10 * 1.5)
#             _loss = 1.0 * np.random.sample() / (1 - epoch / 10 * 0.8)
#             S.update(
#                 _split,
#                 acc=_acc,
#                 loss=_loss,
#                 lr=_lr,
#                 sample=(step + 1) * bs,
#                 epoch=epoch,
#             )
#             S.print()
#             _accs[-1].append(_acc)
#             time.sleep(np.random.sample() * 0.02 + 0.01)
#         S.finish_round(save=True)
#         print()

# S.finish()

# # %%


# # # %%
# # a = [
# #     {
# #         k: v.avg
# #         for k, v in d.items()
# #     }
# #     for d in SM.all_metrics
# # ]
# # a

# # %%
# import pandas as pd

# # %%
# df = pd.DataFrame([
#     {
#         'split': _split,
#         **{
#             k: v % 1000
#             for k, v in m['time'].get_dict().items()
#         },
#     }
#     for _split, _sm in S.SM.items()
#     for m in _sm.all_metrics
# ]).sort_values(['time_start'])
# df

# %%

