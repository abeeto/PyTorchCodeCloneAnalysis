# %%
import time, json, re, string, datetime

# %%
class ProgressBarText:
    def __init__(self, value=''):
        self.update(value)
        self.update_txt()
    
    def update(self, value):
        self.value = value
    
    def update_txt(self):
        if self.value is None:
            self.txt = ''
            return False
        self.txt = str(self.value)
    
    def __call__(self):
        self.update_txt()
        if not isinstance(self.txt, str):
            self.txt = None
            return ''
        return self.txt

class ProgressBarInt(ProgressBarText):
    def __init__(self, value=0, fill_length=None, fill_char=' '):
        self.fill_length = fill_length
        self.fill_char = str(fill_char)[0]
        self.update(value)
        self.update_txt()
    
    def update(self, value):
        self.value = value
    
    def update_txt(self):
        if self.value is None:
            self.txt = ''
            return False
        self.txt = str(self.value)
        if isinstance(self.fill_length, int):
            self.txt = self.txt.rjust(self.fill_length, self.fill_char)

class ProgressBarFloat(ProgressBarText):
    def __init__(self, value=0.0, format='{:.1f}'):
        self.format = str(format)
        self.update(value)
        self.update_txt()
    
    def update(self, value):
        self.value = value
    
    def update_txt(self):
        if self.value is None:
            self.txt = ''
            return False
        self.txt = self.format.format(self.value)

class ProgressBarTime(ProgressBarText):
    def __init__(self, value=0.0, format='{:.1f}', rounding=False):
        self.format = str(format)
        self.rounding = bool(rounding)
        self.update(value)
        self.update_txt()
    
    def update(self, value):
        self.value = value
    
    def update_txt(self):
        if self.value is None:
            self.txt = ''
            return False
        magnitudes = {
            'y': 31536000,
            'm': 86400 * 30,
            # 'w': 86400 * 7,
            'd': 86400,
            'h': 3600,
            'm': 60,
            's': 1,
            'ms': 0.001,
        }
        for i, (m, v) in enumerate(magnitudes.items()):
            if self.value >= v * 1.0 or i == len(magnitudes) - 1:
                _value = self.value / v
                _mag = m
                break
        self.txt = self.format.format(_value) + str(_mag)
    

class ProgressBarTextMulti(ProgressBarText):
    def __init__(self):
        # super
        pass

class ProgressBarTextMultiInt(ProgressBarTextMulti):
    def __init__(self, value=[], format=None, fill_length=1, fill_char=' '):
        self.fill_length = fill_length
        self.fill_char = str(fill_char)[0]
        self.value = []
        self.update(value)
        self.format = format
        if not isinstance(self.format, str):
            self.format = '[{}, {}]' * len(self.value)
        self.update_txt()
    
    def update(self, value):
        if not isinstance(value, list):
            value = [value]
        for i, v in enumerate(list(value)):
            if i < len(self.value):
                self.value[i] = v
            else:
                self.value.append(v)
        if isinstance(self.fill_length, int):
            self.fill_length = max(self.fill_length, *[len(str(v)) for v in self.value], 1)
    
    def update_txt(self):
        # self.txt = str(self.value)
        self.txt = self.format.format(*[v for v in self.value])
        if isinstance(self.fill_length, int):
            self.txt = self.format.format(*[str(v).rjust(self.fill_length, self.fill_char) for v in self.value])

class ProgressBar:
    def __init__(self, s='', *args, **kwargs):
        self.s = s
        self.args = []
        self.kwargs = {}
        self.update(*args, **kwargs)
        self.fill_length = 1
    
    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        return self.print(True)
    
    def update(self, *args, **kwargs):
        for i, v in enumerate(args):
            if i >= len(self.args):
                self.args.append(v)
            elif isinstance(self.args[i], ProgressBarText):
                self.args[i].update(v)
            else:
                self.args[i] = v
        for k, v in kwargs.items():
            if k not in self.kwargs:
                self.kwargs[k] = v
            elif isinstance(self.kwargs[k], ProgressBarText):
                self.kwargs[k].update(v)
            else:
                self.kwargs[k] = v
        return True
    
    def get_strs_from_Text(self, v):
        if isinstance(v, ProgressBarTextMulti):
            return [v()]
        elif isinstance(v, ProgressBarText):
            return [v()]
        else:
            return [str(v)]
    
    def print(self, fill_length=1, printing=True):
        str_args = [
            v1
            for v in self.args
            for v1 in self.get_strs_from_Text(v)
        ]
        str_kwargs = {
            k: self.get_strs_from_Text(v)[0]
            for k, v in self.kwargs.items()
        }
        _str = self.s.format(*str_args, **str_kwargs)
        self.fill_length = max(fill_length, self.fill_length, len(_str))
        if printing:
            print('\r' + _str.ljust(self.fill_length, ' '), end='')
        return _str
