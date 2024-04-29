import time, argparse, json, os

class ARGS:
    def __init__(self, config=[]):
        self._config = list(config)
        self.args_list = []
        self.args = {}
        # self.update_from_dict(self._config)
        
        self.info = {}
        self.update_from_list(self._config)
        self.update_info(self._config)
    
    @classmethod
    def isnotebook(cls,):
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False
    
    def update_info(self, config=[]):
        for v in config:
            _key = v[0]
            if isinstance(v[0], (list, tuple)):
                _key = list(v[0])[0]
            _value = v[1]
            self.info = {
                **self.info,
                _key: _value,
            }
    
    def update_from_list(self, config=[]):
        for v in config:
            _keys = [v[0]]
            if isinstance(v[0], (list, tuple)):
                _keys = list(v[0])
            _value = v[1]
            if len(v) >= 3:
                _type = v[2]
                # print(_keys, _value, type(_value), _type)
                # print(config)
                if _type is bool:
                    _value = bool(_value)
                if not isinstance(_value, list):
                    assert isinstance(_value, _type), 'arg `{}` must be of type <{}>, got {}'.format(_keys[0], v[2], _value)
            if len(v) >= 4:
                _set = v[3]
                if isinstance(_set, (list, tuple, set)):
                    _set = list(_set)
                    if len(_set) >= 1:
                        assert _value in _set, 'arg `{}` must be one of [{}]'.format(_keys[0], ' | '.join(v[3]))
            self.args = {
                **self.args,
                **{
                    _key: _value
                    for _key in _keys
                },
            }
        self.update_info(config)
    
    def update_from_dict(self, _dict={}):
        for k, v in _dict.items():
            _keys = [k]
            if isinstance(k, (list, tuple)):
                _keys = list(k)
            self.args = {
                **self.args,
                **{
                    _key: v
                    for _key in _keys
                },
            }
    
    def set_and_parse_args(self, name='ARGS'):
        if self.isnotebook():
            print('[warning] notebook runtime detected, skip parsing args')
            return None
        _config_indices = {}
        self._parser = argparse.ArgumentParser(name)
        for i, v in enumerate(self._config):
            _kwargs = {}
            _keys = [v[0]]
            if isinstance(v[0], (list, tuple)):
                _keys = list(v[0])
            _value = v[1]
            _type = None
            if len(v) >= 3:
                _type = v[2]
            elif len(v) == 2:
                _type = type(_value)
            if _type is bool:
                if bool(_value):
                    _kwargs['action'] = 'store_false'
                else:
                    _kwargs['action'] = 'store_true'
            else:
                _kwargs['type'] = _type
            if type(_value) == list:
                _kwargs['nargs'] = '+'
            self._parser.add_argument(
                *['--{}'.format(_key) for _key in _keys],
                default=_value,
                help=v[3] if len(v) >= 4 else None,
                **_kwargs,
            )
            for _key in _keys:
                _config_indices[_key] = i
        
        _args = self._parser.parse_args()
        self.parsed_args = _args
        
        _config_parsed = [list(v) for v in self._config]
        for _key, _index in _config_indices.items():
            _config_parsed[_index][1] = _args.__dict__[_key]
        _config_parsed = [tuple(v) for v in _config_parsed]
        self.args_list = _config_parsed
        # print('parsed')
        # print(_config_parsed)
        # print()
        # self.update_from_dict(_dict=_args.__dict__)
        self.update_from_list(config=_config_parsed)
        return _args
