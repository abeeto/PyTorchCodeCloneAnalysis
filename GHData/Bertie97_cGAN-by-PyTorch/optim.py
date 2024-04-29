

class Optim:
    
    def __init__(self, kind, params, lr = 1e-3, momentum = 0):
        self.hyper = {'lr': lr, "momentum": momentum}
        if not isinstance(params, list): params = [dict(params = params, **self.hyper)]
        if not isinstance(params[0], dict) or 'params' not in params[0]:
            params = [dict(params = x, **self.hyper) for x in params]
        tocallable = lambda x: x if callable(x) else (lambda _: x)
        self.param_groups = [{k: tocallable(entry.get(k, self.hyper.get(k, None))) for k in entry} for entry in params]
        self.param_groups_i = lambda i: [{k: entry[k](i) for k in entry} for entry in self.param_groups]
        init_params = self.param_groups_i(0)
        self.generator = kind
        self.optimizer = kind(init_params)
        self.curI = 0
    
    def change_params(self, i):
        self.curI = i
        new_params = self.param_groups_i(i)
        tmp_state_dict = self.optimizer.state_dict()
        for i in range(len(tmp_state_dict['param_groups'])):
            for k in self.hyper:
                tmp_state_dict['param_groups'][i][k] = new_params[i][k]
        self.optimizer.load_state_dict(tmp_state_dict)
        return self.optimizer

    def __getitem__(self, i): return self.change_params(i)
    def __call__(self, i = 0): return self.change_params(i)
    def __next__(self): return self.change_params(self.curI + 1)