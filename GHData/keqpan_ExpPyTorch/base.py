import os
from common import CKPT_DIR, LOGS_DIR
from utils import save_model, load_model, dt
import numpy as np

stop_flag = False
main_pid = os.getpid()
def handler(signum, frame):
    if main_pid == os.getpid():
        print("Shutting down at " + dt() + " ...")
        global stop_flag
        stop_flag = True

import signal
# signal.signal(signal.SIGTERM, handler) for 15 signal
signal.signal(signal.SIGINT, handler)


class BaseExpRunner():
    def __init__(self, name, models_dict, schedulers_dict, optimizers_dict, losses_dict, log_names, multigpu_mode=False):
        self.name = name
        self.cur_logs_path = os.path.join(LOGS_DIR, name)
        self.cur_ckpt_path = os.path.join(CKPT_DIR, name)
        
        self.models_dict = models_dict
        self.schedulers_dict = schedulers_dict
        self.optimizers_dict = optimizers_dict
        self.losses_dict = losses_dict
        self.log_names = log_names
        
        self.logs_dict = {}
        self.tmp_logs_dict = {}
        
        self.multigpu_mode = multigpu_mode
        self.total_loss = 0
        self.cur_epoch = 0

        self.load_ckpts()
        self.create_ckpt_dirs()
        self.create_log_dirs()

    def load_ckpts(self):
        for model_dict in self.models_dict.values():
            model_ckpt_path = model_dict.get('load_ckpt')
            if model_ckpt_path is not None:
                model = model_dict['model']
                model_dict['model'] = load_model(model, model_ckpt_path, self.multigpu_mode)

    def create_ckpt_dirs(self):
        for model_name in self.models_dict.keys():
            model_ckpt_path = os.path.join(self.cur_ckpt_path, model_name)
            self.models_dict[model_name]['ckpt_path'] = model_ckpt_path
            os.makedirs(model_ckpt_path, exist_ok=True)
    
    def create_log_dirs(self):
        os.makedirs(self.cur_logs_path, exist_ok=True)
        for log_name in self.log_names:
            self.logs_dict[log_name] = {
                                            'path': os.path.join(self.cur_logs_path, log_name + "_" + self.name),
                                            'data': []
                                        }
            self.tmp_logs_dict[log_name] = []
            
    def save_ckpt(self, epoch):
        for model_dict in self.models_dict.values():
            save_model(model_dict['model'], os.path.join(model_dict['ckpt_path'], "weights_%d" % epoch), self.multigpu_mode)
            
    def save_logs(self):
        for log in self.logs_dict.values():
            log_path = log['path'] 
            log_data = log['data']
            np.save(log_path, np.asarray(log_data))
    
    def global_forward(self, sample, batch_idx):
        raise NotImplementedError("Each trainer should define global_forward() method")
        
    def schedulers_step(self):
        for scheduler in self.schedulers_dict.values():
            scheduler.step()
    
    def avg_losses(self):
        for log_name, log_dict in self.tmp_logs_dict.items():
            self.logs_dict[log_name]['data'].append(np.mean(log_dict))
            
    
    def train_epoch(self, dataloader_train):
        self.total_loss = 0
        for batch_idx, sample in enumerate(dataloader_train):
            if stop_flag:
                break
            self.global_forward(sample, batch_idx)
        print("\n")
        self.avg_losses()

    def train(self, dataloader_train, n_epochs):
        for epoch in range(n_epochs):
            if stop_flag:
                break
            self.schedulers_step()
            self.train_epoch(dataloader_train)

            self.save_ckpt(epoch)
            self.save_logs()
            self.cur_epoch += 1