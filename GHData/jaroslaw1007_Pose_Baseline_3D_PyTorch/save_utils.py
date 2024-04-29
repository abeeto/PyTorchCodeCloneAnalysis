import numpy as np
import torch
import json
import os

class Logger(object):
    def __init__(self, file_path, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if not title else title
        
        if file_path is not None:
            if resume:
                self.file = open(file_path, 'r')
                name = self.file.readline()
                self.names = name.rstrip()
                self.numbers = {}
                
                for _, name in enumerate(self.names):
                    self.numbers[name] = []
                    
                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numubers[self.names[i]].append(numbers[i])
                        
                self.file.close()
                self.file = open(file_path, 'a') # Appending
            else:
                self.file = open(file_path, 'w')
                
    def set_names(self, names):
        if self.resume:
            pass
        
        self.numbers = {}
        self.names = names
        
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        
        self.file.write('\n')
        self.file.flush()
        
    def append(self, member, member_type):
        assert len(self.names) == len(member)
        
        for idx, mem in enumerate(member):
            if member_type[idx] == 'int':
                self.file.write("{}".format(mem))
            else:
                self.file.write("{0:.5f}".format(mem))
                
            self.file.write('\n')
            self.file.flush()
        
    def close(self):
        if self.file:
            self.file.close()
            
def save_options(opts, save_path):
    file_path = os.path.join(save_path, 'opts.json')
    with open(file_path, 'w') as f:
        f.write(json.dumps(vars(opts), sort_keys=True, indent=4))
    
def save_ckpt(state, save_path, is_best=True):
    if is_best:
        file_path = os.path.join(save_path, 'ckpt_best.pt')
        torch.save(state, file_path)
    else:
        file_path = os.path.join(save_path, 'ckpt_last.pt')
        torch.save(state, file_path)