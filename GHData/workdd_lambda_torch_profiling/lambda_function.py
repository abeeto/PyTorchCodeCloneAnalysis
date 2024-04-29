import time
from json import load
import numpy as np
import torch
import torchvision.models as models
import os
from torch.profiler import profile, record_function, ProfilerActivity


model_name = os.environ['model_name']
batch_size = int(os.environ['batch_size'])
framework = 'torch'

efs_path = '/mnt/efs/'
model_path = efs_path + f'{framework}/base/arm/{model_name}'

image_size = 224
if model_name == "inception_v3":
    image_size = 299
channel = 3

load_start = time.time()
model = torch.load(model_path + '/model.pt')
model.load_state_dict(torch.load(model_path + '/model_state_dict.pt'))
model.eval()
load_time = time.time() - load_start

def make_dataset():
    data = torch.randn(batch_size, 3, 224, 224)
    if "inception_v3" in model_name:
        data = torch.randn(batch_size, 3, 299, 299)
    return data
  
data = make_dataset()

def lambda_handler(event, context):
    print(*torch.__config__.show().split("\n"), sep="\n")
    # record_shapes=True 추가하면 input demension check 가능
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        model(data)
            
    res = prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total")
    print(res)
    
    return res
  
  

