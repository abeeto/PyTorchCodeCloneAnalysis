import os
import torch

def save_checkpoints(state,model_folder,model_name):
    torch.save(state,os.path.join(model_folder,model_name))