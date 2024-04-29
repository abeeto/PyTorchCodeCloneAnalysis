import os
import pandas as pd
import torch

class MeasureProgression():
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)

    def getMeasure(self, mode='avg'):
        if mode == 'avg':
            return 1.0 * sum(self.items) / len(self.items)
        else:
            return None

# Static methods
def generate_metadata_file(image_dir, img_sub_dir='images', delimiter = '_', index = 0, output = 'metadata.csv', img_format = 'jpg'):
    
    img_sub_dir = '' if img_sub_dir is None else img_sub_dir
    
    filenames = [f for f in os.listdir(os.path.join(image_dir, img_sub_dir)) if f[-len(img_format):] == img_format]
    labels = [f.split(delimiter)[index] for f in filenames]

    df = pd.DataFrame(list(zip(filenames, labels)), columns =['filename','label'])
    df.to_csv(os.path.join(image_dir, output), index=False)

def save_as_model(epoch, model_name, model_state, storage_dir):
    filepath = os.path.join(storage_dir, "_{0}_{1}.model".format(epoch, model_name))
    if not os.path.exists(storage_dir):
        os.mkdir(storage_dir)

    torch.save(model_state, filepath)

def save_as_best_model(model_name, model_state, storage_dir):
    filepath = os.path.join(storage_dir, "_best_{0}.model".format(model_name))
    if not os.path.exists(storage_dir):
        os.mkdir(storage_dir)

    torch.save(model_state, filepath)

def load_model(model_name, storage_dir, model, optimizer=None, model_type=-1):
    prefix = 'best' if model_type == -1 else model_type
    filepath = os.path.join(storage_dir, "_{0}_{1}.model".format(prefix, model_name))

    if not os.path.exists(filepath):
        print("Model doesn't exist {}".format(filepath))

    model_state = torch.load(filepath)
    model.load_state_dict(model_state['state_dict'])

    if optimizer:
        optimizer.load_state_dict(model_state['optim_dict'])
