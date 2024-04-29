import os
import time
import yaml
import multiprocessing as mp
from pretrained_classifier import MODEL_NAMES


def yaml_load(fpath):
    with open(fpath) as fp:
        return yaml.load(fp)


def yaml_write(data, fpath):
    with open(fpath, 'w') as fp:
        yaml.dump(data, fp, default_flow_style=False)


def generate_yml(model):
    moxel_yml = yaml_load('moxel.yml')

    for i, cmd in enumerate(moxel_yml['cmd']):
        if 'serve_model.py' in cmd:
            moxel_yml['cmd'][i] = 'python serve_model.py ' + model

    yaml_write(moxel_yml, 'moxel.yml')


model = 'ResNet-18'

def moxel_push(model):
    generate_yml(model)
    os.system('yes | moxel push -f moxel.yml {}:latest'.format(model))

pool = []

for model in MODEL_NAMES:
    if model in ['AlexNet', 'ResNet-18']:
        continue
    print('launching model', model)
    p = mp.Process(target=moxel_push, args=[model])
    p.start()
    time.sleep(2)
    pool.append(p)

for p in pool:
    p.join()
