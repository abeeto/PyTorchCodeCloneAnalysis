
from flask import Flask, render_template, make_response, send_file
from os import listdir
import json

app = Flask(__name__)

from PIL import Image

from os import mkdir, remove
from os.path import join, exists
from shutil import rmtree

import torch
from collections import OrderedDict
import numpy as np


class Tracer:
    def __init__(self, save_dir = "model_weight_imgs", scaling = 5):
        self.first_loop = True
        self.tensors = OrderedDict()
        self.save_dir = join("static",save_dir)
        self.struct_path = save_dir + "_struct.json"
        self.MAX_REPEAT = 100
        self.scaling = scaling

        if exists(self.save_dir):
            rmtree(self.save_dir)
        mkdir(self.save_dir)

    def trace(self, weights, name, group=""):
        if isinstance(weights,torch.Tensor):
            self.trace_tensor(weights, name, group)
        elif isinstance(weights,torch.nn.Module):
            self.trace_module(weights, name, group)
        elif isinstance(weights,torch.nn.parameter.Parameter):
            self.trace_param(weights, name, group)

    def trace_tensor(self, tensor, name, group=""):
        name = self._add_entry(name, tensor.shape, group)
        self._create_image(self._squeeze(tensor),name, scaling = self.scaling)

    def trace_param(self, tensor, name, group=""):
        name = self._add_entry(name, tensor.shape, group)
        self._create_image(self._squeeze(tensor),name, scaling = self.scaling)

    def trace_module(self, module, name, group=""):
        for param in module.named_parameters():
            self.trace_param(param[1], ".".join([name,param[0]]))

    def end_loop(self):
        if self.first_loop:
            self._save_struct()
            self.first_loop = False
        self.tensors = OrderedDict()

    def _add_entry(self, name, shape, group=""):
        if not(name in self.tensors):
            self.tensors[name] = {"shape":shape,"group":group}
            return name
        else:
            index = 1
            while name + str(index) in self.tensors:
                index += 1
                if index > self.MAX_REPEAT:
                    raise RuntimeError('End loop point not found. Tracer.end_loop should be called at the end of the loop.')

            self.tensors[name + str(index)] = {"shape":shape, "group":group}
            return name + str(index)

    def _squeeze(self, tensor):
        if len(tensor.shape) == 0: 
            tensor = tensor.view(1,1)
        elif len(tensor.shape) == 1:
            tensor = torch.unsqueeze(tensor,0)
        elif len(tensor.shape) != 2:
            tensor = tensor.squeeze()
            if len(tensor.shape) == 0: 
                tensor = tensor.view(1,1)
            elif len(tensor.shape) == 1:
                tensor = torch.unsqueeze(tensor,0)
            else:
                while len(tensor.shape)!= 2:
                    tensor = tensor[0]
        return tensor

    def _create_image(self, weights, name, scaling = 5):
        img = np.clip((weights.cpu().detach().numpy()*scaling + 0.5)*255, 0., 255.).astype(float)
        img = Image.fromarray(img).convert("L")
        img.save(join(self.save_dir, name + ".bmp"),format="bmp")

    def _save_struct(self):
        with open(self.save_dir + "_struct.json","w") as f:
            json.dump([{**{"key":x},**self.tensors[x]} for x in self.tensors],f)


@app.route('/images/<string:img>/<string:step>')
def get_image(img,step):
    return send_file("static/model_weight_imgs/{}.bmp".format(img), mimetype='image/bmp')

@app.route('/step')
def step():
    with open("static/step.txt") as f:
        response = make_response(f.read(), 200)
    response.mimetype = "text/plain"
    return response

@app.route('/')
def home():
    with open (join("static","model_weight_imgs_struct.json")) as f:
        imgs = json.load(f)
    return render_template('home.html', img_list = [x["key"] for x in imgs], imgs = imgs)

def run():
    app.run(host='0.0.0.0',debug=True)

if __name__ == "__main__":
    run()
