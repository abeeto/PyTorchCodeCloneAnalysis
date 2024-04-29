import tvm
from tvm import relay
from tvm import rpc
from tvm import autotvm
from tvm.contrib import util
from tvm.contrib.util import tempdir
from tvm.contrib import graph_runtime as runtime

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
separator = "****************************************************************************************"

traced_path = 'cnn_net_quantized_traced_ptq.pt'

pytorch_model = torch.jit.load(traced_path).eval()

input_shape = [1, 1, 28, 28]

irmodule, params = relay.frontend.from_pytorch(pytorch_model,[("input0", input_shape)])

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(irmodule, target="llvm", params=params)

######################################################################
# Compile the kernel as a shared library into a local temp directory
tmp = util.tempdir()
kbinname = tmp.relpath('tvmmodel.tar')
lib.export_library(kbinname)

######################################################################
# Open remote session and upload the kernel to the target device
remote = rpc.LocalSession()

remote.upload(kbinname)
rlib = remote.load_module('tvmmodel.tar')
ctx  = remote.cpu(0)


rtmodule = runtime.create(graph, rlib, ctx)

byteparams = relay.save_param_dict(params)
rtmodule.load_params(byteparams)


#Import PyTorch Data
transform = transforms.Compose([torchvision.transforms.ToTensor()])
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)


correct = 0
print(separator)
print("Measuring Performance...")


for idx, (data, target) in enumerate(testloader):
    rtmodule.set_input("input0", tvm.nd.array(data))

    if idx % 1000 == 0:
        print(f"Done with {idx}/{len(testloader.dataset)}")
    rtmodule.run()
   
    out = rtmodule.get_output(0)

    if np.argmax(out.asnumpy()[0]) == target.data.tolist()[0]:
        correct +=1


print('Accuracy: %.2f %%' % (correct / len(testloader.dataset) * 100))