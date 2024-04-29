from collections import namedtuple, OrderedDict

import numpy as np
import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
import torch
import torchvision

mnist_data = torchvision.datasets.MNIST('./data/',
                                        train=False,
                                        download=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                        ]))

one_data = mnist_data[0][0].unsqueeze(0)
one_data_label = mnist_data[0][1]

print(trt.__version__ >= '7.0.0')  # require tensorrt>=7.0.0
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
logger = trt.Logger(trt.Logger.INFO)
with open("./model.engine", 'rb') as f, trt.Runtime(logger) as runtime:
    model = runtime.deserialize_cuda_engine(f.read())
bindings = OrderedDict()
fp16 = False  # default updated below
for index in range(model.num_bindings):
    name = model.get_binding_name(index)
    dtype = trt.nptype(model.get_binding_dtype(index))
    shape = tuple(model.get_binding_shape(index))
    data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(torch.device("cuda:0"))
    bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    if model.binding_is_input(index) and dtype == np.float16:
        fp16 = True

binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
context = model.create_execution_context()

im = one_data
im.to(torch.device("cuda:0"))
assert im.shape == bindings['images'].shape, (im.shape, bindings['images'].shape)
binding_addrs['images'] = int(im.data_ptr())
context.execute_v2(list(binding_addrs.values()))
y = bindings['output'].data
print(y)
pass
