#%% imports
import torch.nn as nn
import torch
from carafe import CARAFEPack


#%% custom class
class MyClass(nn.Module):
    def __init__(self):
        super(MyClass, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        self.upsampling = CARAFEPack(channels=1, scale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsampling(x)
        return x


# %% the class in action
with torch.no_grad():
  my_nn = MyClass()
  my_nn = torch.jit.script(my_nn)

'''
RuntimeError:
Python builtin <built-in method forward of PyCapsule object at 0x7fa48f8c4ed0> is currently not supported in Torchscript:
  File "/home/ubuntu/.virtualenvs/temp/lib/python3.6/site-packages/carafe/carafe.py", line 109
        rmasks = masks.new_zeros(masks.size())
        if features.is_cuda:
            carafe_ext.forward(features, rfeatures, masks, rmasks, self.up_kernel,
            ~~~~~~~~~~~~~~~~~~ <--- HERE
                            self.up_group, self.scale_factor, routput, output)
        else:
'CARAFEPack.forward_carafe' is being compiled since it was called from 'CARAFEPack.feature_reassemble'
  File "/home/ubuntu/.virtualenvs/temp/lib/python3.6/site-packages/carafe/carafe.py", line 92
    def feature_reassemble(self, x: Tensor, mask: Tensor):
        x = self.forward_carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
        return x
'CARAFEPack.feature_reassemble' is being compiled since it was called from 'CARAFEPack.forward'
  File "/home/ubuntu/.virtualenvs/temp/lib/python3.6/site-packages/carafe/carafe.py", line 121
        mask = self.kernel_normalizer(mask)

        x = self.feature_reassemble(x, mask)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
        return x
'''
