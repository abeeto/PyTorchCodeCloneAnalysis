import torchvision.models as models
import torch
import torch.nn as nn
import os
from hardnet import get_pose_net as get_hardnet
from decode import ctdet_decode

_model_factory = {
  'hardnet': get_hardnet,
}

def create_model(arch, heads, head_conv):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model


arch = "hardnet_85"
heads = {'hm': 10, 'wh': 4}
head_conv = 256

if __name__ == "__main__":
    model = create_model(arch, heads, head_conv)
    x = torch.randn(2,3,512,512)
    out = model(x)[-1]
    # print(out[0])
    # for i in out[0]:
        # print(i)
        # print()
        # print()
    # print(out[0]["hm"].shape)
    # print(out[0]["wh"].shape)

    output = model(x)[-1]
    hm = output['hm'].sigmoid_()
    wh = output['wh']
    # print(hm.shape, wh.shape)
    reg = wh[:,2:,:,:]
    wh  = wh[:,:2,:,:]
            
    torch.cuda.synchronize()
    dets = ctdet_decode(hm, wh, reg=reg)
    print(dets.shape)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    print(dets.shape)
    dets[:, :, :4] *= 4
    print(dets.shape)
    # print(dets.shape)

      