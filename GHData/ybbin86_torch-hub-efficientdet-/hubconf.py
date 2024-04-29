import torch
from torch.backends import cudnn


from backbone import EfficientDetBackbone

dependencies = ["torch", "torchvision"]

def _make_efficientdet():
    compound_coef = 0

    obj_list = ['cancer']
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 # replace this part with your project's anchor config
                                 ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                 scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    return model



def efficientdet_d0():

    model = _make_efficientdet()
    model.load_state_dict(torch.load('/home/Cyberlogitec/YB/src/EfficientDet/logs/colonCT/efficientdet-d0_397_13500.pth'), strict=False)

    return model
