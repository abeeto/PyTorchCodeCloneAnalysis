# -*- coding: utf-8 -*-
from models import MobileNet_v2


def mobilenet_v2(pretrained=False, *args, **kwargs):
    model = MobileNet_v2()

    if pretrained:
        raise NotImplementedError

    return model
