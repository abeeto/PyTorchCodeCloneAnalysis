# -*- coding: utf-8 -*-

class config:
    # train
    root = "./data"
    batch_size = 128
    num_workers= 4
    lr = 1e-3
    output = "output"
    size = 32
    num_epochs = 50
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    interval = 100
    valinterval = 10
    fp16 = True # 是否采用fp16进行训练

    # onnx. must be run python train.py
    # fp32
    pt = f"./{output}/resnet50-50-ckpt.t7"
    onnx = f"./{output}/resnet50.onnx"
    engine = f"./{output}/resnet50.engine"
    # fp16
    pt_fp16 = f"./{output}/resnet50-50-fp16-ckpt.t7"
    onnx_fp16 = f"./{output}/resnet50-fp16.onnx"
    engine_fp16 = f"./{output}/resnet50-fp16.onnx"

    # torch2trt



