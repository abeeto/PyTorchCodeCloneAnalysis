from enum import Enum

class Config(Enum):
    BATCHSIZE = 128
    NUM_WORKERS = 4
    RESIZED_HEIGHT = 32
    RESIZED_WIDTH = 32
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD = [0.2023, 0.1994, 0.2010]
    CIFAR100_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    CIFAR100_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    EPOCH_LR_SCHEDULER = [100,200,300]
    EVAL_TRAIN_STEP = 10
    MODEL_TYPE = [
        "vgg11",
        "vgg13",
        "vgg16",
        "vgg19",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152"
    ]