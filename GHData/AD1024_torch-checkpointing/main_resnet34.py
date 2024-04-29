from utils import checkpointing
from utils import src_builder
import torch
import torchvision
import sys

if __name__ == '__main__':
    model = torchvision.models.resnet34()
    inp = torch.zeros([32, 3, 224, 224])
    output = checkpointing.auto_checkpoint(model, inp, int(sys.argv[1]), verbose=True)
    with open('./models/resnet34.py', 'w') as fp:
        fp.write(src_builder.to_python_src('ResNet34', output.params, output.start,\
                                           output.graph, output.checkpoints))