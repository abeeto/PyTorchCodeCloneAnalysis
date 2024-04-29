from utils import checkpointing
from utils import src_builder
import torch
import torchvision

if __name__ == '__main__':
    model_resnet18 = torchvision.models.resnet18()
    inputs_resnet18 = torch.zeros([64, 3, 7, 7])
    output = checkpointing.auto_checkpoint(model_resnet18, inputs_resnet18, 1024000)
    with open('./models/resnet18.py', 'w') as fp:
        fp.write(src_builder.to_python_src('ResNet18', output.params, output.start,\
                                           output.graph, output.checkpoints))