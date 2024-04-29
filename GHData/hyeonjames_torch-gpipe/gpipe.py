from typing import Iterable, Union, Any, Optional

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as md
import torchvision.transforms as transforms
from torch.utils import checkpoint
import torchgpipe


def device(d: Union[int, str, torch.device]) -> Union[torch.device]:
    if isinstance(d, torch.device):
        return d
    elif isinstance(d, str):
        return torch.device(d)
    elif isinstance(d, int):
        return torch.device(f'cuda:{d}')


def detach_tensors(tensors):
    if isinstance(tensors, tuple):
        out = []
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                out.append(tensor)
            new_tensor = tensor.detach()
            new_tensor.requires_grad = tensor.requires_grad
            out.append(new_tensor)
        return tuple(out)
    else:
        raise RuntimeError("")


class CheckPoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, module, *args):
        with torch.no_grad():
            result = module(*args)
        ctx.module = module
        ctx.save_for_backward(*args)
        return result

    @staticmethod
    def backward(ctx, *args):
        i_args = detach_tensors(ctx.saved_tensors)
        with torch.enable_grad():
            outputs = ctx.module(*i_args)
        torch.autograd.backward(outputs, args)
        return tuple([None] + [x.grad if isinstance(x, torch.Tensor) else x for x in i_args])


def my_checkpoint(model, *args):
    return CheckPoint.apply(model, *args)


class GPipe(nn.Module):

    def __init__(self, *models: nn.Module, balance: Optional[Iterable[float]] = None,
                 devices: Optional[Iterable[Union[int, str, torch.device]]] = None,
                 chunks: Optional[int] = None) -> None:
        super(GPipe, self).__init__()
        if not devices:
            devices = range(torch.cuda.device_count())
        if chunks is None:
            chunks = len(balance)
        assert chunks > 0
        devices = list(devices)
        if balance is None:
            balance = [1. / len(devices)] * len(devices)
        assert sum(balance) == len(models)
        self.chunks = chunks
        self.devices = [device(d) for d in devices]
        self.models = nn.ModuleList()
        index = 0
        models = list(models)
        for i, bal in enumerate(balance):
            if bal > 1:
                model = nn.Sequential(*models[index:index+bal])
            else:
                model = models[index]
            self.models.append(model.to(self.devices[i], non_blocking=True))
            index += bal
        #print(list(enumerate(self.models)))
    def __len__(self):
        return sum([len(x) for x in self.models])

    def forward(self, x: torch.Tensor) -> Any:
        chunks = list(x.chunk(self.chunks))
        ret = []
        for chunk in chunks:
            for index, model in enumerate(self.models):
                chunk = model(chunk.to(self.devices[index]))
            ret.append(chunk)

        return torch.cat(ret)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ])
    data = datasets.CIFAR10('/data/private/datasets', train=True, download=False, transform=transform)
    torch.manual_seed(42)
    resnet = md.resnet50(True)

    trainloader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=8192)

    model2 = nn.Sequential(resnet.conv1,
                           resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
                           resnet.layer3,
                           resnet.layer4,
                           resnet.avgpool, nn.Flatten(), resnet.fc).cuda(0)

    model = torchgpipe.GPipe(copy.deepcopy(model2), balance=[3, 3, 5], devices=[0, 1, 2], chunks=3)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer2 = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1):
        for inputs, target in trainloader:
            with torch.no_grad():
                outputs = model(inputs.to(model.devices[0]))
                outputs2 = model2(inputs.cuda(0))
            print(outputs.cuda(0)-outputs2)

        print(f'epoch {epoch}')
