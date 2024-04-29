def resnet18(pretrained = True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock,[2,2,2,2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir ='./')) 
    return model

import torch
import torch.nn as nn
import torchvision.models as models
model = models.resnet18(pretrained=True)

def save_model(net, epoch):
    PATH = "./resnet18_pretrain.pth"
    torch.save(net.state_dict(),PATH)

def load_model(net, pretrained_epoch):
    PATH = "./pretrained_models/model_epoch" + str(pretrained_epoch) + ".pth"
    net.load_state_dict(torch.load(PATH))
    net.to(device)
    net.eval()


save_model(model,1)

