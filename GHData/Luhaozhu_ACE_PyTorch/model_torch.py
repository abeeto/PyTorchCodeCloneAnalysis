import torch
import torch.nn as nn
from torchvision import models
from ace_torch import hook_feature,hook_grad

# get feature map and grad information
# grad_blobs = []
# def hook_grad(module, grad_input, grad_output):
#     grad_blobs.append(grad_output[0].data.cpu())

# features_blobs = []
# def hook_feature(module, input, output):
#     features_blobs.append(output.data.cpu())

def load_model(model_name,feature_names,num_classes=1000):
    if model_name == "resnet50":  # weights=default代表的是默认的预训练模型
        model = models.resnet50(weights="DEFAULT",num_classes=num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights="DEFAULT",num_classes=num_classes)
    elif model_name == "googlenet":
        model = models.googlenet(weights="DEFAULT",num_classes=num_classes)
    elif model_name == "inceptionv3":
        model = models.inception_v3(weights="DEFAULT",num_classes=num_classes)
    elif model_name == "vgg19":
        model = models.vgg19(weights="DEFAULT",num_classes=num_classes)
    elif model_name == "densenet":
        model = models.densenet121(weights="DEFAULT",num_classes=num_classes)
    for name in feature_names:
        model._modules.get(name).register_forward_hook(hook_feature)
        model._modules.get(name).register_full_backward_hook(hook_grad)
    return model


# if __name__ == "__main__":
#     model = load_model("googlenet",['inception4c'])
#     x = torch.rand(50,3,224,224)
#     y = model(x)
#     x.requires_grad_(True)
#     model.zero_grad()
#     y.backward(torch.ones(50,1000))
#     print(grad_blobs[0].shape)