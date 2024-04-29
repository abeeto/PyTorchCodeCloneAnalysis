import  torchvision
import torch

# vgg16 = torchvision.models.vgg16(pretrained=False)

# 方式1，保存了模型的结构和参数
# torch.save(vgg16, "vgg16_method1.pth")

# 方式2，保存模型参数（官方推荐）
# torch.save(vgg16.state_dict(), "vgg16_method2.pth")

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)