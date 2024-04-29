import torch
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
model.eval()
example = torch.rand(1, 3, 520, 781)
traced_script_module = torch.jit.trace(model, example, strict=False)
traced_script_module.save("traced_fcn_resnet50_model.pt")