"""
    Exporting the model
    @author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""

import torch
from torchvision.models import resnet34


model = resnet34(pretrained=True)
example_input = torch.rand(1, 3, 224, 224)
model.eval()
traced_model = torch.jit.trace(model, example_input)
traced_model.save("./resnet34.pt")
