import torch
from torch import nn

print (torch.hub.list("msk-mind/luna-ml:main"))

model = torch.hub.load("msk-mind/luna-ml:main",  "logistic_regression_model", weight_tag="main:logistic_regression_ones.pth")
print (model.lin1.weight)

model = torch.hub.load("msk-mind/luna-ml:main",  "logistic_regression_model", weight_tag="main:logistic_regression_random.pth")
print (model.lin1.weight)

model = torch.hub.load("msk-mind/luna-ml:main", "tissue_tile_net_model", weight_tag="main:tissue_net_2021-01-19_21.05.24-e17.pth",
                            activation=nn.Sigmoid(), num_classes=5)
print (model)

transform = torch.hub.load("msk-mind/luna-ml:main", "tissue_tile_net_transform")
print (transform)