import torch
import torch.nn as nn
import torch.nn.functional as F
from opt import models
from tensorboardX import SummaryWriter

net = models.rgb_seresnet50(pretrained=True, num_classes=101)
net.load_state_dict(torch.load('./checkpoints/ucf101_rgb_rgb_seresnet50_split1.pth.tar')params)


# with SummaryWriter(comment='Rgb_seresnet50') as w:
#     w.add_graph(net)