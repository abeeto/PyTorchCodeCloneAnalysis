import torch
import torch.nn as nn

class CEL_Jaccard:
    def __init__(self, jaccard_weight=0.3, num_classes=8):
        self.nll_loss = nn.NLLLoss()
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
       loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
       if self.jaccard_weight:
           eps = 1e-15 # Statistical stability
           for cls in range(self.num_classes):
               jaccard_target = (targets == cls).float()
               jaccard_output = outputs[:, cls].exp()
               intersection = (jaccard_output * jaccard_target).sum()
               union = jaccard_output.sum() + jaccard_target.sum()
               denom = union - intersection
               loss -= torch.log((intersection + eps) / (denom + eps)) * self.jaccard_weight
       return loss
