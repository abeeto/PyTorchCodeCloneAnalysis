# %%
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cosine_loss(p, z):
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class PredictionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SiamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.projector = ProjectionMLP(input_dim=512, output_dim=128)
        self.predictor = PredictionMLP(input_dim=128, hidden_dim=64, output_dim=128)

    def f(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        return x

    def h(self, x):
        x = self.predictor(x)
        return x

    def forward(self, x):
        return self.encoder(x)

    def train(self, x1, x2):
        z1, z2 = self.f(x1), self.f(x2)
        p1, p2 = self.h(z1), self.h(z2)

        loss_a = cosine_loss(p1, z2)
        loss_b = cosine_loss(p2, z1)

        loss = (loss_a / 2) + (loss_b / 2)
        return loss


class ClassifierModel(nn.Module):
    def __init__(self, checkpoint=None, train_backbone=False):
        super().__init__()

        if checkpoint:
            self.backbone = torch.load(checkpoint, map_location=DEVICE)
        else:
            self.backbone = models.resnet18(pretrained=False)
            self.backbone.fc = nn.Identity()

        if train_backbone == False:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)
