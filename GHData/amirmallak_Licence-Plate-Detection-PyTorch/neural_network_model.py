import torch
import torch.nn.functional as nn_func

from torch import nn
from torch import sigmoid
from torch.optim import Adam
from torchsummary import summary
from torchvision.models import vgg16
from torch.nn import Sequential, Linear, Dropout


class LicensePlateDetectionNN(nn.Module):
    def __init__(self):
        super(LicensePlateDetectionNN, self).__init__()
        self.vgg16 = vgg16(pretrained=True)

        # Neglecting (dis-including) the last 3 fully connected layers (Not including the 7x7x512 Layer)
        self.classifier = Sequential(*list(self.vgg16.features.children()),
                                     self.vgg16.avgpool)

        # Freezing the weights in the transferred VGG16 Neural Network
        # for param in self.classifier.parameters():
        #     param.requires_grad = False

        self.fc1 = Linear((7 * 7 * 512), 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = Linear(64, 4)  # Our output coordinates (x_t, y_t, x_b, y_b)
        self.drop_out = Dropout(0.25)  # Dropout neurons with probability P = 0.25

    def forward(self, x):
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        x = nn_func.relu(self.bn1(self.fc1(x)))
        x = self.drop_out(x)
        x = nn_func.relu(self.bn2(self.fc2(x)))
        x = self.drop_out(x)
        x = nn_func.relu(self.bn3(self.fc3(x)))
        x = sigmoid(self.fc4(x))

        return x


# TODO: Merge train_step() function into LicensePlateDetectionNN class
def _make_train_step(model, loss_fn, optimizer):
    def _train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(model, y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    return _train_step


# Defining the loss function of the Neural Network Model
def _loss_function(model, y, y_hat, lambda1=1e-6, lambda2=1e-6, lambda3=1e-6, lambda4=1e-6):
    loss_func = nn.MSELoss(reduction='mean')
    all_linear1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
    all_linear2_params = torch.cat([x.view(-1) for x in model.fc2.parameters()])
    all_linear3_params = torch.cat([x.view(-1) for x in model.fc3.parameters()])
    all_linear4_params = torch.cat([x.view(-1) for x in model.fc4.parameters()])
    l1_regularization = lambda1 * torch.norm(all_linear1_params, p=2)
    l2_regularization = lambda2 * torch.norm(all_linear2_params, p=2)
    l3_regularization = lambda3 * torch.norm(all_linear3_params, p=2)
    l4_regularization = lambda4 * torch.norm(all_linear4_params, p=2)

    total_loss = loss_func(y, y_hat) + l1_regularization + l2_regularization + l3_regularization + l4_regularization

    return total_loss


def neural_network(channel: int, width: int, height: int) -> tuple:
    """
    Q: Why seed(42)?
    A: It's a pop-culture reference! In Douglas Adams's popular 1979 science-fiction novel The Hitchhiker's Guide to
    the Galaxy, towards the end of the book, the supercomputer Deep Thought reveals that the answer to the great
    question of “life, the universe and everything” is 42.
    """
    torch.manual_seed(42)

    model = LicensePlateDetectionNN()  # In case of cuda GPU usage, add: .to(device)
    print('\nSummary of our Neural Network Model:')
    summary(model, (channel, width, height))
    print('\n')

    # loss_fn = nn.MSELoss(reduction='mean')
    loss_fn = _loss_function
    optimizer = Adam(model.parameters(), lr=3e-4)  # Adam's good learning rate (1/3 of 1e-3)
    train_step = _make_train_step(model, loss_fn, optimizer)

    return model, train_step, loss_fn
