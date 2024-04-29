import math
import torch
import torch.nn as nn

class CapsNet(nn.Module):
    def __init__(self, patch_size=(1, 28, 28), route_iters=1, with_reconstruction=False):
        super(CapsNet, self).__init__()
        self.in_channels, self.height, self.width = patch_size
        self.num_primaryCaps = 32 * (self.height - 16) * (self.width - 16) // 4
        self.conv1 = nn.Conv2d(self.in_channels, 256, 9) # In, Out, Kernel
        self.primaryCaps = nn.Conv2d(256, 256, 9, stride=2) # In, Out, Kernel
        self.squash = Squash(dim=1) # Capsule dimension
        self.digitCaps = LinearCapsule(self.num_primaryCaps, 8, 10, 16, route_iters=route_iters)
        self.relu = nn.ReLU()
        self.with_reconstruction = with_reconstruction
        if self.with_reconstruction:
            self.decoder = LinearDecoder()

    def forward(self, input, rc_target=None):
        # conv1
        x = self.conv1(input)
        x = self.relu(x)

        # PrimaryCaps: conv2d -> view -> squash
        # Out Tensor Size: (batch, caps dim, channels * h * w -> num capsules)
        x = self.primaryCaps(x).view(-1, 8, self.num_primaryCaps)
        x = self.squash(x)

        # DigitCaps
        # This layer view Digit caps as a fully connected capsule layer, thus first flatten it to (batch, #caps, caps dim)
        x = x.transpose(1, 2)
        x = self.digitCaps(x)
        if self.with_reconstruction and rc_target is not None:
            rc = self.decoder(x[torch.arange(x.size(0), dtype=torch.long),
                                rc_target.argmax(dim=1), :])
        else:
            rc = None

        return x, x.norm(dim=2), rc

    def reconstruct(self, x):
        return self.decoder(x)


class LinearCapsule(nn.Module):
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, route_iters=1):
        super(LinearCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.route_iters = route_iters
        self.weight = nn.Parameter(torch.Tensor(out_num_caps, in_num_caps, in_dim_caps, out_dim_caps))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.prior_softmax = nn.Softmax(dim=2)
        self.squash = Squash(dim=2)

    def forward(self, input):
        # In this Linear (Fully Connected) Capsule Layer, we heavily use batch matrix
        # multiplication defined in pytorch tensor matmul
        # Each of tensor used in this forward is listed as following,
        # it would help a lot for understanding the routing algorithm
        # weight: (out_num_caps, in_num_caps, in_dim caps, out_dim_caps)
        # u_hat: (minibatch, out_num_caps, in_num_caps, out_dim_caps)
        # b_logits: (minibatch, out_num_caps, in_num_caps)
        # c_couple: (minibatch, out_num_caps, in_num_caps)
        # s_input: (minibatch, out_num_caps, out_dim_caps)
        # v_pred: (minibatch, out_num_caps, out_dim_caps)

        u_hat = input.unsqueeze(1).unsqueeze(-2).matmul(self.weight).squeeze(-2)
        b_logits = torch.zeros(input.size(0), self.out_num_caps, self.in_num_caps).to(input.device)
        for i in range(self.route_iters):
            c_couple = self.prior_softmax(b_logits)
            s_input = c_couple.unsqueeze(2).matmul(u_hat).squeeze(2)
            v_pred = self.squash(s_input)
            b_logits = b_logits.add(u_hat.matmul(v_pred.unsqueeze(-1)).squeeze(-1))
        c_couple = self.prior_softmax(b_logits)
        s_input = c_couple.unsqueeze(2).matmul(u_hat).squeeze(2)
        v_pred = self.squash(s_input)
        return v_pred

class LinearDecoder(nn.Module):
    def __init__(self, dim_caps=16, out_patch=(1, 28, 28)):
        super(LinearDecoder, self).__init__()
        self.fc1 = nn.Linear(dim_caps, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, out_patch[0]*out_patch[1]*out_patch[2])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.out_patch = out_patch


    def forward(self, input):
        x = self.fc1(input)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x.view(-1, *self.out_patch)

class DigitExistenceMarginLoss(nn.Module):
    def __init__(self, neg_weight=0.5):
        super(DigitExistenceMarginLoss, self).__init__()
        self.neg_weight = neg_weight
        self.relu = nn.ReLU() # Use for max(0, x)

    def forward(self, v, t):
        return (t*(self.relu(0.9 - v).pow(2)) + self.neg_weight*(1 - t)*(self.relu(v - 0.1).pow(2))).sum(dim=1).mean()

class Squash(nn.Module):
    def __init__(self, dim=None, inplace=False):
        super(Squash, self).__init__()
        self.inplace = inplace
        self.dim = dim
        self.keepdim = (dim is not None)

    def forward(self, input):
        norm = input.norm(dim=self.dim, keepdim=self.keepdim)
        if self.keepdim:
            norm = norm.expand_as(input)
        if self.inplace:
            return input.div_(norm*(1 + norm.pow(2))).mul_(norm.pow(2))
        else:
            return input.div(norm*(1 + norm.pow(2))).mul(norm.pow(2))
