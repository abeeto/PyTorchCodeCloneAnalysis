import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.style_networks_contrastive import StyleEncoder

class ContrastiveModel(nn.Module):
    def __init__(self, input_channels_a, use_decoder_a, input_channels_b, use_decoder_b, attribute_channels):
        super(ContrastiveModel, self).__init__()
        
        self.hidden_dim = 128
        self.mlp_dim = 4096
        self.dim = 256
        
        self.front_end_shared = list(models.resnet18(pretrained=True).children())[5][1]
        self.momentum_shared = list(models.resnet18(pretrained=True).children())[5][1]
        
        self.encoder_a = StyleEncoder(input_channels_a, self.front_end_shared,
                                               attribute_channels, use_decoder_a)
        self.encoder_b = StyleEncoder(input_channels_b, self.front_end_shared,
                                               attribute_channels, use_decoder_b)
        self.momencoder_a = StyleEncoder(input_channels_a, self.momentum_shared,
                                               attribute_channels, use_decoder_a)
        self.momencoder_b = StyleEncoder(input_channels_b, self.momentum_shared,
                                               attribute_channels, use_decoder_b)
        
        
        self.predictor = self.build_mlp(2, self.dim, self.mlp_dim, self.dim, False)
        self.initial_parameter()
        
    def initial_parameter(self):
        """Initialize parameter for moment encoder"""
        for param_b, param_m in zip(self.encoder_a.parameters(), 
                                    self.momencoder_a.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        for param_b, param_m in zip(self.encoder_b.parameters(), 
                                    self.momencoder_b.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        
    def build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        """Build mlp for base encoder and predictor"""
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp)

    @torch.no_grad()
    def update_momencoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.encoder_a.parameters(), 
                                    self.momencoder_a.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
        for param_b, param_m in zip(self.encoder_b.parameters(), 
                                    self.momencoder_b.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
    
    def forward(self, data_a, data_b, m):
#         print(type(self.encoder_a(data_a)))
#         print(self.encoder_a(data_a))
        r1 = self.encoder_a(data_a)
        r2 = self.encoder_b(data_b)
#         print(data_a.shape)
#         print(torch.tensor(r1[0]).shape)
        q1 = self.predictor(torch.tensor(r1[0]))
        q2 = self.predictor(torch.tensor(r2[0]))
        
        with torch.no_grad():  # no gradient
            self.update_momencoder(m)  # update the momentum encoder
            # compute momentum features as targets
            k1 = self.momencoder_a(data_a)[0]
            k2 = self.momencoder_b(data_b)[0]
        
        loss = self.contrastive_loss(q1,k2) + self.contrastive_loss(q2,k1)
        return loss
    
    def contrastive_loss(self, q, k):
        """Compute contrastive loss for query and key"""
        N, _ = q.shape
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.einsum('ij,kj->ik', [q, k])
        print(logits)
        logits = F.softmax(logits)
        print(logits)
        labels = torch.arange(N, dtype=torch.long)
        # labels = torch.eye(N)
        return nn.CrossEntropyLoss()(logits, labels.cuda())