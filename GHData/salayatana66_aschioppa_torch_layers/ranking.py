import torch.nn as nn
import torch

class SimpleFactorRanker(nn.Module):
    def __init__(self, num_items, num_users, num_latent_factors):
        super(SimpleFactorRanker, self).__init__()
        self.num_items = num_items
        self.num_users = num_users
        self.num_latent_factors = num_latent_factors
        # we normalize the weights to have l2 norm 1 for each vector
#        self.item_weights = #  nn.Parameter(torch.Tensor(self.num_items,
        #                                               self.num_latent_factors))
        # nn.init.normal_(self.item_weights)
        self.item_weights = nn.Embedding(self.num_items, self.num_latent_factors,sparse=True)#,
                                         #max_norm=1.0)
#        self.user_weights = nn.Parameter(torch.Tensor(self.num_users,
 #                                                     self.num_latent_factors))
  #      nn.init.normal_(self.user_weights)

        self.user_weights = nn.Embedding(self.num_users, self.num_latent_factors,sparse=True)#,
#                                         max_norm=1.0)

    def forward(self,inputUsers,inputItems,negativeItems):
        iI = self.item_weights(inputItems)
        iU = self.user_weights(inputUsers)
        iN = self.item_weights(negativeItems)

        # iI = self.item_weights[inputItems,:]
        # iU = self.user_weights[inputUsers,:]
        # iN = self.item_weights[negativeItems,:]
        
        itemScore = torch.einsum('bl,bl->b',(iU,iI))
        negScore = torch.einsum('bl,bnl->bn',(iU,iN))

        return itemScore, negScore

    def bpr(self,itemScore, negScore):
        bpr0 = itemScore.view(-1,1)-negScore
        bpr1 = -torch.sigmoid(bpr0)
        bpr2 = torch.mean(bpr1.view(-1))

        return bpr2
