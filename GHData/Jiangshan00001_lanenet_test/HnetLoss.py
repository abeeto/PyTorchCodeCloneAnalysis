
#https://github.com/stesha2016/lanenet-enet-hnet/blob/master/lanenet_model/hnet_loss.py
import torch
class HnetLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, image, h_matrix):
        pass
        #apply matrix to image points
        # get different value points
        # fit
        # map to old image pos
        # compare the different






