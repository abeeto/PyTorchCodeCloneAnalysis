import torch
from torch.autograd import Variable
import torch.nn as nn

def A(n, in_channels=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, 2**(n+4), kernel_size=3, bias=True, padding=1),
        nn.Conv2d(2**(n+4), 2**(n+4), kernel_size=3, bias=True, padding=1),
        nn.ReLU()
    )

class PixelFusionNet(nn.Module):
    def __init__(self, N):
        super(PixelFusionNet, self).__init__()
        
        self.layers = N
        self.pyr = []
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, im1, im2):

        # Initialize the image pyramid
        self.pyr.append((im1, im2))
        for _ in range(self.layers):
            im1_l, im2_l = self.pyr[-1]
            im1_k, im2_k = self.avgpool(im1_l), self.avgpool(im2_l)
            self.pyr.append((im1_k, im2_k))
        
        encoder_pyr = []

        # Feature Extraction Layer-1
        im1, im2 = self.pyr[0]
        A0_0, A0_1 = A(0, im1.shape[1]), A(0, im2.shape[1])
        s0, t0 = A0_0(im1), A0_1(im2)
        encoder_pyr.append((s0, t0))


        # Feature Extraction Layer-2
        im1_1, im2_1 = self.pyr[1]
        A0_0, A0_1 = A(0, im1_1.shape[1]), A(0, im2_1.shape[1])
        A1_0, A1_1 = A(1, 2**4), A(1, 2**4)

        A0_im1_1, A0_im2_1 = A0_0(im1_1), A0_1(im2_1)
        A1_im1_1, A1_im2_1 = self.avgpool(A1_0(s0)), self.avgpool(A1_1(t0))
        s1, t1 = torch.cat((A0_im1_1, A1_im1_1), dim=1), torch.cat((A0_im2_1, A1_im2_1), dim=1)
        encoder_pyr.append((s1, t1))
        

        # Feature Extraction Layer 2 to N
        A0_0, A0_1 = A(0, im1.shape[1]), A(0, im2.shape[1])
        A1_0, A1_1 = A(1, 2**4), A(1, 2**4)
        A2_0, A2_1 = A(2, 2**5), A(2, 2**5)

        A1_im1_k_1, A1_im2_k_1 = A1_im1_1, A1_im2_1
        A0_im1_k_1, A0_im2_k_1 = A0_im1_1, A0_im2_1
        for k in range(2, self.layers):
            im1_k, im2_k = self.pyr[k]
            A2_im1_k, A2_im2_k = self.avgpool(A2_0(A1_im1_k_1)), self.avgpool(A2_1(A1_im2_k_1))
            A1_im1_k, A1_im2_k = self.avgpool(A1_0(A0_im1_k_1)), self.avgpool(A1_1(A0_im2_k_1))
            A0_im1_k, A0_im2_k = A0_0(im1_k), A0_1(im2_k)

            s_k, t_k = torch.cat((A0_im1_k, A1_im1_k, A2_im1_k),dim=1), torch.cat((A0_im2_k, A1_im2_k, A2_im2_k),dim=1)
            encoder_pyr.append((s_k, t_k))
            
            A1_im1_k_1, A1_im2_k_1 = A1_im1_k, A1_im2_k
            A0_im1_k_1, A0_im2_k_1 = A0_im1_k, A0_im2_k

        
        

        return self.pyr, encoder_pyr

    def warp(self, x, flow):

        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flow

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

def test():
    N = 8
    net = PixelFusionNet(N)
    inp1 = torch.randn((1, 3, 1536, 1536))
    inp2 = torch.randn((1, 1, 1536, 1536))
    pyr, enc_pyr = net(inp1, inp2)

    for k in range(N):
        im1_k, im2_k = pyr[k]
        enc_1, enc_2 = enc_pyr[k]
        print(im1_k.shape, im2_k.shape)
        print(enc_1.shape, enc_2.shape)
        print('')
     
    # enc_1, enc_2 = enc_pyr[0]
    # print(enc_1.shape, enc_2.shape)

    # enc_1, enc_2 = enc_pyr[1]
    # print(enc_1.shape, enc_2.shape)

test()



