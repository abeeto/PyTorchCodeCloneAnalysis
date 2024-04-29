import torch
import torch.nn as nn

class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()
        
        self.in_conv=nn.Conv2d(in_channels=3, out_channels=64, kernel_size = 7, padding =3, bias= False)
        self.relu=nn.ReLU()
        
        self.denseblock1 = self.add_dense_block(dense_block, 64)
        self.transitionLayer1 = self.add_transition_block(transition_block, in_channels=160, out_channels=128)
         
        self.denseblock2 = self.add_dense_block(dense_block, 128)
        self.transitionLayer2 = self.add_transition_block(transition_block, in_channels=160, out_channels=128)
        
        self.denseblock3 = self.add_dense_block(dense_block, 128)
        self.transitionLayer3 = self.add_transition_block(transition_block, in_channels=160, out_channels=64)
        
        self.denseblock4 = self.add_dense_block(dense_block, 64)
        
        self.bn=nn.BatchNorm2d(num_features=64)
        self.lastlayer=nn.Linear(64*4*4, 512)
        self.final = nn.Linear (512, num_classes)
       
    
    def add_dense_block(self, block, in_channels):
        layer=[]
        layer.append(block(in_channels))
        D_seq=nn.Sequential(*layer)
        return D_seq
    
    def add_transition_block(self, layers, in_channels, out_channels):
        trans=[]
        trans.append(layers(in_channels, out_channels))
        T_seq=nn.Sequential(*trans)
        return T_seq
    
    
    
    def forward(self, x):
        out= self.relu(self.in_conv(x))
        out= self.denseblock1(out)
        out=self.transitionLayer1(out)
        
        out= self.denseblock2(out)
        out=self.transitionLayer2(out)
        
        out= self.denseblock3(out)
        out=self.transitionLayer3(out)
        
        out=self.bn(out)
        out=out.view(-1, 64*4*4)
        
        out=self.lastlayer(out)
        out=self.final(out)
        
        return out
