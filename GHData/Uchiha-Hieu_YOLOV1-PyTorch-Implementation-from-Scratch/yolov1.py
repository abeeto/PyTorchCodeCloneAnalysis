import torch
import torch.nn as nn
from .backbones.components import conv_bn_relu

class YOLOv1(nn.Module):
    def __init__(self,backbone,cell_size=7,num_boxes=2,num_classes=3):
        super(YOLOv1,self).__init__()
        self.backbone = backbone
        self.conv = conv_bn_relu(in_c=backbone.out_c,out_c=1024,k=1,s=1,p=0,use_bn=True,use_relu=True)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*backbone.out_size*backbone.out_size,512),
            nn.Dropout2d(0.3),
            nn.Linear(512,cell_size*cell_size*(5*num_boxes+num_classes)),
            # nn.Sigmoid()
        )
        self.S = cell_size
        self.B = num_boxes
        self.C = num_classes
        
    def forward(self,x):
        out = self.backbone(x)
        out = self.conv(out)
        out = self.fc(out).clone()
        return out.view(-1,self.S,self.S,self.B*5+self.C)
        # self.B*5 + self.C : 
        # B first elements are confident score of each box
        # 4*B next elements is coordinates of each box (box1_x_cell,box1_y_cell,box1_w_cell,box1_h_cell,box2_x_cell,box2_y_cell,box2_w_cell,box2_h_cell,....)]
        # C last elements : predict probability of class for boxes

