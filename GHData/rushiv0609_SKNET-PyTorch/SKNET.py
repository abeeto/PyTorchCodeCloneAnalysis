import torch
import torch.nn as nn
from torchsummary import summary

class SKConv2(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32, use_1x1 = False):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv2, self).__init__()
        d = max(int(features / r), L)
        self.features = features
        self.M = M
        self.convs = nn.ModuleList([])
        start_index = 0
        dilation_cnt = 0
        if use_1x1:
            start_index += 1
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=1, stride=stride, groups=G),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False))
                )
            
        for i in range(start_index, M):
            dilation_cnt += 1
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d( features, features, kernel_size=3, dilation = i+1, stride=stride, padding = i+1, groups=G),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False))
            )
            # print(dilation_cnt)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv1d(features, d, kernel_size=1, stride=1),
                                nn.BatchNorm1d(d))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Conv1d(d, features, kernel_size=1, stride=1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size,in_ch,W,H = x.shape
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_U).view([batch_size, self.features, 1])
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors).squeeze()
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class SKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 , stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(#nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                #nn.Linear(features, d),
                                nn.Conv1d(features, d, kernel_size=1, stride=1),
                                nn.BatchNorm1d(d),
                                nn.ReLU(inplace=False),
                                nn.Conv1d(d, 2 * features, kernel_size=1, stride=1))
        # self.fcs = nn.ModuleList([])
        # for i in range(M):
        #     self.fcs.append(
        #           nn.Conv1d(d, features, kernel_size=1, stride=1)
        #           #nn.Linear(d, features)
        #     )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size,in_ch,W,H = x.shape
        # Split
        feats_3x3 = self.convs[0](x)
        feats_5x5 = self.convs[1](x)
        # feats = [conv(x) for conv in self.convs]
        # feats = torch.cat(feats, dim=1)
        # feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        
        # Fuse
        feats_U = feats_5x5.add(feats_3x3)
        feats_S = self.gap(feats_U).view([batch_size, self.features])
        #print(feats_S.shape)
        feats_Z = self.fc(feats_S.unsqueeze(2))
        
        # Select
        # attention_vectors = [fc(feats_Z) for fc in self.fcs]
        # attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = feats_Z
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        # print(attention_vectors.shape)
        attention_vectors = self.softmax(attention_vectors)
        att_3x3 = attention_vectors[:,0]
        att_5x5 = attention_vectors[:,1]
        # feats_V = torch.sum(feats*attention_vectors, dim=1)
        feats_3x3 = feats_3x3.mul(att_3x3)
        feats_5x5 = feats_5x5.mul(att_5x5)
        feats_V = feats_3x3.add_(feats_5x5)
        
        return feats_V
    
class SKConv1x1(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv1x1, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=(1 + 2*i) , stride=stride, padding=i, groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(#nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                #nn.Linear(features, d),
                                nn.Conv1d(features, d, kernel_size=1, stride=1),
                                nn.BatchNorm1d(d),
                                nn.ReLU(inplace=False),
                                nn.Conv1d(d, 2 * features, kernel_size=1, stride=1))
        # self.fcs = nn.ModuleList([])
        # for i in range(M):
        #     self.fcs.append(
        #           nn.Conv1d(d, features, kernel_size=1, stride=1)
        #           #nn.Linear(d, features)
        #     )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        batch_size,in_ch,W,H = x.shape
        # Split
        feats_1x1 = self.convs[0](x)
        feats_3x3 = self.convs[1](x)
        # feats = [conv(x) for conv in self.convs]
        # feats = torch.cat(feats, dim=1)
        # feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])
        
        # Fuse
        feats_U = feats_3x3.add(feats_1x1)
        feats_S = self.gap(feats_U).view([batch_size, self.features])
        #print(feats_S.shape)
        feats_Z = self.fc(feats_S.unsqueeze(2))
        
        # Select
        # attention_vectors = [fc(feats_Z) for fc in self.fcs]
        # attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = feats_Z
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        # print(attention_vectors.shape)
        attention_vectors = self.softmax(attention_vectors)
        att_1x1 = attention_vectors[:,0]
        att_3x3 = attention_vectors[:,1]
        # feats_V = torch.sum(feats*attention_vectors, dim=1)
        feats_1x1 = feats_1x1.mul(att_1x1)
        feats_3x3 = feats_3x3.mul(att_3x3)
        feats_V = feats_1x1.add_(feats_3x3)
        
        return feats_V


class SKUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, M=2, G=32, r=16, stride=1, L=32, use_1x1=False):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            )
        # self.conv2_sk =  SKConv1x1(mid_features, M=M, G=G, r=r, stride=stride, L=L)
        # if use_1x1 :
        #     self.conv2_sk = SKConv1x1(mid_features, M=M, G=G, r=r, stride=stride, L=L)
            
    
        self.conv2_sk = SKConv2(mid_features, M=M, G=G, r=r, stride=stride, L=L, use_1x1 = use_1x1)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
            )
        

        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2_sk(out)
        out = self.conv3(out)
        
        return self.relu(out + self.shortcut(residual))
    
class BasicBlock(nn.Module):
    def __init__(self, in_features, mid_features, out_features, M=2, G=32, r=16, stride=1, L=32, use_1x1=False):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=stride, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            )
        # self.conv2_sk =  SKConv1x1(mid_features, M=M, G=G, r=r, stride=stride, L=L)
        # if use_1x1 :
        #     self.conv2_sk = SKConv1x1(mid_features, M=M, G=G, r=r, stride=stride, L=L)
            
    
        # self.conv2_sk = SKConv2(mid_features, M=M, G=G, r=r, stride=stride, L=L, use_1x1 = use_1x1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_features, mid_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_features)
            )
        

        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        return self.relu(out + self.shortcut(residual))

class SKNet(nn.Module):
    def __init__(self, class_num, nums_block_list = [3, 4, 6,3], strides_list = [1, 2, 2, 2], G = 32, use_1x1 = False, M = 2, block = BasicBlock):
        '''
        Parameters
        ----------
        class_num : INT, output layer size(number of classes)
        nums_block_list : List, number of SKUnit in each block(max size of list = 4), default is [3, 4, 6, 3].
        strides_list : List, number of strides for SKUnit in each block, default is [1, 2, 2, 2].

        '''
        super(SKNet, self).__init__()
        self.block = block
        self.use_1x1 = use_1x1
        self.M = M
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding = 1, bias=False, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
                
        self.stage_1 = self._make_layer(64, 64, 64, nums_block=nums_block_list[0], stride=strides_list[0], G = G)
        self.stage_2 = self._make_layer(64, 128, 128, nums_block=nums_block_list[1], stride=strides_list[1], G = G)
        self.stage_3 = self._make_layer(128, 256, 256, nums_block=nums_block_list[2], stride=strides_list[2], G = G)
        self.stage_4 = self._make_layer(256, 512, 512, nums_block=nums_block_list[3], stride=strides_list[3], G = G)
     
        self.gap = nn.AvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512, class_num))#,
            # nn.Softmax(dim = 1))
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
    def _make_layer(self, in_feats, mid_feats, out_feats, nums_block, stride=1, G = 32):
        layers=[self.block(in_feats, mid_feats, out_feats, stride=stride, M = self.M, G = G, use_1x1 = self.use_1x1)]
        for _ in range(1,nums_block):
            layers.append(self.block(out_feats, 
                                 mid_feats, 
                                 out_feats, 
                                 M = self.M, 
                                 G = G, 
                                 use_1x1=self.use_1x1))
        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        # print(fea.shape)
        fea = self.stage_4(fea)
        fea = self.gap(fea)
        # return fea
        fea = fea.view(fea.shape[0], -1)
        fea = self.classifier(fea)
        return fea
    
def sknet29(num_classes, skconv = False):
    if skconv:
        return SKNet(200, [2,2,2,2], [1,2,2,2], G = 1, M = 2, block = SKUnit)
    
    return SKNet(200, [2,2,2,2], [1,2,2,2], G = 1, M = 2)

    
if __name__ == '__main__':
    # net = SKNet(200, [2,2,2,2], [1,2,2,2], G = 1, M = 3).cuda()
    # print(summary(net, (3, 64, 64)))
    net = sknet29(200, True).cuda()
    print(summary(net, (3, 56, 56)))
    torch.cuda.empty_cache()
    # c = SKConv(128)
    # x = torch.zeros(8,3,56,56).cuda()
    # print(net(x).shape)   