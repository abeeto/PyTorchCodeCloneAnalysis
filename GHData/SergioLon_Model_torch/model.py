from torch_geometric.nn import FeaStConv,GCNConv,InstanceNorm,BatchNorm
import torch
import numpy as np
import time 
#from torch_geometric.utils import 
class GCN(torch.nn.Module):
    def __init__(self, add_self_loops=True,bias=True,heads=1):
        super(GCN, self).__init__()
        torch.manual_seed(time.time())
        
        self.linear_1 = torch.nn.Linear(6,
                                6,
                                #heads=1,
                                )
        #torch.nn.init.normal_(self.linear_1.weight,mean=0,std=0.3)
        
        
        self.g_conv1 = FeaStConv(6,
                                128,
                                add_self_loops=False,
                                bias=True,
                                heads=1,
                                )
        #torch.nn.init.xavier_uniform_(self.g_conv1.weight) 
        #torch.nn.init.normal_(self.g_conv1.weight,mean=0,std=0.3)
        
        
        self.g_conv2 = FeaStConv(128,
                                128,
                                add_self_loops=False,
                                bias=True,
                                #heads=1,
                                )
        #torch.nn.init.xavier_uniform_(self.g_conv2.weight)
        #torch.nn.init.normal_(self.g_conv2.weight,mean=0,std=0.3)
        
        
        self.g_conv3 = FeaStConv(128,
                                128,
                                add_self_loops=False,
                                bias=True,
                                #heads=1,
                                
                                 )
        #torch.nn.init.xavier_uniform_(self.g_conv3.weight)
        #torch.nn.init.normal_(self.g_conv3.weight,mean=0,std=0.3)
        
        
        self.g_conv4 = FeaStConv(128,
                                128,
                                add_self_loops=False,
                                bias=True,
                                #heads=1,
                                )
        #torch.nn.init.normal_(self.g_conv4.weight,mean=0,std=0.3)
        
        self.linear_2 = torch.nn.Linear(128,
                                3,
                                #heads=1,
                                )
        #torch.nn.init.normal_(self.linear_2.weight,mean=0,std=0.3)
        # torch.nn.init.xavier_uniform_(self.g_conv4.weight)
        #self.dropout=torch.nn.Dropout(p=0.9)
        #self.softplus=torch.nn.Softplus()
        self.b_norm_1=BatchNorm(128)
        self.b_norm_2=BatchNorm(128)
        self.b_norm_3=BatchNorm(128)
        self.b_norm_4=BatchNorm(128)
    def forward(self, data):
    
        x,edge_index=torch.cat([data.pos,data.norm],dim=1),data.edge_index
        #x,edge_index=torch.cat([data.pos,data.norm],dim=0),data.edge_index
        #x,edge_index=data.norm,data.edge_index
        #print(x)
        #adj=to_dense_adj(edge_index)
        #GCN layer
        #x=torch.tensor(np.expand_dims(x[:,0].cpu().detach().numpy(),axis=-1))
        #x=x[:,0].unsqueeze(1)
        #x=x.to('cuda')
        # x=self.linear_1(x)
        # x=x.relu()
        x=self.g_conv1(x,edge_index)
        #x=self.b_norm_1(x)
        x=x.relu()
        
        #x=self.dropout(x)
        x=self.g_conv2(x,edge_index)
        #x=self.b_norm_2(x)
        x=x.relu()
        # #x=self.dropout(x)
        x=self.g_conv3(x,edge_index)
        #x=self.b_norm_3(x)
        x=x.relu()
        # # #x=self.dropout(x)
        x=self.g_conv4(x,edge_index)
        #x=self.b_norm_4(x)
        x=x.relu()
        x=self.linear_2(x)
        #a=torch.zeros((x.size(0),1))
        #a=torch.sqrt(x[:,0]**2+x[:,1]**2+x[:,2]**2)
        #x=torch.cat([x,a.unsqueeze(1)],dim=1)
        return x