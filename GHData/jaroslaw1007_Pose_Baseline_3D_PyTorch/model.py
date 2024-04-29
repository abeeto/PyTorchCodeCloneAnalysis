import numpy as np
import torch
import torch.nn as nn

def weights_init(model):
    if isinstance(model, nn.Linear):
        nn.init.kaiming_normal(model.weight)
        
class LinearBlock(nn.Module):
    def __init__(self, linear_size, drop_rate=0.5):
        super(LinearBlock, self).__init__()
        
        self.linear_size = linear_size
        self.drop_rate = drop_rate
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.drop_rate)
        
        self.w1 = nn.Linear(self.linear_size, self.linear_size)
        self.bn1 = nn.BatchNorm1d(self.linear_size)
        self.w2 = nn.Linear(self.linear_size, self.linear_size)
        self.bn2 = nn.BatchNorm1d(self.linear_size)
        
    def forward(self, x):
        y = self.dropout(self.relu(self.bn1(self.w1(x))))
        y = self.dropout(self.relu(self.bn2(self.w2(y))))
        
        output = x + y
        
        return output
        
class LinearModel(nn.Module):
    def __init__(self, linear_size=1024, num_stage=2, drop_rate=0.5):
        super(LinearModel, self).__init__()
        
        self.linear_size = linear_size
        self.num_stage = num_stage
        self.drop_rate = drop_rate
        
        self.input_size = 16 * 2
        self.output_size = 16 * 3
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.drop_rate)
        
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.bn1 = nn.BatchNorm1d(self.linear_size)
        
        self.linear_stages = []
        
        for s in range(num_stage):
            self.linear_stages.append(LinearBlock(self.linear_size, self.drop_rate))
        self.linear_stages = nn.ModuleList(self.linear_stages)
        
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        
    def forward(self, x):
        y = self.dropout(self.relu(self.bn1(self.w1(x))))
        
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
            
        output = self.w2(y)
        
        return output
    
if __name__ == '__main__':
    model = LinearModel()
    rand_input = torch.randn((64, 16 * 2), dtype=torch.float)
    out = model(rand_input)
    
    print(out.shape)