import torch.nn as nn
import torch

class OurModule(nn.Module):
    def __init__(self, num_inputs, num_classes, dropout_prob = 0.3): #constructor
        super(OurModule, self).__init__() #call parent's constructor
        self.pipe = nn.Sequential( #Creates pipe with layers
            nn.Linear(num_inputs,5),
            nn.ReLU(),
            nn.Linear(5,20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax()
        )
    def forward(self, x): #our implementation of data transformation
        return self.pipe(x) #uses callable property

if __name__ == "__main__":
    net = OurModule(num_inputs=2, num_classes=3)
    print(net)
    v = torch.FloatTensor([[2, 3]])
    out = net(v)
    print(out)
    print("Cuda's availability is %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Data from cuda: %s" % out.to('cuda'))
