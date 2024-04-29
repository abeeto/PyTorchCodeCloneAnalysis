import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os

class MeinNetz(nn.Module):
    def __init__(self):
        super(MeinNetz, self).__init__()
        self.lin1 = nn.Linear(10, 10) #Schicht1
        self.lin2 = nn.Linear(10, 10) #Schicht2
    
    def forward(self, x):
        x = F.relu(self.lin1(x)) #Input durch Schicht1 schicken
        x = self.lin2(x) #Dasselbe für Schicht2
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num 

netz = MeinNetz()

if os.path.isfile('meinNetz.pt'):
    netz = torch.load('meinNetz.pt')

for i in range(100):
    
    x = [1,0,0,0,1,0,0,0,1,1]
    input = Variable(torch.Tensor([x for _ in range(10)]))
    output = netz(input) #input durch das Netz schicken
    
    x = [0,1,1,1,0,1,1,1,0,0]
    target = Variable(torch.Tensor([x for _ in range(10)]))
    criterion = nn.MSELoss() 
    loss = criterion(output, target) #Berechnung des Fehlers
    print(loss)
    
    netz.zero_grad() #Gradient auf 0 setzen?
    loss.backward() #backpropagation
    
    optimizer = optim.SGD(netz.parameters(), lr=0.1) #Einstellung des Lerners
    optimizer.step() # Ausführen des Lernens

#Speichern und Laden
torch.save(netz, 'meinNetz.pt')