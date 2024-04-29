import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer

device=torch.device("cpu")

#input parameters
input_size=30
hidden_size=500
num_classes=2
num_epoch=200
learning_rate=1e-5
input,output=load_breast_cancer(return_X_y=True)
# print(input)
# print(input.shape)
# print(output)
# print(output.shape)

#from numpy to torch
train_input=torch.from_numpy(input).float() # wihtout float it will give this error: expected scalar type Float but found Double
train_output=torch.from_numpy(output).long()

class NeuralNet(nn.Module):
    
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.fc1=nn.Linear(input_size,hidden_size)
        self.lrelu=nn.LeakyReLU(negative_slope=0.02)
        self.fc2=nn.Linear(hidden_size,num_classes)
    
    def forward(self,input):
        output_fc1=self.fc1(input)
        output_lrelu=self.lrelu(output_fc1)
        output=self.fc2(output_lrelu)
        return output

model=NeuralNet(input_size,hidden_size,num_classes)

lossf=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range (num_epoch):
    outputs=model(train_input)
    loss=lossf(outputs,train_output)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}/{num_epoch}, Loss: {loss:.4f}")
