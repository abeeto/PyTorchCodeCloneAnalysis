# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# # CORE of TUTORIAL:
# #I can not run this because I do not have the GPU on this PC
# #to put the model on a GPU it is enough
# device = torch.device("cuda:0")
# model.to(device)
#
# #all the tensors created can be moved to the GPU
# #my_tensor.to(device) returns a new copy of my_tensor on GPU and it is not rewritten
# #if I want a new my_tensor I need to assign it to a new tensor
# mytensor = my_tensor.to(device)
#
# #the BACKWARD and FORWARD can be computed on multiple GPU. default is on one GPU
# #to make my model run parallel I can do:
# model = nn.DataParallel(model)
# -------------------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100


#device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#make a dummy dataset

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)


#simple model

class Model(nn.Module):
    def __init__(self,input_size, output_size):
        super(Model,self).__init__()
        self.fc = nn.Linear(input_size,output_size)


    def forward(self,input):
        output = self.fc(input)
        print("\tIn Model: input size " , input.size(),
              "output size ", output.size())

        return output


#create a model and dataparallel

model = Model(input_size,output_size)
print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use ", torch.cuda.device_count(),"GPUs")
    model = nn.DataParallel(model)

model.to(device)


# #run model
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size ", input.size(),
          "output size ", output.size())

#dataparallel will split automatically the data and send jobs orders to multiple models on several GPUS