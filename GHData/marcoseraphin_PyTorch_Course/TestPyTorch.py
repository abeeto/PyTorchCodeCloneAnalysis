import torch
print ('Torch Version: ' + torch.__version__)

t1 = torch.Tensor(5,3)
print(t1)

x = torch.randn(2,3)
print(x)
     
y = torch.randn(2,3)
print(y)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print('Using GPU')

if torch.cuda.is_available():
    x = x.to(mps_device)
    y = y.to(mps_device)

print(torch.add(x,y))