
def activation(x):
    return 1/(1+torch.exp(-x))

inputs = images.view(images.shape[0],-1)
w1 = torch.randn(784, 512)
b1 = torch.randn(512)

w2 = torch.randn(512, 10)
b2 = torch.randn(10)

h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2
print (out)

Caluculating probability of each number:
def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x),dim=1).view(-1,1)
