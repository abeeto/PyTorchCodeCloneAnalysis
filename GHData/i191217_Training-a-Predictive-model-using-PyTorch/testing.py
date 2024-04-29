import torch
from torch.autograd import Variable
import numpy as np

Xtt=np.loadtxt("TestData.csv")

X_test=Variable(torch.FloatTensor(Xtt), requires_grad=True)

model = torch.load('myModel.pth')
model.eval()

ytest = model.forward(X_test)
np.savetxt("myPredictions.csv", ytest.detach().numpy())