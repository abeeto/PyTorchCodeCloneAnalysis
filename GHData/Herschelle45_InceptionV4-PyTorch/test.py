import torch
from model import InceptionV4

def test(m):
    model = m(3, 1000)
    x = torch.randn(1, 3, 299, 299)
    return model(x), model 

out, model = test(InceptionV4)
print(out.shape, model, sep='\n')
