import torch

y_fun = lambda x:x + 2*x**2 + x**3
sample_size = 10000

daten = list(torch.linspace(-2, 2, sample_size))

input = daten
output = [y_fun(x) for x in daten]