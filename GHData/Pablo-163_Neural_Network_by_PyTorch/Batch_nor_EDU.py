import torch
import torch.nn as nn

input_size = 7
batch_size = 5
input_tensor = torch.randn(batch_size, input_size, dtype=torch.float)

eps = 1e-3

def custom_batch_norm1d(input_tensor, weight, bias, eps):
    normed_tensor = torch.zeros(input_tensor.shape)
    for i in range(input_tensor.shape[1]):
        avg = input_tensor[:,i].sum() / input_tensor.shape[0]
        mse = ((input_tensor[:,i] - avg)**2).sum() / input_tensor.shape[0]
        normed_tensor[:, i] = (input_tensor[:,i] - avg) / (mse + eps)**0.5 * weight[i] + bias[i]
    return normed_tensor

# Check that works fine
batch_norm = nn.BatchNorm1d(input_size, eps=eps)
batch_norm.bias.data = torch.randn(input_size, dtype=torch.float)
batch_norm.weight.data = torch.randn(input_size, dtype=torch.float)
batch_norm_out = batch_norm(input_tensor)
custom_batch_norm_out = custom_batch_norm1d(input_tensor, batch_norm.weight.data, batch_norm.bias.data, eps)
print(torch.allclose(batch_norm_out, custom_batch_norm_out) \
      and batch_norm_out.shape == custom_batch_norm_out.shape)
