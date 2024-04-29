import torch
import torch.nn as nn

input_size = 3
batch_size = 5
eps = 1e-1

class CustomBatchNorm1d:
    def __init__(self, weight, bias, eps, momentum):
        self.running_mean = torch.zeros(weight.shape[0])
        self.running_var = torch.ones(weight.shape[0])
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.momentum = momentum
        self.eva = False

    # конструктор
    def __call__(self, input_tensor):
        normed_tensor =  torch.zeros(input_tensor.shape) # Напишите в этом месте нормирование входного тензора.
        for i in range(input_tensor.shape[1]):
            avg = input_tensor[:, i].sum() / input_tensor.shape[0]
            mse = ((input_tensor[:, i] - avg) ** 2).sum() / input_tensor.shape[0]
            #self.running_mean[i] = avg * (1 - self.momentum) + self.momentum * self.running_mean[i]
            #self.running_var[i] = mse * (1 - self.momentum) * input_tensor.shape[0] / (input_tensor.shape[0]-1) + self.momentum * self.running_var[i]
            if self.eva:
                normed_tensor[:, i] = (input_tensor[:,i] - self.running_mean[i]) / (self.running_var[i] + self.eps) ** 0.5 * self.weight[i] + self.bias[i]
            else:
                normed_tensor[:, i] = (input_tensor[:,i] - avg) / (mse + self.eps)**0.5 * self.weight[i] + self.bias[i]
                self.running_mean[i] = avg * (1 - self.momentum) + self.momentum * self.running_mean[i]
                self.running_var[i] = mse * (1 - self.momentum) * input_tensor.shape[0] / (
                            input_tensor.shape[0] - 1) + self.momentum * self.running_var[i]
        return normed_tensor

    # переключение в режим предикта.
    def eval(self):
        self.eva = True
       


batch_norm = nn.BatchNorm1d(input_size, eps=eps)
batch_norm.bias.data = torch.randn(input_size, dtype=torch.float)
batch_norm.weight.data = torch.randn(input_size, dtype=torch.float)
batch_norm.momentum = 0.5

custom_batch_norm1d = CustomBatchNorm1d(batch_norm.weight.data,
                                        batch_norm.bias.data, eps, batch_norm.momentum)

# Проверка происходит автоматически вызовом следующего кода
all_correct = True

for i in range(8):
    torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
    norm_output = batch_norm(torch_input)
    custom_output = custom_batch_norm1d(torch_input)
    all_correct &= torch.allclose(norm_output, custom_output, atol=1e-06) \
        and norm_output.shape == custom_output.shape

print(all_correct)

batch_norm.eval()
custom_batch_norm1d.eval()

for i in range(8):
    torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
    norm_output = batch_norm(torch_input)
    custom_output = custom_batch_norm1d(torch_input)
    all_correct &= torch.allclose(norm_output, custom_output, atol=1e-06) \
        and norm_output.shape == custom_output.shape
    #print("custom avg is", custom_batch_norm1d.running_mean)
    #print("custom mse is", custom_batch_norm1d.running_var)
    #print(batch_norm._buffers)
    #print("norm_output:", norm_output)
    #print( "custom:", custom_output)
print(all_correct)
