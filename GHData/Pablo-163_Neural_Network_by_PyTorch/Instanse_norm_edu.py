import torch
import torch.nn as nn

eps = 1e-3

batch_size = 5
input_channels = 2
input_length = 30

instance_norm = nn.InstanceNorm1d(input_channels, affine=False, eps=eps)

input_tensor = torch.randn(batch_size, input_channels, input_length, dtype=torch.float)


def custom_instance_norm1d(input_tensor, eps):
    normed_tensor = torch.zeros(input_tensor.shape)
    for num_batch, batch in enumerate(input_tensor):
        for num_channel, channel in enumerate(batch):
            avg = torch.mean(channel)
            mse = torch.var(channel, unbiased=False)
            normed_tensor[num_batch][num_channel] = (channel - avg) / (mse + eps) ** 0.5
    return normed_tensor


# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
norm_output = instance_norm(input_tensor)
custom_output = custom_instance_norm1d(input_tensor, eps)
print(torch.allclose(norm_output, custom_output, atol=1e-06) and norm_output.shape == custom_output.shape)