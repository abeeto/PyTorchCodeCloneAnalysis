import torch
import torch.nn as nn

inputs=torch.Tensor(1, 1, 28, 28)
print("텐서 크기 : {}".format(inputs.shape))

# create Convolution Layers
conv1=nn.Conv2d(1, 32, 3, padding=1)
print("1st Conv Layer: ", conv1)

conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
print("2st Conv Layer: ", conv2)

# Create MaxPooling Layer
pool=nn.MaxPool2d(2)

out = conv1(inputs)
print("Pooling Layer: ", out.shape)

out = pool(out)
print("Output of 1st Conv: ", out.shape)

out = conv2(out)
print("Output of 2st Conv: ", out.shape)

out = pool(out)
print("Output of Pooling: ",out.shape)

print("Output size of 0-Dim", out.size(0))

print("Output size of 1-Dim", out.size(1))

print("Output size of 2-Dim", out.size(2))

print("Output size of 3-Dim", out.size(3))

# 첫번째 차원인 배치 차원은 그대로 두고 나머지는 펼쳐라
out = out.view(out.size(0), -1) 
print("Output of Flattening", out.shape)

fc = nn.Linear(3136, 10) # input_dim = 3,136, output_dim = 10
out = fc(out)
print("Output of Connected", out.shape)