from impl_forward_method import MyNet
import torch

net = MyNet()
print("~~~Analysing Shapes of output and weight tensors~~~")

bs = 20 # batch size
color = 1
H = W = 12
input = torch.rand([bs, color, H, W])

conv1_out = net.conv1(input)
conv2_out = net.conv2(conv1_out)
fc1_out = net.fc1(conv2_out.flatten(1))
fc2_out = net.fc2(fc1_out)
out = net.out(fc2_out)

print("input shape: ", input.shape)
print("Conv1 out shape: ", conv1_out.shape)
print("Conv2 out shape: ", conv2_out.shape)
print("FC1 out shape: ", fc1_out.shape)
print("FC2 out shape: ", fc2_out.shape)
print("Output shape: ", out.shape)

print("->Weights:")
print(net.conv1.weight.shape)
print(net.conv2.weight.shape)
print(net.fc1.weight.shape)
print(net.fc2.weight.shape)
print(net.out.weight.shape)