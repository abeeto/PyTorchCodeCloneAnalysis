# -*- coding: utf-8 -*-
import torch
import torchvision

from modelx import Net

if __name__ == '__main__':
    network = Net()
    network.load_state_dict(torch.load('model.pth'))
    network.eval()
    mnist_data = torchvision.datasets.MNIST('./data/',
                                            train=False,
                                            download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,))
                                            ]))

    one_data = mnist_data[0][0].unsqueeze(0)
    one_data_label = mnist_data[0][1]

    with torch.no_grad():
        output = network(one_data)
        print("预测的类别是否正确：", output.max(1, keepdim=True)[1].item() == one_data_label)
        print("模型的输出数值是：", output[0].numpy())

# 预测的类别是否正确： True
# 模型的输出数值是： [-2.29137955e+01 -1.76951065e+01 -1.34065456e+01 -1.56550188e+01
#  -2.26447525e+01 -2.39984474e+01 -3.70633507e+01 -2.50339190e-06
#  -2.10068398e+01 -1.40329485e+01]