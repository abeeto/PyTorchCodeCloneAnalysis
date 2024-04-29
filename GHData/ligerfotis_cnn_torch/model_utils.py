import torch

input_height = 32
stride = 1
kernel = 3
padding = 1


def create_conv_encoder(in_channels, hidden_layers, device="cpu"):
    # create an empty list to carry the encoder layers
    layers = []
    feature_map_size = input_height

    # Build Encoder
    for h_dim in hidden_layers[:-1]:
        # print(h_dim)
        if h_dim == 'M':
            layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
        else:

            layers.append(
                torch.nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=kernel, stride=stride, padding=padding))
            layers.append(torch.nn.BatchNorm2d(h_dim))
            layers.append(torch.nn.ReLU(True))
            feature_map_size = int((feature_map_size - kernel + 2 * padding) / stride) + 1
            in_channels = h_dim

    layers.append(
        torch.nn.Conv2d(in_channels, out_channels=hidden_layers[-1], kernel_size=kernel, stride=stride, padding=0))
    feature_map_size = int((feature_map_size - kernel) / stride) + 1

    # torch.nn.BatchNorm2d(hidden_layers[-1]),
    layers.append(torch.nn.ReLU(True))
    return torch.nn.Sequential(*layers).to(device), feature_map_size


def test_model(net, testloader, classes, device):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
