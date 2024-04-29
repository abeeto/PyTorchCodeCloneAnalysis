import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def matrices():

    # Given a fully connected network
    #
    #     (1a)   (2a)   (3a)   (4a)
    #         \\  |  \ / |   //
    #            (1b)   (2b)

    # The values of a layer can be
    # expressed as a matrix

    layer_1 = numpy.array([
        [-0.5, 0.9, 0.1, -0.7]
        ])

    # So can the weights between the layers
    # (1a 1b) (1a 2b) (2a 1b) ...

    weights = numpy.array([
        [-0.3, 0.9],
        [ 0.2, 0.1],
        [-0.4, 0.6],
        [-0.7, 0.2]
        ])

    # Multiplying the matrices gives
    # the next layer values

    layer_2 = layer_1.dot(weights)
    print(layer_2)

    #    [[ 0.78 -0.44]]

def tensors():

    # Tensors are multiplied on the GPU

    device = torch.device("cuda:0")

    layer_1 = torch.tensor([
        [-0.5, 0.9, 0.1, -0.7]
        ], device=device, dtype=torch.float)

    # Tensors can track their influence
    # on the ongoing calculations

    weights = torch.tensor([
        [-0.3, 0.9],
        [ 0.2, 0.1],
        [-0.4, 0.6],
        [-0.7, 0.2]
        ], device=device, dtype=torch.float, requires_grad=True)

    # Calculate a numeric loss based on the difference
    # between the actual and the expected output

    layer_2 = layer_1.mm(weights)

    layer_2_expected = torch.tensor([
        [0.1, -0.7]
        ], device=device, dtype=torch.float)

    loss = (layer_2 - layer_2_expected).sum()

    # Ask the weights to adjust themselves
    # in a way that minimizes that value

    loss.backward()

    with torch.no_grad():
        learning_rate = 0.0001
        weights -= learning_rate * weights.grad
        weights.grad.zero_()

    # We get better weights

    print(weights);

    # tensor([[-0.3000,  0.9000],
    #         [ 0.1999,  0.0999],
    #         [-0.4000,  0.6000],
    #         [-0.6999,  0.2001]], device='cuda:0', requires_grad=True)

def networks():

    # Networks describe a set of tensors
    # Here we say three fully connected layers
    # That is three tensors of 1x10, 10x5 and 5x2 respectively

    network = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2))

    device = torch.device("cuda:0")
    network.to(device)

    # Activation functions can be used to modulate output values
    # Here we say the second layer nodes use the ReLU function
    # That is a fancy name for max(0, x)

    # Send values through the network

    layer_1 = torch.tensor([
        [-0.5, 0.9, 0.1, -0.7, 0.2, 0.6, -0.3, 0, -0.4, -0.1]
        ], device=device, dtype=torch.float)

    layer_3 = network(layer_1)
    print(layer_3)

    # Instead of summing differences as we did above
    # We will use the Mean Squared Error function
    # to calculate a loss

    layer_3_expected = torch.tensor([
        [0.1, -0.7]
        ], device=device, dtype=torch.float)

    loss_function = nn.MSELoss(reduction='sum')

    loss = loss_function(layer_3, layer_3_expected)
    print(loss.item())

    # Adjust the weights in a way that minimizes that value
    # Here we use the Adam optimizer which is given
    # the tensors and a learning rate
    optimizer = optim.Adam(network.parameters(), lr=0.0001)
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

def classification():

    # Load images

    image_transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    image_data = datasets.ImageFolder(
        root="../Learning-Data/cat-dog/train",
        transform=image_transforms)

    image_loader = data.DataLoader(image_data, shuffle=True)

    # Describe our network

    network = nn.Sequential(
        nn.Linear(12288, 64),
        nn.ReLU(),
        nn.Linear(64, 2))

    device = torch.device("cuda:0")
    network.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    # Do ourselves some learning

    network.train()

    for epoch in range(1000):

        cummulative_loss = 0
        average_loss = 0

        for (index, batch) in enumerate(image_loader):
            optimizer.zero_grad()

            input, output_expected = batch
            input = input.view(-1, 12288).to(device)
            output_expected = output_expected.to(device)
            output = network(input)

            loss = loss_function(output, output_expected)
            loss.backward()

            optimizer.step()

            cummulative_loss += loss.item()
            average_loss = cummulative_loss / (index + 1)

            if index % 1000 == 0:
                print("Epoch: {} Index: {} Average Loss: {:.5f}".format(
                    epoch, index, average_loss))

matrices()
tensors()
networks()
classification()
