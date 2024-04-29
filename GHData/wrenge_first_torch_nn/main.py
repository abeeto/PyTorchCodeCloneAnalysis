import torch
import torch.optim as optim
import numpy as np
from torch import nn
from netwok import Net
from data_reader import read_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
net = Net().to(device)

x_train, y_train = read_data("data/1")
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).long()
x_train, y_train = x_train.to(device), y_train.to(device)

x_test = x_train
y_test = y_train

batches = 7
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)
learning_rate = 0.1
epochs = 1000

# optimizer = torch.optim.Adam(net.parameters(), learning_rate)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    running_loss = np.empty(len(x_train_batches))
    for batch in range(len(x_train_batches)):
        optimizer.zero_grad()
        outputs = net(x_train_batches[batch])
        loss = criterion(outputs, y_train_batches[batch])
        loss.backward()
        optimizer.step()

        running_loss[batch] = loss.item()

    print('[%d] loss: %.3f' %
          (epoch + 1, running_loss.mean()))

# print("accuracy = %s" % net.accuracy(x_test, y_test))
print("accuracy = %s" % torch.mean(torch.eq(torch.max(net(x_test), 1).indices, y_test).float()))

torch.onnx.export(net,                       # model being run
                  torch.split(x_train, 1)[0],                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

