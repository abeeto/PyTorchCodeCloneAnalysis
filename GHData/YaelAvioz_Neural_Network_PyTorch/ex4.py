import sys
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

#creating test_y
def write_res(results):
    out_file = open("test_y", "w")
    for batch in results:
        for pred in batch:
            out_file.write("%s\n" % pred.numpy()[0])
    out_file.close()

#training the model
def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()

#testing the model
def test(model, test_loader):
    model.eval()
    pred_list = []
    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            pred = output.max(1, keepdim = True)[1]

            #adding the prediction to the file
            pred_list.append(pred)

    return pred_list

#defining the model
class Model(nn.Module):
    def __init__(self, image_size):
        super(Model, self).__init__()
        self.image_size = image_size

        # Neural Network with two hidden layers: [128,64,10]
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    #activation function - ReLU
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#main func
if __name__ == "__main__":
    training_examples_path, training_labels_path, test_examples_path, out_file_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    #loading the data
    training_examples = numpy.loadtxt(training_examples_path, dtype ='int')
    training_labels = numpy.loadtxt(training_labels_path, dtype ='int')

    #converting to tensor + normalization
    tensor_x = torch.Tensor(training_examples/255)
    tensor_y = torch.tensor(training_labels, dtype=torch.long)

    labeled_training_examples_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = torch.utils.data.DataLoader(labeled_training_examples_dataset, batch_size=64, shuffle=True)

    #loading the test
    test_examples = numpy.loadtxt(test_examples_path, dtype='int')
    test_tensor = torch.Tensor(test_examples/255)
    test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=64, shuffle=False)

    # The Model with Adam optimizer
    epochs = 10
    lr = 0.00075
    model = Model(image_size = 28 * 28)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs):
        #training
        train(model, train_loader, optimizer)

        #shuffeling after every epoch
        train_loader = torch.utils.data.DataLoader(labeled_training_examples_dataset, batch_size=64, shuffle=True)

    #testing + creating test_y
    pred_list = test(model, test_loader)
    write_res(pred_list)
