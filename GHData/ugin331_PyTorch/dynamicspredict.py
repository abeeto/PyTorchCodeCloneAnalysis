#TODO: Generate more data?


import torch as torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import OrderedDict
import math


# DEPRECIATED NORMALIZATION
"""
def normalize_data(norm_tensor, col, tensor_max, tensor_min):
    sizetuple = norm_tensor.size()
    numrows = sizetuple[0]
    for i in range(0, numrows):
        # print(norm_tensor[i][col])
        norm_tensor[i][col] -= tensor_min
        norm_tensor[i][col] *= 2
        norm_tensor[i][col] /= (tensor_max - tensor_min)
        norm_tensor[i][col] -= 1
        # print(norm_tensor[i][col])
    return norm_tensor


def denormalize_data(norm_tensor, col, tensor_max, tensor_min):
    sizetuple = norm_tensor.size()
    numrows = sizetuple[0]
    for i in range(0, numrows):
        norm_tensor[i][col] += 1
        norm_tensor[i][col] *= (tensor_max-tensor_min)
        norm_tensor[i][col] /= 2
        norm_tensor[i][col] += tensor_min
    return norm_tensor
"""


col_mean_data = []
col_std_data = []
col_mean_tgt = []
col_std_tgt = []


# remove all outliers greater than 5 standard deviations away from center
def remove_outliers(data_tensor, target_tensor):
    print("removing outliers")
    rows_to_elim = []

    # loop thru each data column, get mean and std of column
    numcols = data_tensor.size()[1]
    print("number of columns:", numcols)
    numrows = data_tensor.size()[0]
    for colnum in range(0, numcols):
        temp_vals = np.empty(numrows)
        # generate temp column tensor
        for rownum in range(0, numrows):
            temp_vals[rownum] = data_tensor[rownum][colnum]

        col_tensor = torch.from_numpy(temp_vals).float()
        # if(colnum == 1):
        #     for temprownum  in range(0, numrows):
        #         print("row: ", temprownum, "value:", temp_vals[temprownum])
        # get mean and std of data tensor
        col_mean = torch.mean(col_tensor)
        col_std = torch.std(col_tensor)
        # col_mean_data.append(col_mean)
        # col_std_data.append(col_std)
        offset = col_std * 4
        print("column mean:", col_mean)
        print("column std:", col_std)
        # print("outlier offset:", offset)

        # go through and identify outliers
        for rownum in range(0, numrows):
            element = data_tensor[rownum][colnum]
            if element <= col_mean-offset or element >= col_mean+offset:
                # TODO: Change up order of columns see if its still all one column
                print("outlier element:", element, "from column:", colnum)
                if not rownum in rows_to_elim:
                    print("eliminating row:", rownum)
                    # print(data_tensor[rownum])
                    rows_to_elim.append(rownum)

    # loop thru each target column, get mean and std of column
    print(target_tensor.size())
    numcols = target_tensor.size()[1]
    numrows = target_tensor.size()[0]
    # print(numcols)
    for colnum in range(0, numcols):
        temp_vals = np.empty(numrows)
        # generate temp column tensor
        for rownum in range(0, numrows):
            temp_vals[rownum] = target_tensor[rownum][colnum]

        col_tensor = torch.from_numpy(temp_vals).float()
        # get mean and std of data tensor
        col_mean = torch.mean(col_tensor)
        col_std = torch.std(col_tensor)
        # col_mean_tgt.append(col_mean)
        # col_std_tgt.append(col_std)
        offset = col_std * 4
        # print("column mean:", col_mean)
        # print("column std:", col_std)
        # print("outlier offset:", offset)

        # go through and identify outliers
        for rownum in range(0, numrows):
            element = target_tensor[rownum][colnum]
            if element <= col_mean - offset or element >= col_mean + offset:
                print("outlier element:", element, "from column:", colnum)
                if not rownum in rows_to_elim:
                    print("eliminating row:", rownum)
                    # print(data_tensor[rownum])
                    rows_to_elim.append(rownum)

    # eliminate rows from data and target here
    # convert to np, delete all relevant rows, then return to tensor of type FloatTensor
    np_data_tensor = data_tensor.numpy()
    np_target_tensor = target_tensor.numpy()
    np_data_tensor = np.delete(np_data_tensor, rows_to_elim, 0)
    np_target_tensor = np.delete(np_target_tensor, rows_to_elim, 0)

    data_tensor = torch.from_numpy(np_data_tensor)
    target_tensor =  torch.from_numpy(np_target_tensor)

    data_tensor, target_tensor = data_tensor.type(torch.FloatTensor), target_tensor.type(torch.FloatTensor)

    # print("outliers removed:")
    # print(data_tensor)
    # print(target_tensor)

    return data_tensor, target_tensor


# z_score normalization on data & targets
def z_score(data_tensor, target_tensor):
    # z-score on data
    print("data:")
    numcols = data_tensor.size()[1]
    numrows = data_tensor.size()[0]
    for colnum in range(0, numcols):
        temp_vals = np.empty(numrows)
        # generate temp column tensor
        for rownum in range(0, numrows):
            temp_vals[rownum] = data_tensor[rownum][colnum]

        col_tensor = torch.from_numpy(temp_vals).float()
        # get mean and std of data tensor
        col_mean = torch.mean(col_tensor)
        col_std = torch.std(col_tensor)
        col_mean_data.append(col_mean)
        col_std_data.append(col_std)
        # col_std = col_std_data[colnum]
        # col_mean = col_mean_data[colnum]
        print("column mean:", col_mean)
        print("column std:", col_std)

        # normalize
        # TODO: MAKE THIS NOT WRONG
        for rownum in range(0, numrows):
            data_tensor[rownum][colnum] -= col_mean
            data_tensor[rownum][colnum] /= col_std

    # z-score norm on targets
    print("targets:")
    numcols = target_tensor.size()[1]
    numrows = target_tensor.size()[0]
    for colnum in range(0, numcols):
        temp_vals = np.empty(numrows)
        # generate temp column tensor
        for rownum in range(0, numrows):
            temp_vals[rownum] = target_tensor[rownum][colnum]

        col_tensor = torch.from_numpy(temp_vals).float()
        # get mean and std of data tensor
        tgt_mean = torch.mean(col_tensor)
        tgt_std = torch.std(col_tensor)
        col_mean_tgt.append(tgt_mean)
        col_std_tgt.append(tgt_std)
        # tgt_std = col_std_tgt[colnum]
        # tgt_mean = col_mean_tgt[colnum]
        print("column mean:", tgt_mean)
        print("column std:", tgt_std)

        # normalize
        # TODO: MAKE THIS NOT WRONG
        for rownum in range(0, numrows):
            target_tensor[rownum] -= tgt_mean
            target_tensor[rownum] /= tgt_std

    print("normalized tensors:")
    print(data_tensor)
    print(target_tensor)
    return data_tensor, target_tensor


def denormalize_data(output_tensor, target_tensor):
    # de-normalize data
    '''numcols = data_tensor.size()[1]
    numrows = data_tensor.size()[0]
    for colnum in range(0, numcols):
        col_std = col_std_data[colnum]
        col_mean = col_mean_data[colnum]
        for rownum in range(0, numrows):
            data_tensor[rownum][colnum] *= col_std
            data_tensor[rownum][colnum] += col_mean'''
    # de-normalize targets
    numcols = target_tensor.size()[1]
    numrows = target_tensor.size()[0]
    for colnum in range(0, numcols):
        tgt_std = col_std_tgt[colnum]
        tgt_mean = col_mean_tgt[colnum]
        # reverse operations
        for rownum in range(0, numrows):
            target_tensor[rownum] *= tgt_std
            target_tensor[rownum] += tgt_mean
            output_tensor[rownum] *= tgt_std
            output_tensor[rownum] += tgt_mean
    print("final denormalized targets:")
    print(target_tensor)
    print(target_tensor)
    return output_tensor, target_tensor


# get min + max in tensor column for better normalization(tm)
def get_col_minmax(tensor, col):
    data_max = torch.min(tensor)
    data_max = data_max.item()
    data_min = torch.max(tensor)
    data_min = data_min.item()
    for row in tensor:
        if row[col].item() > data_max:
            data_max = row[col].item()
        if row[col].item() < data_min:
            data_min = row[col].item()
    # print("column:", col, "column max:", data_max, "column min: ", data_min)
    return data_max, data_min

diffs = []


# compare tensors together, return number correct (used in compfunc below)
# TODO: Redo this
def comp_tensor(predict, target, diffpercent):
    correct = 0
    num = torch.numel(predict)
    # tempcorrect = 0
    for i in range(0, num):
        predict_val = predict[i].item()
        target_val = target[i].item()
        print("predict val:", predict_val, "target_val:",target_val)
        diffs.append(100*abs(abs(target_val - predict_val) / target_val))
        if abs(abs(target_val - predict_val) / target_val) <= diffpercent:
            correct += 1
    print(correct, "correct out of 6")
    # if tempcorrect == 6:
    #    correct += 1
    return correct


# compare two tensors for validation
# TODO: Redo this
def compfunc(predict, target, diffpercent):
    correct = 0
    predict_copy = predict.numpy()
    len_outer = len(predict_copy)
    for i in range(0, len_outer):
        correct += comp_tensor(predict[i], target[i], diffpercent)
    return correct


# delta column min/max, depreciated
delta_col_minmax = []


# dataset class for data
class simdata(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # load npy files
        actions = np.load(root_dir+'/actions.npy')
        states = np.load(root_dir+'/states.npy')
        deltas = np.load(root_dir+'/deltas.npy')

        # turn them to tensors
        actions = torch.from_numpy(actions)
        actions = actions.type(torch.float64)
        states = torch.from_numpy(states)
        # use first line for original dataset (original deltas unreliable, self-generate)
        deltas = states[1:, :] - states[:-1, :]
        # use second line for others
        # deltas = torch.from_numpy(deltas)
        deltas = deltas.type(torch.float64)

        # concat states and deltas
        data = torch.cat((states, actions), dim=1)
        data = data[:-1]

        # temparraything = [0, 2, 3, 4, 5, 6, 7, 8, 9, 1]
        # data = torch.index_select(data, 1, torch.LongTensor(temparraything))

        targets = deltas
        print(targets)
        # print(deltas.size())
        # print(data.size())

        data, targets = remove_outliers(data, targets)
        data, targets = z_score(data, targets)

        # DEPRECIATED CODE
        """
        numcols = data.size()[1]
        for column in range(0, numcols):
            data_max, data_min = get_col_minmax(data, column)
            data = normalize_data(data, column, data_max, data_min)

        numcols = targets.size()[1]
        for column in range(0, numcols):
            data_max, data_min = get_col_minmax(targets, column)
            delta_col_minmax.append((data_max, data_min))
            targets = normalize_data(targets, column, data_max, data_min)
        """

        self.data = []
        for i in range(0, len(data)):
            temp = [data[i], targets[i]]
            self.data.append(temp)

        # print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        return x, y


# neural network class
class testNet(nn.Module):
    def __init__(self, n_in, hidden_w, depth, n_out):
        super(testNet, self).__init__()
        self.n_in = n_in
        self.hidden_w = hidden_w
        self.depth = depth
        self.activation = nn.Softmax()
        self.n_out = n_out
        layers = []
        layers.append(('dynm_input_lin', nn.Linear(self.n_in, self.hidden_w)))
        layers.append(('dynm_input_act', self.activation))
        for d in range(self.depth):
            layers.append(('dynm_lin_' + str(d), nn.Linear(self.hidden_w, self.hidden_w)))
            layers.append(('dynm_act_' + str(d), self.activation))

        layers.append(('dynm_out_lin', nn.Linear(self.hidden_w, self.n_out)))
        self.features = nn.Sequential(OrderedDict([*layers]))

    def forward(self, x):
        x = self.features(x)
        return x

    def optimize(self, dataset):
        print("stuff")
        # TODO: Optimize Dataset


# custom collate function for dataset
def my_collate(batch):
    data = []
    target = []
    for item in batch:
        data_item = item[0].tolist()
        data.append(data_item)
        target_item = item[1].tolist()
        target.append(target_item)

    data = torch.FloatTensor(data)
    target = torch.FloatTensor(target)

    return data, target


# variables and things
epochs = 100
split = 0.8
bs = 16
lr = 0.04

dataset = simdata(root_dir='./data/simdata')
print("dataset length:")
print(len(dataset))
# print(math.floor(split*len(dataset)))
# print(math.ceil((1-split)*len(dataset)))
train_set, test_set = random_split(dataset, [math.floor(split*len(dataset)), math.ceil((1-split)*len(dataset))])
print("train_set length:")
print(len(train_set))
print("test_set length:")
print(len(test_set))

# train_set_len = int(split*len(dataset))
model = testNet(10, 50, 2, 6)
# print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

test_errors = []
train_errors = []

# load data and do things here
trainLoader = DataLoader(train_set, batch_size=bs, shuffle=True, collate_fn=my_collate)
testLoader = DataLoader(test_set, batch_size=bs, shuffle=True, collate_fn=my_collate)

for epoch in range(epochs):

    # testing and training loss
    train_error = 0
    test_error = 0

    for i, data in enumerate(trainLoader):

        inputs, targets = data
        inputs = Variable(inputs, requires_grad=True)
        targets = Variable(targets, requires_grad=True)

        optimizer.zero_grad()
        predict = model(inputs)
        loss = criterion(predict, targets)
        train_error += loss.item() / len(trainLoader)

        loss.backward()
        optimizer.step()

    for i, data in enumerate(testLoader):
        inputs, targets = data
        outputs = model(inputs)
        loss = criterion(outputs.float(), targets.float())
        test_error += loss.item() / (len(testLoader))

    # step lr scheduler
    scheduler.step(test_error)

    print(f"Epoch {epoch + 1}, Train loss: {train_error}, Test loss: {test_error}")
    train_errors.append(train_error)
    test_errors.append(test_error)

train_legend = mlines.Line2D([], [], color='blue', label='Train Loss')
test_legend = mlines.Line2D([], [], color='orange', label='Test Loss')
plt.plot(train_errors, label="train loss")
plt.plot(test_errors, color="orange", label="test loss")
plt.legend(handles=[train_legend, test_legend])
plt.savefig('dynamicslossgraph.png')
plt.close()

correct = 0
total = 0
diffpercent = 0.01
model.eval()  # prep model for testing

# validation set here
# TODO: REWRITE THIS
with torch.no_grad():
    for data, targets in testLoader:
        # print(data)
        outputs = model(data)
        numcols = targets.size()[1]
        outputs, targets = denormalize_data(outputs, targets)
        # print("targets")
        # print(targets)
        # print("outputs")
        # print(outputs)

        tgtcpy = targets.numpy()
        tgtlen = len(tgtcpy)
        total += tgtlen*6
        correct += compfunc(outputs, targets, diffpercent)

print('threshold is  %%%f' % ((diffpercent*100)))
print('Accuracy of the network on the test set: %d out of %d' % (correct, total))
print('Accuracy of the network on the test set: %d %%' % ((100 * correct / total)))

diffFig = plt.figure()
ax1 = diffFig.add_subplot()
ax1.hist(diffs, 20, (0, 100), rwidth=0.7)
ax1.set_xlabel("% difference between predicted and target value")
ax1.set_ylabel("number of occurrences")
plt.savefig('diffpercent.png')
plt.show()

# true error vs % error


# with torch.no_grad():
    # for data, targets in trainLoader:
        # print(data)
        # outputs = model(data)
        # print("targets")
        # print(targets)
        # print("outputs")
        # print(outputs)
        # tgtcpy = targets.numpy()
        # tgtlen = len(tgtcpy)
        # total += tgtlen
        # correct += compfunc(outputs, targets, diffpercent)
#
# print('threshold is  %%%f' % ((diffpercent*100)))
# print('Accuracy of the network on the train set: %d out of %d' % (correct, total))
# print('Accuracy of the network on the train set: %d %%' % ((100 * correct / total)))

# diffFig = plt.figure()
# ax1 = diffFig.add_subplot()
# ax1.hist(diffs, 20, (0, 100))
# ax1.set_xlabel("% difference between predicted and target value")
# ax1.set_ylabel("number of occurrences")
# plt.show()
