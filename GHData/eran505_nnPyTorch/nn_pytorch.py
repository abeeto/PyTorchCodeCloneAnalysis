import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import normalize
from sklearn import preprocessing
import pandas as pd
import torchvision.datasets as dsets
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from os.path import expanduser
import preprocessor as pr
import helper as hlp
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
from sklearn.datasets import make_regression, make_classification
from my_losses import XTanhLoss, LogCoshLoss, XSigmoidLoss
from fast_data_loader import FastTensorDataLoader
import matplotlib.pyplot as plt

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')


def n_normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def normalize_d(d, target=1.0):
    raw = sum(set(d.values()))
    factor = target / raw
    return {key: value * factor for key, value in d.items()}


class LR(nn.Module):
    def __init__(self, dim, out=27, hidden=400, sec_hidden=400, a=-1.0, b=1.0):
        super(LR, self).__init__()
        # intialize parameters

        self.classifier = nn.Sequential(

            self.make_linear(dim, hidden, a, b),
            #nn.BatchNorm1d(hidden),  # applying batch norm
            nn.ReLU(),

            self.make_linear(hidden, hidden, a, b),
            # nn.BatchNorm1d(hidden),  # applying batch norm
            nn.ReLU(),

            self.make_linear(hidden, sec_hidden, a, b),
            # nn.BatchNorm1d(sec_hidden),  # applying batch norm
            nn.ReLU(), #ReLU

            self.make_linear(sec_hidden, out, a, b)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
        # return x.squeeze()

    def get_weights(self):
        return self.linear1.weight, self.linear2.weight

    def print_weights(self):
        print("linear1_weight=\n{}".format(self.linear1.weight))
        print("linear2_weight=\n{}".format(self.linear2.weight))

    def make_linear(self, in_input, out_output, a_w, b_w):
        layer = torch.nn.Linear(in_input, out_output)
        #torch.nn.init.uniform_(layer.weight, a=a_w, b=b_w)
        # torch.nn.init.uniform_(layer.bias, a=a_w, b=b_w)
        return layer


class NeuralNetwork(object):

    def __init__(self, loss_func, model, optimizer_object):
        self.loss_function = loss_func
        self.nn_model = model
        self.optimizer = optimizer_object
        self.scheduler = optim.lr_scheduler.StepLR(optimizer_object, step_size=100, gamma=0.1)
        self.losses_train = []
        self.losses_test = []
        self.home = None
        self.get_home()
        self.ctr = 0
        self.debug_D = {}

    def get_home(self):
        str_home = expanduser("~")
        if str_home.__contains__('lab2'):
            str_home = "/home/lab2/eranher"
        self.home = str_home

    def learn_Q_value(self, x, y):
        # Sets model to TRAIN mode
        losses = np.zeros(y.shape[1])
        for i in range(y.shape[1]):
            self.ctr += 1
            action_tensor = torch.empty(y.shape[0], 1).fill_(i)

            new_y = y[:, i]

            new_x = torch.cat([x, action_tensor], dim=1)

            new_x = new_x.to(device)
            new_y = new_y.to(device)

            yhat = self.nn_model(new_x)

            self.optimizer.zero_grad()
            # print(yhat)
            # print(new_y)
            loss = self.loss_function(yhat, new_y)

            loss.backward()

            self.optimizer.step()

            losses[i] = loss.item()
        # Returns the loss
        return np.max(losses)

    def train_step(self, x, y):
        # Sets model to TRAIN mode
        x = x.to(device)
        y = y.to(device)
        self.ctr += 1
        # Makes predictions
        # print("--Step--")
        # check if req grad = T and where device
        # print(x.requires_grad)
        yhat = self.nn_model(x)

        # for param in self.nn_model.parameters():
        #     print(type(param.data), param.size(),list(param))

        self.optimizer.zero_grad()
        # Computes loss
        #print("y={0} \nyhat={1}\n".format(y.tolist(),yhat.tolist()))

        loss = self.loss_function(yhat, y)

        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        self.optimizer.step()

        # Returns the loss
        return loss.item()

    def fit_model(self, n_epochs, train_dataset, validtion_datatest=None):
        # For each epoch...
        ctr = 0
        l_time = []
        data_loader_time = []
        loss_tmp = []
        sampels_size_batch = len(train_dataset)
        for epoch in range(n_epochs):
            # Performs one train step and returns the corresponding loss
            training_loader_iter = iter(train_dataset)
            ctr = 0
            for x_train_tensor, y_train_tensor in training_loader_iter:
                # print(x_train_tensor[:,0])
                # print("--"*10)
                self.nn_model.train()
                t = time.process_time()

                # x_train_tensor, y_train_tensor = next(training_loader_iter)
                data_loader_time.append(time.process_time() - t)

                # print(Counter(y_batch.tolist()))
                # print("x_batch={} \t y_batch={}".format(x_batch.requires_grad,y_batch.requires_grad))

                # for auto computing the auto grad
                t = time.process_time()
                losser = []
                # loss = self.learn_Q_value(x_train_tensor,y_train_tensor)
                loss = self.train_step(x_train_tensor, y_train_tensor)

                l_time.append(time.process_time() - t)

                self.losses_train.append([loss, epoch])
                # if ctr % 10000 == 0:
                #     test_loader_iter = iter(validtion_datatest)
                #     self.eval_nn(test_loader_iter)

                # decay the learning rate
                loss_tmp.append(loss)
                if ctr % 10000 == 0:
                    print('Training loss: {2} Iter-{3} Epoch-{0} lr: {1}  Avg-Time:{4} DataLoader(time):{5} '.format(
                        epoch, self.optimizer.param_groups[0]['lr'], np.mean(loss_tmp), ctr / sampels_size_batch,
                        np.mean(l_time), np.mean(data_loader_time)))
                    l_time.clear()
                    data_loader_time.clear()
                    loss_tmp.clear()
                    # print(100 * "-")
                    # print(list(self.nn_model.parameters()))
                    self.eval_nn(validtion_datatest)
                # self.log_to_files()
                # torch.save(self.nn_model.state_dict(), "{}/car_model/nn/nn{}.pt".format(self.home, epoch + int(ctr/1000)))

                ctr = ctr + 1

            self.log_to_files()
            torch.save(self.nn_model.state_dict(), "{}/car_model/nn/nn{}.pt".format(self.home, epoch))
            self.scheduler.step()

    def eval_nn(self, validtion_datatest):
        self.losses_test = []
        with torch.no_grad():
            for x_val, y_val in validtion_datatest:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                self.nn_model.eval()

                yhat = self.nn_model(x_val)
                # print("X:{}\t\tY^:{}\t\tY:{}".format(x_val.tolist(),yhat.tolist(),y_val.tolist()))
                val_loss = self.loss_function(y_val, yhat)
                self.losses_test.append(val_loss.item())
                break
            print("test loss= avg:{}  max:{}".format(sum(self.losses_test) / len(self.losses_test),
                                                     max(self.losses_test)))

    def log_to_files(self):
        hlp.log_file(self.losses_train, "{}/car_model/nn/loss_train.csv".format(self.home), ["loss", "epoch"])
        # hlp.log_file(self.losses_test, "{}/car_model/nn/loss_test.csv".format(self.home), ["loss"])

    def log_dict_debug(self, ep):
        print("debug_d = ", len(self.debug_D))
        with open('/home/ERANHER/car_model/generalization/test.csv', 'w') as f:
            for key in self.debug_D.keys():
                f.write("%s,%s\n" % (key, self.debug_D[key] / ep))


class DataSet(object):

    def __init__(self, data, targets, W=[]):
        print("data shape:{}\ntarget shape:{}\n".format(data.shape, targets.shape))
        self.data = data
        self.targets = targets
        self.weights = W
        self.debug_d = None
        self.data = self.norm_without_negative(self.data)
        self.data = np.nan_to_num(self.data) # for nan -> 0
        print(len(self.targets))
        self.targets = self.scale_negtive_one_to_one(self.targets)
        print(len(self.targets))
        print("done")
        if len(W) == 0:
            self.imbalanced_data_set_weight()
        else:
            # self.weights = self.min_max_zero_to_one(self.weights)
            print()
            # print(Counter(self.weights))
            # exit(0)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)

    def scale_negtive_one_to_one(self, foo):
        return preprocessing.maxabs_scale(foo)

    def min_max_zero_to_one(self, foo):
        return preprocessing.minmax_scale(foo)

    def norm(self):
        self.data = normalize(self.data, axis=0, norm='l1')

    def to_bin(self, occurrence_matrix, bins=None):
        d_bin = {}
        d_bin_occ = {}
        ctr = 0
        elements_array = occurrence_matrix[:, 0]
        sum_all = elements_array.sum()
        if bins is None:
            bins = np.arange(np.min(elements_array), np.max(elements_array), 0.1)
        for i in range(1, 100):
            z = elements_array[np.digitize(elements_array, bins) == i]
            if ctr == len(elements_array):
                break
            if len(z) == 0:
                continue
            for e in z:
                d_bin[e] = {"bin": i, "occ": occurrence_matrix[ctr, 1], 'w': 0}
                if i not in d_bin_occ:
                    d_bin_occ[i] = 0
                d_bin_occ[i] += occurrence_matrix[ctr, 1]
                ctr = ctr + 1
        dixt_W = {}
        for ky in d_bin:
            d_bin[ky]['w'] = sum_all / d_bin_occ[d_bin[ky]['bin']]
            dixt_W[ky] = d_bin[ky]['w']
        print(d_bin_occ)
        print("d_bin_occ")
        dixt_W[1.0] = dixt_W[1.0]
        return normalize_d(dixt_W)

    def imbalanced_data_set_weight(self, bins=None):
        # if bins is None:
        #     bins = [0, 0.001, 1.1]
        labels = self.targets
        labels_avg = labels.mean(1)
        print(labels_avg.shape)
        print("labels {}".format(len(labels)))
        unique, counts = np.unique(labels.mean(1), return_counts=True)
        unique_arr = np.asarray((unique, counts)).T
        # W_bin = self.to_bin(unique_arr, bins)
        sum = unique_arr[:, 1].sum()
        W = sum / unique_arr[:, 1]
        W = n_normalize(W)
        pairs = zip(unique_arr[:, 0], W)
        d_all_w = {i[0]: i[1] for i in pairs}
        a = np.vectorize(d_all_w.get)(labels_avg)  # or W_bin --> d_all_w
        self.weights = a
        # self.debug_d = result

    def norm_without_negative(self, table_data):
        print("####" * 50)
        min_arr = table_data.min(0)
        ptp_arr = table_data.ptp(0)
        np.array(min_arr).tofile("{}/min.csv".format(folder_dir),sep=',')
        np.array(ptp_arr).tofile("{}/ptp.csv".format(folder_dir),sep=',')
        print(list(min_arr))
        print(list(ptp_arr))

        print("####" * 50)
        return (table_data - min_arr) / ptp_arr

    def split_test_train(self, raito=0.001):
        self.weights = np.ones(self.data.shape[0])
        X_train, X_test, y_train, y_test, w_train, w_test = \
            train_test_split(self.data, self.targets, self.weights, test_size=raito, random_state=0)
        loader_train = DataSet.make_DataSet(X_train, y_train, size_batch=batch_size, is_shuffle=False,
                                            samples_weights=w_train)
        # l = torch.multinomial(torch.tensor(w_test),len(w_test),False).tolist()

        loader_test = DataSet.make_DataSet(X_test, y_test, size_batch=16, samples_weights=w_test)

        return loader_train, loader_test

    @staticmethod
    def make_DataSet(X_data, y_data, size_batch=1, is_shuffle=False, samples_weights=None
                     , pin_memo=False, over_sample=False):
        print(Counter(samples_weights))
        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=False)
        print("-----------batch size = {}".format(size_batch))
        if device.type != 'cpu':
            pin_memo = True

        tensor_x = torch.tensor(X_data, requires_grad=False, dtype=torch.float64)
        tensor_y = torch.tensor(y_data, dtype=torch.float64).contiguous()
        my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
        if over_sample:
            return DataLoader(my_dataset, shuffle=is_shuffle, batch_size=size_batch, num_workers=0
                              , sampler=sampler, pin_memory=pin_memo)  # ),)  # create your dataloader
        else:
            return DataLoader(my_dataset, shuffle=is_shuffle, batch_size=size_batch, num_workers=0)


def main(in_dim, train_dataset, test_dataset=None):
    print(device)
    SEED = 2809
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    ## hyperparams
    num_iterations = 900
    lrmodel = LR(in_dim).double()
    lrmodel = lrmodel.to(device)

    # loss = nn.NLLLoss
    # loss = nn.functional.kl_div
    # loss= nn.KLDivLoss()
    # loss = XSigmoidLoss()
    # loss=F.

    loss = nn.BCEWithLogitsLoss()  # pos_weight=torch.from_numpy(positive_class_pr))
    #loss = nn.CrossEntropyLoss()
    # SGD/Adam
    optimizer = torch.optim.SGD(lrmodel.parameters(), lr=0.0955,momentum=0.9) #0.0955

    my_nn = NeuralNetwork(loss_func=loss,
                          optimizer_object=optimizer,
                          model=lrmodel)

    # hlp.get_the_best_lr(lrmodel,loss,train_dataset)
    # exit()
    my_nn.fit_model(num_iterations, train_dataset, test_dataset)


def test_main(path_to_model):
    df = pd.read_csv("/home/eranhe/car_model/generalization/4data/nn_DATA/all.csv")
    matrix_df = df.to_numpy()  # [756251:756254]
    print(len(matrix_df[:, :-27]))
    print(matrix_df)
    obj = DataSet(matrix_df[:, :-27], matrix_df[:, -27:])
    test_loader = DataSet.make_DataSet(obj.data, obj.targets, size_batch=1)
    in_p = matrix_df.shape[-1] - 27
    my_model = LR(in_p).double()

    my_model.load_state_dict(torch.load(path_to_model, map_location=device))
    my_model.cpu()
    # self.nn = self.nn.double()
    my_model.eval()
    sum = 0
    ctr = 0
    with torch.no_grad():
        for x_val, y_val in iter(test_loader):
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            my_model.eval()
            yhat = my_model(x_val)
            k = 0
            for i, j in list(zip(yhat, y_val.squeeze())):
                print("{}| Y^:{} | Y:{}".format(k, i, j))
                k += 1

            sum += F.smooth_l1_loss(y_val.squeeze(), yhat).item()
            if ctr % 1000 == 0:
                print("{}".format(ctr))
            ctr += 1
            print("---- losses --------")
            print("MAE: ", F.l1_loss(y_val.squeeze(), yhat).item())
            print("MSE: ", F.mse_loss(y_val.squeeze(), yhat).item())
            exit()
    print("SUM -> ", sum)
    exit()


batch_size = 2

# 756253:756251 index

folder_dir=None


if __name__ == "__main__":
    np.random.seed(3)
    str_home = expanduser("~")
    if str_home.__contains__('lab2'):
        str_home = "/home/lab2/eranher"

    data_file = "all.csv"
    folder='new/1'
    #folder="new/small/600"
    folder_dir = "{}/car_model/generalization/new/4".format(str_home,folder)
    p_path_data = "{}/{}".format(folder_dir, data_file)
    df = pd.read_csv(p_path_data)

    colz = list(df)

    # take only the relevant
    #print(len(df))
    df = df.loc[df[colz[-1]] >= 1]
    #print(len(df))
    #exit()
    # make multi one hot encoding
    s = len(df)
    df = pr.only_max_value(df,first=True)
    print(len(df),":",s)
    z = df[colz[-2]].value_counts()
    false_count = len(df[colz[-28:-1]]) / df[colz[-28:-1]].sum()
    positive_class_pr = false_count.values

    print(np.sum(z.values))

    print(len(df))

    #df.to_csv("{}/car_model/generalization/{}/cut.csv".format(str_home, folder))

    # ax = df[df.columns[-3]].hist()
    # plt.show()

    matrix_df = df.to_numpy()
    print(matrix_df.shape)

    DataLoder = DataSet(matrix_df[:, :-28], matrix_df[:, -28:-1], matrix_df[:, -1])
    train_loader, test_loader = DataLoder.split_test_train(0.13)

    # df = pd.read_csv("{}/car_model/generalization/{}/all.csv".format(str_home,folder))
    # add index
    # df.insert(0, 'idz', range(1, len(df) + 1))

    # df = pr.only_max_value(df)
    # matrix_df = df.to_numpy()
    # DataLoder = DataSet(matrix_df[:, :-28], matrix_df[:, -28:-1],matrix_df[:,-1])
    # _, test_loader = DataLoder.split_test_train(0.1)

    print("len - train_loader:", len(train_loader))
    print("len - test:", len(test_loader))

    main(matrix_df.shape[-1] - 28, train_loader, test_loader)
