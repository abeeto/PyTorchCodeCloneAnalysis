import pickle
import pandas as pd
import os
from matplotlib import pyplot as plt
from glob import glob
import torch
import os
from prepare_datasets import get_raw_data
from torch_cnn_model import  Config, torch_model, EarlyStopping, My_Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, recall_score
from torch.utils.data import DataLoader


def cross_inference(experiments_path, x, y):
    net = torch.load(os.path.join(experiments_path, 'net.pb'))
    batcher_test = DataLoader(My_Dataset(x, y), batch_size=config.batch_size, shuffle=True)

    loss = 0.0
    correct = 0
    iterations = 0
    f_scores = 0
    unweighted_accuracy_scores = 0
    weighted_accuracy_scores = 0


    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    torch.manual_seed(7)
    h = net.init_hidden(config.batch_size)

    for i, (items, classes) in enumerate(batcher_test):
        if classes.shape[0] != config.batch_size:
            break
        items = items.to('cuda')
        classes = classes.to('cuda')
        h = tuple([each.data for each in h])

        outputs, h = net(items, h)
        loss += criterion(outputs, classes.long()).item()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == classes.data.long()).sum()

        predicted_cpu, classes_cpu = predicted.cpu().numpy(), classes.data.cpu().numpy()
        f_scores += f1_score(predicted_cpu, classes_cpu, average='macro')
        unweighted_accuracy_scores += accuracy_score(predicted_cpu, classes_cpu)
        weighted_accuracy_scores += accuracy_score(predicted_cpu, classes_cpu, )
        iterations += 1

    return loss / iterations, f_scores / iterations



config = Config(lr=0.0001, batch_size=128, n_classes=4, num_epochs=300)
list_of_bases = [r'C:\Users\kotov-d\Documents\BASES\friends',
          r'C:\Users\kotov-d\Documents\BASES\RAMAS\ramas',
          r'C:\Users\kotov-d\Documents\BASES\iemocap_last']

for p in [r'C:\Users\kotov-d\Documents\BASES\iemocap_last']:
    for ip in list_of_bases:

        experiments_path = os.path.join(r'C:\Users\kotov-d\Documents\TASKS\cross_inference',
                                        os.path.basename(ip))
        [_, _, x_test, _, _, y_test] = get_raw_data(ip, experiments_path, 4)

        experiments_path = os.path.join(r'C:\Users\kotov-d\Documents\TASKS\cross_inference',
                                        os.path.basename(p))

        loss, f_score = cross_inference(experiments_path, x_test, y_test)
        print('модель обученная на {}, инференс на {}: loss {}, f-score {}'.format(os.path.basename(p),
                                                                                   os.path.basename(ip),
                                                                                   loss,
                                                                                   f_score))