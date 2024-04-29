import torch
import torchvision
import torch.nn  as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torch import Tensor
import os
import glob
import shutil
import random
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import statistics

class TestModels():

    def test_cae(net, test_loader, criterion):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        test_result = []
        diff = []
        original = []
        test_loss = 0
        loss_lst = []
        label_lst = []
        net.eval()
        for counter, (images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, images)
            test_loss += loss
            label_lst.extend(labels)
            loss_lst.extend(loss)
            original.extend(images.to('cpu'))
            test_result.extend(outputs.to('cpu'))
            diff.extend(abs(images - outputs).to('cpu'))
        avg_test_loss = test_loss / counter
        print("test_loss: {0:.4f}".format(avg_test_loss))
        return label_lst, loss_lst, original, diff, test_result

    def test_cnn(net, test_loader, criterion):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        test_loss = 0.0
        test_acc = 0.0
        outputs_lst = []
        labels_lst = []
        net.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() / len(test_loader.dataset)
                test_acc += (outputs.max(1)[1] == labels).sum() / len(test_loader.dataset)
                outputs_lst += list(outputs.argmax(1).to('cpu').detach().numpy().copy())
                labels_lst += list(labels.to('cpu').detach().numpy().copy())
            arv_test_loss = test_loss / len(test_loader.dataset)
            arv_test_acc = test_acc/ len(test_loader.dataset)
            print("test_loss: {0:4f}, test_acc: {1:4f}".format(arv_test_loss, arv_test_acc))
        return labels_lst, outputs_lst

    def test_mc_drop(net, test_loader):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        label_lst = []
        predict_lst = []
        data_uncertainty_lst = []
        model_uncertainty_lst = []
        softmax = nn.Softmax(dim=1)
        net.eval()
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            class0_lst = []
            class1_lst = []
            class2_lst = []
            for i in range(100):
                output = softmax(net(images))[0]
                class0_lst.append(float(output[0]))
                class1_lst.append(float(output[1]))
                class2_lst.append(float(output[2]))
            mean_lst = [
                statistics.mean(class0_lst),
                statistics.mean(class1_lst),
                statistics.mean(class2_lst)
            ]
            max_mean = max(mean_lst)
            predict_lst.append(mean_lst.index(max_mean))
            data_uncertainty_lst.append(max_mean)
            var_lst = [
                statistics.variance(class0_lst),
                statistics.variance(class1_lst),
                statistics.variance(class2_lst)
            ]
            model_uncertainty_lst.append(sum(var_lst))
            label_lst.append(int(labels.to('cpu').detach().numpy().copy()))
        return label_lst, predict_lst, data_uncertainty_lst, model_uncertainty_lst

class TrainModels():

    def train_cae(net, num_epoch, train_loader, vali_loader, optimizer, criterion, model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_loss_list = []
        val_loss_list = []
        best_loss = 10000
        for epoch in range(num_epoch):
            train_loss = 0
            val_loss = 0
            net.train()
            print("Epoch [{}/{}]".format(epoch+1, num_epoch))
            print("~training~")
            for counter, (images, _) in enumerate(tqdm(train_loader)):
                images = images.to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / counter
            print("~validating~")
            net.eval()
            with torch.no_grad():
                for counter, (images, _) in enumerate(tqdm(vali_loader)):
                    images = images.to(device)
                    outputs = net(images)
                    loss = criterion(outputs, images)
                    val_loss += loss.item()
                avg_val_loss = val_loss / counter
            print("train_loss: {0:.4f}, val_loss: {1:.4f}".format(avg_train_loss, avg_val_loss))
            train_loss_list.append(avg_train_loss)
            val_loss_list.append(avg_val_loss)
            if (avg_val_loss < best_loss) & (epoch > 80):
                best_loss = avg_val_loss
                best_model_wts = copy.deepcopy(net.state_dict())
                net.load_state_dict(best_model_wts)
                torch.save(net.state_dict(), model_path)
        return net, train_loss_list, val_loss_list


    def train_cnn(net, num_epoch, train_loader, vali_loader, optimizer, criterion, model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        best_loss = 10000000
        for epoch in range(num_epoch):
            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0
            net.train()
            for _, (images, labels) in enumerate(tqdm(train_loader)):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                train_loss += loss.item()
                train_acc += (outputs.max(1)[1] == labels).sum().item()
                loss.backward()
                optimizer.step()
            avg_train_loss = train_loss / len(train_loader.dataset)
            avg_train_acc = train_acc / len(train_loader.dataset)
            net.eval()
            with torch.no_grad():
                for images, labels in vali_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc += (outputs.max(1)[1] == labels).sum().item()
                avg_val_loss = val_loss / len(vali_loader.dataset)
                avg_val_acc = val_acc / len(vali_loader.dataset)
            print("Epoch [{}/{}], train_loss: {:4f}, val_loss: {:4f}, train_acc: {:4f}, val_acc: {:4f}".format(
                epoch+1, num_epoch, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc
            ))
            train_loss_list.append(avg_train_loss)
            train_acc_list.append(avg_train_acc)
            val_loss_list.append(avg_val_loss)
            val_acc_list.append(avg_val_acc)
            if (avg_val_loss < best_loss) & (epoch > 10):
                best_loss = avg_val_loss
                best_model_wts = copy.deepcopy(net.state_dict())
                net.load_state_dict(best_model_wts)
                torch.save(net.state_dict(), model_path)
        return net, train_loss_list, train_acc_list, val_loss_list, val_acc_list
