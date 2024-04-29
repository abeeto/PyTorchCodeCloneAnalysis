import sys
import numpy as np
# import chainer
# from chainer import cuda
# import chainer.functions as F
import time
from torch.optim.lr_scheduler import StepLR
import utils



# ---------------


import torch.optim as optim
import torch
import time
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler


class Trainer:
    def __init__(self, model, optimizer, trainloader, valloader, opt):
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader
        self.opt = opt
        self.lr_scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)
        # self.n_batches = (len(train_iter.dataset) - 1) // opt.batchSize + 1
        self.start_time = time.time() 

    def train(self, epoch):

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        train_loss = 0
        train_acc = 0
        count = 0
        for i, data in enumerate(self.trainloader, 0):
            inputs, labels = data
            inputs = inputs.float()
            inputs = inputs.cuda()
            labels = labels.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            if self.opt.BC:
                loss = utils.kl_divergence(labels, outputs)
                acc = torch.eq(labels, np.argmax(outputs, axis=1)).float().mean()
            else:
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                acc = torch.eq(preds, labels).float().mean()
            count += len(labels)
            loss.backward()
            self.optimizer.step()
            train_loss += float(loss.data) * len(labels)
            train_acc += float(acc.data) * len(labels)
        
        self.lr_scheduler.step()
        train_loss /= count
        train_top1 = 100 * (1 - train_acc / count)
        
        return train_loss, train_top1

    def val(self):
        count = 0
        self.model.eval()
        val_acc = 0
        for i, data in enumerate(self.valloader, 0):
            inputs, labels = data
            count += len(labels)
            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.cuda()
            labels = labels.cuda()
            if self.opt.nCrops > 1:
                inputs = torch.reshape(inputs,(inputs.size(0)* self.opt.nCrops, 1, 1, inputs.size(3)))
            outputs = self.model(inputs)
            outputs = outputs.reshape((outputs.shape[0] // self.opt.nCrops, self.opt.nCrops, outputs.shape[1]))
            outputs = torch.mean(outputs, axis=1)
#             print(outputs.shape)
            _, preds = torch.max(outputs, 1)
#             print(preds.shape)
        
            
            preds = preds.float()
#             preds = torch.mean(preds, axis=1)
            
            acc = torch.eq(preds, labels).float().mean()
            val_acc += float(acc.data) * len(labels)

        # self.val_iter.reset()
        # self.model.train = True
#         print(len(self.valloader))
        val_top1 = 100 * (1 - val_acc / count)
        
        return val_top1






# def train(self, epoch):
#         self.optimizer.lr = self.lr_schedule(epoch)
#         # train_loss = 0
#         # train_acc = 0
#         # for i, batch in enumerate(self.train_iter):
#         #     x_array, t_array = chainer.dataset.concat_examples(batch)
#         #     x = chainer.Variable(cuda.to_gpu(x_array[:, None, None, :]))
#         #     t = chainer.Variable(cuda.to_gpu(t_array))
#         #     y = self.model(x)
#         #     if self.opt.BC:
#         #         loss = utils.kl_divergence(y, t)
#         #         acc = F.accuracy(y, F.argmax(t, axis=1))
#         #     else:
#         #         loss = F.softmax_cross_entropy(y, t)
#         #         acc = F.accuracy(y, t)

#         #     loss.backward()
#         #     self.optimizer.update()
#         #     train_loss += float(loss.data) * len(t.data)
#         #     train_acc += float(acc.data) * len(t.data)

#             elapsed_time = time.time() - self.start_time
#             progress = (self.n_batches * (epoch - 1) + i + 1) * 1.0 / (self.n_batches * self.opt.nEpochs)
#             eta = elapsed_time / progress - elapsed_time

#             line = '* Epoch: {}/{} ({}/{}) | Train: LR {} | Time: {} (ETA: {})'.format(
#                 epoch, self.opt.nEpochs, i + 1, self.n_batches,
#                 self.optimizer.lr, utils.to_hms(elapsed_time), utils.to_hms(eta))
#             sys.stderr.write('\r\033[K' + line)
#             sys.stderr.flush()

#         self.train_iter.reset()
#         train_loss /= len(self.train_iter.dataset)
#         train_top1 = 100 * (1 - train_acc / len(self.train_iter.dataset))

#         return train_loss, train_top1


#     def lr_schedule(self, epoch):
#         divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule])
#         decay = sum(epoch > divide_epoch)
#         if epoch <= self.opt.warmup:
#             decay = 1

#         return self.opt.LR * np.power(0.1, decay)



