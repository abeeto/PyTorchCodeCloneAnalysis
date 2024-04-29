# Training
import numpy as np
import torch
from torch.autograd import Variable

import utils


class ModelTraining():
    def __init__(self, use_cuda=False):
        # self.model = model
        # self.optimizer = optimizer
        # self.criterion = criterion
        # self.scheduler = scheduler
        self.use_cuda = use_cuda
        global PublicTest_acc
        global best_PublicTest_acc
        global best_PublicTest_acc_epoch
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []

    def train(self, epoch, trainloader, model, optimizer, criterion):
        print('\nEpoch: %d' % epoch)
        global Train_acc
        model.train()
        train_loss = 0
        epoch_loss = 0
        correct = 0
        total = 0

        print('learning_rate: %s' % str(utils.get_lr(optimizer)))

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            utils.clip_gradient(optimizer, 0.1)
            optimizer.step()
            train_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            epoch_loss = train_loss / (batch_idx + 1)
            utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        self.train_losses.append(epoch_loss)
        print('loss: ', self.train_losses)
        Train_acc = 100. * correct / total
        self.train_acc.append(Train_acc)

    def PublicTest(self, epoch, PublicTestloader, model, criterion):
        self.best_PublicTest_acc = 0
        self.best_PublicTest_acc_epoch = 0
        model.eval()
        PublicTest_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(PublicTestloader):

            bs, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            with torch.no_grad():
                outputs = model(inputs)
            outputs_avg = outputs.view(bs, 1, -1).mean(1)  # avg over crops
            loss = criterion(outputs_avg, targets)
            PublicTest_loss += loss.data
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        PublicTest_acc = 100. * correct / total
        if PublicTest_acc > self.best_PublicTest_acc:
            print('Saving..')
            print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
            state = {
                'net': model.state_dict() if self.use_cuda else model,
                'acc': PublicTest_acc,
                'epoch': epoch,
            }
            # if not os.path.isdir(path):
            #     os.mkdir(path)
            # torch.save(state, os.path.join(path,'PublicTest_model.t7'))
            self.best_PublicTest_acc = PublicTest_acc
            self.best_PublicTest_acc_epoch = epoch

    def PrivateTest(self, epoch, PrivateTestloader, model, criterion):
        global PrivateTest_acc
        best_PrivateTest_acc = 0
        best_PrivateTest_acc_epoch = 0
        model.eval()
        PrivateTest_loss = 0
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
            # bs, ncrops, c, h, w = np.shape(inputs)
            bs, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            with torch.no_grad():
                outputs = model(inputs)
            outputs_avg = outputs.view(bs, 1, -1).mean(1)  # avg over crops
            loss = criterion(outputs_avg, targets)
            PrivateTest_loss += loss.data
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            epoch_loss = PrivateTest_loss / (batch_idx + 1)
            utils.progress_bar(batch_idx, len(PrivateTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (epoch_loss, 100. * correct / total, correct, total))
        # Save checkpoint.
        PrivateTest_acc = 100. * correct / total
        self.val_losses.append(epoch_loss)
        self.val_acc.append(PrivateTest_acc)

        if PrivateTest_acc > best_PrivateTest_acc:
            print('Saving..')
            print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
            # state = {
            #     'net': self.model.state_dict() if self.use_cuda else self.model,
            #     'best_PublicTest_acc': self.best_PublicTest_acc,
            #     'best_PrivateTest_acc': PrivateTest_acc,
            #     'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            #     'best_PrivateTest_acc_epoch': epoch,
            # }
            # if not os.path.isdir(path):
            #     os.mkdir(path)
            # torch.save(state, os.path.join(path,'PrivateTest_model.t7'))
            best_PrivateTest_acc = PrivateTest_acc
            best_PrivateTest_acc_epoch = epoch

    def schedulerStep(self, scheduler):
        scheduler.step(self.val_losses[-1])