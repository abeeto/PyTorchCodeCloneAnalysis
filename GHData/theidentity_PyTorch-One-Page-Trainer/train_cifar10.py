import os
import torch
from torchvision import datasets
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
import resnet
import shake_shake
import numpy as np
import pickle


# https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones([h, w], dtype=torch.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = mask.expand(img.shape)
        img = img * mask

        return img


class Average_Meter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val.item()
        self.count += n

    def avg(self):
        return self.sum / (self.count + 1e-9)


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


class EMA_Averager():
    def __init__(self, model, ema_decay=.999):
        self.model = model
        self.ema_decay = ema_decay
        self.ema_model = self.get_ema_model()

    def step(self):
        alpha = self.ema_decay
        one_minus_alpha = 1 - alpha
        for src, dst in zip(self.model.parameters(), self.ema_model.parameters()):
            dst.mul_(alpha)
            dst.add_(one_minus_alpha*src.data)

    def update_bn(self):
        tmp_model = pickle.loads(pickle.dumps(self.ema_model))
        self.ema_model.load_state_dict(self.model.state_dict())
        for src, dst in zip(tmp_model.parameters(), self.ema_model.parameters()):
            dst[:] = src[:]

    def get_ema_model(self):
        ema_model = pickle.loads(pickle.dumps(self.model))
        ema_model.load_state_dict(self.model.state_dict())
        for src in ema_model.parameters():
            src.detach_()
        return ema_model


class Classifier():
    def __init__(self):
        self.num_classes = 10
        self.input_size = (3, 32, 32)
        self.name = 'res18_cutout'
        self.model_save_path = 'models/%s/%s_best_acc.pth' % (self.name, self.name)
        self.best_acc = 0
        self.global_step = 0
        self.progress_bar = None
        create_folder('models/%s/' % (self.name))

    def get_dataloaders(self):
        mean, std = [0.4914, 0.4824, 0.4467], [0.1965, 0.1984, 0.1992]
        eval_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_tfms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout(n_holes=1, length=16)
        ])
        train_dataset = datasets.CIFAR10(root='data/', train=True, transform=train_tfms, download=False)
        test_dataset = datasets.CIFAR10(root='data/', train=False, transform=eval_tfms, download=False)
        train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
        test_dl = DataLoader(test_dataset, batch_size=1024, shuffle=False, pin_memory=True, num_workers=4)
        return train_dl, test_dl

    def get_model(self, resume=False):
        criterion = nn.CrossEntropyLoss()
        model = resnet.ResNet18(num_classes=self.num_classes)
        # model = shake_shake.ShakeResNet(depth=26, w_base=32, label=self.num_classes)
        # model = shake_shake.ShakeResNet(depth=26, w_base=64, label=self.num_classes)
        model = nn.DataParallel(model).cuda()
        opt = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, nesterov=True, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [60, 120, 160], gamma=.2)
        epoch = self.global_step = self.best_acc = 0
        ema_opt = EMA_Averager(model, ema_decay=.999)
        ema_model = ema_opt.ema_model

        if resume:
            state = torch.load(self.model_save_path)
            model.load_state_dict(state['model_state_dict'])
            opt.load_state_dict(state['opt_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])
            ema_model.load_state_dict(state['ema_model_state_dict'])
            epoch, self.global_step, self.best_acc = state['epoch'], state['global_step'], state['best_acc']

        return epoch, model, opt, scheduler, criterion, ema_model, ema_opt

    def train_epoch(self, model, opt, scheduler, criterion, dataloader, ema_model, ema_opt):
        model.train()
        train_loss, train_acc = Average_Meter(), Average_Meter()
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            op = model(x)
            opt.zero_grad()
            loss = criterion(op, y)
            loss.backward()
            opt.step()
            ema_opt.step()
            pred = torch.argmax(op, dim=1)
            num_correct = torch.sum(pred == y)
            train_loss.update(loss, n=len(y))
            train_acc.update(num_correct, n=len(y))
            self.global_step += 1
            self.progress_bar.set_postfix(Train_Xe='%.6f' % train_loss.avg(), Train_Acc='%.4f' % train_acc.avg())
        scheduler.step()
        ema_opt.update_bn()
        return train_loss.avg(), train_acc.avg()

    def test(self, model, criterion, dataloader):
        model.eval()
        test_loss, test_acc = Average_Meter(), Average_Meter()
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.cuda(), y.cuda()
                op = model(x)
                loss = criterion(op, y)
                pred = torch.argmax(op, dim=1)
                num_correct = torch.sum(pred == y)
                test_loss.update(loss, n=len(y))
                test_acc.update(num_correct, n=len(y))
            return test_loss.avg(), test_acc.avg()

    def train(self, num_epochs, resume=False):
        start_epoch, model, opt, scheduler, criterion, ema_model, ema_opt = self.get_model(resume)
        train_dl, test_dl = self.get_dataloaders()

        self.progress_bar = tqdm(range(start_epoch, num_epochs), total=num_epochs, initial=start_epoch)
        for epoch in self.progress_bar:
            self.progress_bar.set_description('[%d/%d]' % (epoch, num_epochs))

            train_loss, train_acc = self.train_epoch(model, opt, scheduler, criterion, train_dl, ema_model, ema_opt)
            test_loss, test_acc = self.test(model, criterion, test_dl)
            ema_test_loss, ema_test_acc = self.test(ema_model, criterion, test_dl)

            test_acc = max(test_acc, ema_test_acc)
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                state = {'model_state_dict': model.state_dict(), 'ema_model_state_dict': ema_model.state_dict(), 'opt_state_dict': opt.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'epoch': epoch, 'global_step': self.global_step, 'best_acc': self.best_acc}
                torch.save(state, self.model_save_path)

            mode = 'a' if epoch > 0 else 'w'
            with open('models/%s/%s_logs.txt' % (self.name, self.name), mode=mode) as file:
                log = '[%d/%d %d]Train Xe %.6f Train Acc %.4f Test Xe %.6f/%.6f Test Acc %.4f/%.4f' % (epoch, num_epochs, self.global_step, train_loss, train_acc, test_loss, ema_test_loss, test_acc, ema_test_acc)
                print(log,file=file)
                tqdm.write(log)

    def evaluate(self):
        start_epoch, model, opt, scheduler, criterion, ema_model, ema_opt = self.get_model(resume=True)
        train_dl, test_dl = self.get_dataloaders()
        train_loss, train_acc = self.test(model, criterion, train_dl)
        test_loss, test_acc = self.test(model, criterion, test_dl)
        ema_test_loss, ema_test_acc = self.test(ema_model, criterion, test_dl)
        print('Acc(Train %.4f Test %.4f Ema Test %.4f) Loss (Train %.6f Test %.6f Ema Test %.6f)' % (train_acc, test_acc, ema_test_acc, train_loss, test_loss, ema_test_loss))


if __name__ == '__main__':
    clf = Classifier()
    clf.train(num_epochs=300, resume=False)
    clf.evaluate()
