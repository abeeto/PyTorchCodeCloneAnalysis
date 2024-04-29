__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from sklearn import metrics
from data_utils import get_dataloader
from cnn_model import ModelConfig, TextCNN
from train_config import TrainConfig
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir='cnn_log')


def train(data_path):
    train_loader, test_loader, vocab = get_dataloader(data_path=data_path, bs=32, seq_len=50)
    model = TextCNN(ModelConfig())

    print(model)

    config = TrainConfig()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=1) # Ignoring <PAD> Token

    model.train()

    gs = 0

    for epoch in tqdm(range(config.num_epochs)):
        for idx, batch in tqdm(enumerate(train_loader)):
            gs += 1
            inputs, targets = batch.text, batch.label

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            if gs % 500 == 0:
                writer.add_scalar('train/loss', loss.item(), gs)
                print(f'{gs} loss : {loss.item()}')

        train_acc, train_f1, test_acc, test_f1 = evaluate(model, './rsc/data/')
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('train/f1', train_f1, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)
        writer.add_scalar('test/f1', test_f1, epoch)


def evaluate(model: TextCNN, data_path):
    print('evaluate ...')
    train_loader, test_loader, vocab = get_dataloader(data_path=data_path, bs=32, seq_len=50)

    train_y_true, train_y_pred = [], []
    test_y_true, test_y_pred = [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs, targets = batch.text, batch.label
            output = model(inputs)
            pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
            test_y_pred.extend(pred)
            test_y_true.extend(targets.data)

        for batch in tqdm(train_loader):
            inputs, targets = batch.text, batch.label
            output = model(inputs)
            pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
            train_y_pred.extend(pred)
            train_y_true.extend(targets.data)
    model.train()

    test_acc = metrics.accuracy_score(test_y_true, test_y_pred)
    test_f1 = metrics.f1_score(test_y_true, test_y_pred, average='macro')

    train_acc = metrics.accuracy_score(train_y_true, train_y_pred)
    train_f1 = metrics.f1_score(train_y_true, train_y_pred, average='macro')

    print(f'Train Accuracy: {train_acc}, F1-Score: {train_f1}')
    print(f'Test Accuracy: {test_acc}, F1-Score: {test_f1}')

    return train_acc, train_f1, test_acc, test_f1


if __name__ == '__main__':
    print('trainer loading ...')
    train('./rsc/data')