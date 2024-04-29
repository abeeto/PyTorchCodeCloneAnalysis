import torch
import torch.nn as nn
import numpy as np

from datahandler import build_dataloader
from model import my_lstm, my_lstm_regressor


def train(train_data, epochs=5):
    model.train()

    for i in range(epochs):
        epoch_loss = []
        for data, labels in train_data:
            optimizer.zero_grad()
            hidden = model.init_hidden(data.shape[0])
            
            # LSTMCell expects batch at dim=1
            data = data.transpose(0,1).cuda()
            
            preds = model(data, hidden)
            
            # flatten labels to match with predictions
            #loss = loss_func(preds, labels.view(-1).cuda())
            
            # don't need to flatten for MSE
            loss = loss_func(preds, labels.cuda())

            loss.backward()

            optimizer.step()

            epoch_loss.append(loss.item())
        
        print(f'Loss for epoch #{i}: {np.mean(epoch_loss)}')


def test(test_data):
    model.eval()
    with torch.no_grad():
        test_loss = []
        for data, labels in test_data:
            hidden = model.init_hidden(data.shape[0])
            data = data.transpose(0,1).cuda()

            preds = model(data, hidden)
            #loss = loss_func(preds, labels.view(-1).cuda())
            loss = loss_func(preds, labels.cuda())
            test_loss.append(loss.item())

        print(f'Loss for test: {np.mean(test_loss)}')


if __name__ == '__main__':
    # gather train/test data
    train_data = build_dataloader(5, False)
    test_data = build_dataloader(5, False)

    # instantiate model, optimizer, and loss function
    model = my_lstm_regressor(300, 128)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    #loss_func = nn.CrossEntropyLoss()
    loss_func = nn.MSELoss()

    # move model to GPU if possible
    if torch.cuda.is_available():
        model.cuda()

    train(train_data)
    test(test_data)
