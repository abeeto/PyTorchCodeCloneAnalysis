import torch
import numpy as np
import sys
import os

from model import Model
from dataset import DatasetTrain, DatasetValid

def train(dataset_train, dataset_valid, model, batch_size, max_epochs, sequence_length):
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # else:
    #     torch.set_default_tensor_type('torch.FloatTensor')
    
    device = torch.device('cuda' if torch.cuda.is_avilable() else 'cpu')
    model.to(device)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=dataset_valid.__len__())
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(max_epochs):
        state_h, state_c = model.init_state(sequence_length)
        model.train()
        
        best_loss = sys.maxsize
        for batch, (x, y) in enumerate(dataloader_train):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x.to(device), (state_h, state_c))
            print(y.shape)
            #loss = criterion(y_pred.transpose(1, 2), y)
            train_loss = criterion(y_pred, y.float())
            state_h = state_h.detach()
            state_c = state_c.detach()

            train_loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'training loss': train_loss.item() })

        model.eval()
        for batch, (x, y) in enumerate(dataloader_valid):
            path = './saved_model'
            isExist = os.path.exists(path)

            if not isExist:
                os.makedirs(path)

            with torch.no_grad():
                y_pred, (state_h, state_c) = model(x, (state_h, state_c))

            valid_loss = criterion(y_pred, y.float())
            print({'validation loss': valid_loss.item() })
            save_path = os.path.join(path,'best_model.pt')
            if valid_loss < best_loss:
                best_loss = valid_loss
                print('validation loss decreased, model being saved')
                torch.save(model, save_path)




if __name__ == '__main__':
    dataset_train = DatasetTrain()
    dataset_valid = DatasetValid()
    model = Model(embedding_dim = 128, hidden_dim = 1000)

    # print(dataset_train.__len__())
    # print(dataset_valid.__len__())
   
    train(dataset_train, dataset_valid, model, batch_size = 1028, max_epochs = 50, sequence_length = 3)