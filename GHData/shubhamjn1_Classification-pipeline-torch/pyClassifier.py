import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import classifierDNN

class pyClassifier():
    """
    """
    def __init__(self, X_train,X_val, y_train, y_val):
        # super.__init__()
        self.X_train = X_train
        # self.X_test = X_test
        self.y_train = y_train
        # self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val

        # set parameters for NN
        self.NUM_FEATURES = (self.X_train.shape[1])
        self.NUM_CLASSES = len(np.unique(self.y_train))
    
    
    def processData(self, batch_size = 16):
        """
        docstring
        """

        # prep data into tensors
        train_dataset = ClassifierDataset(torch.from_numpy(self.X_train).float(), \
            torch.from_numpy(self.y_train).long())

        val_dataset = ClassifierDataset(torch.from_numpy(self.X_val).float(), \
            torch.from_numpy(self.y_val).long())

        # dataloaders
        train_loader = DataLoader(dataset = train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        
        val_loader = DataLoader(dataset = val_dataset, batch_size=1)
        return train_loader,val_loader

    
    def fit(self, epoch = 10, lr = 0.01):
        """
        docstring
        """
        # dataprep
        train_loader,val_loader = self.processData()
        
        # setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print (f"Using {self.device}")
        
        # model call and loss
        model = classifierDNN(num_features=self.NUM_FEATURES, num_class=self.NUM_CLASSES)
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = lr)

        print (model)

        # saving acc and losses per epoch
        accuracy_stats = {'train':[], 'val':[]}
        loss_stats = {'train':[], 'val':[]}

        # training starts
        print ("Begin Training")
        for e in (range(1, epoch+1)):
            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0

            model.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(self.device), y_train_batch.to(self.device)
                optimizer.zero_grad()

                y_train_pred = model(X_train_batch)
                train_loss = criterion(y_train_pred, y_train_batch)
                train_acc = self.multi_acc(y_train_pred, y_train_batch)
                
                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()

            # VALIDATION
            with torch.no_grad():
                val_epoch_loss = 0
                val_epoch_acc = 0

                model.eval()
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)

                    y_val_pred = model(X_val_batch)
                    val_loss = criterion(y_val_pred, y_val_batch)
                    val_acc = self.multi_acc(y_val_pred, y_val_batch)
                    
                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()

            loss_stats['train'].append(train_epoch_loss/len(train_loader))
            accuracy_stats['train'].append(train_epoch_acc/len(train_loader))

            loss_stats['val'].append(val_epoch_acc/len(val_loader))
            accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
            
            print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

        return model, accuracy_stats, loss_stats

    def predict(self, model, X_test):
        """To predict for the test data

        Args:
            model ([object]): [Trained torch model object]
            X_test ([type]): [Testing data for inferencing]
        """

        test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(np.array(range(0, len(X_test)))).long())
        test_loader = DataLoader(dataset = test_dataset, batch_size=1)
        y_pred_list = []

        # prediction
        with torch.no_grad():
            model.eval()
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                y_test_pred = model(X_batch)

                y_test_pred = torch.log_softmax(y_test_pred, dim = 1)
                _, y_pred_tags = torch.max(y_test_pred, dim = 1)
                y_pred_list.append(y_pred_tags.cpu().numpy())

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        return y_pred_list

    def multi_acc(self, y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
        _, y_pred_labels = torch.max(y_pred_softmax, dim = 1)

        correct_predictions = (y_pred_labels == y_test).float()
        acc = correct_predictions.sum() / len(correct_predictions)

        acc = torch.round(acc)*100
        return acc
        
class ClassifierDataset(Dataset):
    """To prepare item level dataset for NN

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


# main
if __name__ == "__main__":
    data = load_wine(as_frame=True)
    target_values = data['target']
    data = data['data']

    # split into train test and val
    X_train, X_test, y_train, y_test = train_test_split(data, target_values, test_size=0.2, \
        stratify=target_values, random_state=2020)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, \
        stratify=y_test, random_state=2020)
    
    # normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_val, y_val = np.array(X_val), np.array(y_val)

    print (f"Train size:{X_train.shape}, Test size:{X_test.shape}, Val size: {X_val.shape}")
    classifier = pyClassifier(X_train, X_val, y_train, y_val)
    fitted_model, accuracy_dict, loss_dict = classifier.fit()

    # prediction and classification report
    predictions_test = classifier.predict(fitted_model, X_test)
    print (classification_report(y_test, predictions_test))

