#!/usr/bin/env python
# coding: utf-8

# In[62]:

import torch
import torchvision
from torch.optim.lr_scheduler import StepLR
import numpy as np

# class DeviceDataLoader():
#     """
#         This is a class that helps to move data to available device
#     """
#     def __init__(self, dl):
#         import imp
#         try:
#             imp.find_module('torch')
#             found = True
#         except:
#             found = False
        
#         if found is False: raise Exception('torch module not present')
#         self.dl = dl
        
    # def get_device(self):
    #     """Picks GPU if available or else CPU"""
    #     if torch.cuda.is_available():
    #         return torch.device('cuda')
    #     else: return torch.device('cpu')
#     #moving data to available device gpu or cpu
#     def to_device(self,data, device):
#         if isinstance(data, (list, tuple)):
#             return [to_device(x, device) for x in data ]
#         return data.to(device, non_blocking=True)
    
#     def __iter__(self):
#         for b in self.dl:
#             yield self.to_device(b, self.get_device())
#     def __len__(self):
#         return len(self.dl)
    
        

class SebTorchTrainer():
    """
        A helper class for training Image Classifier Neurals <<<<<<<< Development stage {Beta version} >>>>>>>>>>>>>>
    """
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        import imp
        try:
            imp.find_module('torch')
            found = True
        except:
            found = False
        
        if found is False: raise Exception('torch module not present')

    def trainable_neurons(self):
        neurons =  sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return '<<<<<<----Trainable neurons for {} Computer Vision Model : {} ---->>>>>'.format( self.model_name ,neurons)
        
    def spliter(self, train_size, dataset):
        """
            This method helps to split data randomly, by passing a float of the percentage of the training set
            Parameters: \n
                train_size: (float): fraction(percentage of traininig required), must be less than 1
                dataset: whole dataset

            returns: Train set and Validation set
        """
        train_size = float(train_size)
        from torch.utils.data import random_split
        self.dataset = dataset
        
        if float(train_size) >= 1.00:
            raise TypeError('Size of train must be less than 1.00 {} '.format(train_size))
        else:
            train = round(len(self.dataset)*train_size)
            validation = len(self.dataset) - train
            train_set, validation_Set = random_split(self.dataset, [ train, validation])
            
            return train_set, validation_Set
        
    def prepare(self, valid_set,train_set, train_bs=3, valid_bs=3, shuf_train=False, shuf_test=False, shuf_valid=False):
        """
            Converts data to batches for feeding to the Network \n
            Parameters: \n
                valid_set: Validation dataset
                train_set: raining dataset
                train_bs:(int) default is 3
                valid_bs:(int) default is 3
                shuf_train: default is False
                            Helps to shuffle the train dataloader
                shuf_test: default is False
                            Helps to shuffle the test dataloader

            returns: train_loader, valid_loader (Data in batches)
        """
        from torch.utils.data import DataLoader
        
        # if train_bs is None: train_bs = 3
        # if valid_bs is None: valid_bs = 3   
        # if test_bs is None: test_bs = 10
        
        # if train_set is None or valid_set is None and type(train_split)== 'float':
        #     train_set,valid_set = self.spliter(train_split, dataset )
        # else: raise Exception('Train_split is probably not a float')
            
        train_loader = DataLoader(dataset=train_set, batch_size=train_bs, shuffle=shuf_train)
        valid_loader = DataLoader(dataset=valid_set, batch_size=valid_bs, shuffle=shuf_valid)
        
        return train_loader, valid_loader
        
    
    def fit(self, epochs, train_batch, validation_batch, loss_func, metric, opt_func, lr, patience=[None, 'val_loss'], checkpoint=[True, 'val_loss'], scheduler=False, step=1, gamma=0.1):
        """
            Fits the model for training process \n
            Parameters: \n
                epochs: (int) Number of training cycle
                train_batch: training data in batches
                validation_batch: validation data in batches
                loss_func: (function) Loss function for computing losses
                metric: (function) For measuring accuracy
                opt_func: (function) Optimization function 
                lr:(float) Learning rate
                patience: (List) : default: ([None, 'val_loss']), the first index of the list accept number of epoch patience to stop training,
                the second index accept the criterion for the process, it could be 'val_loss' or 'val_acc'
                checkpoint: (list): default: [True, 'val_loss']: the first index allows model checkpoints to be created, the second index accepts the criterion for checkpoint 'val_loss' or 'val_acc'
                scheduler: default: (False) learning rate scheduler
                step:(int) decay step after each epoch
                gamma: (float) decaying constant i.e Lr_(t) = Lr_(t-1) * gamma 
            
            returns: training losses, training accuracies, validation losses , validation accuracies
        """

        val_loss_memory = np.inf
        val_acc_memory = np.inf
        counter = []


        sch_lr = None
        opt = opt_func(self.model.parameters(), lr=lr)
        if scheduler == True:
            sch_lr = StepLR(opt, step_size=step,gamma= gamma)

        # train_batch = DeviceDataLoader(train_batch)
        # validation_batch = DeviceDataLoader(validation_batch)
        
        #<------------- Tracking results --------------------->
        train_losses, train_acces = [], []

        valid_losses , valid_acces = [], []
        
        for epoch in range(epochs):

            train_acc = 0.0
            train_loss = 0.0
            valid_acc = 0.0
            valid_loss = 0.0
            batch_train = 0.0
            batch_val = 0.0

            for xb, yb in train_batch:

                #  ###### Checking for GPU or CPU #######
                # if torch.cuda.is_available():
                #     xb.cuda()
                #     validation_batch.cuda()
                # else:
                #     train_batch.cpu()
                #     validation_batch.cpu()
                self.model.train()
                # make predictions
                preds = self.model(xb)
                # calculate the loss
                loss = loss_func(preds, yb)
                # Calculate the derivatives
                loss.backward()
                # update training parameters
                opt.step()
                # reset gradients to x=zero

                #----------Decaying lr----------
                if sch_lr is not None: sch_lr.step()
                
                opt.zero_grad()
                train_loss += loss.item() * len(xb)
                # validation on training
                acc = metric(preds, yb)

                train_acc += acc 
                batch_train += len(xb)      

            # Validation accuracy
            with torch.no_grad():
                for xval, yval in validation_batch:
                
                    self.model.eval()
                    val_pred = self.model(xval)

                    val_loss = loss_func(val_pred, yval).item()

                    val_acc = metric(val_pred, yval)

                    valid_acc += val_acc 
                    valid_loss += val_loss * len(xval)
                    batch_val += len(xval)              

            #<-------------------calculate average metrics (losses, accuracies)------------------>
            train_loss = train_loss/batch_train
            valid_loss = valid_loss/batch_val
            train_acc = train_acc/len(train_batch)
            valid_acc = valid_acc/len(validation_batch)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_acces.append(train_acc)
            valid_acces.append(valid_acc)


            if checkpoint[0] == True and len(checkpoint[1]) == 1:
                if valid_loss < val_loss_memory:
                    print('Model saved at Epoch: {} watching by val_loss'.format(epoch))
                    torch.save(self.model_name, 'model_at_epoch{}.pkl'.format(epoch))
                    val_loss_memory = valid_loss
            elif checkpoint[0] == True and len(checkpoint) == 2:
                if checkpoint[1] == 'val_loss':
                    if valid_loss < val_loss_memory:
                        print('Model saved at Epoch: {} watching by val_loss'.format(epoch))
                        torch.save(self.model_name, 'model_at_epoch{}.pkl'.format(epoch))
                        val_loss_memory = valid_loss
                elif checkpoint[1] == 'val_acc':
                    if val_acc > val_acc_memory:
                        print('Model saved at Epoch: {} watching by val_acc'.format(epoch))
                        torch.save(self.model_name, 'model_at_epoch{}.pkl'.format(epoch))
                        val_acc_memory = valid_acc
            if patience[0] is not None and type(patience[0]) is not int: pass
            elif patience[0] is not None and type(patience[0]) is int:
                if len(patience) == 1:
                    #using validation loss
                    counter.append(val_loss)
                    if len(counter) > patience[0]:
                        break
                elif len(patience) == 2:
                    if patience[1] == 'val_loss':
                        counter.append(val_loss)
                        if len(counter) > patience[0]:
                            break
                        elif patience[1] == 'val_acc':
                            counter.append(val_acc)
                        if len(counter) > patience[0]:
                            break

            print(' Epoch [{}/{}], training_loss: {:.4f}, training_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format( epoch+1, epochs, train_loss, train_acc, valid_loss, valid_acc ))
        return train_losses, train_acces, valid_losses , valid_acces
    
    def evaluate(self, validation_batch, loss_func, metric):
        """
            evaluates the model on unseen data \n
            Parameters: \n
                validation_batch: validation data in batches
                loss_func: (function) Loss function for computing losses
                metric: (function) For measuring accuracy
               
            
            returns: validation accuracies, validation losses
        """
        
        valid_acc = 0.0
        valid_loss = 0.0
       
        
        with torch.no_grad():
            for xval, yval in validation_batch:
                self.model.eval()
                val_pred = self.model(xval)

                val_loss = loss_func(val_pred, yval).item()

                val_acc = metric(val_pred, yval)

                valid_acc += val_acc 
                valid_loss += val_loss * len(xval)
                
           
        valid_loss = valid_loss/len(validation_batch)
        valid_acc = valid_acc/len(validation_batch)

        return {'Acc: {:.4f}, loss: {:.4f}'.format(valid_acc, valid_loss)}   
    


# Author: @Sebago
# Mail: humiditii45@gmail.com
# Github: github.com/Humiditii
                 


# In[ ]:




