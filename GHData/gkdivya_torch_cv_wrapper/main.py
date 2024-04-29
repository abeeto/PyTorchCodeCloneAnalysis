import torchvision
from torch_cv_wrapper.dataloader.load_data import *
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR,OneCycleLR,ReduceLROnPlateau
from torch_cv_wrapper.utils import train as trn
from torch_cv_wrapper.utils import test as tst
from torchsummary import summary
import yaml
from pprint import pprint
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch_lr_finder import LRFinder


class TriggerEngine:
    def __init__(self, config):
        self.config = config
        self.loader = config['data_loader']['type']
        self.image_dataset=eval(self.loader)(self.config)
        self.device = self.set_device()
        self.writer = SummaryWriter()
        self.l2_factor = self.config['training_params']['l2_factor']

        
    def dataloader(self):
        #Get dataloaders
        return self.image_dataset.get_dataloader()
       
    def get_classes(self): 
        return self.image_dataset.classes()
        
    def set_device(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        return device
        
    def run_experiment(self,model,train_loader,test_loader,lrmin=None,lrmax=None):
        
        model.to(self.device) 
        dropout=self.config['model_params']['dropout']
        epochs=self.config['training_params']['epochs']
        
        l1_factor = self.config['training_params']['l1_factor']
        max_epoch = self.config['lr_finder']['max_epoch']
        
        criterion = nn.CrossEntropyLoss() if self.config['criterion'] == 'CrossEntropyLoss' else F.nll_loss()
        opt_func = optim.Adam if self.config['optimizer']['type'] == 'optim.Adam' else optim.SGD
        lr = self.config['optimizer']['args']['lr']
        
        grad_clip = 0.1
            
        train_losses = []
        test_losses = []
        train_accuracy = []
        test_accuracy = []
        plot_train_acc=[]
        lrs=[]
            
        
        
        if lrmax is not None:
            optimizer = optim.SGD(model.parameters(), lr=lrmin, momentum=0.90,weight_decay=self.l2_factor)
            if self.config['lr_scheduler'] == 'OneCycleLR': 
                scheduler = OneCycleLR(optimizer=optimizer, max_lr=lrmax,
                                      epochs=epochs, steps_per_epoch=len(train_loader),
                                      pct_start=max_epoch/epochs,div_factor=8)
            else:
                scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3,verbose=True,mode='max')
        else:
            optimizer = opt_func(model.parameters(), lr=lr, momentum=0.90,weight_decay=self.l2_factor)
            if self.config['lr_scheduler'] == 'OneCycleLR':
                scheduler = OneCycleLR(optimizer, max_lr=lr,epochs=epochs,steps_per_epoch=len(train_loader))
            else:
                scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3,verbose=True,mode='max')

            
        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}:')
            trn.train(model, self.device, train_loader, optimizer,epoch, train_accuracy, train_losses, l1_factor,scheduler,criterion,lrs,self.writer,grad_clip)
            tst.test(model, self.device, test_loader,test_accuracy,test_losses,criterion)
            
            self.writer.add_scalar('Epoch/Train/train_loss', train_losses[-1], epoch)
            self.writer.add_scalar('Epoch/Test/test_loss', test_losses[-1], epoch)
            self.writer.add_scalar('Epoch/Train/train_accuracy', train_accuracy[-1], epoch)
            self.writer.add_scalar('Epoch/Train/test_accuracy', test_accuracy[-1], epoch)
            plot_train_acc.append(train_accuracy[-1])

            if "ReduceLROnPlateau" in str(scheduler):
                scheduler.step(test_accuracy[-1])

            self.writer.flush()
        return (plot_train_acc,train_losses,test_accuracy,test_losses)
    
    def find_lr(self,model,train_loader, test_loader, start_lr, end_lr):
        
        
        lr_epochs = self.config['lr_finder']['lr_epochs']
        num_iterations = len(test_loader) * lr_epochs

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=0.90, weight_decay=self.l2_factor)
        lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
        lr_finder.range_test(train_loader, val_loader=test_loader, end_lr=end_lr, num_iter=num_iterations, step_mode="linear",diverge_th=50)
        
        # Plot
        max_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(lr_finder.best_loss)]
        #max_lr = lr_finder.plot(suggest_lr=True,skip_start=0, skip_end=0)

        # Reset graph
        lr_finder.reset()
        return max_lr
    
        
    def save_experiment(self,model, experiment_name,path):
        print(f"Saving the model for {experiment_name}")
        torch.save(model, f'{path}/{experiment_name}.pt')
    
    def model_summary(self,model, input_size):
        result = summary(model, input_size=input_size)
        print(result)    
        
    def wrong_predictions(self,model,test_loader,num_img):
        wrong_images=[]
        wrong_label=[]
        correct_label=[]
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)        
                pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

                wrong_pred = (pred.eq(target.view_as(pred)) == False)
                wrong_images.append(data[wrong_pred])
                wrong_label.append(pred[wrong_pred])
                correct_label.append(target.view_as(pred)[wrong_pred])  
      
                wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))    
            print(f'Total wrong predictions are {len(wrong_predictions)}')
            
            self.plot_misclassified(wrong_predictions,num_img)
      
        return wrong_predictions
        
    def plot_misclassified(self,wrong_predictions,num_img):
        fig = plt.figure(figsize=(15,12))
        fig.tight_layout()
        mean,std = self.image_dataset.calculate_mean_std()
        for i, (img, pred, correct) in enumerate(wrong_predictions[:num_img]):
            img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
            for j in range(img.shape[0]):
                img[j] = (img[j]*std[j])+mean[j]
            
            img = np.transpose(img, (1, 2, 0)) 
            ax = fig.add_subplot(5, 5, i+1)
            fig.subplots_adjust(hspace=.5)
            ax.axis('off')
            self.class_names,_ = self.get_classes()
            
            ax.set_title(f'\nActual : {self.class_names[target.item()]}\nPredicted : {self.class_names[pred.item()]}',fontsize=10)  
            ax.imshow(img)  
          
        plt.show()
