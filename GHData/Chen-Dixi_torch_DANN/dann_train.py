import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os
from dixitool.data import datasetsFactory
from dixitool.pytorch.optim import functional as optimizerF
from dixitool.pytorch.module import functional as moduleF
import torch.nn as nn
import torch.optim as optim
from models.models import Feature_Extractor,Class_classifier,Domain_classifier
import numpy as np
# experiment parameter
mnistdata_path= "../data"
mnistmdata_path= "../data/MNIST_M"
gamma=10
batch_size = 512
theta = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

label_fromsrc = 1
label_fromtgt = 0
#代码有点长，训练和验证同时进行
def dcnn_mnist2mnistm(src_trainloader,src_testloader,tgt_trainloader,tgt_testloader, n_epoch=100 ):
    
    # Network Variable
    feature_extractor = Feature_Extractor()
    class_classifier = Class_classifier()
    domain_classifier = Domain_classifier()
    #Remember to class fy
    feature_extractor.to(device)
    class_classifier.to(device)
    domain_classifier.to(device)
    #NLL_Loss 
    domain_criterion = nn.BCELoss()
    class_criterion = nn.NLLLoss()

    #init optimizer
    #三个params 放在SGD父类Optimizer(object)的self.param_groups里面
    optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                            {'params': class_classifier.parameters()},
                            {'params': domain_classifier.parameters()}], lr= 0.01, momentum= 0.9)
    src_loader={'train':src_trainloader,'val':src_testloader}
    tgt_loader={'train':tgt_trainloader,'val':tgt_testloader}

    for epoch in range(n_epoch):
        print('Epoch: {}/{}'.format(epoch,n_epoch))
        
        for phase in ['train','val']:
            src_corrects = 0
            domain_corrects = 0
            src_data_sum = 0
            tgt_corrects = 0
            tgt_data_sum = 0

            # Set the mode of 3 networks
            if phase == 'train':
                feature_extractor.train()
                class_classifier.train()
                domain_classifier.train()
            else:
                feature_extractor.eval()
                class_classifier.eval()
                domain_classifier.eval()
            # get inputs from two dataloader in an iteration
            zip_loader = zip(src_loader[phase],tgt_loader[phase])
            zip_loader_len = min(len(src_loader[phase]),len(tgt_loader[phase]))
            start_steps = epoch * zip_loader_len
            total_steps = n_epoch * zip_loader_len
            for batch_idx, (src_data, tgt_data) in enumerate(zip_loader):

                #'p' is the training progress linearly changing from 0 to 1
                p = (batch_idx + start_steps) / total_steps
                #'λ' is initiated at 0 and is gradually changed to 1 using the following schedule:
                lambda_p = 2.0 / (1. + np.exp(-gamma*p)) - 1.0

                src_input, src_label = src_data
                tgt_input, tgt_label = tgt_data
                src_data_sum += src_input.size(0)
                tgt_data_sum += tgt_input.size(0)
                #set cpu or cuda
                src_input = src_input.to(device)
                src_label = src_label.to(device)
                tgt_input = tgt_input.to(device)
                tgt_label = tgt_label.to(device)

                #adjust learning rate
                optimizerF.optimizer_scheduler(optimizer,p)
                optimizer.zero_grad()

                # data running through feature extractor
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    src_feature = feature_extractor(src_input)
                    tgt_feature = feature_extractor(tgt_input)

                    ############################
                    # (1) Update classifier network: 
                    ###########################
                    ## Train with all-source batch
                    # get log_softmax of source data, only source data running through class classifier
                    src_classifier_output = class_classifier(src_feature)

                    _, src_preds = torch.max(src_classifier_output,1)
                    src_corrects += torch.sum(src_preds == src_label)

                    # loss objectie of classifer
                    lossY = class_criterion(src_classifier_output, src_label)
                    # if phase == 'train':
                    #     lossY.backward()

                    ############################
                    # (2) Update domain_classifier network: 
                    ###########################
                    ## Train with all source batch
                    b_size = src_input.size(0)
                    label = torch.full((b_size,), label_fromsrc ,device=device)
                    # domain classifier has two input parameters
                    src_domain_output = domain_classifier(src_feature,lambda_p).view(-1)
                    domain_corrects += torch.sum(src_domain_output.detach().squeeze() >= 0.5)

                    lossD_src = domain_criterion(src_domain_output, label)
                    # if phase == 'train':
                    #     lossD_src.backward()

                    ## Train with all target batch
                    b_size = tgt_input.size(0)
                    label = torch.full((b_size,), label_fromtgt ,device=device)

                    tgt_domain_output = domain_classifier(tgt_feature,lambda_p).view(-1)
                    domain_corrects += torch.sum(tgt_domain_output.detach().squeeze() < 0.5)
                    lossD_tgt = domain_criterion(tgt_domain_output, label)
                    
                    # if phase == 'train':
                    #     lossD_tgt.backward()
                    #     optimizer.step()

                    ## pytorch_DANN的训练 procedure
                    if phase == 'train':
                        domain_loss = lossD_src+lossD_tgt
                        loss = lossY + theta*domain_loss
                        loss.backward()
                        optimizer.step()

                    if phase == 'train' and ((batch_idx + 1) % 20 == 0 or batch_idx == (zip_loader_len-1)):
                        #print loss
                        print('training:')
                        print('[batch:{}/ total_batch:{} ]\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(batch_idx+1,zip_loader_len,lossY.item(), (lossD_tgt+lossD_src).item() ) )

                    ############################
                    # Val state
                    ###########################                   
                    if phase == 'val':
                        #source domain accuracy
                        tgt_classifier_output = class_classifier(tgt_feature)
                        _, tgt_preds = torch.max(tgt_classifier_output,1)
                        tgt_corrects += torch.sum(tgt_preds == tgt_label)

            
            if phase == 'val':
                #see target accuracy
                print("val:")
                print('source acc: {:.4f}%\ttarget acc:{:.4f}%\tdomain acc:{:.4f}%'.format( 100. * float(src_corrects) / src_data_sum, float(tgt_corrects)/tgt_data_sum,float(domain_corrects)/(src_data_sum+tgt_data_sum)))
                if (epoch+1) % 10 == 0 or epoch == n_epoch - 1:
                    print("saving model====")
                    moduleF.save_model('model','_epoch:_{}'.format(epoch),feature_extractor)
                    moduleF.save_model('model','_epoch:_{}'.format(epoch),class_classifier)
                    moduleF.save_model('model','_epoch:_{}'.format(epoch),domain_classifier)


source_trainloader, source_testloader = datasetsFactory.create_data_loader("MNIST",mnistdata_path,batch_size)

target_trainloader, target_testloader = datasetsFactory.create_data_loader("MNIST_M",mnistmdata_path,batch_size)


dcnn_mnist2mnistm(source_trainloader,source_testloader,target_trainloader,target_testloader,n_epoch=50)




