import torch
import numpy as np
from dixitool.pytorch.optim import functional as optimizerF
from dixitool.pytorch.module import functional as moduleF
import torchvision.utils 
import torch.optim as optim
internal_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
label_fromsrc = 1
label_fromtgt = 0
gamma=10
#a training process in one epoch
def dann_train(feature_extractor, class_classifier, domain_classifier, 
    class_criterion,domain_criterion ,src_trainloader,tgt_trainloader,optimizer,
    epoch, n_epoch, theta = 1,device = internal_device, closure=None):
    """
    Args:
        model:
        extractor:
        class_classifier:
        domain_classifier:
        class_criterion:
        domain_criterion:
        src_trainloader:
        tgt_trainloader:
        optimizer:
        epoch (int): current number of epoch
        n_epoch (int): total epochs
        device (torch.device): cpu or gpu
        closure (func): closure for print train state information
    """

    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()

    #I will try to use all of the data from source and target domain,
    # But now just keep this way
    zip_loader = zip(src_trainloader,tgt_trainloader)
    zip_loader_len = min(len(src_trainloader),len(tgt_trainloader))
    start_steps = epoch * zip_loader_len
    total_steps = n_epoch * zip_loader_len
    for batch_idx, (src_data, tgt_data) in enumerate(zip_loader):

        #'p' is the training progress linearly changing from 0 to 1
        p = (batch_idx + start_steps) / total_steps
        #'λ' is initiated at 0 and is gradually changed to 1 using the following schedule:
        lambda_p = 2.0 / (1. + np.exp(-gamma*p)) - 1.0

        src_input, src_label = src_data
        tgt_input, tgt_label = tgt_data
                #set cpu or cuda
        src_input = src_input.to(device)
        src_label = src_label.to(device)
        tgt_input = tgt_input.to(device)
        tgt_label = tgt_label.to(device)
        
        #make mnist image to have same shape as mnist-m image
        src_input = torch.cat((src_input, src_input, src_input), 1)
        #adjust learning rate
        optimizerF.optimizer_scheduler(optimizer,p)
        optimizer.zero_grad()

        # data running through feature extractor
        # forward
        # track history if only in train
        
        src_feature = feature_extractor(src_input)
        tgt_feature = feature_extractor(tgt_input)

        ############################
        # (1) Update classifier network: 
        ###########################
        ## Train with all-source batch
        # get log_softmax of source data, only source data running through class classifier
        src_classifier_output = class_classifier(src_feature)

        _, src_preds = torch.max(src_classifier_output,1)
        

        # loss objectie of classifer
        lossY = class_criterion(src_classifier_output, src_label)
        # if phase == 'train':
        #     lossY.backward()

        ############################
        # (2) Update domain_classifier network: 
        ###########################
        ## Train with all source batch
        b_size = src_input.size(0)
        src_label = torch.full((b_size,), label_fromsrc ,device=device)
        # domain classifier has two input parameters
        src_domain_output = domain_classifier(src_feature,lambda_p).view(-1)
        

        lossD_src = domain_criterion(src_domain_output, src_label)
        # if phase == 'train':
        #     lossD_src.backward()

        ## Train with all target batch
        b_size = tgt_input.size(0)
        tgt_label = torch.full((b_size,), label_fromtgt ,device=device)

        tgt_domain_output = domain_classifier(tgt_feature,lambda_p).view(-1)
        lossD_tgt = domain_criterion(tgt_domain_output, tgt_label)
        
        # if phase == 'train':
        #     lossD_tgt.backward()
        #     optimizer.step()

        ## pytorch_DANN的训练 procedure
        
        domain_loss = lossD_src+lossD_tgt
        loss = lossY + theta*domain_loss
        loss.backward()
        optimizer.step()

        if closure is not None:
             closure(batch_idx, zip_loader_len, epoch, n_epoch, lossY.item() ,domain_loss.item())

        ############################
        # Val state
        ###########################                   
    #     if phase == 'val':
    #         #source domain accuracy
    #         tgt_classifier_output = class_classifier(tgt_feature)
    #         _, tgt_preds = torch.max(tgt_classifier_output,1)
    #         tgt_corrects += torch.sum(tgt_preds == tgt_label)

    
    # if phase == 'val':
    #     #see target accuracy
    #     print("val:")
    #     print('source acc: {:.4f}%\ttarget acc:{:.4f}%\tdomain acc:{:.4f}%'.format( 100. * float(src_corrects) / src_data_sum, float(tgt_corrects)/tgt_data_sum,float(domain_corrects)/(src_data_sum+tgt_data_sum)))
    #     if (epoch+1) % 10 == 0 or epoch == n_epoch - 1:
    #         print("saving model====")
    #         moduleF.save_model('model','_epoch:_{}'.format(epoch),feature_extractor)
    #         moduleF.save_model('model','_epoch:_{}'.format(epoch),class_classifier)
    #         moduleF.save_model('model','_epoch:_{}'.format(epoch),domain_classifier)




#
def dcnn_test(feature_extractor, class_classifier, domain_classifier, 
    class_criterion,domain_criterion ,src_testloader,tgt_testloader,
    epoch,n_epoch,device=internal_device, closure=None):
    """
    Args:
        model:
        extractor:
        class_classifier:
        domain_classifier:
        class_criterion:
        domain_criterion:
        src_trainloader:
        tgt_trainloader:
        optimizer:
        epoch (int): current number of epoch
        n_epoch (int): total epochs
        device (torch.device): cpu or gpu
        closure (func): closure for print train state information
    """
    #Placeholder
    src_corrects = 0
    domain_corrects = 0
    src_data_sum = 0
    tgt_corrects = 0
    tgt_data_sum = 0

    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()

    # source test data
    for batch_idx, sdata in enumerate(src_testloader):
        p = float(batch_idx) / len(src_testloader)
        #'λ' is initiated at 0 and is gradually changed to 1 using the following schedule:
        lambda_p = 2.0 / (1. + np.exp(-gamma*p)) - 1.0
        inputs, labels = sdata
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        src_data_sum += inputs.size(0)
        inputs = torch.cat((inputs,inputs,inputs),1)
        
        src_feature = feature_extractor(inputs)

        src_classifier_output = class_classifier(src_feature)
        _, src_preds = torch.max(src_classifier_output,1)
        src_corrects += torch.sum(src_preds == labels)

        src_domain_output = domain_classifier(src_feature,lambda_p).view(-1)
        domain_corrects += torch.sum(src_domain_output.detach().squeeze() >= 0.5)
    
    #target test data
    for batch_idx, tdata in enumerate(tgt_testloader):
        p = float(batch_idx) / len(tgt_testloader)
        #'λ' is initiated at 0 and is gradually changed to 1 using the following schedule:
        lambda_p = 2.0 / (1. + np.exp(-gamma*p)) - 1.0

        inputs, labels = tdata
        inputs = inputs.to(device)
        labels = labels.to(device)
        tgt_data_sum += inputs.size(0)

        tgt_feature = feature_extractor(inputs)
        

        tgt_classifier_output = class_classifier(tgt_feature)
        _, tgt_preds = torch.max(tgt_classifier_output,1)
        tgt_corrects += torch.sum(tgt_preds == labels)

        tgt_domain_output = domain_classifier(tgt_feature,lambda_p).view(-1)
        domain_corrects += torch.sum(src_domain_output.detach().squeeze() < 0.5)

    if closure is not None:
        closure(epoch, n_epoch,src_corrects, src_data_sum, tgt_corrects, tgt_data_sum,domain_corrects)









