from cProfile import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os, time, random, pdb
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import pickle
import tqdm, pdb
from sklearn.metrics import roc_auc_score,  roc_curve
from multiprocessing import Pool
import config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_roc_curve(labels, predicted_vals, generator, when = ''):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve ' + when)
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    # plt.show()
    return auc_roc_vals

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        
        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss_pos = -1 * torch.mean(pos_weights[i] * y_true[:, i] * torch.log(y_pred[:, i] + epsilon))
            loss_neg = -1 * torch.mean(neg_weights[i] * (1 - y_true[:, i]) * torch.log(1 - y_pred[:, i] + epsilon))
            loss += loss_pos + loss_neg
        return loss

    return weighted_loss

def get_roc_auc_score(y_true, y_probs):
    '''
    Uses roc_auc_score function from sklearn.metrics to calculate the micro ROC AUC score for a given y_true and y_probs.
    '''

    with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'rb') as handle:
        all_classes = pickle.load(handle)
    
    NoFindingIndex = all_classes.index('No Finding')

    if True:
        print('\nNoFindingIndex: ', NoFindingIndex)
        print('y_true.shape, y_probs.shape ', y_true.shape, y_probs.shape)
        GT_and_probs = {'y_true': y_true, 'y_probs': y_probs}
        with open('GT_and_probs', 'wb') as handle:
            pickle.dump(GT_and_probs, handle, protocol = pickle.HIGHEST_PROTOCOL)

    class_roc_auc_list = []    
    useful_classes_roc_auc_list = []
    
    for i in range(y_true.shape[1]):
        try:
            class_roc_auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        except:
            return 0
        class_roc_auc_list.append(class_roc_auc) 
        if i != NoFindingIndex:
            useful_classes_roc_auc_list.append(class_roc_auc)
    if True:
        print('\nclass_roc_auc_list: ', class_roc_auc_list)
        print('\nuseful_classes_roc_auc_list', useful_classes_roc_auc_list)

    return np.mean(np.array(useful_classes_roc_auc_list))

def make_plot(epoch_train_loss, epoch_val_loss, total_train_loss_list, total_val_loss_list, save_name):
    '''
    This function makes the following 4 different plots-
    1. mean train loss VS number of epochs
    2. mean val   loss VS number of epochs
    3. batch train loss for all the training   batches VS number of batches
    4. batch val   loss for all the validation batches VS number of batches
    '''
    fig = plt.figure(figsize=(16,16))
    fig.suptitle('loss trends', fontsize=20)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.title.set_text('epoch train loss VS #epochs')
    ax1.set_xlabel('#epochs')
    ax1.set_ylabel('epoch train loss')
    ax1.plot(epoch_train_loss)

    ax2.title.set_text('epoch val loss VS #epochs')
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel('epoch val loss')
    ax2.plot(epoch_val_loss)

    ax3.title.set_text('batch train loss VS #batches')
    ax3.set_xlabel('#batches')
    ax3.set_ylabel('batch train loss')
    ax3.plot(total_train_loss_list)

    ax4.title.set_text('batch val loss VS #batches')
    ax4.set_xlabel('#batches')
    ax4.set_ylabel('batch val loss')
    ax4.plot(total_val_loss_list)
    
    plt.savefig(os.path.join(config.models_dir,'losses_{}.png'.format(save_name)))

def get_resampled_train_val_dataloaders(XRayTrain_dataset, transform, bs):
    '''
    Resamples the XRaysTrainDataset class object and returns a training and a validation dataloaders, by splitting the sampled dataset in 80-20 ratio.
    '''
    XRayTrain_dataset.resample()

    train_percentage = 0.8
    train_dataset, val_dataset = torch.utils.data.random_split(XRayTrain_dataset, [int(len(XRayTrain_dataset)*train_percentage), len(XRayTrain_dataset)-int(len(XRayTrain_dataset)*train_percentage)])

    print('\n-----Resampled Dataset Information-----')
    print('num images in train_dataset   : {}'.format(len(train_dataset)))
    print('num images in val_dataset     : {}'.format(len(val_dataset)))
    print('---------------------------------------')

    # make dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, shuffle = True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size = bs, shuffle = not True)

    print('\n-----Resampled Batchloaders Information -----')
    print('num batches in train_loader: {}'.format(len(train_loader)))
    print('num batches in val_loader  : {}'.format(len(val_loader)))
    print('---------------------------------------------\n')

    return train_loader, val_loader

def transmitting_matrix(fm1, fm2):
    if fm1.size(2) > fm2.size(2):
        fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))
    fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
    fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

    fsp = torch.bmm(fm1, fm2) / fm1.size(2)
    return fsp

def top_eigenvalue(K, n_power_iterations=10, dim=1):
    v = torch.ones(K.shape[0], K.shape[1], 1).to(device)
    for _ in range(n_power_iterations):
        m = torch.bmm(K, v)
        n = torch.norm(m, dim=1).unsqueeze(1)
        v = m / n
    top_eigenvalue = torch.sqrt(n / torch.norm(v, dim=1).unsqueeze(1))
    return top_eigenvalue

def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def train_epoch(device, train_loader, model, loss_fn, optimizer, epochs_till_now, log_interval):
    '''
    Takes in the data from the 'train_loader', calculates the loss over it using the 'loss_fn' 
    and optimizes the 'model' using the 'optimizer'  
    
    Also prints the loss and the ROC AUC score for the batches, after every 'log_interval' batches. 
    '''
     
    model.train()
    class_num = 12
    running_train_loss = 0
    train_loss_list = []

    sigmoid = torch.nn.Sigmoid()
    total = 0
    correct = 0

    start_time = time.time()
    for batch_idx, (img, target) in enumerate(train_loader):
        # print(type(img), img.shape) # , np.unique(img))

        img = img.to(device)
        target = target.float().to(device)

        if torch.sum(target) == 0:
            continue
        
        optimizer.zero_grad()    
        # t_feats, out = model.extract_feature(img)
        out = torch.log_softmax(model(img), dim=1)  
        print(out)
        loss = loss_fn(out, target)
        running_train_loss += loss*img.shape[0]
        train_loss_list.append(loss.item())
        ###########################################
        prediction = out.max(1, keepdim=True)[1]
        print(label.view_as(prediction))
        correct += prediction.eq(label.view_as(prediction)).sum().item()
        ###############################################
        preds = np.round(sigmoid(out).cpu().detach().numpy())
        targets = target.cpu().detach().numpy()

        # total += len(targets)*14
        total += len(targets)*class_num
        correct += (preds == targets).sum()
        # .detach().tolist()

        loss.backward()
        optimizer.step()


        
        if (batch_idx+1)%log_interval == 0:

            batch_time = time.time() - start_time
            m, s = divmod(batch_time, 60)
            print('Train Loss for batch {}/{} @epoch{}: {} in {} mins {} secs'.format(str(batch_idx+1).zfill(3), str(len(train_loader)).zfill(3), epochs_till_now, round(loss.item(), 5), int(m), round(s, 2)))
        start_time = time.time()

    print("Training Accuracy: ", correct/total)
            
    return train_loss_list, running_train_loss/float(len(train_loader.dataset))

def val_epoch(device, val_loader, model, loss_fn, epochs_till_now = None, log_interval = 1, test_only = False):
    '''
    It essentially takes in the val_loader/test_loader, the model and the loss function and evaluates
    the loss and the ROC AUC score for all the data in the dataloader.
    
    It also prints the loss and the ROC AUC score for every 'log_interval'th batch, only when 'test_only' is False
    '''
    model.eval()
    class_num = 12
    running_val_loss = 0
    val_loss_list = []
    val_loader_examples_num = len(val_loader.dataset)
    sigmoid = torch.nn.Sigmoid()

    # probs = np.zeros((val_loader_examples_num, 14), dtype = np.float32)
    # gt    = np.zeros((val_loader_examples_num, 14), dtype = np.float32)

    probs = np.zeros((val_loader_examples_num, class_num), dtype = np.float32)
    gt    = np.zeros((val_loader_examples_num, class_num), dtype = np.float32)
    k=0

    total = 0
    correct = 0
    total_target = []
    total_preds = []

    with torch.no_grad():
        batch_start_time = time.time()    
        for batch_idx, (img, target) in enumerate(val_loader):
            if test_only:
                per = ((batch_idx+1)/len(val_loader))*100
                a_, b_ = divmod(per, 1)
                print(f'{str(batch_idx+1).zfill(len(str(len(val_loader))))}/{str(len(val_loader)).zfill(len(str(len(val_loader))))} ({str(int(a_)).zfill(2)}.{str(int(100*b_)).zfill(2)} %)', end = '\r')
    #         print(type(img), img.shape) # , np.unique(img))

            img = img.to(device)
            target = target.to(device)    
    
            out = model(img)       
            # sig_out = sigmoid(out) 
            # loss = loss_fn(sig_out, target) 
            loss = loss_fn(out, target)    

            preds = np.round(sigmoid(out).cpu().detach().numpy())
            targets = target.cpu().detach().numpy()

            # total += len(targets)*14
            total += len(targets)*class_num
            correct += (preds == targets).sum()

            running_val_loss += loss.item()*img.shape[0]
            # val_loss_list.append(loss.item())
            # weighted
            val_loss_list.append(loss.cpu().detach().numpy())

            # storing model predictions for metric evaluat`ion 
            probs[k: k + out.shape[0], :] = out.cpu()
            gt[   k: k + out.shape[0], :] = target.cpu()
            k += out.shape[0]

            if ((batch_idx+1)%log_interval == 0) and (not test_only): # only when ((batch_idx + 1) is divisible by log_interval) and (when test_only = False)
                # batch metric evaluation
#                 batch_roc_auc_score = get_roc_auc_score(target, out)

                batch_time = time.time() - batch_start_time
                m, s = divmod(batch_time, 60)
                print('Val Loss   for batch {}/{}: {} in {} mins {} secs'.format(str(batch_idx+1).zfill(3), str(len(val_loader)).zfill(3), round(loss.item(), 5), int(m), round(s, 2)))
            
            batch_start_time = time.time()  
            total_target+= targets.tolist()
            total_preds += sigmoid(out).cpu().detach().numpy().tolist()
            
    # metric scenes
    accuracy = correct/total
    print("Test Accuracy: ", correct/total)
    # try:
    #     roc_auc = roc_auc_score(gt, probs)
    # except:
    #     roc_auc = 0


    return val_loss_list, running_val_loss/float(len(val_loader.dataset)), accuracy

def fit(device, train_loader, val_loader, test_loader, model,
                                         loss_fn, optimizer, losses_dict,
                                         epochs_till_now, 
                                         log_interval, save_interval, 
                                         test_only = False, c_num = None):
    '''
    Trains or Tests the 'model' on the given 'train_loader', 'val_loader', 'test_loader' for 'epochs' number of epochs.
    If training ('test_only' = False), it saves the optimized 'model' and  the loss plots ,after every 'save_interval'th epoch.
    '''
    epoch_train_loss, epoch_val_loss, total_train_loss_list, total_val_loss_list = losses_dict['epoch_train_loss'], losses_dict['epoch_val_loss'], losses_dict['total_train_loss_list'], losses_dict['total_val_loss_list']

    if test_only:
        print('\n======= Testing... =======\n')
        test_start_time = time.time()
        test_loss, mean_running_test_loss, test_acc = val_epoch(device, test_loader, model, loss_fn, log_interval, test_only = test_only)
        total_test_time = time.time() - test_start_time
        m, s = divmod(total_test_time, 60)
        print('test_roc_auc: {} in {} mins {} secs'.format(int(m), int(s)))
        return 0, test_acc
        sys.exit()

    starting_epoch  = epochs_till_now
    # print('\n======= Training after epoch #{}... =======\n'.format(epochs_till_now))

    # epoch_train_loss = []
    # epoch_val_loss = []
    
    # total_train_loss_list = []
    # total_val_loss_list = []
    epoch_start_time = time.time()
    
    # outputs_async = pool.map_async(train_epoch, inputs)
    print('TRAINING')
    train_loss, mean_running_train_loss = train_epoch(device, train_loader, model, loss_fn, optimizer, epochs_till_now, log_interval)
    print('VALIDATION')
    val_loss, mean_running_val_loss, val_acc = val_epoch(device, val_loader, model, loss_fn, epochs_till_now, log_interval)
        
    epoch_train_loss.append(mean_running_train_loss)
    epoch_val_loss.append(mean_running_val_loss)

    total_train_loss_list.extend(train_loss)
    total_val_loss_list.extend(val_loss)

    save_name = 'temp.pth'

    # the follwoing piece of codw needs to be worked on !!! LATEST DEVELOPMENT TILL HERE
    print('\nTRAIN LOSS : {}'.format(mean_running_train_loss))
    print('VAL   LOSS : {}'.format(mean_running_val_loss))

    total_epoch_time = time.time() - epoch_start_time
    m, s = divmod(total_epoch_time, 60)
    h, m = divmod(m, 60)
    print('\nEpoch {} took {} h {} m'.format(epochs_till_now,int(h), int(m)))

    return model.state_dict()
