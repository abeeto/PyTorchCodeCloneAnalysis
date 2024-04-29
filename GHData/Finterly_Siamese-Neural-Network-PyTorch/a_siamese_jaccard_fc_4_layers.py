"""
Created on June 24 2019

@author: Finterly
"""
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.autograd import Variable   
import numpy as np
import pandas as pd
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
pd.set_option("display.max_colwidth", -1) #so pandas does not truncate strings
torch.manual_seed(1)

database_1 = "supfam"
database_2 = "pfam"
database_3 = "gene3d"
my_dir = os.path.join(os.path.expanduser("~"), "thesis", "thesis_eclipse")
benchmark_dir = os.path.join(my_dir, "data","csv", "csv_siamese") 
benchmark_hom = os.path.join(benchmark_dir, "hom_nonhom", "siamese_file_" + database_1 + "_max50_hom.csv")
benchmark_nonhom = os.path.join(benchmark_dir, "hom_nonhom", "siamese_file_" + database_1 + "_max50_nonhom.csv")
benchmark_2 = os.path.join(benchmark_dir, "siamese_file_" + database_2 + "_max50.csv")
benchmark_3 = os.path.join(benchmark_dir, "siamese_file_" + database_3 + "_max50.csv")
emb_csv =  os.path.join(my_dir, "data", "csv", "emb", "blos62.csv")
amino_acid_to_idx = {"0": 0, "A" :1, "R":2, "N":3, "D":4, "C":5, "Q":6, "E":7, "G":8, "H":9, "I":10, "L":11, "K":12, "M":13, "F":14, "P":15, "S":16, "T":17, "W":18, "Y":19, "V":20, "B":21, "Z":22, "X":23, "J":24, "U":25}  

class myDataset(Dataset):   
    def __init__(self, csv_file):  # download, read data, etc. 
        self.xy_data= pd.read_csv(csv_file, sep=",", skiprows=1, header=None)
        print(type(self.xy_data))
    def __len__(self):  # return data length    
        return len(self.xy_data) 
    def __getitem__(self, index):  # return one item on the index
        max_len = 1368 #pfam 1330, gene3d 1248, supfam 1368
        self.seq1 = pd.Series.to_string(self.xy_data.iloc[index, 1:2], index=False).strip()
        print(type(self.seq1))

        self.seq2 = pd.Series.to_string(self.xy_data.iloc[index, 2:3], index=False).strip()
        self.x_seq1 = torch.tensor([amino_acid_to_idx[i] for i in self.seq1] , dtype=torch.long)
        print(type(self.x_seq1))
        self.x_seq2 = torch.tensor([amino_acid_to_idx[i] for i in self.seq2] , dtype=torch.long)
        self.x_seq1 = (F.pad(self.x_seq1, pad=(0, max_len-len(self.x_seq1)), mode="constant", value=0))
        print(type(self.x_seq1))
        self.x_seq2 = (F.pad(self.x_seq2, pad=(0, max_len-len(self.x_seq2)), mode="constant", value=0))
        self.y_label = self.xy_data.iloc[index, 0]
        sample = self.x_seq1, self.x_seq2, self.y_label
        return sample 
    
def partition_five_fold(dataset):
    n = int(np.floor(len(dataset)/5))
    remainder = len(dataset)%5
    if remainder == 0: 
        n1, n2, n3, n4, n5 = n, n, n, n, n
    elif remainder == 1: 
        n1, n2, n3, n4, n5 = n, n, n, n, n+1
    elif remainder == 2: 
        n1, n2, n3, n4, n5 = n, n, n, n+1, n+1
    elif remainder == 3: 
        n1, n2, n3, n4, n5 = n, n, n+1, n+1, n+1
    elif remainder == 4: 
        n1, n2, n3, n4, n5 = n, n+1, n+1, n+1, n+1
    return n1, n2, n3, n4, n5

def create_test_sets(csv_file_hom, csv_file_nonhom):
    hom_set = myDataset(csv_file_hom) 
    nonhom_set = myDataset(csv_file_nonhom) 
    hom_partition = partition_five_fold(hom_set)
    nonhom_partition = partition_five_fold(nonhom_set)
    hom_1, hom_2, hom_3, hom_4, hom_5 = torch.utils.data.random_split(hom_set, (hom_partition))
    nhom_1, nhom_2, nhom_3, nhom_4, nhom_5 = torch.utils.data.random_split(nonhom_set, (nonhom_partition))
    test_set_1 = ConcatDataset((hom_1, nhom_1))
    test_set_2 = ConcatDataset((hom_2, nhom_2))
    test_set_3 = ConcatDataset((hom_3, nhom_3))
    test_set_4 = ConcatDataset((hom_4, nhom_4))
    test_set_5 = ConcatDataset((hom_5, nhom_5))
    return test_set_1, test_set_2, test_set_3, test_set_4, test_set_5
    
def retrieve_train_valid_idx(dataset, valid_pct): 
    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_pct * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx

def create_train_sets(test_set_1, test_set_2, test_set_3, test_set_4, test_set_5):
    train_set_1 = ConcatDataset((test_set_2, test_set_3, test_set_4, test_set_5))
    train_set_2 = ConcatDataset((test_set_1, test_set_3, test_set_4, test_set_5))
    train_set_3 = ConcatDataset((test_set_1, test_set_2, test_set_4, test_set_5))
    train_set_4 = ConcatDataset((test_set_1, test_set_2, test_set_3, test_set_5))
    train_set_5 = ConcatDataset((test_set_1, test_set_2, test_set_3, test_set_4))
    return train_set_1, train_set_2, train_set_3, train_set_4, train_set_5

def load_train_valid_data(train_set, batch_size, valid_pct):  
    valid_pct = valid_pct
    train_idx, valid_idx = retrieve_train_valid_idx(train_set, valid_pct)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler) 
    valid_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=valid_sampler)
    print("This train set contains {} inputs, in {} batches".format(len(train_sampler), len(train_loader)))
    print("This validation set contains {} inputs, in {} batches".format(len(valid_sampler), len(valid_loader)))
    return train_loader, valid_loader 

def load_test_data(test_set):
    test_loader = DataLoader(dataset=test_set, batch_size=len(test_set), shuffle=True) 
    print("This test set contains {} inputs, in {} batches".format(len(test_set), len(test_loader)))
    return test_loader

def load_full_dataset(benchmark_n):
    dataset_n = myDataset(benchmark_n) 
    test_loader_n = DataLoader(dataset=dataset_n, batch_size=len(dataset_n), shuffle=False)
    print("This test set contains {} inputs, in {} batches".format(len(dataset_n), len(test_loader_n)))
    return test_loader_n

class MaxMinout(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, embedding1, embedding2):
        shape = list(embedding1.size())
        flat1 = embedding1.view(1, -1)
        flat2 = embedding2.view(1, -1)
        combined = torch.cat((flat1, flat2), 0)
        maxout = combined.max(0)[0].view(*shape)
#         minout = combined.min(0)[0].view(*shape)
        minout = ((combined * -1).max(0)[0].view(*shape) * -1) # workaround for memory leak bug
        return maxout, minout
    
class myModel(nn.Module):
    def __init__(self):      
        super(myModel, self).__init__()
        weight_vec = np.loadtxt(emb_csv, delimiter=",", usecols=range(0, 21), dtype=np.float32)
        weight = torch.FloatTensor(weight_vec) 
        self.embeddings = nn.Embedding.from_pretrained(weight, freeze=True, sparse=False) 
        self.maxminout = MaxMinout()
        self.conv1 = nn.Conv1d(21, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 48, 3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(48, 64, 3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(20)    
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.05)
        self.fc0 = nn.Linear(20, 1)
        self.fc1 = nn.Linear(64, 64) #different for each batch automatic?    
#         self.fc2 = nn.Linear(640,320)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(48)
        self.bn4 = nn.BatchNorm1d(64)
        
    def forward_one_side(self, x):
        x = self.embeddings(x)  # .view((1, -1))   
        x = torch.transpose(x, 1, 2)
        x = self.relu(self.pool(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.relu(self.pool(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.relu(self.pool(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = self.relu(self.pool(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        x= self.fc0(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
#         x = self.fc2(x)
        return x
 
    def forward(self, input1, input2):
        embedding1 = self.forward_one_side(input1)
        embedding2 = self.forward_one_side(input2)
        maxout, minout = self.maxminout(embedding1, embedding2)
        return maxout, minout

def weight_func(dist):
    return 1.0
#     return 100.0 if dist < 0.2 else 1.0

class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()   
    def forward(self, maxout, minout, align_dist):
        align_dist = align_dist.float()
        weight = Variable(torch.FloatTensor([weight_func(x) for x in align_dist.data]), requires_grad=False)
        loss_contrastive = torch.mean(torch.mul(weight, torch.pow(1 - minout.sum(1)/maxout.sum(1) - align_dist, 2)))
        return loss_contrastive

def train(model, train_set, criterion, optimizer, train_losses, val_losses, batch_size, out_file, scheduler, tb, model_checkpoint, epoch): 
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []  
    start_time = time.time()
    for epoch in range(epoch):
        train_loader, valid_loader = load_train_valid_data(train_set, batch_size, 0.2)
        scheduler.step()
        #TRAINING
        model.train()
        N = 0 
        train_loss = 0.0
        for i, data in enumerate(train_loader, 0): 
            x1, x2, y_label = data              
            optimizer.zero_grad()   
            output1, output2  = model(x1, x2)
            N += x1.shape[0] 
            train_loss = criterion(output1, output2, y_label)
            train_loss.backward()
            optimizer.step()       
            train_losses.append(train_loss.item())   
        #VALIDATION 
        val_loss, valid_losses, val_auc= test(model, valid_loader, criterion, valid_losses, False, out_file) #for model selection 
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss) 
        avg_valid_losses.append(valid_loss) 
        train_losses = []
        valid_losses = []
        tb.add_scalar("Train/Loss", train_loss, epoch)
        tb.add_scalar("Val/Loss", val_loss, epoch) 
        tb.add_histogram("conv1.bias", model.conv1.bias, epoch)
        tb.add_histogram("conv1.weight", model.conv1.weight, epoch)
        tb.add_histogram("conv1.weght.grad", model.conv1.weight.grad, epoch)
        print(epoch+1, ": Train: Avg Loss:" ,  round(train_loss, 3))
        print(epoch+1, ": Val: Loss:", round(val_loss, 3), ", AUC/AUPRC:", val_auc)
        print("epoch ", epoch+1, " complete, learning rate: ", scheduler.get_lr(), "time(sec):", time.time() - start_time)
        model_checkpoint.update(val_loss)    #select by val_loss or train_loss
    print("-------------------average time per epoch:" + str(round(float(time.time() - start_time)/100,2)) + "-----------------------")
    return avg_train_losses, avg_valid_losses
             
def test(model, loader, criterion, test_losses, write_pred, out_file):
    with torch.no_grad():
        model.eval() 
        N = 0
        test_loss = 0.0
        for i, (x1, x2, targets) in enumerate(loader):
            output1, output2 = model(x1,x2) 
            N += x1.shape[0] 
            test_loss += x1.shape[0] * criterion(output1, output2 , targets).item() #problem here
            test_losses.append(test_loss/N)
            eudist = F.pairwise_distance(output1, output2, keepdim = True)
            auc_results = auc_scores(targets, eudist[:,0])
            target_log = targets.numpy()
            pred_log = eudist[:,0].numpy()
            assert len(target_log) == len(pred_log)
            if write_pred is True: 
                np.savetxt(out_file, np.c_[target_log, pred_log], fmt='\t'.join(['%i'] + ['%1.15f']))
        return test_loss/N, test_losses, auc_results

def auc_scores(y_true, y_scores):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores) 
    auc = metrics.auc(fpr, tpr) 
    auprc = metrics.average_precision_score(y_true, y_scores) 
    return auc, auprc

def generate_unique_logpath(logdir, raw_run_name):
    i = 1
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

class ModelCheckpoint:
    def __init__(self, filepath, model):
        self.min_loss = None
        self.filepath = filepath
        self.model = model
    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            torch.save(self.model.state_dict(), self.filepath) 
            #Saving the model is done here by calling torch.save on model.state_dict()
            #torch.save(self.model, self.filepath)
            self.min_loss = loss

def my_dir(my_dir):
    my_dir = my_dir
    if not os.path.exists(my_dir): 
        os.makedirs(my_dir)
    return my_dir
###########################################
# DIRECTORY
my_log_dir = my_dir(os.path.join(os.path.expanduser("~"), "thesis", "thesis_eclipse", "data","results", database_1 +"_jaccard_fc_results_4_layer"))
    
#CREATE DATASETS
test_set_1, test_set_2, test_set_3, test_set_4, test_set_5 = create_test_sets(benchmark_hom, benchmark_nonhom)
train_set_1, train_set_2, train_set_3, train_set_4, train_set_5 = create_train_sets(test_set_1, test_set_2, test_set_3, test_set_4, test_set_5)
model_1, model_2, model_3, model_4, model_5 = myModel(), myModel(), myModel(), myModel(), myModel()
out_file_1 = os.path.join(my_log_dir, database_1+ "_outfile_iter_1" )
out_file_2 = os.path.join(my_log_dir, database_1+ "_outfile_iter_2" )
out_file_3 = os.path.join(my_log_dir, database_1+ "_outfile_iter_3" )
out_file_4 = os.path.join(my_log_dir, database_1+ "_outfile_iter_4" )
out_file_5 = os.path.join(my_log_dir, database_1+ "_outfile_iter_5" )

tensorboard_dir = my_dir(os.path.join(my_log_dir, "jaccard_tensorboard" ))
model_dir = my_dir(os.path.join(my_log_dir, "jaccard_models" ))
plot_dir = my_dir(os.path.join(my_log_dir, "jaccard_plots" ))

def main_one_iter(model, train_set, test_set, out_file, idx, epoch):
    tensorboard_log = generate_unique_logpath(tensorboard_dir, "tensorboard_log")
    batch_size = 32
    best_model = os.path.join(model_dir, "best_model" + str(idx)+ ".pt")
    model_checkpoint = ModelCheckpoint(best_model, model)
    criterion = ContrastiveLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3) #, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    tb = SummaryWriter(tensorboard_log)

    #TRAINING
    train_losses, val_losses = [], []
    train_losses, val_losses = train(model, train_set, criterion, optimizer, train_losses, val_losses, batch_size, out_file, scheduler, tb, model_checkpoint, epoch)
    
    #PLOT
    my_plot = os.path.join(plot_dir, "Figure_" + str(idx)+ ".png")
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend(frameon=False) #plt.show()
    plt.savefig(my_plot)
    plt.clf()
    
    #LOAD BEST MODEL
    model =  myModel()
    model.load_state_dict(torch.load(best_model))
    
    #EVALUATION
    print("----------------------------------------------------------------------------------")
    print("------------------------------testing now------------------------------------")
    print("----------------------------------------------------------------------------------")
    train_auc_losses = []
    model.eval()
    full_train_loader, null_valid_loader = load_train_valid_data(train_set, batch_size, 0)
    train_auc_loss, train_auc_losses, train_auc_results = test(model, full_train_loader, criterion, train_auc_losses, False, out_file)
    print(database_1 + " Train: Loss:", round(train_auc_loss, 3),  ", AUC/AUPRC:", train_auc_results)
    print("----------------------------------------------------------------------------------")
    test_losses = []
    model.eval()
    test_loader = load_test_data(test_set)
    test_loss, test_losses, auc_results = test(model, test_loader, criterion, test_losses, True, out_file)
    print(database_1 + " Test: Loss:", round(test_loss, 3), ", AUC/AUPRC:", auc_results)
    print("----------------------------------------------------------------------------------")
    test_losses_2 = []
    model.eval()
    test_loader_2= load_full_dataset(benchmark_2)
    test_loss_2, test_losses_2, auc_results_2 = test(model, test_loader_2, criterion, test_losses_2, False, out_file)
    print(database_2 + ": Loss:", round(test_loss_2, 3), ", AUC/AUPRC:", auc_results_2)
    print("----------------------------------------------------------------------------------")
    test_losses_3 = []
    model.eval()
    test_loader_3 = load_full_dataset(benchmark_3)
    test_loss_3, test_losses_3, auc_results_3 = test(model, test_loader_3, criterion, test_losses_3, False, out_file)
    print(database_3 + ": Loss:", round(test_loss_3, 3), ", AUC/AUPRC:", auc_results_3)
    print("----------------------------------------------------------------------------------")
    print("----------------------------------- yay! ----------------------------------------")


#MAIN
print("Iteration 1:")
main_one_iter(model_1, train_set_1, test_set_1, out_file_1, idx=1, epoch=100)
print("Iteration 2:")
main_one_iter(model_2, train_set_2, test_set_2, out_file_2, idx=2, epoch=100)
print("Iteration 3:")
main_one_iter(model_3, train_set_3, test_set_3, out_file_3, idx=3, epoch=100)
print("Iteration 4:")
main_one_iter(model_4, train_set_4, test_set_4, out_file_4, idx=4, epoch=100)
print("Iteration 5:")
main_one_iter(model_5, train_set_5, test_set_5, out_file_5, idx=5, epoch=100)

combined_out_file = os.path.join(my_log_dir, database_1+ "_combined_outfile")
filenames = [out_file_1, out_file_2, out_file_3, out_file_4, out_file_5]
with open(combined_out_file, 'w') as outfile:
    for fname in filenames:
        with open(fname, 'r') as infile:
            outfile.write(infile.read())