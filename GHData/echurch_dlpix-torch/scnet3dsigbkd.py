import torch.nn as nn
import torch as torch
import math
import torch.utils.model_zoo as model_zoo
import horovod.torch as hvd
import os, glob
import numpy as np
import threading
import h5py
import hvd_util as hu
import pdb


hvd.init()
seed = 314159
print("hvd.size() is: " + str(hvd.size()))
print("hvd.local_rank() is: " + str(hvd.local_rank()))
print("hvd.rank() is: " + str(hvd.rank()))

print("Number of gpus per rank {:d}".format(torch.cuda.device_count()))
# Horovod: pin GPU to local rank.
#torch.cuda.set_device(hvd.local_rank())

os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())
torch.cuda.manual_seed(seed)
dtype = 'torch.cuda.FloatTensor' 
dtypei = 'torch.cuda.LongTensor'                                                                     


global_Nclass = 3 # bkgd, 0vbb, 2vbb
global_n_iterations_per_epoch = 100
global_n_iterations_val = 4
global_n_epochs = 40
global_batch_size = 4  ## Can be at least 32, but need this many files to pick evts from in DataLoader

vox = 10 # int divisor of 1500 and 1500 and 3000. Cubic voxel edge size in mm.
nvox = int(1500/vox) # num bins in x,y dimension 
nvoxz = int(3000/vox) # num bins in z dimension 
voxels = (int(1500/vox),int(1500/vox),int(3000/vox) ) # These are 1x1x1cm^3 voxels



def accuracy(output, target):
    """Computes the accuracy. we want the aggregate accuracy along with accuracies for the different labels. easiest to just use numpy..."""
    profile = False

    maxk = 1
    batch_size = target.size(0)
    _, pred = output.max( 1, keepdim=False) # on gpu
    targetex = target.resize_( pred.size() ) # expanded view, should not include copy

    correct = pred.eq( targetex.type(dtypei))  #.to(torch.device("cuda")) ) # on gpu
    #print "correct ",correct.size(), " iscuda=",correct.is_cuda    

    # make result vector
    res = float(correct.sum())/float(len(correct))

    return res
                                                            

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import sparseconvnet as scn
import time
import sys
import math


dimension = 3
nPlanes = 1


'''
Model below is an example, inspired by 
https://github.com/facebookresearch/SparseConvNet/blob/master/examples/3d_segmentation/fully_convolutional.py
Not yet even debugged!
EC, 24-March-2019
'''

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension, (nvox,nvox,nvoxz), mode=3)).add(
                scn.SubmanifoldConvolution(dimension, nPlanes, 4, 3, False)).add(
                    scn.MaxPooling(dimension, 3, 3)).add(
                    scn.SparseResNet(dimension, 4, [
                        ['b', 8, 2, 1],
                        ['b', 16, 2, 1],
                        ['b', 24, 2, 1]])).add(
#                        ['b', 32, 2, 1]])).add(
                            scn.Convolution(dimension, 24, 32, 5, 1, False)).add(
                                scn.BatchNormReLU(32)).add(
                                    scn.SparseToDense(dimension, 32))
#        self.spatial_size = self.sparseModel.input_spatial_size(torch.LongTensor([1, 1]))
        self.linear = nn.Linear(int(32*46*46*96), 32)
        self.linear2 = nn.Linear(32,global_Nclass)
    def forward(self,x):
        x = self.sparseModel(x)
        x = x.view(-1, 32*46*46*96)
        x = nn.functional.elu(self.linear(x))
        x = self.linear2(x)
        x = nn.functional.softmax(x, dim=1)
        return x
 
net = Model()
# print(net) # this is lots of info
Net = net.cuda()

tensor_list = []
for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))


# Horovod: broadcast parameters.
hvd.broadcast_parameters(net.state_dict(), root_rank=0)


try:
    print ("Reading weights from file")
    net.load_state_dict(torch.load('./model-scn3dsigbkd.pkl'))
    net.eval()
    print("Succeeded.")
except:
    print ("Failed to read pkl model. Proceeding from scratch.")
#    raise 

# Next two functions taken from Taritree's train_wlarcv1.py
# We define a pixel wise L2 loss

# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

import torch.nn.functional as F    

loss = torch.nn.NLLLoss().cuda()

learning_rate = 0.001 # 0.010
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(net.parameters(), lr=learning_rate * hvd.size(),
                      momentum=0.9)
# Horovod: wrap optimizer with DistributedOptimizer.
compression = hvd.Compression.none  # .fp16 # don't use compression
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=net.named_parameters(),
                                     compression=hvd.Compression.none)  # to start
hvd.broadcast_parameters(net.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

lr_step = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) # lr drops to lr*0.9^N after 5N epochs
#val_gen = gen_waveform(n_iterations_per_epoch=global_n_iterations_per_epoch,mini_batch_size=global_batch_size)


class BinnedDataset(Dataset):

    def __init__(self, path, frac_train, train=True, thresh=3, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #ftype = [ "bb0nu/*.h5","bb2nu/*.h5","Bi214/*.h5","Tl208/*-OUT_PLANES.h5","Tl208/*-VESSEL.h5" ]
        ftype = [ "bb0nu/*.h5","bb2nu/*.h5" ]

        self.files = []
        for ft in ftype:
            self.files.extend( glob.glob(path+"/"+ft) )
        print('Found %s files.'%len(self.files))
        self.file_lengths = [ len((h5py.File(fname,'r'))['MC']['extents']) for fname in self.files ]
        dim3 = np.array(( nvox,nvox,nvoxz))

        self.frac_train = frac_train
        self.frac_dataset = self.frac_train if train else (1 - self.frac_train)
        self.train = train
        self.path = path
        self.thresh = thresh
        self.lock = threading.Lock()

    def __len__(self):
        return sum([ int(self.frac_dataset*flength) for flength in self.file_lengths ])

    def __getitem__(self, idx):

        with self.lock:
            for ifile in range(len(self.file_lengths)):
                if idx < sum([ int(self.frac_dataset*flength) for flength in self.file_lengths[0 : ifile + 1] ]):
                    ind_file = ifile
                    ind_evt = idx - sum([ int(self.frac_dataset*flength) for flength in self.file_lengths[0 : ifile] ])
                    break

            x = np.ndarray(shape=( 1, nvox, nvox, nvoxz))
            #print ("ind_file: " +str(ind_file)+ " self.files is " + str(self.files[ind_file]))

            current_file = h5py.File(self.files[ind_file],'r')
            sigbkd = "bb0" in current_file.filename
            bkd2nu = "bb2" in current_file.filename
            self.np_labels = 0
            if (sigbkd):
                self.np_labels = 1
            if (bkd2nu):
                self.np_labels = 2


            extentset = current_file['MC']['extents']

            if self.train:
                current_index = ind_evt
            else:
                current_index = ind_evt + int(self.file_lengths[ind_file]*self.frac_train)

            if current_index != 0:
                current_starthit = int(extentset[current_index - 1]['last_hit'] + 1)
            else:
                current_starthit = 0

            current_endhit = int(extentset[current_index]['last_hit'])
            if current_starthit >= current_endhit:
                print('current start >= current end!!!')
                print('file: %s'%self.files[ind_file])
                print('evtidx: %s'%current_index)

            hitset = current_file['MC']['hits'][current_starthit:current_endhit]            

            data = np.array((hitset['hit_position'][:,0]+750.,hitset['hit_position'][:,1]+750.,hitset['hit_position'][:,2]+1500.))
            weights=hitset['hit_energy']
            dataT = data.T
            
            if dataT.sum() is 0:
                print("Problem! Image is empty for current_index " + str(current_index))
                raise

            '''  Try to use whole pixelated volume now with scn. EC, 15-Apr-2019.            
            xmin = np.argmin(dataT[:,0])
            ymin = np.argmin(dataT[:,1])
            zmin = np.argmin(dataT[:,2])
            ##  view,chan,x
            '''
            ranges = tuple(vox*float(x) for x in voxels)
#            H,edges = np.histogramdd(dataT,bins=voxels,range=ranges, weights=weights)
            H,edges = np.histogramdd(dataT,bins=voxels,range=((0.,1500.),(0.,1500.),(0.,3000.)), weights=weights)

            return ( H, self.np_labels )

'''
            (xmax,ymax,zmax) = np.unravel_index(np.argmax(H,axis=None),H.shape)
            # Crop this back to central 2mx2mx2m about max activity point
            ix = np.maximum(xmax-nvox/vox,0); ix = int(np.minimum(ix,voxels[0]-nvox))
            iy = np.maximum(ymax-nvox/vox,0); iy = int(np.minimum(iy,voxels[1]-nvox))
            iz = np.maximum(zmax-nvoxz/vox,0); iz = int(np.minimum(iz,voxels[2]-nvoxz))

            x[0] = H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvoxz] # The 0th element is for 1st (only) layer.

            return ( x[0], self.np_labels )
'''
            

#binned_tdata = BinnedDataset(path=[os.environ['HOME']+'/NEXT1Ton',os.environ['HOME']+'/NEXT1Ton/Bi214'],frac_train=0.8,train=True)
#binned_vdata = BinnedDataset(path=[os.environ['HOME']+'/NEXT1Ton',os.environ['HOME']+'/NEXT1Ton/Bi214'],frac_train=0.8,train=False)
binned_tdata = BinnedDataset(path=os.environ['HOME']+'/NEXT1Ton',frac_train=0.8,train=True)
binned_vdata = BinnedDataset(path=os.environ['HOME']+'/NEXT1Ton',frac_train=0.8,train=False)

import csv
if hvd.rank()==0:
    filename = os.environ['MEMBERWORK']+'/nph133/'+os.environ['USER']+'/next1t/'+'history.csv'
    csvfile = open(filename,'w')
#with open('history.csv','w') as csvfile:


fieldnames = ['Training_Validation', 'Iteration', 'Epoch', 'Loss',
              'Accuracy', "Learning Rate"]

# only let one core write to this file.
if hvd.rank()==0:
    history_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    history_writer.writeheader()
thresh = 0.003 # deposited energy

train_loss = hu.Metric('train_loss')
train_accuracy = hu.Metric('train_accuracy')
val_loss = hu.Metric('val_loss')
val_accuracy = hu.Metric('val_accuracy')

for epoch in range (global_n_epochs):

        train_gen = DataLoader(dataset=binned_tdata, batch_size=global_batch_size,
                               shuffle=True, num_workers=global_batch_size)
        print('len(train_gen): %s'%len(train_gen))
        lr_step.step()

        for iteration, minibatch in enumerate(train_gen):
            net.train()
            optimizer.zero_grad()

            feats, labels_var = minibatch            

            tmp = np.nonzero(feats>thresh)
            # below 4-lines are torch-urous equivalent of numpy moveaxis to get batch indx on far right column.
            coords = tmp
            bno = coords[:,0].clone() # wo clone this won't copy, it seems.
            coords[:,0:3] = tmp[:,1:4]
            coords[:,3] = bno
            indspgen = feats>thresh

            yhat = net([coords,feats[indspgen].type(dtype).unsqueeze(1), global_batch_size])

#            train_loss = loss(yhat, labels_var.type(dtypei)) #, weight_var[indspgen].type(dtype)) 
#            train_accuracy = accuracy(yhat, labels_var)
            acc = hu.accuracy(yhat, labels_var.cuda(), weighted=True, nclass=global_Nclass)
            train_accuracy.update(acc)
            loss = nn.functional.cross_entropy(yhat, labels_var.cuda())
            train_loss.update(loss)

            loss.backward()
#            optimizer.synchronize()

            optimizer.step()

            net.eval()

            print("Train.Rank,Epoch: {},{}, Iteration: {}, Loss: [{:.4g}], Accuracy: [{:.4g}]".format(hvd.rank(), epoch, iteration,float(train_loss.avg), train_accuracy.avg))

            output = {'Training_Validation':'Training', 'Iteration':iteration, 'Epoch':epoch, 'Loss': float(train_loss.avg),
                      'Accuracy':train_accuracy.avg.data, "Learning Rate":learning_rate}
            if hvd.rank()==0:
                history_writer.writerow(output)
                csvfile.flush()

            # below is to keep this from exceeding 4 hrs
            if iteration > global_n_iterations_per_epoch:
                break



        # done with iterations within a training epoch
        val_gen = DataLoader(dataset=binned_vdata, batch_size=global_batch_size,
                                 shuffle=True, num_workers=global_batch_size)

        for iteration, minibatch in enumerate(val_gen):

            pdb.set_trace()
            feats, labels_var = minibatch            

            tmp = np.nonzero(feats>thresh)
            # below 4-lines are torch-urous equivalent of numpy moveaxis to get batch indx on far right column.
            coords = tmp
            bno = coords[:,0].clone() # wo clone this won't copy, it seems.
            coords[:,0:3] = tmp[:,1:4]
            coords[:,3] = bno
            indspgen = feats>thresh

            yhat = net([coords,feats[indspgen].type(dtype).unsqueeze(1), global_batch_size])
            
            #            val_accuracy = accuracy(y, yhat)
            acc = hu.accuracy(yhat, labels_var.cuda())   
            val_accuracy.update(acc)
            loss = nn.functional.cross_entropy(yhat, labels_var.cuda())
            val_loss.update(loss)

            print("Val.Epoch: {}, Iteration: {}, Train,Val Loss: [{:.4g},{:.4g}], *** Train,Val Accuracy: [{:.4g},{:.4g}] ***".format(epoch, iteration,float(train_loss.avg), val_loss.avg, train_accuracy.avg, val_accuracy.avg ))

            
            #            for g in optimizer.param_groups:
            #                learning_rate = g['lr']
            output = {'Training_Validation':'Validation','Iteration':iteration, 'Epoch':epoch, 
                      'Loss':float(val_loss.avg), 'Accuracy':val_accuracy.avg, "Learning Rate":learning_rate}
            if hvd.rank()==0:
                history_writer.writerow(output)
            if iteration>=global_n_iterations_val:
                break # Just check val for 4 iterations and pop out

        if hvd.rank()==0:        
            csvfile.flush()

hostname = "hidden"
try:
    hostname = os.environ["HOSTNAME"]
except:
    pass
print("host: hvd.rank()/hvd.local_rank() are: " + str(hostname) + ": " + str(hvd.rank())+"/"+str(hvd.local_rank()) ) 


print("end of epoch")
torch.save(net.state_dict(), 'model-scn3dsigbkd.pkl')

