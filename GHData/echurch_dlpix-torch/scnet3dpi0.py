import torch.nn as nn
import torch as torch
import math
import torch.utils.model_zoo as model_zoo
import horovod.torch as hvd
import os, glob
import numpy as np
import threading
import pdb

'''
###########################################################
#
# Mid-transition to a NEXT-appropriate scn to classify sig/bkd. 
# 
# 
  (1) Need to read up actual NEXT 1t MC data instaead of LArTPC data
  (2) BinnedDataSet class below needs to change from labeling, weighing pixels to one that just labels the image, 
      no weights. Want to keep the histogramming and still pass that out.
  (3) accuracy() function needs to just return tuple 2-long, 1/0 for sig, bkd
  (4) whole thing needs to be debugged!
  
# EC, 24-Mar-2019.
###########################################################
'''


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


global_Nclass = 3 
global_n_iterations_per_epoch = 20
global_batch_size = 32
vox = 2 # int divisor of 250 and 600 and 200. Cubic voxel edge size in cm.
nvox = int(200/vox) # num bins in each dimension 
voxels = (int(250/vox),int(600/vox),int(250/vox) ) # These are 2x2x2cm^3 voxels

plotted = np.zeros((8),dtype=bool) # 8 files 
plotted = np.ones((59999),dtype=bool) # 

#from mpi4py import MPI
#print("hvd.size()/MPI_COMM_RANK are: " + str(hvd.size()) + "/" + str(MPI.COMM_WORLD.Get_size())) 


# next 4 are globals
fit_pi0 = True
Epi = np.empty([1])
mpi_pca = np.empty([1])
mpi_constrained = np.empty([1])
Epi_cuts = np.empty([1])
mpi_pca_cuts = np.empty([1])
mpi_constrained_cuts = np.empty([1])






def accuracy(output, target, imgdata):
    """Computes the accuracy. we want the aggregate accuracy along with accuracies for the different labels. easiest to just use numpy..."""
    profile = False
    # needs to be as gpu as possible!
    maxk = 1


    batch_size = target.size(0)
    if profile:
        torch.cuda.synchronize()
        start = time.time()    
    #_, pred = output.topk(maxk, 1, True, False) # on gpu. slow AF
    _, pred = output.max( 1, keepdim=False) # on gpu
    if profile:
        torch.cuda.synchronize()
        print ("time for topk: "+str(time.time()-start)+" secs")

    if profile:
        start = time.time()
    #print "pred ",pred.size()," iscuda=",pred.is_cuda
    #print "target ",target.size(), "iscuda=",target.is_cuda
    targetex = target.resize_( pred.size() ) # expanded view, should not include copy


    correct = pred.eq( targetex.type(dtypei))  #.to(torch.device("cuda")) ) # on gpu
    #print "correct ",correct.size(), " iscuda=",correct.is_cuda    
    if profile:
        torch.cuda.synchronize()
        print ("time to calc correction matrix: "+str(time.time()-start)+" secs")

    # we want counts for elements wise

    num_per_class = {}
    corr_per_class = {}
    total_corr = 0
    total_pix  = 0

    if profile:
        torch.cuda.synchronize()            
        start = time.time()
    for c in range(output.size(1)):
        # loop over classes
        classmat = targetex.eq(int(c)).long() # elements where class is labeled
        #print "classmat: ",classmat.size()," iscuda=",classmat.is_cuda
        num_per_class[c] = classmat.long().sum()
        corr_per_class[c] = (correct.long()*classmat.type(dtypei)).long().sum() # mask by class matrix, then sum
        total_corr += corr_per_class[c].long()
        total_pix  += num_per_class[c].long()
    print ("total_pix: " + str(total_pix))
    print ("total_corr: " + str(total_corr))

    if profile:
        torch.cuda.synchronize()                
        print ("time to reduce: "+str(time.time()-start)+" secs")
        
    # make result vector
    res = []



    for c in range(output.size(1)):
        if num_per_class[c]>0:
            res.append( float(corr_per_class[c])/float(num_per_class[c])*100.0 )
        else:
            res.append( 0.0 )

    # totals
    if total_pix==0:
        res.append(0.0)
        print ("Mysteriously in here - total_pix: " +str(total_pix)  )
    else:
        res.append( 100.0*float(total_corr)/float(total_pix) )


    if num_per_class[1]==0 and num_per_class[2]==0:
        res.append(0.0)
        print ("Mysteriously in here: num-per-class" +str(num_per_class[1]) +", " +str(num_per_class[2]) )
    else:
        res.append( 100.0*float(corr_per_class[1]+corr_per_class[2])/float(num_per_class[1]+num_per_class[2]) ) # track/shower acc

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
            scn.InputLayer(dimension, torch.LongTensor([nvox]*3), mode=3)).add(
                scn.SubmanifoldConvolution(dimension, nPlanes, 16, 3, False)).add(
#                    scn.SparseResNet(dimension, 16, [
#                        ['b', 16, 2, 1],
#                        ['b', 32, 2, 2],
#                        ['b', 48, 2, 2],
#                        ['b', 96, 2, 2]]                )).add(
                            scn.BatchNormReLU(16)).add(
                                scn.OutputLayer(dimension))
        self.linear = nn.Linear(16, global_Nclass)
    def forward(self,x):
        x=self.sparseModel(x)
        x=self.linear(x)
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
    net.load_state_dict(torch.load('./model-scn3dpi0.pkl'))
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
        ftype = "*ana*"

        self.files = [ i for i in glob.glob(path+"/"+ftype)]
        dim3 = np.array(( nvox,nvox,nvox))
        self.np_labels  = np.zeros( dim3, dtype=np.int )
        self.np_weights = np.zeros( dim3, dtype=np.float32 )
        self.frac_train = frac_train
        self.valid_train = 1.0 - self.frac_train
        self.train = train
        self.path = path
        self.thresh = thresh
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        with self.lock:
            x = np.ndarray(shape=( 1, nvox, nvox, nvox))
#            print ("ind_file: " +str(idx)+ " self.files is " + str(self.files))
            ind_file = idx
            current_file = np.load(self.files[ind_file])
            if self.train:
                current_index = np.random.randint(int(current_file.shape[0]*self.frac_train), size=1)[0]
            else:
                current_index = np.random.randint(int(current_file.shape[0]*self.frac_train),int(current_file.shape[0]), size=1)[0]



            data = np.array((current_file[current_index,]['sdX'][current_file[current_index,]['sdTPC']==3],current_file[current_index,]['sdY'][current_file[current_index,]['sdTPC']==3], current_file[current_index,]['sdZ'][current_file[current_index,]['sdTPC']==3] ))
            dataT = data.T
            
            if dataT.sum() is 0:
                print("Problem! Image is empty for current_index " + str(current_index))
                raise
            
            xmin = np.argmin(dataT[:,0][dataT[:,0]>2])
            ymin = np.argmin(dataT[:,1][dataT[:,1]>2])
            zmin = np.argmin(dataT[:,2][dataT[:,2]>2])
            weights=current_file[current_index,]['sdElec'][current_file[current_index,]['sdTPC']==3]
            ##  view,chan,x

            voxels = (int(250/vox),int(600/vox),int(250/vox) ) # These are 2x2x2cm^3 voxels
            
            H,edges = np.histogramdd(dataT,bins=voxels,range=((0.,250.),(0.,600.),(0.,250.)),weights=weights)
            
            pixlabs1 = current_file[current_index,]['sdgamma1'][current_file[current_index,]['sdTPC']==3]
            pixlabs2 = current_file[current_index,]['sdgamma2'][current_file[current_index,]['sdTPC']==3]

            Hpl1,edges = np.histogramdd(dataT,bins=voxels,range=((0.,250.),(0.,600.),(0.,250.)),weights=pixlabs1) # original pixel value 1
            Hpl2,edges = np.histogramdd(dataT,bins=voxels,range=((0.,250.),(0.,600.),(0.,250.)),weights=pixlabs2/2.) # original pixel value 2

            (xmax,ymax,zmax) = np.unravel_index(np.argmax(H,axis=None),H.shape)
            # Crop this back to central 2mx2mx2m about max activity point
            ix = np.maximum(xmax-nvox/vox,0); ix = int(np.minimum(ix,voxels[0]-nvox))
            iy = np.maximum(ymax-nvox/vox,0); iy = int(np.minimum(iy,voxels[1]-nvox))
            iz = np.maximum(zmax-nvox/vox,0); iz = int(np.minimum(iz,voxels[2]-nvox))

            x[0] = H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox] # The 0th element is for 1st (only) layer.
            self.np_labels = np.zeros(H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox].shape)
            indx1 = np.where(Hpl1[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox]!=0)
            indx2 = np.where(Hpl2[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox]!=0)
            self.np_labels[indx2] = 2 # Eg1 pixels

            self.np_weights = self.np_labels.astype('bool').astype('int')
            self.np_weights[self.np_weights==0] = 1.0/np.prod(Hpl1.shape)/1000.


            return ( x[0], self.np_labels, self.np_weights )
            

        ''' Note that scn.InputLayer() expects 1 input layer not potentially N of them, so collapse x now.'''
'''
        return (torch.from_numpy(x.reshape((nvox,nvox,nvox))).float(),
                torch.from_numpy(self.np_labels).long(),
                torch.from_numpy(self.np_weights).float())
'''


binned_tdata = BinnedDataset(path='/ccs/home/echurch/pi0',frac_train=0.8,train=True)
binned_vdata = BinnedDataset(path='/ccs/home/echurch/pi0',frac_train=0.8,train=False)

import csv
with open('history.csv','w') as csvfile:
    fieldnames = ['Iteration', 'Epoch', 'Train Loss',
                  'Validation Loss', 'Train Accuracy', 'Validation Accuracy', "Learning Rate"]
    history_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    history_writer.writeheader()
    thresh = 3

    for epoch in range (150):  # (400)
#        train_gen = gen_waveform(n_iterations_per_epoch=global_n_iterations_per_epoch,mini_batch_size=global_batch_size)

        train_gen = DataLoader(dataset=binned_tdata, batch_size=global_batch_size,
                               shuffle=False, num_workers=global_batch_size)
        lr_step.step()

        for iteration, minibatch in enumerate(train_gen):

            net.train()
#            torch.distributed.init_process_group(backend='nccl',
#                                                 init_method='env://',
#                                                 world_size=4,
#                                                 rank = hvd.rank())

            optimizer.zero_grad()

            feats, labels_var, weight_var = minibatch            

            tmp = np.nonzero(feats>thresh)
            # below 4-lines are torch-urous equivalent of numpy moveaxis to get batch indx on far right column.
            coords = tmp
            bno = coords[:,0].clone() # wo clone this won't copy, it seems.
            coords[:,0:3] = tmp[:,1:4]
            coords[:,3] = bno
            indspgen = feats>thresh

            yhat = net([coords,feats[indspgen].type(dtype).unsqueeze(1), global_batch_size])
            
            train_loss = loss(yhat, labels_var[indspgen].type(dtypei)) #, weight_var[indspgen].type(dtype)) 
            train_loss.backward()
#            optimizer.synchronize()
            optimizer.step()



            ''' After some diagnostic period let's put this outside the Iteration loop'''
            train_accuracy = accuracy(yhat, labels_var[indspgen], feats[indspgen])    # None)
            net.eval()

            print("Epoch: {}, Iteration: {}, Loss: [{:.4g}], Accuracy: [{:.4g},{:.4g},{:.4g}]".format(epoch, iteration,float(train_loss.data), train_accuracy[0], train_accuracy[1], train_accuracy[2]))


        # done with iterations within a training epoch
        val_gen = DataLoader(dataset=binned_vdata, batch_size=global_batch_size,
                                 shuffle=True, num_workers=global_batch_size)

        for iteration, minibatch in enumerate(val_gen):
            feats, labels_var, weight_var = minibatch            

            tmp = np.nonzero(feats>thresh)
            # below 4-lines are torch-urous equivalent of numpy moveaxis to get batch indx on far right column.
            coords = tmp
            bno = coords[:,0].clone() # wo clone this won't copy, it seems.
            coords[:,0:3] = tmp[:,1:4]
            coords[:,3] = bno
            indspgen = feats>thresh

            yhat = net([coords,feats[indspgen].type(dtype).unsqueeze(1), global_batch_size])
            
            val_loss = loss(yhat,labels_var[indspgen].type(dtypei) ) #, weight_var[indspgen].type(dtype)) 
            #            val_accuracy = accuracy(y, yhat)
            val_accuracy = accuracy(yhat, labels_var[indspgen], feats[indspgen])   

            print("Epoch: {}, Iteration: {}, Loss: [{:.4g},{:.4g}], *** Train Accuracy: [{:.4g},{:.4g},{:.4g}, ***,  Val Accuracy: [{:.4g},{:.4g},{:.4g}]".format(epoch, iteration,float(train_loss.data), val_loss, train_accuracy[0], train_accuracy[1], train_accuracy[2], val_accuracy[0], val_accuracy[1], val_accuracy[2]))

            
            #                if (iteration%1 ==0) and (iteration>0):
                
            #            for g in optimizer.param_groups:
            #                learning_rate = g['lr']
            output = {'Iteration':iteration, 'Epoch':epoch, 'Train Loss': float(train_loss.data),
                      'Validation Loss':val_loss, 'Train Accuracy':train_accuracy, 'Validation Accuracy':val_accuracy, "Learning Rate":learning_rate}
            history_writer.writerow(output)
            break # Just do it once and pop out

        csvfile.flush()

        hostname = "hidden"
        try:
            hostname = os.environ["HOSTNAME"]
        except:
            pass
        print("host: hvd.rank()/hvd.local_rank() are: " + str(hostname) + ": " + str(hvd.rank())+"/"+str(hvd.local_rank()) ) 


    print("end of epoch")
    torch.save(net.state_dict(), 'model-scn3dpi0.pkl')

