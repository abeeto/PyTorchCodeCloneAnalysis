from ldrd_onsensor.synthesis import *
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import numpy as np
import torch
import csv
import glob
import uresnet2d
import time
import pdb
from matplotlib.mlab import PCA
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
## weird complaints about mkl trying to use below
#from scipy import linalg
from scipy.optimize import minimize
from scipy.optimize import leastsq


global_Nclass = 5
global_n_iterations_per_epoch = 10
global_batch_size = 3 # 32
vox = 2 # int divisor of 250 and 600 and 200. Cubic voxel edge size in cm.
nvox = int(200/vox) # num bins in each dimension 
plotted = np.zeros((8),dtype=bool) # 8 files

# next 4 are globals
fit_pi0 = True
Epi = np.empty([1])
mpi_pca = np.empty([1])
mpi_constrained = np.empty([1])
Epi_cuts = np.empty([1])
mpi_pca_cuts = np.empty([1])
mpi_constrained_cuts = np.empty([1])


def eigenFunc( ix,iy,iz, H):
    indices = np.where(H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox])
    arr = np.array((np.arange(ix,ix+nvox)[indices[0]],np.arange(iy,iy+nvox)[indices[1]],np.arange(iz,iz+nvox)[indices[2]]))
    rep = np.array(H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox][indices],dtype='int64')
    xyz = np.repeat(arr,rep,axis=1)  # This is a 3xN matrix with each row of N one x,y,z coordinate wtd by energy deposition .
    xyz = arr # effectively don't wt by population of charge created in this voxel by this gamma, afterall
    pca = None
    try:
        pca = PCA(xyz.T)
    except:
        pass
    return pca

    '''
# The linalg calls give mkl runtime errors
    R = np.cov(xyz.T, rowvar=False)
    evals, evecs = linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    return evals,evecs
'''


def energyGamma( ix,iy,iz, Hg, H):
    indices = np.where(Hg[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox])
    E = H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox][indices].sum()
    return E

def gen_waveform(n_iterations_per_epoch=10, mini_batch_size=6):

    Nclass = global_Nclass

    x = np.ndarray(shape=(mini_batch_size, 1, nvox, nvox, nvox))
    y = np.ndarray(shape=(mini_batch_size, ))
    datapaths = "/microboone/ec/dune-root3/pi0/*ana*"

    elec2MeV = 42700   # mip electrons/MeV
    # birks = 0.65 # rough-rough recombination
    birks = 1. # LArSoft has not imposed this, so don't correct for it. 19-July-2018
    
    files = [ i for i in glob.glob(datapaths)]
    truth = ["pi0"]
    labels = torch.FloatTensor([mini_batch_size, nvox, nvox, nvox])
    weight = torch.FloatTensor([mini_batch_size, nvox, nvox, nvox])
    global fit_pi0
    
    for iteration in range(n_iterations_per_epoch):
        for mini_batch in range(mini_batch_size):

            ind_file = np.random.randint(len(files)-1,size=1)[0] # save the last file for validation
            current_file = np.load(files[ind_file])
            current_index = np.random.randint(current_file.shape[0], size=1)[0]
            dim2 = np.array((mini_batch_size, nvox,nvox))
            np_labels  = np.zeros( dim2, dtype=np.int )
            np_weights = np.zeros( dim2, dtype=np.float32 )

            # Form labels ala H below.
            
            # sdTPC==3 is the DUNE TPC in 6x2x1 geometry in which most of these events are fully contained
            data = np.array((current_file[current_index,]['sdX'][current_file[current_index,]['sdTPC']==3],current_file[current_index,]['sdY'][current_file[current_index,]['sdTPC']==3], current_file[current_index,]['sdZ'][current_file[current_index,]['sdTPC']==3] ))
            dataT = data.T
            
            if dataT.sum() is 0:
                print("Problem! Image is empty for current_index " + str(current_index))
                raise
            
            xmin = np.argmin(dataT[:,0][dataT[:,0]>2])
            ymin = np.argmin(dataT[:,1][dataT[:,1]>2])

            weights=current_file[current_index,]['sdElec'][current_file[current_index,]['sdTPC']==3]
            ##  view,chan,x

            voxels = (int(250/vox),int(600/vox),int(250/vox) ) # These are 2x2x2cm^3 voxels


            ''' There's no longer a desire to histogram anything, as our CCD data will be in its pixels already'''
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


            
            x[mini_batch][0] = H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox] # The 0th element is for 1st (only) layer.
#            y[mini_batch] = labelvectmp
#            x[mini_batch][0] = H
#            y[mini_batch] = truth.index(ptype)
            np_labels[mini_batch] = np.zeros(H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox].shape)
            indx1 = np.where(Hpl1[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox]!=0)
            indx2 = np.where(Hpl2[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox]!=0)
            np_labels[mini_batch][indx1] = 1 # Eg1 pixels. Eg1>Eg2
            np_labels[mini_batch][indx2] = 2 # Eg1 pixels

            np_weights[mini_batch] = np_labels[mini_batch].astype('bool').astype('int')
            np_weights[mini_batch][np_weights[mini_batch]==0] = 1.0/np.prod(Hpl1.shape)/1000.


#            labels[mini_batch] = torch.from_numpy(np_labels[np_labels])
#            weight[mini_batch] = torch.from_numpy(np_weights[np_weights])

            # replace y's truth.index(ptype) with pixel labels array. Further, must add pixel weight array too.

        yield (Variable(torch.from_numpy(x).float().cuda()),
               Variable(torch.from_numpy(np_labels).long().cuda(),requires_grad=False),
               Variable(torch.from_numpy(np_weights).float().cuda(),requires_grad=False))



        
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
    correct = pred.eq( targetex ) # on gpu
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
        classmat = targetex.eq(int(c)) # elements where class is labeled
        #print "classmat: ",classmat.size()," iscuda=",classmat.is_cuda
        num_per_class[c] = classmat.sum()
        corr_per_class[c] = (correct*classmat).sum() # mask by class matrix, then sum
        total_corr += corr_per_class[c]
        total_pix  += num_per_class[c]
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
    res.append( 100.0*float(total_corr)/float(total_pix) )
    res.append( 100.0*float(corr_per_class[1]+corr_per_class[2])/float(num_per_class[1]+num_per_class[2]) ) # track/shower acc

    return res
                                                            


net = uresnet2d.UResNet(inplanes=16,input_channels=1,num_classes=3,showsizes=True)
# uncomment dump network definition
# print ("net: "+str(net))

Net = net.cuda()

# load existing weights


try:
    print ("Reading weights from file")
    net.load_state_dict(torch.load('./model-uresCCD.pkl'))
    net.eval()
except:
    print ("Failed. Quitting.")
    raise 


#loss = nn.BCELoss().cuda()
#loss = nn.CrossEntropyLoss().cuda()

# create loss function
# Loss Function

# Next two functions taken from Taritree's train_wlarcv1.py
# We define a pixel wise L2 loss

# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

import torch.nn.functional as F    
class PixelWiseNLLLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,weight=None, size_average=True, ignore_index=-100 ):
        super(PixelWiseNLLLoss,self).__init__(weight,size_average)
        self.ignore_index = ignore_index
        self.reduce = False

        self.mean = torch.mean # torch.mean.cuda() fails with 'has no attribute cuda" ....
        
    def forward(self,predict,target,pixelweights):
        """
        predict: (b,c,h,w) tensor with output from logsoftmax
        target:  (b,h,w) tensor with correct class
        pixelweights: (b,h,w) tensor with weights for each pixel
        """
        _assert_no_grad(target)
        _assert_no_grad(pixelweights)
        # reduce for below is false, so returns (b,h,w)
        pixelloss = F.nll_loss(predict,target, self.weight, self.size_average, self.ignore_index, self.reduce)

        return self.mean(pixelloss*pixelweights)

loss = PixelWiseNLLLoss().cuda()

learning_rate = 0.001 # 0.010
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
lr_step = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) # lr drops to lr*0.9^N after 5N epochs
val_gen = gen_waveform(n_iterations_per_epoch=global_n_iterations_per_epoch,mini_batch_size=global_batch_size)

with open('history.csv','w') as csvfile:
    fieldnames = ['Iteration', 'Epoch', 'Train Loss',
                  'Validation Loss', 'Train Accuracy', 'Validation Accuracy', "Learning Rate"]
    history_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    history_writer.writeheader()

    for epoch in range (4000):  # (400)
        train_gen = gen_waveform(n_iterations_per_epoch=global_n_iterations_per_epoch,mini_batch_size=global_batch_size)
        lr_step.step()

        for iteration, minibatch in enumerate(train_gen):
            net.train()
            optimizer.zero_grad()
            x, labels_var, weight_var = minibatch

            yhat = net(x)

            train_loss = loss(yhat, labels_var, weight_var) 
            train_loss.backward()
            optimizer.step()
#            train_accuracy = accuracy(y, yhat)

            train_accuracy = accuracy(yhat.data, labels_var.data, x)    # None)
            net.eval()

            print("Epoch: {}, Iteration: {}, Loss: [{:.4g}], Accuracy: [{:.4g},{:.4g},{:.4g}]".format(epoch, iteration,float(train_loss.data[0]), train_accuracy[0], train_accuracy[1], train_accuracy[2]))
#            print("Epoch: {}, Iteration: {}, Loss: [{:.4g}]".format(epoch, iteration,float(train_loss.data[0])))

            
            try:
                x,y,_ = next(val_gen)
            except StopIteration:
                val_gen = gen_waveform(n_iterations_per_epoch=1,mini_batch_size=global_batch_size)
                x,y,_ = next(val_gen)
                print("re-upping the validation generator")
                    
            yhat = net(x)
            val_loss = loss(yhat, labels_var, weight_var).data[0]
#            val_accuracy = accuracy(y, yhat)
            val_accuracy = accuracy(yhat.data, labels_var.data, x)    # images_var.data)

            print("Epoch: {}, Iteration: {}, Loss: [{:.4g},{:.4g}], *** Train Accuracy: [{:.4g},{:.4g},{:.4g}, ***,  Val Accuracy: [{:.4g},{:.4g},{:.4g}]".format(epoch, iteration,float(train_loss.data[0]), val_loss, train_accuracy[0], train_accuracy[1], train_accuracy[2], val_accuracy[0], val_accuracy[1], val_accuracy[2]))

            
            #                if (iteration%1 ==0) and (iteration>0):

            for g in optimizer.param_groups:
                learning_rate = g['lr']
            output = {'Iteration':iteration, 'Epoch':epoch, 'Train Loss': float(train_loss.data[0]),
                      'Validation Loss':val_loss, 'Train Accuracy':train_accuracy, 'Validation Accuracy':val_accuracy, "Learning Rate":learning_rate}
            history_writer.writerow(output)
            csvfile.flush()

        print("end of epoch")


# The larger N in the below epoch==N the better stats in the mpi0,Epi0 plots, but also slower, cuz fits are slow and run for N epochs.
# Once plot_pi0 is set True the fitting above is shut off and the Network training speeds up significantly.
        plot_pi0 = False
        if epoch==400:
            plot_pi0 = True
        if plot_pi0:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            counts,bin_edges = np.histogram(Epi,bins=np.arange(0.,5000.,100.0))
            bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
            plt.clf()
            plt.hist(Epi,bins=np.arange(0.,5000.,100.0),facecolor='b', alpha=0.75)
            plt.errorbar(bin_centres, counts, yerr=np.sqrt(counts), fmt='o')
            plt.savefig("Epi.png")
            plt.close()


        torch.save(net.state_dict(), 'model-uresCCD.pkl')

        
