from ldrd_onsensor.synthesis import *
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import numpy as np
import torch
import csv
import glob
import pdb

global_Nclass = 5
global_n_iterations_per_epoch = 10
global_batch_size = 32
plotted = np.zeros((5),dtype=bool)

def gen_waveform(n_iterations_per_epoch=10, mini_batch_size=12):

    Nclass = global_Nclass

    x = np.ndarray(shape=(mini_batch_size, 1, 200, 200, 200))
    y = np.ndarray(shape=(mini_batch_size))
    datapaths = "/microboone/ec/dune-root2/*/*ana*"

    files = [ i for i in glob.glob(datapaths)]
    truth = ["e-", "pi0", "gamma","p","mu"]

    
    for iteration in range(n_iterations_per_epoch):
        for mini_batch in range(mini_batch_size):

            ind_file = np.random.randint(Nclass,size=1)[0]
            current_file = np.load(files[ind_file])
            current_index = np.random.randint(current_file.shape[0], size=1)[0]

            labelvec = np.zeros(Nclass)
            labelvectmp = np.array(labelvec)
            ptype = files[ind_file].split("/")[-1].split("_")[-1].split(".")[0]

            labelvectmp[truth.index(ptype)] = 1

            # sdTPC==3 is the DUNE TPC in 6x2x1 geometry in which most of these events are fully contained
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
#            H,edges = np.histogramdd(dataT,bins=(250,600,250),range=((0.,250.),(0.,600.),(0.,250.)),weights=weights)
            H,edges = np.histogramdd(dataT,bins=(200,200,200),range=((0.,250.),(0.,600.),(0.,250.)),weights=weights)

            (xmax,ymax,zmax) = np.unravel_index(np.argmax(H,axis=None),H.shape)
            # Crop this back to central 2mx2mx2m about max activity point
            ix = np.maximum(xmax-100,0); ix = np.minimum(ix,250-200)
            iy = np.maximum(ymax-100,0); iy = np.minimum(iy,600-200)
            iz = np.maximum(zmax-100,0); iz = np.minimum(iz,250-200)


            if not plotted[ind_file]:
                plotted[ind_file] = True
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(1)
                fig.clf()
                ax = Axes3D(fig)
                xp,yp,zp = H[ix:ix+200,iy:iy+200,iz:iz+200].nonzero()            
                ax.scatter(xp,yp,zp, s=3) # s=np.log10(weights/np.mean(weights))+3 )
                ax.set_xlabel('X (drift)')
                ax.set_ylabel('Y (up-down)')
                ax.set_zlabel('Z (beam)')
                ax.set_xlim(0.,250.)
                ax.set_ylim(0.,600.)
                ax.set_zlim(0.,250.)
                plt.draw()
                plt.savefig('scat_xyz_'+str(ptype)+'_'+str(current_index)+'.png')
                plt.close()


            
#            x[mini_batch][0] = H[ix:ix+200,iy:iy+200,iz:iz+200] # The 0th element is for 1st (only) layer.
#            y[mini_batch] = labelvectmp
            x[mini_batch][0] = H
            y[mini_batch] = truth.index(ptype)

            

        yield (Variable(torch.from_numpy(x).float().cuda()),
               Variable(torch.from_numpy(y).long().cuda()))


def accuracy(target, output):
    target = target.cpu().data.numpy()
    output = output.cpu().data.numpy()
    n_total = len(target)
    n_correct = 0.0
    for index in range(n_total):
        softmax = output[index]
        yhat = np.argmax(softmax)
#        y = np.argmax(target[index])
        y = target[index]
        correct = ( y == yhat )
        n_correct += int(correct)
    return float(n_correct) / float(n_total)


class Net(nn.Module):

    def __init__(self):

# obvi, change all this.
        
        super(Net, self).__init__()
        self.flat_features = 46656 # product of final x.shape dimensions, except for first one, which is batch-size
        self.relu = nn.LeakyReLU() 

        self.ll = nn.Softmax(dim=1)

        self.conv0p = nn.MaxPool3d( 2, 2)
        self.fc00 = nn.Linear(100, 1024)
        self.output_layer00 = nn.Linear(1024,global_Nclass)        
        
        self.conv0n = nn.BatchNorm3d( 1 )
        
        self.conv1 = nn.Conv3d(1 , 16, 3, stride=(1,1,1))
        self.conv2 = nn.Conv3d(16, 16, 3, stride=(1,1,1))
        self.conv3 = nn.Conv3d(16, 32, 3, stride=(1,1,1))
        self.conv4 = nn.Conv3d(32, 32, 3, stride=(1,1,1))
        self.conv5 = nn.Conv3d(32, 64, 3, stride=(1,1,1))

        self.pool1 = nn.MaxPool3d( 3, 2)

        self.fc1 = nn.Linear(self.flat_features, 512)
        self.drop = nn.Dropout(p=0.5)
        
#        self.output_layer = nn.Linear(self.flat_features,5)
        self.output_layer = nn.Linear(512,global_Nclass)

    def forward(self, x):

        x = self.conv0p(x)
        x = self.conv0n(x)
        x = self.conv1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)        
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = x.view(-1, self.flat_features)  # flatten before using Linear
        x = self.fc1(x)
#        x = self.drop(x)

        x = self.output_layer(x)
        x = self.ll(x)
        return x

net = Net()
net = net.cuda()

# load existing weights
net.load_state_dict(torch.load('./model-chinanormal.pkl'))
#net.eval()

#loss = nn.BCELoss().cuda()
loss = nn.CrossEntropyLoss().cuda()
learning_rate = 0.001 # 0.010
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
lr_step = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) # lr drops to lr*0.9^N after 5N epochs
val_gen = gen_waveform(n_iterations_per_epoch=1,mini_batch_size=5)

with open('history.csv','w') as csvfile:
    fieldnames = ['Iteration', 'Epoch', 'Train Loss',
                  'Validation Loss', 'Train Recall', 'Validation Recall', "Learning Rate"]
    history_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    history_writer.writeheader()

    for epoch in range(400): # (1000):
        train_gen = gen_waveform(n_iterations_per_epoch=global_n_iterations_per_epoch,mini_batch_size=global_batch_size)
        lr_step.step()
        for iteration, minibatch in enumerate(train_gen):
            net.train()
            optimizer.zero_grad()
            x, y = minibatch

            yhat = net(x)

            train_loss = loss(yhat, y) 
            train_loss.backward()
            optimizer.step()
            train_accuracy = accuracy(y, yhat)
            net.eval()

            print("Epoch: {}, Iteration: {}, Loss: [{:.4g}], Accuracy: [{:.4g}]".format(epoch, iteration,float(train_loss.data[0]), train_accuracy))

            
            try:
                x,y = next(val_gen)
            except StopIteration:
                val_gen = gen_waveform(n_iterations_per_epoch=1,mini_batch_size=5)
                x,y = next(val_gen)
                print("re-upping the validation generator")
                    
            yhat = net(x)
            val_loss = loss(yhat, y).data[0]
            val_accuracy = accuracy(y, yhat)


            print("Epoch: {}, Iteration: {}, Loss: [{:.4g},{:.4g}], Accuracy: [{:.4g},{:.4g}]".format(epoch, iteration,float(train_loss.data[0]), val_loss, train_accuracy, val_accuracy))
            print ("prediction, truth: " + str(yhat) + " " + str(y))
            
            #                if (iteration%1 ==0) and (iteration>0):

            for g in optimizer.param_groups:
                learning_rate = g['lr']
            output = {'Iteration':iteration, 'Epoch':epoch, 'Train Loss': float(train_loss.data[0]),
                      'Validation Loss':val_loss, 'Train Recall':train_accuracy, 'Validation Recall':val_accuracy, "Learning Rate":learning_rate}
            history_writer.writerow(output)
            csvfile.flush()
        print("end of epoch")
        torch.save(net.state_dict(), 'model-chinanormal.pkl')
