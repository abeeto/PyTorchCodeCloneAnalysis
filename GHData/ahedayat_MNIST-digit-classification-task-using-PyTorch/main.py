from net.nets import FFNetwork
from utils.activations import Relu, Sigmoid, Identity, Softplus, Leaky_Relu
from utils.losses import MSE_loss,CrossEntropy,Hinge_loss
from dataloader import DataLoader
from utils.util import plot_loss, plot_accuracy, plot_accuracy_test, plot_var_accuracy
import matplotlib.pyplot as plt
import datetime
from utils.regularization import L2_Reg
from mpl_toolkits import mplot3d
import numpy as np

def printTest(num_correct,num_validation = 10000):
    print('correct results : {}/{}({})%'.format(num_correct,num_validation,f'{((num_correct/num_validation)*100):.2f}'))


epoch = 4
lr = 0.001
layers = [784,200,10]

activations = [Relu,Sigmoid]
loss = MSE_loss()
momentums = [0.99,0.9,0.5,0]
# momentums = [0.5]
num_batch_lr_sample_per_momentum = 4

lr_samples = np.linspace(0.001,0.01,num=num_batch_lr_sample_per_momentum).tolist()

nets = list()
nets_type = list()
batch_lrs = list()

data_loader = DataLoader('./dataset/', pca_n_components = None , normalize = False)

for ix in range(num_batch_lr_sample_per_momentum):
    batch_lrs.append( (100+10*ix, lr_samples[ix]) )

for momentum in momentums:
    nets.append( FFNetwork(layers,activations,loss,lr=lr,momentum=momentum, correct_score=1, incorrect_score=0) )
    nets_type.append('Momemntum = {}'.format(momentum))

test_accuracies_after = list()
batches = list()
lrs = list()

for (net,net_type) in zip(nets,nets_type):
    current_batches = list()
    current_lrs = list()
    current_test_accuracies_after = list()
    for batch_size, lr in batch_lrs:
        net.lr = lr
        current_batches.append(batch_size)
        current_lrs.append(lr)

        print('-'*60)
        print(net)

        net_test_accuracy_before = net.test_net(data_loader.test_data)

        losses,val_accuracies_net = net.train(data_loader, epoch, batch_size)

        net_test_accuracy_after = net.test_net(data_loader.test_data)

        current_test_accuracies_after.append( net_test_accuracy_after )

    test_accuracies_after.append( current_test_accuracies_after )
    batches.append(current_batches)
    lrs.append(current_lrs)

print('lrs : {}'.format(len(lrs)))
if len(lrs)>0:
    print('lrs[0] : {}'.format(len(lrs)))


print('batches : {}'.format(len(batches)))
if len(lrs)>0:
    print('batches[0] : {}'.format(len(batches)))


print('test_accuracies_after : {}'.format(len(test_accuracies_after)))
if len(lrs)>0:
    print('test_accuracies_after[0] : {}'.format(len(test_accuracies_after)))

for ix,momentum in enumerate(momentums):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xdata = np.array(lrs[ix])
    ydata = np.array(batches[ix])
    zdata = np.array(test_accuracies_after[ix])
    print('ydata.shape : {}'.format(len(ydata)))
    print('xdata.shape : {}'.format(len(xdata)))
    print('ydata.shape : {}'.format(len(ydata)))
    ax.scatter3D(xdata,ydata,zdata, c=zdata)

plt.show()