from random import sample
import torch
from torch import nn
from torchsummary import summary
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import copy
import cnn
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ' + str(device))
model = cnn.CNN()
model.load_state_dict(torch.load('/Users/sean/Documents/Neubauer_Research/MNIST/cnn_model', map_location=device))
print('Model loaded...')

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = transforms.ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transforms.ToTensor()
)
print('Retrieved dataset...')

# getting a predictions and labels of sample data from test_data
def predict_activation(sample_idx):
    return torch.max(model(test_data[sample_idx][0].unsqueeze(0)).detach())

def predict_label(sample_idx):
    max_ = predict_activation(sample_idx)
    x = model(test_data[sample_idx][0].unsqueeze(0)).detach()[0]
    return (x==max_).nonzero().item()

def plot_sample_image(sample_idx):
    sample_image = test_data[sample_idx][0]
    sample_label = test_data[sample_idx][1] 

    plt.imshow(sample_image.reshape(28, 28))
    plt.show()
    print('Label: ' + str(sample_label))
    print('Prediction: ' + str(predict_label(sample_idx)))
    print('Activation: ' + str(predict_activation(sample_idx)))
    for i in range(len(model(sample_image.unsqueeze(0)).detach().flatten())):
        print(str(i) + ' : ' + str(model(sample_image.unsqueeze(0)).detach().flatten()[i]))

def rho(w,l):  
    return w + [None,0.1,0.0,0.0][l] * np.maximum(0,w)

def incr(z,l): 
    return z + [None,0.0,0.1,0.0][l] * (z**2).mean()**.5+1e-9

def standardize_image(sample_image, mean, std):
    # return (torch.FloatTensor(sample_image[np.newaxis].permute([0, 3, 1, 2])*1) - mean) / std
    return (torch.FloatTensor(sample_image[np.newaxis]*1) - mean) / std

def newlayer(layer, g):
    layer = copy.deepcopy(layer)

    try: layer.weight = nn.Parameter(g(layer.weight))
    except AttributeError: pass

    try: layer.bias = nn.Parameter(g(layer.bias))
    except AttributeError: pass
    return layer

def plot_heatmap(sample_image, sample_label, layers):
    # standardize image
    mean = torch.Tensor([0.1307]).reshape(1,-1,1,1)
    std  = torch.Tensor([0.3081]).reshape(1,-1,1,1)
    X = standardize_image(sample_image=sample_image, mean=mean, std=std)

    # activations at each layer
    A = [X]+[None]*L
    for l in range(L):
        A[l+1] = layers[l].forward(A[l])

    # Relevance scores for the last layer
    scores = np.array(A[-1].data.view(-1))
    print('Relevance Scores:')
    for i, score in enumerate(scores):
        print(str(i) + ' : ' + str(score))

    fig = plt.figure()
    ax0 = sns.barplot(x=np.arange(0, 10), y=scores)
    ax0.set(xlabel='Classes', ylabel='Scores', title='Relevance Scores')
    plt.show()
    path = os.getcwd() + '/PNGS/'
    try: 
        os.mkdir(path) 
    except OSError as error: 
        print(error)  
    fig.savefig(path + str(sample_label) + '_relevance_scores')
    plt.close()
    
    R = [None]*L + [A[-1].data]

    for l in range(1,L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)

        # try using vanilla rho() and incr()
        rho = lambda p: p
        incr = lambda z: z+1e-9

        if isinstance(layers[l],torch.nn.MaxPool2d): 
            layers[l] = torch.nn.AvgPool2d(kernel_size=2)
        if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):
            z = incr(newlayer(layers[l], rho).forward(A[l]))         # step 1
            s = (R[l+1] / z).data                                    # step 2
            (z*s).sum().backward() 
            c = A[l].grad                                            # step 3
            R[l] = (A[l]*c).data                                     # step 4
        else:
            z = incr(newlayer(layers[l],rho).forward(A[l]))        # step 1
            s = (R[l+1]/z).data                                    # step 2
            (z*s).sum().backward(); c = A[l].grad                  # step 3
            R[l] = (A[l]*c).data

    # getting relevance scores for input layer
    A[0] = (A[0].data).requires_grad_(True)

    lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
    hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

    z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
    z -= newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)          # step 1 (b)
    z -= newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)          # step 1 (c)
    s = (R[1]/z).data                                                      # step 2
    (z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
    R[0] = (A[0]*c+lb*cp+hb*cm).data                                       # step 4

    fig, axs = plt.subplots(nrows = 2, ncols=2)
    ax1 = sns.heatmap(R[6][0].sum(axis=0), xticklabels=False, yticklabels=False, ax=axs[0, 0]).set(title='Layer 6')
    ax2 = sns.heatmap(R[4][0].sum(axis=0), xticklabels=False, yticklabels=False, ax=axs[0, 1]).set(title='Layer 4')
    ax3 = sns.heatmap(R[2][0].sum(axis=0), xticklabels=False, yticklabels=False, ax=axs[1, 0]).set(title='Layer 2')
    ax4 = sns.heatmap(R[0][0].sum(axis=0), xticklabels=False, yticklabels=False, ax=axs[1, 1]).set(title='Layer 0')
    plt.show()
    fig.savefig(path + str(sample_label) + '_heatmaps')
    plt.close()

layers = [layer for layer in model.modules() if not isinstance(layer, nn.Sequential) and not isinstance(layer, cnn.CNN) and not isinstance(layer, nn.Softmax)]
print('Layers: ' + str(layers))
L = len(layers)
print('Number of layers: ' + str(L))

# sample image and label
sample_idx = 0
sample_image = test_data[sample_idx][0]
sample_label = test_data[sample_idx][1] 
print('Label: ' + str(sample_label))
print('Prediction: ' + str(predict_label(sample_idx)))
sns.set()
plot_heatmap(sample_image=sample_image, sample_label=sample_label, layers=layers)

# getting a sample image of 8
for i, data in enumerate(test_data):
    if data[1] == 8:
        sample_idx = i
        break
sample_image = test_data[sample_idx][0]
sample_label = test_data[sample_idx][1] 
print('Label: ' + str(sample_label))
print('Prediction: ' + str(predict_label(sample_idx)))
plot_heatmap(sample_image=sample_image, sample_label=sample_label, layers=layers)