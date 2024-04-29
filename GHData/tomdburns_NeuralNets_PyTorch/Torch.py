"""
Runs a fitting with PyTorch

v2 is the version of the code that will be used for Network Optimizations

v3 - this update incorporates the optimizer's control of the Learning Rate
       and the dropoff probability
"""


import torch
import numpy as np
import pandas as pd
import pickle as pkl
import torch.nn as nn
from tqdm import tqdm
from glob import glob
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from scipy.optimize import curve_fit as cfit
from sklearn.preprocessing import StandardScaler


# Initial Data Parameters
OMIT    = ['mof', 'EPSA', 'Etot', 'EComp', 'Purity', 'Recovery', 'Productivity', 'DoE Flag']
#TARG    = 'Productivity'
#TARG = 'EPSA'
TARG = 'DoE Flag'
#INFILE  = 'combined_ML_Names.csv'
#INFILE  = 'Subsets\\DoE_Only.csv'
INFILE  = 'DoE_Classified_ALLDATA.csv'
TEST    = 0.1
DEV     = 0.1
DP_PROB = 0.1
LAYERS  = 4

# Are there any features that need to be log transformed?
LOG  = (True, ['b0_c', 'd0_c', 'q1_c', 'q2_c',
               'b0_n', 'd0_n', 'q1_n', 'q2_n',
               'Pint', 'Plow', 'tblow'])
OFFS = ('LogOffsets.pkl', 100) # offset reference and magnitude

# Identify device for fitting
use_cuda = torch.cuda.is_available()
device   = torch.device('cuda:0' if use_cuda else 'cpu')
#print('Device:', device)

TRIM   = (False, 330)

# Optimizer Parameters
loss_fn = nn.L1Loss()
#loss_fn = nn.MSELoss()
LR      = 0.0001 #10**(-3.5)
AMSGRAD = False
WEIGHT_DECAY = 0
num_epochs   = 20000
SAVE         = True


class Net(nn.Module):
    """Neural Net Class"""

    def __init__(self, n_features=20, nhidden1=20, nhidden2=20,
                 nhidden3=20, nhidden4=20, act_fn=F.relu, dp_prob=0.1):
        super().__init__()
        # Initializa all net specific parameters
        self.n_features = n_features
        self.nhidden1   = nhidden1
        self.nhidden2   = nhidden2
        self.nhidden3   = nhidden3
        self.nhidden4   = nhidden4
        self.act_fn     = act_fn   # Define the non-linear activation function
        self.dp_prob    = dp_prob  # Dropout layer probability

        # Initialize Layers of the Net:
        # Note:  First Fully Connected Layer (fc1) where nn.Learnear has (X, Y)
        #        and X = number of features fed into payer, Y = number of features returned
        #        by layer (X=input, Y=output)
        self.fc1     = nn.Linear(n_features, nhidden1)
        self.fc2     = nn.Linear(nhidden1,   nhidden2)
        self.fc3     = nn.Linear(nhidden2,   nhidden3)
        self.fc4     = nn.Linear(nhidden3,   nhidden4)
        self.output  = nn.Linear(nhidden4,   1)
        self.dropout = nn.Dropout(p=dp_prob)

    def forward(self, x):
        """forward propagation in the NN"""
        if device == 'cuda:0':
            x.cuda(device)
        x = self.dropout(self.act_fn(self.fc1(x)))
        x = self.dropout(self.act_fn(self.fc2(x)))
        x = self.dropout(self.act_fn(self.fc3(x)))
        x = self.dropout(self.act_fn(self.fc4(x)))
        x = self.act_fn(F.relu(self.output(x)))
        return x


def import_data():
    """imports the raw data"""
    raw = pd.read_csv(INFILE)
    if TRIM[0]:
        raw = raw.loc[raw[TARG] <= TRIM[1]]
    cols = []
    for col in [c for c in raw]:
        if col in OMIT:
            continue
        cols.append(col)
    cols.append(TARG)
    if LOG[0]:
        offsets = {}
        for col in LOG[1]:
            msub   = raw.loc[raw[col] > 0]
            mini   = min(np.array(msub[col]))
            maxi   = max(np.array(raw[col]))
            offset = mini / OFFS[1]
            if min(np.array(raw[col])) > 0:
                disto = np.log10(np.array(raw[col]))
                offsets[col] = (False, offset)
            else:
                disto = np.log10(np.array(raw[col]) + offset)
                offsets[col] = (True, offset)
            raw[col] = disto
        pkl.dump(offsets, open(OFFS[0], 'wb'))
        print(OFFS[0], 'written.')
    rawd = raw[cols]
    raw  = None
    rawd = shuffle(rawd)
    data = np.array(rawd)
    rawd = None
    return data[:,:-1], data[:,-1]


def preprocess_data():
    """preprocesses the data"""
    X, Y = import_data()

    # Create my subsets
    N, W = X.shape[0], X.shape[1]
    cut2 = int(DEV*N)
    cut  = int((TEST + DEV)*N)
    TrainX, TrainY = X[:-cut,:], Y[:-cut]
    DevX, DevY     = X[-cut:-cut2,:], Y[-cut:-cut2]
    TestX, TestY   = X[-cut2:,:], Y[-cut2:]
    X, Y = None, None

    # Scale my subsets
    scaler = StandardScaler()
    TrainX = scaler.fit_transform(TrainX)
    DevX   = scaler.transform(DevX)
    TestX  = scaler.transform(TestX)
    pkl.dump(scaler, open('Scaler_%s_N%i.pkl' % (TARG, LAYERS), 'wb'))
    print('Scaler_%s_N%i.pkl' % (TARG, LAYERS), 'written.')

    # Convert to and format Tensors
    TrainX = torch.from_numpy(TrainX).float()
    TrainX = TrainX.to(device)
    TrainY = TrainY.reshape(TrainY.shape[0], 1)
    TrainY = torch.from_numpy(TrainY).float()
    TrainY = TrainY.view(-1, 1)
    TrainY = TrainY.to(device)

    DevX = torch.from_numpy(DevX).float()
    DevX = DevX.to(device)
    DevY = DevY.reshape(DevY.shape[0], 1)
    DevY = torch.from_numpy(DevY).float()
    DevY = DevY.view(-1, 1)
    DevY = DevY.to(device)

    TestX = torch.from_numpy(TestX).float()
    TestX = TestX.to(device)
    TestY = TestY.reshape(TestY.shape[0], 1)
    TestY = torch.from_numpy(TestY).float()
    TestY = TestY.view(-1, 1)
    TestY = TestY.to(device)

    return TrainX, TrainY, DevX, DevY, TestX, TestY


def weights_init_uniform(m):
    """initializes the weights"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        y = np.sqrt(1/n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def convergence_check(epochs, losses, check):
    """checks for convergence"""

    erange    = check # How many points will be considered
    converged = False

    def line(x, m=1, b=1):
        """line function"""
        return m*x+b

    # Set convergen threshold values
    #lm = -.05 # Low value of slope for it to be considered
    #hm = .04  # High value for slope
    lm = -.00001 # Low value of slope for it to be considered
    hm = .00001  # High value for slope

    # preprocess the data
    echeck = epochs[-erange:]
    scaler = StandardScaler()
    echeck = scaler.fit_transform([[i] for i in echeck])
    echeck = np.array([i[0] for i in echeck])
    lcheck = losses[-erange:]
    relval = np.mean(losses[-erange:]) # what value will be used for relative

    # Perform a linear fit to get an approximation of trend
    lout  = cfit(line, np.array(echeck),
                       np.array(lcheck/relval))
    m, b  = lout[0][0], lout[0][1]
    lX    = np.linspace(min(echeck), max(echeck), 500)
    lY    = line(lX, m=m, b=b)
    pY    = line(np.array([i for i in echeck]), m=m, b=b)
    stdval = np.std(losses[-erange:])

    # Get the relative values
    rm = 100* m / relval
    rstdval = 100* stdval / relval

    # Check for convergence
    if rm >= lm and rm <= hm:
        #print(m, rm)
        converged = True

    #return converged
    return False


def fit_network(TrainX, TrainY, DevX, DevY, TestX, TestY,
                lr=LR, hn1=20, hn2=20, hn3=10, hn4=20, dp_prob=DP_PROB,
                amsgrad=AMSGRAD, weight_decay=WEIGHT_DECAY):
    """Performs the NN fitting"""
    global use_cuda, device

    #print('>', lr, hn1, hn2, hn3, hn4, dp_prob)

    # Identify device for fitting
    use_cuda = torch.cuda.is_available()
    device   = torch.device('cuda:0' if use_cuda else 'cpu')
    #print('Device:', device)
    # Initialize the Network
    net = Net(n_features=TrainX.shape[1], nhidden1=hn1, nhidden2=hn2,
              nhidden3=hn3, nhidden4=hn4, dp_prob=dp_prob)
    net.apply(weights_init_uniform)
    net = net.to(device)

    # convergence parameters
    check     = 1000  # Check frequency and range
    minchk    = 5000  # mininum epochs to run before check
    converged = False

    # Run the Training
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=amsgrad, weight_decay=weight_decay)
    epX, epY, epR = [], [], []

    #print("Starting Fitting.")
    for epoch in tqdm(range(num_epochs), leave=False):
        #print('Training on Epoch %i' % epoch)
        train_loss, valid_loss = 0, 0

        net.train()
        optimizer.zero_grad()

        # Forward Propagation
        y_train_predict = net(TrainX)

        # Loss Function
        loss = loss_fn(y_train_predict, TrainY)

        # Backward Propagation
        loss.backward()

        # Weight Opt
        optimizer.step()
        train_loss = float(loss) #.detatch())

        # Evaludate the fitting
        net.eval()
        y_dev_predict = net(DevX)
        DevY = DevY.to(device)
        loss = loss_fn(y_dev_predict, DevY)
        valid_loss = float(loss) #.detatch())
        #print('\tDev Loss =', valid_loss)
        epX.append(epoch)
        epY.append(valid_loss)
        if epoch > minchk and epoch % check == 0:
            #print(epoch, check, epoch%check)
            converged = convergence_check(epX, epY, check)
        if converged:
            break
    #print("\tDone")

    if SAVE:
        net.eval()
        y_test_pred = net(TestX).detach()
        TestY = TestY.cpu().numpy()
        y_test_pred = y_test_pred.cpu().numpy()
        pkl.dump((epX, epY, epR, y_test_pred, TestY), open('NewResults_v3_Classifier.pkl', 'wb'))
        model_ini = 'Models\\NN_Model_%iLayers_Targ_%s' % (LAYERS, TARG)
        model_file = model_ini + '_%i.pkl' % len(glob('%s_*.pkl' % model_ini))
        torch.save(net.state_dict(), model_file)

    return min(epY)


def main():
    """main"""
    # Import and prepare the data

    # =========================================================================
    # Parameters to run
    # =========================================================================
    lr = 10**-2.75
    h1 = 50
    h2 = 50
    h3 = 50
    h4 = 50
    dp = 0.
    # =========================================================================

    print('Predicting  :', TARG)
    print('Epochs      :', num_epochs)
    print('Save Results:', SAVE)
    print('Running     :', lr, h1, h2, h3, h4, dp, '\n')
    pkl.dump((lr, h1, h2, h3, h4, dp, AMSGRAD, WEIGHT_DECAY),
             open('%s_%iLayer_NNHyperParameters.pkl' % (TARG, LAYERS), 'wb'))

    print("Importing Data:")
    TrainX, TrainY, DevX, DevY, TestX, TestY = preprocess_data()

    print('\t>: Done.\n\nFitting NN:\n')
    best = fit_network(TrainX, TrainY, DevX, DevY, TestX, TestY,
                       lr=lr, hn1=h1, hn2=h2, hn3=h3, hn4=h4, dp_prob=dp,
                       amsgrad=AMSGRAD, weight_decay=WEIGHT_DECAY)
    #pkl.dump((TrainX.detatch(), TrainY.detatch(), DevX.detatch(),
    #          DevY.detatch(), TestX.detatch(), TestY.detatch()),
    #          open('DataSet_%s_N%i.pkl' % (TARG, LAYERS), 'wb'))
    print('\t>: Done.')


if __name__ in '__main__':
    main()
