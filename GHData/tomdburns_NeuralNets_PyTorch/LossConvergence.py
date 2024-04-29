"""
Code for the development of a loss function of the convergence
in the loss function
"""


import numpy as np
import pickle as pkl
from glob import glob
from math import floor
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit as cfit
from sklearn.preprocessing import StandardScaler


def import_data():
    """imports the loss results"""
    results = glob('Cyclops Results\\ConvTest_NewResults_*20000*.pkl')
    epochs, losses = [], []
    for result in results:
        try:
            x, y, r = pkl.load(open(result, 'rb'))
        except:
            x, y, r, pY, Y = pkl.load(open(result, 'rb'))
        epochs.append(x)
        losses.append(y)
    return epochs, losses


def convergence_check(epochs, losses, col, check):
    """checks for convergence"""

    erange    = check # How many points will be considered
    converged = False

    def line(x, m=1, b=1):
        """line function"""
        return m*x+b

    # Set convergen threshold values
    #lm = -.05 # Low value of slope for it to be considered
    #hm = .04  # High value for slope
    #rv = 0.0 # minimum r squared to check for convergence
    lm = -.00001 # Low value of slope for it to be considered
    hm = .00001  # High value for slope
    rv = 0.0 # minimum r squared to check for convergence

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
    r, p  = pearsonr(pY, lcheck)
    r2    = r**2
    stdval = np.std(losses[-erange:])

    # Get the relative values
    rm = 100* m / relval
    rstdval = 100* stdval / relval

    # Check for convergence
    if rm >= lm and r2 >= rv and rm <= hm:
        print(rm, r2)
        converged = True

    # for testing: plot the results
    plt.plot(echeck, lcheck/relval, color=col)
    plt.plot(lX, lY, color='k', linestyle='--')
    plt.title('Epochs: %i [m=%f, s=%f, R$^2$=%.2f]' % (max(epochs), rm, rstdval, r2))

    return converged


def run_optimization(epochs, losses):
    """simulates the NN optimization. This is basically
    a placeholder function that will be replaced by the
    actual NN optimization in the Torch code
    """
    E, L   = [], []  # Holds the "new" values
    check  = 350     # how often do we check for convergence?
    minchk = 4999    # minimum number of steps to run before starting check
    conv   = False   # Boolean determining whether converged
    maxm   = 20000   # maximum number of epochs
    #N      = int(floor((maxm-check)/check)) + 1

    N, cval, lval = 0, None, None
    for epoch in epochs:
        if epoch > minchk and epoch % check == 0:
            N += 1
        if epoch >= maxm:
            break
    c = ['b', 'r', 'g', 'y', 'c', 'm', 'k']
    verts, ci = [], 0
    sub = 2
    for i, epoch in enumerate(epochs):
        loss = losses[i]
        E.append(epoch)
        L.append(loss)
        if epoch > minchk and epoch % check == 0:
            verts.append(epoch)
            plt.subplot(N,2,sub)
            try:
                col = c[ci]
            except IndexError:
                ci = 0
                col = c[ci]
            ci += 1
            conv = convergence_check(E, L, col, check)
            sub += 2
        if conv:
            print('NN Optimization Converged: %s - %.2f' % (epoch, loss))
            cval, lval = epoch, loss
            break
        if epoch >= maxm:
            break
    if not conv:
        print("Warning: Optimization never converged")
    plt.subplot(1,2,1)
    plt.plot(E, L)

    ci = 0
    for i, vert in enumerate(verts):
        try:
            col = c[ci]
        except IndexError:
            ci = 0
            col = c[ci]
        ci += 1
        plt.axvline(vert, color=col, label='Epoch: %i' % vert)
    plt.legend()
    #plt.show()
    plt.clf()
    return conv, cval, lval


def main():
    """main"""
    epochs, losses = import_data()
    results = []
    for j, epoch in enumerate(epochs):
        #plt.subplot(3,2,j+1)
        print('=' * 50)
        print('Run %i' % (j+1))
        conv, cval, lval = run_optimization(epoch, losses[j])
        results.append((cval, lval))
        #plt.plot(epoch, losses[j])
        #plt.xlabel('Epoch')
        #plt.ylabel('Loss')
    #plt.show()
    print('=' * 50)
    for j, epoch in enumerate(epochs):
        plt.subplot(1,2,j+1)

        plt.plot(epoch, losses[j])
        mini = min(losses[j])
        if results[j][0] is not None:
            plt.axvline(results[j][0], color='k', linestyle='--')
            diff = 100*(results[j][1] - mini) / mini
            plt.title('Relative Difference: %.2f' % diff + ' %')
        else:
            plt.title('No Solution Found.')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
    plt.show()


if __name__ in '__main__':
    main()
