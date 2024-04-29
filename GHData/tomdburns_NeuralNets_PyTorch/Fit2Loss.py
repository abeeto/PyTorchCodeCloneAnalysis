"""
Tries to fit a curve to resulting data
"""


import numpy as np
import pickle as pkl
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cfit


FILE = 'Cyclops Results\\Results_Trimmed90_100k_Epsa.pkl'

N = 20000


def pearsonr2(A, B):
    """calcualtes the pearson r2"""
    a, b = np.mean(A), np.mean(B)
    top, bot1, bot2 = [], [], []
    for i, val in enumerate(A):
        x = (val  - a)*(B[i] - b)
        y = (val  - a)**2
        z = (B[i] - b)**2
        top.append(x)
        bot1.append(y)
        bot2.append(z)
    top, bot1, bot2 = sum(top), sum(bot1), sum(bot2)
    r = top / ((bot1*bot2)**0.5)
    return r**2


def mad(A, B):
    """Calculate the mean absolute deviation"""
    isum = []
    for i, v in enumerate(A):
        isum.append(abs(v - B[i]))
    return np.mean(isum)


def reverse_quad(y, a=1, b=1, c=1):
    """Inverted Quadratic Equation"""
    top = 4*a*(y-c) + b**2
    top = top**0.5 - b
    #print('>>:\t', y, top / (2*a))
    #exit()
    return top / (2*a)


def reverse_line(y, m=1, b=1):
    """Inverted Line Function"""
    #print(y, (y-b) / m)
    return (y-b) / m


def main():
    """main"""

    def line_fn(x, m=1, b=1):
        """linear function"""
        return m*x + b

    def curve_fn(x, a=1, c=1, d=1):
        """First a langmuir type curve"""
        top = a*c*x
        bot = 1+(c*x)
        return (top/bot) + d

    def quadratic_fn(x, n=1, v=1, e=1):
        """quadratic function"""
        one = -1*n*(x**2)
        two = v*x
        return one + two + e

    def log_fn(x, A=1, B=1):
        """logarithmic function"""
        return A+B*np.log10(x)

    try:
        x, y, r = pkl.load(open(FILE, 'rb'))
    except:
        x, y, r, pY, Y = pkl.load(open(FILE, 'rb'))

    nY = np.linspace(min(Y), max(Y), 500)

    # Try a Linear Fit
    lout  = cfit(line_fn, np.array([i[0] for i in Y], dtype='float'),
                          np.array([i[0] for i in pY], dtype='float'))
    m, b  = lout[0][0], lout[0][1]
    cLine = line_fn(nY, m=m, b=b)
    lpY   = line_fn(pY, m=m, b=b)
    pr    = pearsonr2(lpY, Y)
    pmad  = mad(pY, Y)
    lmad  = mad(lpY, Y)
    print('OG MAD:', pmad)
    print('LN MAD:', lmad)

    ## Try a Langmuir Fit
    #cout   = cfit(curve_fn, np.array([i[0] for i in Y], dtype='float'),
    #                        np.array([i[0] for i in pY], dtype='float'))
    #a, c, d = cout[0][0], cout[0][1], cout[0][2]
    #cCurve = curve_fn(nY, a=a, c=c, d=d)
    #cpY = curve_fn(Y, a=a, c=c, d=d)
    #lr = pearsonr2(pY, cpY)

    # Try a Quadratic Fit
    qout    = cfit(quadratic_fn, np.array([i[0] for i in Y], dtype='float'),
                              np.array([i[0] for i in pY], dtype='float'))
    n, v, e = qout[0][0], qout[0][1], qout[0][2]
    qCurve  = quadratic_fn(nY, n=n, v=v, e=e)
    qpY     = quadratic_fn(pY, n=n, v=v, e=e)
    qr      = pearsonr2(qpY, Y)

    # try a logarithmic fit
    pout   = cfit(log_fn, np.array([i[0] for i in Y], dtype='float'),
                          np.array([i[0] for i in pY], dtype='float'))
    A, B   = pout[0][0], pout[0][1]
    LCurve = log_fn(nY, A=A, B=B)
    LpY    = log_fn(pY, A=A, B=B)
    Lqr    = pearsonr2(LpY, Y)

    plt.subplot(131)
    plt.plot(x, y)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(132)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.hexbin(Y, pY, bins='log', mincnt=1, cmap='jet')
    r = pearsonr2(np.array([i[0] for i in Y], dtype='float'), np.array([i[0] for i in pY], dtype='float'))
    plt.plot([0., 330.], [0., 330.], color='k', linestyle='--', label='R$^2$=%.2f' % r)
    plt.xlim(min(Y), max(Y))
    plt.ylim(min(pY), max(pY))

    #plt.plot(nY, cCurve, label='Langmuir = %.2f' % lr)
    plt.plot(nY, cLine, label='Line = %.2f' % pr, color='r', linestyle=':')
    plt.plot(nY, qCurve, label='Quadratic = %.2f' % qr, color='g', linestyle=':')
    plt.plot(nY, LCurve, label='Logarithmic = %.2f' % Lqr, color='g', linestyle=':')
    plt.legend()

    plt.subplot(133)
    #fY = np.array([reverse_quad(i[0], a=n, b=v, c=e) for i in pY])
    fY = np.array([reverse_line(i[0], m=m, b=b) for i in pY])
    plt.hexbin(np.array([i[0] for i in Y], dtype='float'), fY, bins='log', mincnt=1, cmap='jet')
    r = pearsonr2(np.array([i[0] for i in Y], dtype='float'), np.array([i for i in fY], dtype='float'))
    plt.plot([0., 330.], [0., 330.], color='k', linestyle='--', label='R$^2$=%.2f' % r)
    plt.xlabel('Actual')
    plt.ylabel('Predicted + Linear Correction')
    plt.xlim(min(Y), max(Y))
    plt.ylim(min(pY), max(pY))
    plt.legend()
    plt.show()


if __name__ in '__main__':
    main()
