import matplotlib.pyplot as plt

li = [ 0.0000,   1.0684,   0.3949,  -1.8095,  -2.7279,  -5.9943,  -7.6750,
          -7.2702,  -8.5946,  -9.0466,  -8.1321,  -7.3151,  -6.0540,  -7.7078,
          -5.8979,  -4.7726,  -5.3123,  -7.5358,  -7.6888,  -6.3830]
label = -6.2517
axis = [0,5,10,15,20]
xmin = 0
xmax = 19
plt.figure('出力層の様子')
plt.plot(li, label='output')
plt.xticks(axis)
plt.hlines(label, xmin, xmax, linestyle='dashdot', label='label', color='red')
plt.xlim(xmin, xmax)
plt.xlabel('TimeStep')
plt.ylabel('Output[deg/s]')
plt.legend()
# plt.title('出力層の様子')
plt.show()
# -6.2517
