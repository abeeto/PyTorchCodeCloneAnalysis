import numpy as np
import matplotlib.pyplot as plt


d = np.loadtxt('comp.csv')
d_ = np.loadtxt('comp_d.csv')

for i in [16,32,64,128,256]:
    print(d[0][i-5], d[1][i-5], d_[1][i-5])

# plt.scatter(d[0], d[1])
# plt.plot(d[0], d[1], label='LSTM')
#
# plt.scatter(d_[0], d_[1])
# plt.plot(d_[0], d_[1], label='LSTM dropout=0.5')
#
# plt.xlim(np.max(d[0]), np.min(d[0]))
#
# plt.xscale('log', basex=10)
# plt.xlabel('# of Params Remaining')
# plt.ylabel('Accuracy')
# plt.grid(True)
# plt.legend()
# plt.show()