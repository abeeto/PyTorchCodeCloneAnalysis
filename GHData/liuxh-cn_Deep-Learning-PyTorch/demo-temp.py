import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure

figure.Figure ;
# 产生测试数据
x = np.arange(1, 10)
y = x
fig = plt.figure()
ax1 = fig.add_subplot(121)

# title
ax1.set_title('Scatter Plot')
# x axis
plt.xlabel('X')
# y axis
plt.ylabel('Y')
# scatter plot
p1 = plt.scatter(x[0:2], y[0:2], c='r', marker='o')
p2 = plt.scatter(x[2:], y[2:], c='g', marker='D')
# legend
plt.legend([p1, p2], ['label', 'label1'], loc='lower right', scatterpoints=1)

ax2 = fig.add_subplot(122)
ax2.set_title('test')
# x axis
plt.xlabel('X')
# y axis
plt.ylabel('Y')
# scatter plot
p1 = plt.scatter(x[0:2], y[0:2], c='r', marker='o')
p2 = plt.scatter(x[2:], y[2:], c='g', marker='D')
# legend
plt.legend([p1, p2], ['label', 'label1'], loc='lower right', scatterpoints=1)

# show
plt.show()
