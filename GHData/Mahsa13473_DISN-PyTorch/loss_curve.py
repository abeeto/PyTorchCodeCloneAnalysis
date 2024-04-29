import os
import matplotlib.pyplot as plt

data_dir = '/local-scratch/mma/DISN/version1'

loss = []
s = 0
with open(os.path.join(data_dir, 'log.txt'), 'r') as f:
    for line in f:
        if s%2 == 1:
            #print(float(line))
            loss.append(float(line))
        s = s+1

plt.plot(loss)
plt.ylabel('train_loss')
plt.show()
