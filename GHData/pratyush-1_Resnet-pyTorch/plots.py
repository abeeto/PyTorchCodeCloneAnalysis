train_losses=[loss.detach().numpy() for loss in train_loss_1]
val_losses=[loss.detach().numpy() for loss in val_loss_1]

import matplotlib.pyplot as plt
plt.autoscale()
plt.plot(range(1,6),train_losses,label='training loss')
plt.plot(range(1,6),val_losses,label='validation loss')
plt.legend(frameon=False)
plt.show()

plt.autoscale()
plt.plot(range(1,6),train_acc_1,label='training acc')
plt.plot(range(1,6),val_acc_1,label='validation acc')
plt.legend(frameon=False)
plt.show()

