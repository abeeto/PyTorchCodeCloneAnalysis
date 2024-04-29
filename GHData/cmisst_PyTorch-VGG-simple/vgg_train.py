from vgg import *

try:
    confusion_matrices = np.load('vgg.npy')
except IOError:
    confusion_matrices = []
    for i in range(10,101, 2):
        argv=['vgg.py','retrain']
        argv.append(str(i))
        confusion_matrices.append(main(argv))
        np.save('vgg.npy', np.array(confusion_matrices))

print(np.trace(confusion_matrices, axis1=1, axis2=2))

