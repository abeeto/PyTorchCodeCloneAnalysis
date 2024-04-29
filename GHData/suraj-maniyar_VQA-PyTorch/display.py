import matplotlib.pyplot as plt
import pickle

def read(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

ta = read('results/train_accuracy_vgg16.pkl')
tl = read('results/train_loss_vgg16.pkl')
va = read('results/val_accuracy_vgg16.pkl')
vl = read('results/val_loss_vgg16.pkl')


plt.subplot(1,2,1)
plt.title('Cross Entropy Loss (VGG16)')
plt.ylim(0,600)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
tl, = plt.plot(tl, label='Train Loss')
vl, = plt.plot(vl, label='Val Loss')
plt.legend(handles=[tl, vl]) 


plt.subplot(1,2,2)
plt.title('Accuracy (VGG16)')
plt.ylim(0,101)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
ta, = plt.plot(ta, label='Train Acc')
va, = plt.plot(va, label='Val Acc')
plt.legend(handles=[ta, va])
 
plt.show()
