
import torch as T
import torchvision as TV
import scipy.misc as SPM
import sys
import numpy as NP
import pickle
import sh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as PL

imgs = []
with open('imagenet_labels2', 'rb') as f:
    classes = pickle.load(f, encoding='latin1')

sh.mkdir('-p', sys.argv[1])

for filename in sys.argv[2:]:
    x = SPM.imread(filename, mode='RGB')
    x = SPM.imresize(x, (224, 224))
    x = NP.expand_dims(x, 0)

    x = x.transpose([0, 3, 1, 2]) / 255.
    imgs.append(x)
x = NP.concatenate(imgs)
old_x = NP.copy(x)
norm = TV.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#x = T.autograd.Variable(T.stack([norm(T.FloatTensor(_)) for _ in x]))
x = T.autograd.Variable(norm(T.FloatTensor(x)))
vgg19 = TV.models.vgg19(pretrained=True)
vgg19.eval()
y = vgg19(x).data.numpy()
for i, idx in enumerate(y.argmax(axis=1)):
    print(classes[idx])
    PL.imshow(old_x[i].transpose(1, 2, 0))
    PL.title(classes[idx])
    PL.savefig('%s/%d.png' % (sys.argv[1], i))
    PL.close()
