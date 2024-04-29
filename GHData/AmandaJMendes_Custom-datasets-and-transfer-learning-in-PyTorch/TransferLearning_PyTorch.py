import torch
import torchvision
import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
DATA
"""
class myMNIST(torch.utils.data.Dataset):
  def __init__(self, split):
    if split == 'test':
        self.dataset = pd.read_csv('mnist_test.csv')
    else:
        self.dataset = pd.read_csv('mnist_train.csv')

    self.transform = torchvision.transforms.Compose([lambda x: (x - np.min(x))/(np.max(x)-np.min(x)),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.ConvertImageDtype(torch.float32),
                                                     torchvision.transforms.Resize((224, 224)),
                                                     lambda x: x.repeat(3, 1, 1),
                                                     torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                                             std = [0.229, 0.224, 0.225])])
    self.dataset.columns = ['label'] + list(range(28*28))

  def __getitem__(self, index):
      row = self.dataset.iloc[index]
      label = row['label']
      img = row.to_numpy()[1:].reshape(28,28)
      img = self.transform(img)
      return img, label
    
  def __len__(self):
      return len(self.dataset)

    
traindata = myMNIST('train')
testdata = myMNIST('test')
n = len(traindata)
nt = int(0.8*n)
nv = n - nt
traindata, valdata = torch.utils.data.random_split(traindata, [nt, nv])

bs = 128

train = torch.utils.data.DataLoader(dataset = traindata, batch_size = bs)
validation = torch.utils.data.DataLoader(dataset = valdata, batch_size = bs)
test = torch.utils.data.DataLoader(dataset = testdata, batch_size = bs)

"""
MODEL
"""

class NewVGG(torch.nn.Module):
    def __init__(self):
        super(NewVGG, self).__init__()
        self.vgg = torchvision.models.vgg19(pretrained=True)
        for p in self.vgg.parameters():
            p.requires_grad = False
            
        n_in = self.vgg.classifier[6].in_features
        self.vgg.classifier = torch.nn.Sequential(*list(self.vgg.classifier.children())[:-1])
        self.new_layers = torch.nn.Sequential(torch.nn.Linear(n_in, 10))

    def forward(self, x):
      out = self.vgg(x)
      out = self.new_layers(out)
      return out
    
myvgg = NewVGG()
myvgg = myvgg.to(device)

def train_model (train, validation, epochs, loss_func, opt):
    print(int(len(traindata)/bs)*'-')
    for i in range(epochs):
        print('Epoch', i)
        myvgg.train()
        correct = 0
        for img, label in train:
            img = img.to(device)
            label = label.to(device)
            
            out = myvgg(img)
            loss = loss_func(out, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            _, out_idx = torch.max(out, 1)
            correct += (label == out_idx).sum().item()
            print('-', end='')
        print('\nTrain acc: ', correct / len(traindata))
        
        correct=0
        myvgg.eval()
        with torch.no_grad():
            for img, label in validation:
                img = img.to(device)
                label = label.to(device)

                out = myvgg(img)
                _, out_idx = torch.max(out, 1)
                correct += (label == out_idx).sum().item()
        print('Val acc: ', correct / len(valdata))
        
       
"""
TRAINING
"""

loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(myvgg.parameters(), lr=0.1, momentum = 0.9)
epochs = 20
train_model(train, validation, epochs, loss_func, opt)

"""
TESTING
"""

correct=0
myvgg.eval()
with torch.no_grad():
  for img, label in test:
    img = img.to(device)
    label = label.to(device)

    out = myvgg(img)
    _, out_idx = torch.max(out, 1)
    correct += (label == out_idx).sum().item()
print('Test acc: ', correct / len(testdata))
