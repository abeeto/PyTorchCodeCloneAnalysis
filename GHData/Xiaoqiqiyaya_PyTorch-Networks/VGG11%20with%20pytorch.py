import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        self. feature =nn.Sequential(
            nn.Conv2d(3,64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64,128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False),
            nn.Conv2d(128,256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False),
            nn.Conv2d(256,512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False),
            nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,ceil_mode=False)
        )

        self.classifer = nn.Sequential(
            nn.Linear(512,64,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64,64,bias=0.5),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64,10)
        )

    def forward(self,x):
        out = self.feature(x)
        out = out.view(out.size(0),-1)
        out = self.classifer(out)

        return  out



if __name__ =='__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  epoch_num = 40
  learning_rate = 0.001
  train_transform = transforms.Compose([
      transforms.Resize((32,32)),
      transforms.RandomHorizontalFlip(),#防止过拟合，随机反转
      transforms.RandomRotation(30),
      transforms.ToTensor(),
      transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.2225))
  ])

  test_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.2225))
  ])

  train_data = torchvision.datasets.CIFAR10(root='../../data',
                                            train=True,
                                            download=True,
                                            transform=train_transform)

  test_data = torchvision.datasets.CIFAR10(root='../../data',train=False,transform=test_transform)

  train_loader = torch.utils.data.DataLoader(train_data,batch_size=500,shuffle=True)

  test_loader = torch.utils.data.DataLoader(test_data,batch_size=500)

  model = VGG()
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
  total_step = len(train_loader)
  for  epoch  in range(epoch_num):
      for i ,(image,label) in enumerate(train_loader):
          image = image.to(device)
          label = label.to(device)
          output = model(image)
          loss = criterion(output,label)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          if (i + 1) % 100 == 0:
               print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epoch_num, i + 1,total_step, loss.item()))
  with torch.no_grad():
      total =0
      correct = 0
      length = len(test_loader)
      for (image,label) in test_loader:
          image = image.to(device)
          label = label.to(device)

          output = model(image)
          _, prediction = torch.max(output, 1)
          total +=label.size(0)
          correct +=(prediction==label).sum().item()
      print('Test Accuracy of the model on the {} test images: {} %'.format(length,100 * correct / total))

