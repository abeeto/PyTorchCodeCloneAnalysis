import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.set_grad_enabled(False) # Disabling Training stuff / Computational Graph for now to save mem

train_set = torchvision.datasets.FashionMNIST(
    root = './data/',
    train=False, #should be True though
    download=False,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size = 10
)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1   = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2   = nn.Linear(in_features=120,    out_features=60)
        self.out   = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        ''' Flatten before dense layer'''
        t = t.flatten(1, -1) #(start_dim, end_dim)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        #t = F.softmax(t)
        '''we'll use loss fxn = cross_entropy
        which has inbuild softmax calculation.
        [vs cross_entropy_with_logits]'''
        return t

if __name__ == "__main__":
    net = MyNet()
    net.forward(torch.rand(10,1,28,28))
    print(net)

    img, label = next(iter(train_set))
    output = net.forward(img.unsqueeze(axis=0))
    print(output.shape)
    print(output)
    print(output.argmax(dim=1))
    # print(output.argmax(dim=0))
    print("Actual label: ", label)
    # plt.imshow(img.squeeze())
    # plt.show()

    print("~"*10, "Batch", "~"*10)
    img_set, label_set = next(iter(train_loader))
    print("Shape of img set: ", img_set.shape)
    output_set = net.forward(img_set)
    print("Shape of out set: ", output_set.shape)
    print("Predicted output: ", output_set.argmax(dim=1))
    print("Actual Labels: \t", label_set)
    print("Checking match: ", output_set.argmax(dim=1).eq(label_set))
    print("Checking match [No.]: ", output_set.argmax(dim=1).eq(label_set).sum().item())
    '''Can use softmax followed by argmax'''

def get_num_correct(predicted, actual):
    return predicted.argmax(dim=1).eq(actual).sum().item()
