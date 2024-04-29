import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
import torch.optim as optim;
import torch.nn.functional as F;
import random
import pickle;
import numpy as np;
from tqdm import tqdm;

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1=nn.Linear(100,624)
        nn.init.xavier_uniform(self.fc1.weight.data);
        nn.init.constant(self.fc1.bias.data, 0);
        self.convt_1=nn.ConvTranspose2d(3,32,7,stride=1,padding=3)
        nn.init.xavier_uniform(self.convt_1.weight.data);
        nn.init.constant(self.convt_1.bias.data, 0);
        self.convt_2 = nn.ConvTranspose2d(32, 32, 6, stride=2, padding=2)
        nn.init.xavier_uniform(self.convt_2.weight.data);
        nn.init.constant(self.convt_2.bias.data, 0)
        self.convt_3=nn.ConvTranspose2d(32,64,7,stride=1,padding=3)
        nn.init.xavier_uniform(self.convt_3.weight.data);
        nn.init.constant(self.convt_3.bias.data, 0);
        self.convt_4 = nn.ConvTranspose2d(64, 64, 7,stride=3, padding=2)
        nn.init.xavier_uniform(self.convt_4.weight.data);
        nn.init.constant(self.convt_4.bias.data, 0)
        #self.pixelshuffle = nn.PixelShuffle(2)
        self.convt_5=nn.ConvTranspose2d(64,3,7,stride=1,padding=3)
        nn.init.xavier_uniform(self.convt_5.weight.data);
        nn.init.constant(self.convt_5.bias.data, 0)
        self.bn_1=nn.BatchNorm2d(32)
        self.bn_2=nn.BatchNorm2d(32)
        self.bn_3 = nn.BatchNorm2d(64)
        self.bn_4 = nn.BatchNorm2d(64)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=x.view(-1,3,16,13)
        #print(x.size())
        x=F.relu(self.bn_1(self.convt_1(x)))
        #print(x.size())
        x=F.relu(self.bn_2(self.convt_2(x)))
        #print(x.size())
        x=F.relu(self.bn_3(self.convt_3(x)))
        x = F.relu(self.bn_4(self.convt_4(x)))
        #print(x.size())
        #x=self.pixelshuffle(x)
        x=F.tanh(self.convt_5(x))
        #print(x.size())
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv_1=nn.Conv2d(3,16,7,padding=3)
        nn.init.xavier_uniform(self.conv_1.weight.data);
        nn.init.constant(self.conv_1.bias.data, 0)
        self.conv_2 = nn.Conv2d(16, 16, 7, stride=3,padding=3)
        nn.init.xavier_uniform(self.conv_2.weight.data);
        nn.init.constant(self.conv_2.bias.data, 0)
        self.conv_3 = nn.Conv2d(16, 32, 7, padding=3)
        nn.init.xavier_uniform(self.conv_3.weight.data);
        nn.init.constant(self.conv_3.bias.data, 0)
        self.conv_4 = nn.Conv2d(32, 32, 7,stride=2, padding=3)
        nn.init.xavier_uniform(self.conv_4.weight.data);
        nn.init.constant(self.conv_4.bias.data, 0)
        self.fc1=nn.Linear(6656,1024)
        nn.init.xavier_uniform(self.fc1.weight.data);
        nn.init.constant(self.fc1.bias.data, 0)
        self.fc2=nn.Linear(1024,1)
        nn.init.xavier_uniform(self.fc2.weight.data);
        nn.init.constant(self.fc2.bias.data, 0)
    def forward(self,x):
        x=F.relu(self.conv_1(x))
        x=F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        #print(x.size())
        x=x.view(-1,self.flattern_dim(x))
        #print(x.size())
        x=F.relu(self.fc1(x))
        #print(x.size())
        x=F.sigmoid(self.fc2(x))
        return x
    def flattern_dim(self,x):
        size=x.size()[1:]
        count=1;
        for j in size:
            count*=j
        return count




g_net=Generator()
g_net.cuda()
#g_net.eval()
#g_net.load_state_dict(torch.load("g_net",map_location=lambda storage,loc:storage))
g_params=list(g_net.parameters())
d_net=Discriminator()
d_net.cuda()
#d_net.load_state_dict(torch.load("d_net"))
d_params=list(d_net.parameters())
#test_input=Variable(torch.FloatTensor(10,100))
#test_output=g_net(test_input)

'''from PIL import Image
#input_signal_np=np.random.normal(loc=0,scale=1,size=(100,50))
#input_signal=Variable(torch.from_numpy(input_signal_np)).type(torch.FloatTensor)
input_signal=Variable(torch.FloatTensor(16,100).normal_(0,1))
output_image=g_net(input_signal).permute(0,2,3,1).squeeze()
output_image=output_image.data.numpy()*127.5+127.5
output_image=np.uint8(output_image.reshape((16,96,78,3)))
for i in range(16):
    img=Image.fromarray(output_image[i],mode="RGB")
    img.save("predict_"+str(i)+".png")'''



initial_g_lr=0.0001
initial_d_lr=0.001
g_opt=optim.Adam(g_net.parameters(),lr=initial_g_lr);
d_opt=optim.Adam(d_net.parameters(),lr=initial_d_lr);

# loading cifar10 images
x_train=np.load("all_image_matrix.npy")
x_train=x_train[:200*int(len(x_train)/200)]

criterion=nn.BCELoss()
batch_size=200
for epoch in range(8000):
    all_labels=np.arange(0,len(x_train));np.random.shuffle(all_labels)
    batched_labels=np.array_split(all_labels,int(len(x_train)/batch_size))
    batch_g_loss=[];batch_d_loss=[]
    for label in tqdm(range(len(batched_labels))):
        batch_labels=batched_labels[label]
        batch_images_np=np.zeros((batch_size,96,78,3),dtype=np.float32)
        for i,ele in enumerate(batch_labels):
            batch_images_np[i]=(x_train[ele]-127.5)/127.5
        batch_g_input=Variable(torch.FloatTensor(batch_size,100).normal_(0,1)).cuda()
        batch_images_vec=Variable(torch.from_numpy(batch_images_np).permute(0,3,1,2)).type(torch.FloatTensor).cuda()
        d_net.zero_grad()
        d_loss_real=criterion(d_net(batch_images_vec).squeeze(),Variable(0.9*torch.ones(batch_size)).cuda())
        d_loss_real.backward()
        batch_g_output=g_net(batch_g_input)
        d_loss_fake=criterion(d_net(batch_g_output.detach()).squeeze(),Variable(torch.zeros(batch_size)).cuda())
        d_loss_fake.backward()
        d_opt.step()
        g_net.zero_grad()
        g_loss=criterion(d_net(batch_g_output).squeeze(),Variable(torch.ones(batch_size)).cuda())
        g_loss.backward()
        g_opt.step()
        g_loss_here=g_loss.cpu()
        d_loss_real_here=d_loss_real.cpu()
        d_loss_fake_here=d_loss_fake.cpu()
        batch_g_loss.append(g_loss_here.data.numpy())
        batch_d_loss.append(d_loss_real_here.data.numpy()+d_loss_fake_here.data.numpy())
    torch.save(g_net.state_dict(),"g_net")
    torch.save(d_net.state_dict(),"d_net")
    with open("epoch_loss","a") as f:
        f.write("epoch: {}, g_loss: {}, d_loss: {} \n".format(str(epoch),str(np.mean(batch_g_loss)),str(np.mean(batch_d_loss))))