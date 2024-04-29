from discriminator import Discriminator
from generator import Generator
import torch 
from tqdm import tqdm 
from data import HorseZebraDataset 
from torch.utils.data import DataLoader
from torchvision import transforms as t 
import matplotlib.pyplot as plt 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
disch = Discriminator().to(device)
genh = Generator(3,64).to(device)
discz = Discriminator().to(device)
genz = Generator(3,64).to(device)
transforms = t.Compose([
        t.ToTensor()
    ])
trainds = HorseZebraDataset(horsedir='horse2zebra/trainA', zebradir='horse2zebra/trainB', transform=transforms)
testds = HorseZebraDataset(horsedir='horse2zebra/testA', zebradir='horse2zebra/testB', transform=transforms)
trainloader = DataLoader(trainds,batch_size=8, shuffle=True) 
testloader =  DataLoader(testds,batch_size=8, shuffle=True)
mse = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()
optimd = torch.optim.Adam(list(disch.parameters())+list(discz.parameters()), lr=2e-4)
optimg = torch.optim.Adam(list(genh.parameters())+list(genz.parameters()), lr=2e-4)
EPOCHS = 5 
for epoch in range(EPOCHS):
    loop = tqdm(trainloader)
    for horse, zebra in loop:
        horse, zebra = horse.to(device), zebra.to(device)
        #train discrimnator 
        fakeh = genh(zebra)
        dischrealpred = disch(horse)
        dischfakepred = disch(fakeh)
        dischreal_loss = mse(dischrealpred, torch.ones_like(dischrealpred))
        dischfake_loss = mse(dischfakepred, torch.zeros_like(dischfakepred))
        dischloss = dischreal_loss+dischfake_loss
        fakez = genz(horse)
        disczrealpred = discz(zebra)
        disczfakepred = discz(fakez)
        disczreal_loss = mse(disczrealpred, torch.ones_like(dischrealpred))
        disczfake_loss = mse(disczfakepred, torch.zeros_like(dischfakepred))
        disczloss = disczreal_loss+disczfake_loss
        loss = (dischloss+disczloss)/2
        optimd.zero_grad()
        loss.backward(retain_graph=True)
        optimd.step()
        #train generator 
        #adversarial loss 
        pred_H = disch(fakeh) 
        pred_Z = discz(fakez)
        lossGH = mse(pred_H, torch.ones_like(pred_H)) 
        lossGZ = mse(pred_Z, torch.ones_like(pred_Z))
        #cycle loss 
        cycle_H = genh(fakez) 
        cycle_Z = genz(fakeh)
        cyclelossH = l1(horse, cycle_H) #because in l1loss(x, y) = x-y, in paper it mentioned that we have to do F(G(horse)) - true 
        cyclelossZ = l1(zebra, cycle_Z) #because in l1loss(x, y) = x-y, in paper it mentioned that we have to do G(F(zebra)) - true 
        #identity loss 
        identityZ = genz(zebra)
        identityH = genh(horse) 
        identityZloss = l1(zebra, identityZ) #because in l1loss(x, y) = x-y, in paper it mentioned that we have to do pred - true 
        identityHloss = l1(horse, identityH)#because in l1loss(x, y) = x-y, in paper it mentioned that we have to do pred - true 
        #add all (times 10 because lamba cycle)
        G_loss = (
            lossGH+lossGZ+
            cyclelossH*10.0+cyclelossZ*10.0+
            identityZloss*0.0+identityHloss*0.0 
        )
        optimg.zero_grad()
        G_loss.backward()
        optimg.step()









