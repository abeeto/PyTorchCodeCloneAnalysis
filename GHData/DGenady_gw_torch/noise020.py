import torch
import torch.nn as nn
import torch.nn.functional as F
import boto3
from urllib.parse import urlparse
from io import BytesIO
import numpy as np
from torchvision import models
import time
from resnetNoBN import resnet50NoBN

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        return embedded_x, embedded_y, embedded_z

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #resnet = models.resnet50(pretrained=True)
        resnet =  resnet50NoBN(pretrained=True)
        self.resNet = nn.Sequential(*(list(resnet.children())[:-1]))
        self.dimRed = nn.Sequential(nn.Linear(2048,1024),nn.ReLU(),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,256),
                                    nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,32))
    def forward(self, x):
        x = self.resNet(x)
        x = torch.squeeze(x)
        return self.dimRed(x)

def train(numOfFiles, numOfSamples, batchsize, model, loss_fn, optimizer, filePath):
    train_ind = 0 
    size = numOfFiles*numOfSamples//batchsize
    losses = []
    model.train()
    for file in range(numOfFiles):
        data = loadFile(path=filePath, num=file, s3obj=s3)
        for batch in range(numOfSamples//batchsize):
            optimizer.zero_grad()
            img1, img2, img3 = getBatchData(data,batch,batchsize) 
            img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)
            
            #Compute prediction error
            emb1,emb2,emb3 = model(img1,img2,img3)
            loss = loss_fn(emb1,emb2,emb3) + 0.001*(emb1.norm(2)+emb2.norm(2)+emb3.norm(2))
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        
    return np.mean(np.asarray(losses))

def test(firstFile,lastFile, numOfSamples, batchsize, model, loss_fn, filePath):
    train_ind = 0 
    losses = []
    model.eval()
    
    #for m in model.modules():
    #    for child in m.children():
    #        if type(child) == nn.BatchNorm2d:
    #            child.track_running_stats = False
    
    with torch.no_grad():
        for file in range(firstFile, lastFile):
            data = loadFile(path=filePath, num=file, s3obj=s3)
            for batch in range(numOfSamples//batchsize):
                img1, img2, img3 = getBatchData(data,batch,batchsize) 
                img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)

                #Compute prediction error
                emb1,emb2,emb3 = model(img1,img2,img3)
                loss = loss_fn(emb1,emb2,emb3) + 0.001*(emb1.norm(2)+emb2.norm(2)+emb3.norm(2))
                losses.append(loss.item())
           
    return np.mean(np.asarray(losses))
        
            
def loadFile(path, num, s3obj):
    data = np.load(load_to_bytes(s3obj,f's3://tau-astro/gdevit/{path}{num}.npy'), 
                       allow_pickle=True)
    data = torch.from_numpy(data)
    data = torch.permute(data,(0,1,4,2,3))
    data = data.float()
    return data/255

def load_to_bytes(s3,s3_uri:str):
    parsed_s3 = urlparse(s3_uri)
    f = BytesIO()
    s3.meta.client.download_fileobj(parsed_s3.netloc, 
                                    parsed_s3.path[1:], f)
    f.seek(0)
    return f

def getBatchData(data,batchNum,batchsize):
    imgs1 = data[batchNum*batchsize:(batchNum+1)*batchsize,0,:,:,:]
    imgs2 = data[batchNum*batchsize:(batchNum+1)*batchsize,1,:,:,:]
    imgs3 = data[batchNum*batchsize:(batchNum+1)*batchsize,2,:,:,:]
    return imgs1, imgs2 , imgs3
        
start_time = time.perf_counter()

noiseLvl = '020'

s3 = boto3.resource('s3',endpoint_url = 'https://s3-west.nrp-nautilus.io')
    
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = Net()
tnet = Tripletnet(model).to(device)

loss_fn = nn.TripletMarginLoss(margin=10.0, p=2)
optimizer = torch.optim.Adam(tnet.parameters(), lr=1e-7)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

epochs = 20
losses = np.empty((2,epochs))
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    losses[0,t] = train(numOfFiles=80, numOfSamples=1000, batchsize=50, model=tnet, loss_fn=loss_fn, optimizer=optimizer, 
                        filePath=f'triplet_noise/noise{noiseLvl}/file')
    losses[1,t] = test(firstFile=80,lastFile=90, numOfSamples=1000, batchsize=50, model=tnet, loss_fn=loss_fn, 
                       filePath=f'triplet_noise/noise{noiseLvl}/file')
    print(f'Train loss: {losses[0,t]:>4f} Val loss: {losses[1,t]:>4f}')
    scheduler.step()
    
torch.save(model.state_dict(),f'Netnoise{noiseLvl}.pt')
np.save(f'noise{noiseLvl}loss.npy', losses, allow_pickle=True)
print('Done in {} hours'.format((time.perf_counter()-start_time)/3600))
