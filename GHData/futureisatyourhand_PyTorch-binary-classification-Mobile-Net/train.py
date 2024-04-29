import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from argparse import ArgumentParser
from v1 import MobileNetV1
from data import MyData,collate_fn
from torch.backends import cudnn
from torch.autograd import Variable
import os

parse=ArgumentParser()
parse.add_argument('--train_data',type=str,default='/smart/liqian/demo/data/training_set/training_set/',help="")
parse.add_argument('--test_data',type=str,default='/smart/liqian/demo/data/test_set',help="")
parse.add_argument('--batch',type=int,default=32,help="")
parse.add_argument('--models',type=str,default='models/',help="")
parse.add_argument('--lr',type=float,default=1e-3,help="")
parse.add_argument('--beta1',type=float,default=0.99,help="")
parse.add_argument('--beta2',type=float,default=0.99999,help="")
parse.add_argument('--epochs',type=int,default=100,help="")
parse.add_argument('--no_cuda',type=bool,default=False,help="")

args=parse.parse_args()
args.cuda=not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(1)
base_lr=args.lr


model=MobileNetV1(2,0.75)
if args.cuda:
    model.cuda()
    cudnn.benchmark=True
transform=transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
train_dataset=MyData(args.train_data,transforms=transform)
print(args.batch,len(train_dataset))
batch=args.batch
trainloader=DataLoader(train_dataset,batch_size=batch,shuffle=True,num_workers=2,collate_fn=collate_fn)
#test_dataset=MyData(args.test_data)
#testloader=DataLoader(test_dataset,batch_size=args.batch,shuffle=False,num_workers=4,collate_fn=collate_fn)
optimize=torch.optim.Adam([{'params':model.parameters()}],lr=args.lr,betas=(args.beta1,args.beta2))
logs=open('logs.txt','a+')
for epoch in range(args.epochs):
    for iteration,(images,labels) in enumerate(trainloader):
        images,labels=Variable(images.cuda(),requires_grad=False),Variable(labels.cuda(),requires_grad=False)
        optimize.zero_grad()
        loss=model(images,labels)
        loss.backward()
        optimize.step()     
        logs.write(str(loss.item())+"\n")
    if epoch%5==0:
        torch.save(model.state_dict(),args.models+"model_"+str(epoch)+".pth")
logs.close()
