import torch
import numpy as np
from v1 import MobileNetV1
from data import MyData
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from v1 import MobileNetV1

def collate_fn(images):
    image=[]
    label=[]
    for i,data in enumerate(images):
        image.append(data[0])
        label.append(data[1])
    return torch.stack(image,0),torch.from_numpy(np.array(label,dtype=np.int))

parse=ArgumentParser()
parse.add_argument('--batch',type=int,default=32,help="")
parse.add_argument('--data',type=str,default='/smart/liqian/demo/data/test_set/test_set/',help="")
parse.add_argument('--models',type=str,default='models/',help="")

args=parse.parse_args()

model=MobileNetV1(2,0.75)
torch.manual_seed(2)
transform=transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
dataset=MyData(args.data,transform)
dataloader=DataLoader(dataset,shuffle=False,batch_size=args.batch,num_workers=4,collate_fn=collate_fn)
for dirs in os.listdir(args.models):
    file_model=model.load_state_dict(torch.load(args.models+"/"+dirs))
    model.eval()
    sums=0
    for i,(image,label) in enumerate(dataloader):
        image=Variable(image,requires_grad=False)
        output=model(image)
        output=output.detach().numpy()
        pred=output.argmax(-1)
        sums+=sum(np.array(pred,dtype=np.int)==label.detach().numpy())
    print(dirs+" acc:"+str(sums/len(dataset)))
