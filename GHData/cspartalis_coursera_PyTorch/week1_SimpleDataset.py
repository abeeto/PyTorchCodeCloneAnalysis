from torch.utils.data import Dataset
from torchvision import transforms
import torch

class toy_set(Dataset):
    def __init__(self, length=100, transform=None):
        self.x = 2*torch.ones(length, 2)
        self.y = torch.ones(length,1)
        self.len = length
        self.transform = transform

    def __getitem__(self,index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample=self.transform(sample)
        return sample

    def __len__(self):
        return self.len

class add_mult(object):
    def __init__(self,addx=1,muly=1):
        self.addx = addx
        self.muly = muly

    def __call__(self,sample):
        x=sample[0]
        y=sample[1]
        x=x+self.addx
        y=y*self.muly
        sample=x,y
        return sample

class mult(object):
    def __init__(self,mul=100):
        self.mul=mul

    def __call__(self,sample):
        x=sample[0]
        y=sample[1]
        x=x*self.mul
        y=y*self.mul
        sample=x,y
        return sample

def main():
    dataset= toy_set()
    print(len(dataset))
    print(dataset[0])
    for i in range(3):
        x,y=dataset[i]
        print(i,'x:',x,'y:',y)

    a_m = add_mult()
    dataset_ = toy_set(transform=a_m) # trasformed obj -> '_' in the end of name
    print(dataset_[0])

    data_transform = transforms.Compose([add_mult(),mult()])
    x_,y_=data_transform(dataset[0])
    print(x_,y_)

if __name__ == '__main__':
    main()
