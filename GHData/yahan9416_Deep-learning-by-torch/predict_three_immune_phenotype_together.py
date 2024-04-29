import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

'''
Use the expression of immune pathway genes to predict immune phenotype.such as inflamed, excluded, desert
The labels of sample in one column and 0,1,2
'''
def genTrainTest(dat):
    N,L=dat.shape #return the size of dat
    print dat.shape
    np.random.seed(10)
    np.random.shuffle(dat) #random the data by row
    traindat=dat[:226,:L-1] #python's index is from 0 to length-1 ,and if we write :336 is from 0 to 335
    trainlabel=dat[:226,L-1]#use 80% sample as train data and others is test.
    testdat=dat[226:,:L-1]
    testlabel=dat[226:,L-1]
    return traindat,trainlabel,testdat,testlabel

def Accuracy(pred,label):
    pred=pred.cpu().data.numpy()#np.argmax(pred,1) is get the predict classifier result 
    label=label.cpu().data.numpy()
    test_np=(np.argmax(pred,1) == label)#compare the result with real classifier,"argmax" calculate the row("1") max 的位置
    test_np=np.float32(test_np)#translate the true false into numeric
    return np.mean(test_np)

def Test_class(pred):
    pred=pred.cpu().data.numpy()#np.argmax(pred,1) is get the predict classifier result 
    return np.argmax(pred,1)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc=nn.Linear(224,160)
        self.fc2=nn.Linear(160,100)
        self.fc3=nn.Linear(100,40)
        self.fc4=nn.Linear(40,3)
        
    def forward(self,x):
        x=F.relu(self.fc(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.log_softmax(self.fc4(x),dim=1)
        return x
model=Net()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
#because of the number of sample in three immune phenotype is no same ,so we don't to balance the sample number
# but to set the weight for different weight for different group
weight=torch.tensor([3.73,2.12,3.84])
criterion=nn.NLLLoss(weight=weight)
#criterion=nn.NLLLoss()
if __name__=="__main__":
    data=np.loadtxt("/Users/wubingzhang/Desktop/predict_three_immune_subtye/pytroch_predict_immune_phnotype/three_classific_softmax_one_column_label.txt")
    traindat,trainlabel,testdat,testlabel=genTrainTest(data)
    print traindat.shape
    print trainlabel.shape

for i in range(800):
    traindat=Variable(torch.FloatTensor(traindat))
    trainlabel=Variable(torch.LongTensor(trainlabel))       
    outs=model(traindat)
    #print(outs)
    loss=criterion(outs,trainlabel)
    #print loss
    #backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if ((i+1)% 100) == 0:
        Acc=Accuracy(outs,trainlabel)
        class_sample=Test_class(outs)
        print ("Train Accuracy:",Acc)
        print loss



#test model
testdat=Variable(torch.FloatTensor(testdat))
testlabel=Variable(torch.LongTensor(testlabel))
testouts=model(testdat)
print("Test sample immune group",testlabel)
Acc=Accuracy(testouts,testlabel)
print ("Test Accuracy:",Acc)
class_sample=Test_class(testouts)
print ("Predict sample immune group",class_sample)


#0.7068

    
    
