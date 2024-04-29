import numpy as np
from abc import abstractmethod
import math
from scipy.stats import logistic

class Layer:
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def forward(self,inp):
        pass
    @abstractmethod
    def backward(self,deltas):
        pass
class Sigmoid(Layer):
    def __init__(self,inpsize,outsize,bias=False,inp=None):
        self.bias=bias
        self.inpsize=inpsize+bias
        self.outsize=outsize
        self.weights=np.random.random_sample((outsize,inpsize+bias))*2-1
        self.grad=np.zeros((outsize,inpsize+bias))
        self.oldgrad=np.zeros((outsize,inpsize+bias))
        self.inp=inp
        self.type="sigmoid"
        self.dropaut=False
        self.norm=False
    def norm(self,b):
        self.norm=b
    def dropautset(self,prob):
        self.dropaut=True
        self.dropautprob=prob
    def forward(self):
        if self.norm:
            self.int=(self.int-np.mean(self.int))/np.std(self.int)
        if self.dropaut:
            randarr=np.random.random(len(self.inp))
            for i in range(len(self.inp)):
                if randarr[i]<=self.dropautprob:
                    self.inp[i]=0
        if self.bias:
            self.inp=np.append(self.inp,[1])
        self.out=np.dot(self.weights,self.inp)
        self.sigmoid()
        return self.out
    def backward(self,deltas):
        self.grad=np.outer(deltas,self.inp)
        if self.prev!=None:
            deltas=np.multiply(self.weights.T,deltas).T
            deltas=deltas*(1-self.inp)*self.inp
            deltas=np.sum(deltas,axis=0)/len(deltas)
            deltas=deltas[0:len(deltas)-self.bias]
            self.prev.backward(deltas)
    def sigmoid(self):
        self.out=logistic.cdf(self.out)

class Linear(Layer):
    def __init__(self,inpsize,outsize,bias=False,inp=None):
        self.bias=bias
        self.inpsize=inpsize+bias
        self.outsize=outsize
        self.weights=np.random.random_sample((outsize,inpsize+bias))*2-1
        self.grad=np.zeros((outsize,inpsize+bias))
        self.oldgrad=np.zeros((outsize,inpsize+bias))
        self.inp=inp
        self.type="linear"
        self.dropout=False
    def dropautset(self,prob):
        self.dropaut=True
        self.dropautprob=prob
    def forward(self):
        if self.dropaut:
            randarr=np.random.random(len(self.inp))
            for i in range(len(self.inp)):
                if randarr[i]<=self.dropautprob:
                    self.inp[i]=0
        if self.bias:
            self.inp=np.append(self.inp,[1])
        self.out=np.dot(self.weights,self.inp)
        return self.out
    def backward(self,deltas):
        self.grad=np.outer(deltas,self.inp)
        if self.prev!=None:
            deltas=np.multiply(self.weights.T,deltas).T
            deltas=deltas
            deltas=np.sum(deltas,axis=0)/len(deltas)
            deltas=deltas[0:len(deltas)-self.bias]
            self.prev.backward(deltas)

class ReLU(Layer):
    def __init__(self,inpsize,outsize,bias=False,inp=None):
        self.bias=bias
        self.inpsize=inpsize+bias
        self.outsize=outsize
        self.weights=np.random.random_sample((outsize,inpsize+bias))*2-1
        self.grad=np.zeros((outsize,inpsize+bias))
        self.oldgrad=np.zeros((outsize,inpsize+bias))
        self.inp=inp
        self.type="ReLU"
        self.dropout=False
    def dropautset(self,prob):
        self.dropaut=True
        self.dropautprob=prob
    def forward(self):
        if self.dropaut:
            randarr=np.random.random(len(self.inp))
            for i in range(len(self.inp)):
                if randarr[i]<=self.dropautprob:
                    self.inp[i]=0
        if self.bias:
            self.inp=np.append(self.inp,[1])
        self.out=np.dot(self.weights,self.inp)
        self.relu()
        return self.out
    def backward(self,deltas):
        self.grad=np.outer(deltas,self.inp)
        if self.prev!=None:
            deltas=np.multiply(self.weights.T,deltas).T
            deltas[deltas<0]=0
            deltas=np.sum(deltas,axis=0)/len(deltas)
            deltas=deltas[0:len(deltas)-self.bias]
            self.prev.backward(deltas)
    def relu(self):
        self.out[self.out<0]=0

class LogSigmoid(Layer):
    def __init__(self,inpsize,outsize,bias=False,inp=None):
        self.bias=bias
        self.inpsize=inpsize+bias
        self.outsize=outsize
        self.weights=np.random.random_sample((outsize,inpsize+bias))*2-1
        self.grad=np.zeros((outsize,inpsize+bias))
        self.oldgrad=np.zeros((outsize,inpsize+bias))
        self.inp=inp
        self.type="logsigmoid"
        self.dropout=False
    def dropautset(self,prob):
        self.dropaut=True
        self.dropautprob=prob
    def forward(self):
        if self.dropaut:
            randarr=np.random.random(len(self.inp))
            for i in range(len(self.inp)):
                if randarr[i]<=self.dropautprob:
                    self.inp[i]=0
        if self.bias:
            self.inp=np.append(self.inp,[1])
        self.out=np.dot(self.weights,self.inp)
        self.logsigmoid()
        return self.out
    def backward(self,deltas):
        self.grad=np.outer(deltas,self.inp)
        if self.prev!=None:
            deltas=np.multiply(self.weights.T,deltas).T
            self.x=-math.log(math.e**(-self.out)-1)
            deltas=deltas/(1+math.e**(-self.x))
            deltas=np.sum(deltas,axis=0)/len(deltas)
            deltas=deltas[0:len(deltas)-self.bias]
            self.prev.backward(deltas)
    def logsigmoid(self):
        self.out=math.log(1/(1+math.e**(-self.out)))

class Tanh(Layer):
    def __init__(self,inpsize,outsize,bias=False,inp=None):
        self.bias=bias
        self.inpsize=inpsize+bias
        self.outsize=outsize
        self.weights=np.random.random_sample((outsize,inpsize+bias))*2-1
        self.grad=np.zeros((outsize,inpsize+bias))
        self.oldgrad=np.zeros((outsize,inpsize+bias))
        self.inp=inp
        self.type="tanh"
        self.dropout=False
    def dropautset(self,prob):
        self.dropaut=True
        self.dropautprob=prob
    def forward(self):
        if self.dropaut:
            randarr=np.random.random(len(self.inp))
            for i in range(len(self.inp)):
                if randarr[i]<=self.dropautprob:
                    self.inp[i]=0
        if self.bias:
            self.inp=np.append(self.inp,[1])
        self.out=np.dot(self.weights,self.inp)
        self.tanh()
        return self.out
    def backward(self,deltas):
        self.grad=np.outer(deltas,self.inp)
        if self.prev!=None:
            deltas=np.multiply(self.weights.T,deltas).T
            deltas=deltas*(1-self.inp**2)
            deltas=np.sum(deltas,axis=0)/len(deltas)
            deltas=deltas[0:len(deltas)-self.bias]
            self.prev.backward(deltas)
    def tanh(self):
        e=math.e
        self.out=(e**self.out-e**(-self.out))/(e**self.out+e**(-self.out))

class PReLU(Layer):
    def __init__(self,inpsize,outsize,c,bias=False,inp=None):
        self.bias=bias
        self.inpsize=inpsize+bias
        self.outsize=outsize
        self.weights=np.random.random_sample((outsize,inpsize+bias))*2-1
        self.grad=np.zeros((outsize,inpsize+bias))
        self.oldgrad=np.zeros((outsize,inpsize+bias))
        self.inp=inp
        self.type="prelu"
        self.c=c
        self.dropout=False
    def dropautset(self,prob):
        self.dropaut=True
        self.dropautprob=prob
    def forward(self):
        if self.dropaut:
            randarr=np.random.random(len(self.inp))
            for i in range(len(self.inp)):
                if randarr[i]<=self.dropautprob:
                    self.inp[i]=0
        if self.bias:
            self.inp=np.append(self.inp,[1])
        self.out=np.dot(self.weights,self.inp)
        self.prelu()
        return self.out
    def backward(self,deltas):
        self.grad=np.outer(deltas,self.inp)
        if self.prev!=None:
            deltas=np.multiply(self.weights.T,deltas).T
            deltas[deltas<0]=deltas[deltas<0]*self.c
            deltas=np.sum(deltas,axis=0)/len(deltas)
            deltas=deltas[0:len(deltas)-self.bias]
            self.prev.backward(deltas)
    def prelu(self):
        self.out[self.out<0]=self.out*self.c




class flat():
    def __init__(self):
        self.d=1
        self.type="flat"
    def forward(self,arr):
        self.d=len(arr)
        return arr.flatten()
    def backward(self,deltas):
        self.prev.backward(deltas.reshape(self.inp.shape))









class conv2d():
    def __init__(self,arrk):
        self.arrk=arrk
        self.type="conv2d"
    def forward(self,arra):
        arrc=[]
        for arr in arra:
            for k in self.arrk:
                c = [[0 for i in range(len(arr[0]) - len(k) + 1)] for j in range(len(arr) - len(k) + 1)]
                for i in range(len(arr) - len(k) + 1):
                    for j in range(len(arr[0]) - len(k) + 1):
                        c[i][j] = np.sum(arr[i:i + len(k), j:j + len(k)] * k)/255
                arrc.append(c)
        arrc=np.array(arrc)
        self.arrc=arrc
        return np.array(arrc)
    def backward(self,deltas):
        self.grad=np.zeros_like(self.arrk)
        for _k in range(len(deltas) // len(self.inp)):
            for arrind in range(len(self.inp)):
                k=_k*len(self.inp)+arrind
                grad = [[0 for i in range(len(self.inp[arrind][0]) - len(deltas[k]) + 1)] for j in range(len(self.inp[arrind]) - len(deltas[k]) + 1)]
                for i in range(len(self.inp[arrind]) - len(deltas[k]) + 1):
                    for j in range(len(self.inp[arrind][0]) - len(deltas[k]) + 1):
                        grad[i][j] = np.sum(self.inp[arrind][i:i + len(deltas[k]), j:j + len(deltas[k])] * deltas[k]) / 255
                self.grad[_k]=self.grad[_k]+grad
            self.grad[_k]=self.grad[_k]/len(self.inp)
        if self.prev != None:
            paddeltas=[]
            for i in range(len(deltas)):
                paddeltas.append(np.pad(deltas[i],(len(self.arrk[0])-1)//2+1))
            paddeltas=np.array(paddeltas)
            self.deltas=[]
            for paddelta in paddeltas:
                for i in range(len(self.arrk)):
                    for j in range(len(paddeltas[0])-len(self.arrk[i])+1):
                        for k in range(len(paddeltas[0][0])-len(self.arrk[i][0])+1):
                            c1=np.sum(paddelta[j:j+len(self.arrk[i]),k:k+len(self.arrk[i][0])]*self.arrk[i])
                            self.deltas.append(c1)
            self.deltas=np.array(self.deltas)
            _=[]
            c=0
            s=len(self.arrk[0])-1
            s=s*s
            for i in range(len(self.deltas)):
                c=c+self.deltas[i]
                if i%s==s-1:
                    _.append(c)
                    c=c=0
            self.deltas=np.array(_)
            self.prev.backward(self.deltas)



class maxpool2d():
    def __init__(self,s1,s2):
        self.s1=s1
        self.s2=s2
        self.type="maxpool2d"
    def forward(self,arrlist):
        arrc = []
        for arr in arrlist:
            if len(arr[0]) % self.s2 != 0 or len(arr) % self.s1 != 0:
                return "error"
            c = [[0 for i in range(math.ceil(len(arr[0]) / self.s1))] for j in range(math.ceil(len(arr) / self.s2))]
            for i in range(0, len(arr) - self.s1 + 1, self.s1):
                for j in range(0, len(arr[0]) - self.s2 + 1, self.s2):
                    c[i // self.s1][j // self.s2] = np.max(arr[i:i + self.s1, j:j + self.s2])
            arrc.append(c)
        return np.array(arrc)
    def backward(self,deltas):
        deltas=deltas.flatten()
        self.deltas=np.zeros_like(self.inp)
        q=0
        for k in range(len(self.deltas)):
            arr=self.inp[k]
            for i in range(0, len(self.deltas[k]) - self.s1 + 1, self.s1):
                for j in range(0, len(self.deltas[k][0]) - self.s2 + 1, self.s2):
                    smallarr=np.round(arr[i:i + self.s1, j:j + self.s2],5)
                    c=np.round(np.max(smallarr),5)
                    eltochange=np.where(smallarr==c)
                    self.deltas[k][eltochange[0][0]+i][eltochange[1][0]+j]=deltas[q]
                    q=q+1
        self.prev.backward(self.deltas)















class MSE(Layer):
    def __init__(self):
        self.prev=None
    def loss(self,out,expout):
        self.prev=out[0]
        self.out=out[1]
        self.expout=expout
        self.dif=self.expout-self.out
    def backward(self,c=None):
        if (self.prev.type=="sigmoid"):
            self.prev.backward(np.array([self.dif*(self.out)*(1-self.out)]))
        if (self.prev.type=="prelu"):
            if self.dif<0:
                self.prev.backward(np.array([self.dif*c]))
            else:
                self.prev.backward(np.array([self.dif]))
        if (self.prev.type=="linear"):
            self.prev.backward(np.array([self.dif]))
        if (self.prev.type=="relu"):
            if self.dif<0:
                self.prev.backward(np.array([self.dif*0]))
            else:
                self.prev.backward(np.array([self.dif]))
        if (self.prev.type=="tanh"):
            self.prev.backward(np.array([self.dif*(1-self.out**2)]))
        if (self.prev.type=="logsigmoid"):
            self.x=-math.log(math.e**(-self.out)-1)
            self.prev.backward(np.array([self.dif/(math.e**self.x+1)]))

class CrossEntropy(Layer):
    def __init__(self):
        self.prev=None
    def loss(self,out,expout):
        self.prev=out[0]
        self.out=out[1]
        self.expout=expout
        self.dif=-np.log(self.out)*expout+(1-expout)*np.log(1-self.out)

    def backward(self,c=None):
        if (self.prev.type=="sigmoid"):
            self.prev.backward(np.array([self.dif*(self.out)*(1-self.out)]))
        if (self.prev.type=="prelu"):
            if self.dif<0:
                self.prev.backward(np.array([self.dif*c]))
            else:
                self.prev.backward(np.array([self.dif]))
        if (self.prev.type=="linear"):
            self.prev.backward(np.array([self.dif]))
        if (self.prev.type=="relu"):
            if self.dif<0:
                self.prev.backward(np.array([self.dif*0]))
            else:
                self.prev.backward(np.array([self.dif]))
        if (self.prev.type=="tanh"):
            self.prev.backward(np.array([self.dif*(1-self.out**2)]))
        if (self.prev.type=="logsigmoid"):
            self.x=-math.log(math.e**(-self.out)-1)
            self.prev.backward(np.array([self.dif/(math.e**self.x+1)]))