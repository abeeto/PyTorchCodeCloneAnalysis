import optimizer
import layerslib
import numpy as np
class Model:
    def __init__(self,lossfunction,optim):
        self.lossfunction=lossfunction
        self.optim=optim
        self.layers=optim.layers
    def forward(self,inp):
        pass
    def backward(self,out):
        pass


class Seqential(Model):
    def forward(self,inp):
        self.layers[0].inp=inp
        for i in range(len(self.layers)-1):
            if self.layers[i].type == "flat" or self.layers[i].type == "conv2d" or self.layers[i].type == "maxpool2d":
                self.layers[i+1].inp=self.layers[i].forward(self.layers[i].inp)
            else:
                self.layers[i+1].inp=self.layers[i].forward()
        self.out=self.layers[len(self.layers)-1].forward()
        return [self.layers[len(self.layers)-1],self.out]

class Convolv():
    def __init__(self,layers):
        self.layers=layers
    def forward(self,inparr):
        self.arr=[]
        for i in range(len(inparr)):
            if i%100==0:
                print(i)
            self.forw = inparr[i]
            for i in range(len(self.layers) - 1):
                self.forw = self.layers[i].forward(self.forw)
            self.out = self.layers[len(self.layers) - 1].forward(self.forw)
            self.arr.append(self.out)
        self.arr=np.array(self.arr)
        return self.arr



if __name__ == "__main__":
    layer1=layerslib.Sigmoid(2,2,True)
    layer2=layerslib.Sigmoid(2,1,True)
    layerarr=[layer1,layer2]
    opt=optimizer.SGD(0.5,0.1,layerarr)
    model=Seqential(layerslib.MSE(),opt)

    inp=[[0,0],[0,1],[1,0],[1,1]]
    out=[0,1,1,0]

    curloss=layerslib.MSE()


    for i in range(5000):
        for j in range(4):
            opt.zero_grad()
            ourout=model.forward(inp[j])
            c=curloss.loss(ourout,out[j])
            curloss.backward()
            opt.step()


    print("\n\n")
    for i in inp:
        print(model.forward(i)[1])
