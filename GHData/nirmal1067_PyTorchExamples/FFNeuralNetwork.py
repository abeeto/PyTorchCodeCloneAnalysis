class FFNeuralNetwork(nn.Module):
    
    def __init__(self,inFeature=4,outFeature=3,hiddenLayer1=8,hiddenLayer2=9):
        super().__init__()
        self.fc1 = nn.Linear(inFeature,hiddenLayer1)
        self.fc2 = nn.Linear(hiddenLayer1,hiddenLayer2)
        self.out = nn.Linear(hiddenLayer2,outFeature)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        
    def predict(self,x):
        x = f.relu(self.fc1(x)) # Takes the input and pass it through first layer
        x = f.relu(self.fc2(x)) # Take the output of first layer 
        x = self.out(x) # Take input and pass it to final layer and grab the oupput.
        return x
    
    def printObjectValues(self):
        for name, param in self.named_parameters():
            print(name, '\t', param)
    
    def train(self,x,y,epochs=100):
        losses =[]
        i=0
        for i in range(epochs):
            i =i+1
            predict = self.predict(x)
            loss = self.criterion(predict,y)
            losses.append(loss)
            if i%10 == 1:
                print(f'epoch Number: {i:2} Corresponding loss: {loss.item():10.2f}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.drawGD(losses,epochs)
            
    
    def drawGD(self,losses,epoches):
        plt.plot(range(epoches),losses)
        plt.xlabel("Epoches")
        plt.ylabel("Losses")