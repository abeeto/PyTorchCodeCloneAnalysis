import torch.optim as optim
from torch.autograd import Variable

gpu = torch.cuda.is_available()

learning_rate = 0.001
momentum = 0.95
batch_size = 10
num_classes = 10
num_epochs = 10
loss_check = []

Net = DenseNet(num_classes)
if gpu:
    Net.cuda()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss().cuda() if gpu else nn.CrossEntropyLoss()
optimizer = optim.SGD(Net.parameters(), lr=learning_rate, momentum=momentum, nesterov = False)

print("Start Training..")
for epoch in range(num_epochs):  

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = Variable(inputs).cuda() 
        labels = Variable(labels).cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = Net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            loss_check.append(running_loss / 2000)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    
print("^^^^^^^^^^^^^^^^^")
print('Finished Training.')
#print("The total time to train the model on Google Colab is : {:.1f} minutes.".format((end - start)/60))
