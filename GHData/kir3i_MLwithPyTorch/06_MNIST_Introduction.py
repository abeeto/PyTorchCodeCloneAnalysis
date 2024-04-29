import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

# 실행하는 device 인식 (gpu인지 cpu인지)
device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# define hypothesis
linear = torch.nn.Linear(28*28, 10, bias=True).to(device)
# init W -> W를 초기화(랜덤?)
torch.nn.init.normal_(linear.weight)
# define parameters
training_epochs = 20
batch_size = 100
# define cost function
criterion = torch.nn.CrossEntropyLoss().to(device)  # Softmax is internally computed.
# define optimizer
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

# get data
mnist_train = dset.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dset.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)
# define data_loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=100, shuffle=True, drop_last=True)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)  # batch의 수

    for X, Y in data_loader:
        # reshape input image into [batch_size by 784]
        # Label is not one-hot encoded
        X = X.view(-1, 28 * 28).to(device)
        
        # calculate hypothesis
        hypothesis = linear(X)
        # calculate cost
        cost = criterion(hypothesis, Y)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / total_batch
        # batch 단위로 끊어서 적용했으므로 나눠서 적용

    print(f'epoch: {epoch+1:4d} Cost: {avg_cost:.6f}')
print('Learning finished')

with torch.no_grad():
    # test dataset load
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    # calculate hypothesis
    prediction = linear(X_test)
    # calculate accuracy
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy: ', accuracy.item())

    # visualize
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r+1].view(-1, 28*28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r+1].to(device)

    print('Label: ', Y_single_data.item())
    single_hypothesis = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_hypothesis, 1).item())

    plt.imshow(mnist_test.test_data[r:r+1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()