import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import visdom

"""
# network error로 인해 visdom이 동작하지 않음. (추후 점검)
## set visdom
vis = visdom.Visdom()
vis.close(env='main')

## define loss_tracker
def loss_tracker(loss_plot, loss_value, num):
    # num, loss_value are Tensor
    vis.line(X=num, Y=loss_value, win=loss_plot, update='append')
"""

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # 이게 있어야 multiprocess로 돌릴 수 있다.

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(777)
    if device == "cuda":
        torch.cuda.manual_seed_all(777)

    # set hyper parameters
    learning_rate = 0.001
    batch_size = 100

    # get dataset
    mnist_train = dsets.MNIST(
        root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True
    )
    mnist_test = dsets.MNIST(
        root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=True
    )
    # set data loader
    data_loader = torch.utils.data.DataLoader(
        dataset=mnist_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=3,
    )  # num_worker를 추가해서 멀티스레도로 작동하게 만듦

    # set model
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Convolution layer 1
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            # Convolution layer 2
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            # Convolution layer 3
            self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            # Fully Connected layer
            self.linear1 = nn.Linear(128 * 3 * 3, 625)
            self.linear2 = nn.Linear(625, 10)
            nn.init.xavier_uniform_(self.linear1.weight)  # weight initialize
            nn.init.xavier_uniform_(self.linear2.weight)  # weight initialize
            self.fc = nn.Sequential(self.linear1, nn.ReLU(), self.linear2)

        def forward(self, x):
            out = self.layer3(
                self.layer2(self.layer1(x))
            )  # go through convolution layer
            out = out.view(out.size(0), -1)  # out.size(0) == batch_size
            return self.fc(out)  # go through fully connected layer

    # set model
    model = CNN().to(device)
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    nb_epochs = 15
    batches = len(data_loader)
    """
    ## make plot
    loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
    """
    # learning
    for epoch in range(nb_epochs):
        avg_cost = 0
        for X, Y in data_loader:
            # load
            X = X.to(device)
            Y = Y.to(device)

            # hypothesis
            hypothesis = model(X)

            # cost
            cost = F.cross_entropy(hypothesis, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        avg_cost += cost / batches
        print(f"epoch: {epoch+1:2d}/{nb_epochs} cost: {avg_cost:.10f}")
        ## update loss_tracker
        """loss_tracker(loss_plt, torch.Tensor([avg_cost]), torch.Tensor([epoch]))"""
    print("Learning completed")

    # check accuracy
    with torch.no_grad():
        X_test = (
            mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
        )
        Y_test = mnist_test.test_labels.to(device)

        # hypothesis
        hypothesis = model(X_test)
        # accuracy
        isCorrect = torch.argmax(hypothesis, dim=-1) == Y_test
        accuracy = torch.mean(isCorrect.float())

        print(f"Accuracy: {accuracy*100:.3f}")
        # CUDA로 학슴했을 때 99.020%의 정확도를 보인다!!! (11번에 비해 layer 추가)
