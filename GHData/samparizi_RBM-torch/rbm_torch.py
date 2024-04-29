import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim


class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden, k):
        super(RBM, self).__init__()

        self.num_hidden = num_hidden
        self.num_visible = num_visible

        self.v_bias = nn.Parameter(torch.zeros(num_visible))
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))
        self.weights = nn.Parameter(torch.randn(num_visible, num_hidden)*1e-2)

        self.k = k

    def mean_h(self, v):
        return F.sigmoid(F.linear(self.weights, v, self.h_bias))

    def sample_h(self, v):
        return torch.distributions.Bernoulli(self.mean_h(v)).sample()

    def mean_v(self, h):
        return F.sigmoid(F.linear(h, torch.transpose(self.weights, 0 ,1), self.v_bias))

    def sample_v(self, h):
        return torch.distributions.Bernoulli(self.mean_v(h)).sample()

    def forward(self, v):
        pos_sample_h = self.sample_h(v)
        neg_sample_h = pos_sample_h

        for step in range(self.k):
            neg_sample_v = self.sample_v(neg_sample_h)
            neg_sample_h = self.sample_h(neg_sample_v)
        return v, neg_sample_v

    def free_energy(self, v):
        linear_bias_term = torch.dot(v, self.v_bias)
        pre_nonlinear_term = F.linear(v, self.weights, self.h_bias)
        nonlinear_term = torch.sum(torch.log1p(torch.exp(pre_nonlinear_term)))
        return -linear_bias_term - nonlinear_term

    def data_set(self):
        # Loading the  MNIST database
        print('Downloading and Converting ..')
        img_size = 784
        train_img = datasets.MNIST('../input_data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor()]))
        # train_img, label = train_img[0]
        My_list = []

        for _, (train_img, label) in enumerate(train_img):
            train_img = torch.distributions.Bernoulli(Variable(train_img.view(-1, img_size))).sample()
            #print(train_img.shape)
            #My_list.append(train_img)
            #train_img = My_list
        # train_img = torch.distributions.Normal(train_img, torch.Tensor([0.5]), torch.Tensor([0.05]))
        return train_img

    def update_rand_sampling(self, epochs, sample_num):
        train_img = self.data_set()

        fig_v, ax_v = plt.subplots(5, 8, sharex=True, sharey=True)
        ax_v = ax_v.flatten()
        plt.xticks([])
        plt.yticks([])

        # rand_sample = np.random.randint(2, size=[sample_num, self.num_visible])
        rand_sample = torch.bernoulli(torch.Tensor(sample_num, self.num_visible).uniform_(0, 1))

        for m in xrange(10000):
            rand_sample = self.sample_v(self.sample_h(rand_sample))
        img_v = torch.abs(rand_sample - 1)

        for i in xrange(sample_num):
            ax_v[i].imshow(img_v[i, :].reshape(28, 28), cmap='Greys', interpolation='nearest')

        plt.show()


model = RBM(num_visible=784, num_hidden=1000, k=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

model.update_rand_sampling(epochs=10, sample_num=40)
for i in xrange(10):
    # input_data2 = np.random.permutation(input_data)
    # input_data2 = train_img
    loss_ = []
    sample_data = train_img.bernoulli()

    v, v1 = self.model(sample_data)
    loss = model.free_energy(v) - model.free_energy(v1)
    loss_.append(loss.data[0])
    model.zero_grad()
    loss.backward()
    optimizer.step()

    print torch.mean(loss_)










