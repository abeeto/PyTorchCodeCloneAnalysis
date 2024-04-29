import os
import torch
import random

train_file = 'training.pt'
test_file = 'test.pt'
origin_path = './data/MNIST/processed'
output_path = './federated_data/MNIST'
origin_train_path = os.path.join(origin_path, train_file)
origin_test_path = os.path.join(origin_path, test_file)

#   28x28   train: 60000    test: 10000


train_data = torch.load(origin_train_path)
print(type(train_data[0]))
img, label = train_data
print(img.shape)

os.makedirs(output_path, exist_ok=True)
users = 2


def get_distribution(img, label, num_classes=10):
    indices = []
    for i in range(num_classes):
        indices.append((label == i).nonzero().view(-1))
    # indices = torch.stack(indices)
    # print(indices)
    return indices


def _ppp(label):
    print(label.shape)
    print((label == 0).nonzero().shape)
    print((label == 1).nonzero().shape)
    print((label == 2).nonzero().shape)
    print((label == 3).nonzero().shape)
    print((label == 4).nonzero().shape)
    print((label == 5).nonzero().shape)
    print((label == 6).nonzero().shape)
    print((label == 7).nonzero().shape)
    print((label == 8).nonzero().shape)
    print((label == 9).nonzero().shape)


def generate(img, label):
    distribution = get_distribution(img, label)
    user1_img, use1_label = img[torch.cat(distribution[:5])], label[torch.cat(distribution[:5])]
    user2_img, use2_label = img[torch.cat(distribution[5:])], label[torch.cat(distribution[5:])]
    print(user1_img.shape)


def main():
    # print(get_distribution(img, label))
    generate(img, label)


if __name__ == "__main__":
    main()
