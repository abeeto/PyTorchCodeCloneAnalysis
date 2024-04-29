import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
import numpy as np
from classes import Net, DataSet
from functions import load_data
import matplotlib.pyplot as plt
from math import ceil
import seaborn as sns

sns.set_style('whitegrid')

# Define global variables
batch_size = 100
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 90
run_on = 'GPU'
train_network = True

# Dataset paths
image_path = '/home/henryp/PycharmProjects/AlexNetTorch/data/102flowers_images/'
label_path = '/home/henryp/PycharmProjects/AlexNetTorch/data/imagelabels.mat'
nparray_label_path = '/home/henryp/PycharmProjects/AlexNetTorch/labelsArray.npy'

# Set up to train network on CPU or GPU
if run_on == 'GPU':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print('Training on: ' + str(device))

classes = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold',
          'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot",
          'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily',
          'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger',
          'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william',
          'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya',
          'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily',
          'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion',
          'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium',
          'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan',
          'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower',
          'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower',
          'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose',
          'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm', 'ball moss',
          'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower',
          'trumpet creeper', 'blackberry lily']

partition, labels, training_set_size = load_data(data_dir=image_path, label_path=label_path)

# Add some data transforms to prevent over fitting
transform = transforms.Compose([
    transforms.RandomRotation((45, 90)),
    transforms.RandomPerspective(distortion_scale=0.3),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define training and testing datasets and their respective generators
training_set = DataSet(partition['train'], labels, data_dir=image_path, transform=transform)
training_generator = data.DataLoader(training_set, **params)

testing_set = DataSet(partition['test'], labels, data_dir=image_path, transform=transform)
testing_generator = data.DataLoader(testing_set, **params)

# We track the accuracy and loss as the training continues so we can plot it at the end of training
accuracy_track = []
loss_track = []

if train_network:
    net = Net()
    if run_on == 'GPU':
        net.cuda()

    # Using cross entropy loss and Adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Training loop
    for epoch in range(max_epochs):
        # Training
        running_loss = 0.0
        running_accuracy = 0.0
        for local_batch, local_labels in training_generator:
            # Send image, label bacthes to GPU for training
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Run forward and backward pass
            optimizer.zero_grad()
            outputs = net(local_batch)
            loss = criterion(outputs, local_labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy and loss
            _, predicted = torch.max(outputs.data, 1)
            batch_accuracy = predicted.eq(local_labels.data).sum()
            batch_accuracy = torch.Tensor.cpu(batch_accuracy)
            batch_accuracy = np.array(batch_accuracy) / batch_size
            batch_loss = loss.item()

            running_loss += batch_loss
            running_accuracy += batch_accuracy



        print('Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1,
                                                                 running_loss / ceil(training_set_size / batch_size),
                                                                 running_accuracy /
                                                                 ceil(training_set_size / batch_size)))

        accuracy_track.append(running_loss / ceil(training_set_size / batch_size))
        loss_track.append(running_accuracy / ceil(training_set_size / batch_size))

    PATH = './AlexNet.pth'
    torch.save(net.state_dict(), PATH)

    plt.plot(accuracy_track)
    plt.title('Training Accuracy Over ' + str(max_epochs) + ' epochs')
    plt.show()
    plt.plot(loss_track)
    plt.title('Training Loss Over ' + str(max_epochs) + ' epochs')
    plt.show()

else:
    PATH = './AlexNet.pth'
    net = Net()
    if run_on == 'GPU':
        net.cuda()
    print('Loading saved network state...')
    net.load_state_dict(torch.load(PATH))

# Test model on testing data (20% of dataset)
correct = 0
total = 0
with torch.no_grad():
    i = 1
    for local_batch, local_labels in testing_generator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        outputs = net(local_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += local_labels.size(0)
        correct += (predicted == local_labels).sum().item()

        img = local_batch[0]
        img = torch.Tensor.cpu(img)

        label = local_labels[0]
        label = torch.Tensor.cpu(label)
        label = np.array(label)
        label = int(label)

        guess = torch.Tensor.cpu(predicted)
        guess = guess.data[0]
        guess = np.array(guess)
        guess = int(guess)
        guess_label = classes[guess-1]

        # Plot some examples of the models guesses
        if i < 10:
            plt.subplot(3, 3, i)
            plt.imshow(img.permute(1, 2, 0))
            plt.xticks([])
            plt.yticks([])
            if label == guess:
                plt.title(guess_label + '\n (CORRECT)')
            else:
                plt.title(guess_label + '\n (INCORRECT)')
        i += 1
    plt.tight_layout()
    plt.show()

    print('Network accuracy over testing data: {:.4f}'.format(100 * correct / total))

