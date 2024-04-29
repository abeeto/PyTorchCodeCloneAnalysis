import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from config import args
import torch
import pickle
from lime import lime_image
from skimage.segmentation import mark_boundaries


def load_dataset():
    """
    Load chosen benchmark dataset
    :return: trainloader - object to iterate in batches through the training set
             validloader - object to iterate in batches through the validation set
             class_names - class names for the dataset
    """

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    print(args.data)
    dataset = str(args.data).split('.')[3].split("'")[0]
    print(dataset)
    trainset = args.data(root='datasets/{}/train'.format(dataset),
                         train=True,
                         download=True,
                         transform=transform)
    validset = args.data(root='datasets/{}/test'.format(dataset),
                                train=False,
                                download=True,
                                transform=transform)

    if dataset == 'CIFAR10':
        meta = pickle.load(open('datasets/cifar10/train/cifar-10-batches-py/batches.meta', 'rb'))
        class_names = meta['label_names']
    elif dataset == 'CIFAR100':
        meta = pickle.load(open('datasets/cifar100/train/cifar-100-python/meta', 'rb'))
        class_names = meta['label_names']
    elif dataset == 'MNIST':
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # MNIST extended with handwritten letters
    # elif dataset == 'EMNIST':
    # # Kuzushiji MNIST
    # elif dataset == 'KMNIST':
    # # Fashion MNIST
    # elif dataset == 'Fashion-MNIST':

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.workers)
    validloader = torch.utils.data.DataLoader(validset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    return trainloader, validloader, class_names


def save_loss(epoch, batch, how_many_batches, loss, acc, name):
    """
    Save loss and accuracy for each batch in text file
    :param epoch: current epoch
    :param batch: current batch
    :param how_many_batches: number of batches
    :param loss: loss value
    :param acc: accuracy value
    :param name: file name
    """
    file = open('{}/loss_{}.txt'.format(args.folder_name, name), 'a')
    file.write(
        "[Epoch %d/%d] [Batch %d/%d] [Loss: %f] [Accuracy: %f]" % (epoch, args.n_epochs, batch, how_many_batches, loss, acc) + "\n")
    file.close()


def save_plot(epoch, loss, name):
    """
    Save plots for loss or accuracy over epochs
    :param epoch: current epoch
    :param loss:
    :param name:
    :return:
    """
    x = list(range(0, epoch))
    plt.plot(x, loss, '-b', label=name)
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.title(name)
    plt.legend(loc='upper left')
    plt.savefig('{}/{}.png'.format(args.folder_name, name), bbox_inches='tight')
    plt.close()


def save_cm(cm, class_names, epoch):
    """
    Save confusion matrix
    :param cm:
    :param class_names:
    :param epoch:
    :return:
    """
    df_cm = pd.DataFrame(cm, index=[i for i in class_names],
                         columns=[i for i in class_names])
    plt.figure(figsize=(25, 25))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig('{}/confusion_matrix_epoch'.format(args.folder_name) + str(epoch) + '.jpg')
    plt.close()


def save_images(class_names, imgs, predicted, epoch, n_row):
    """
    Save images with labels
###    Used for both real and predicted images
    :param class_names:
    :param imgs:
    :param predicted:
    :param epoch:
    :param n_row:
    :return:
    """
    mini = torch.min(imgs)
    maxi = torch.max(imgs)
    images = (imgs - mini) / (maxi - mini)
    labels = [class_names[predicted[i]] for i in predicted]

    fig, axs = plt.subplots(n_row, n_row, figsize=(150, 150))
    axs = axs.flatten()
    for img, ax, label in zip(images, axs, labels):
        arr = img.numpy()
        arr_im = np.moveaxis(arr, 0, 2)
        ax.imshow(arr_im)
        ax.set_title(str(label), fontsize=60)
        # get rid of axes etc
        plt.show()
    plt.savefig('{}/predictions_epoch'.format(args.folder_name) + str(epoch) + '.jpg')
    plt.close()


#def explain_predictions():
#    explainer = lime_image.LimeImageExplainer()
    # Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
    explanation = explainer.explain_instance(image, predict_fn, top_labels=5, hide_color=0, num_samples=1000)