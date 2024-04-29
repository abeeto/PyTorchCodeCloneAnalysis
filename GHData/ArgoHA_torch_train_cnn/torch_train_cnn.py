import torch
from torch import nn
from torchvision import datasets, transforms
import os
import shutil
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score


# Count images for train valid test split
def get_sets_amount(valid_x, test_x, path_to_folder):
    count_images = 0

    folders = [x for x in os.listdir(path_to_folder) if not x.startswith(".")]
    for folder in folders:
        path = os.path.join(path_to_folder, folder)
        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            if os.path.isfile(image_path) and not image.startswith("."):
                count_images += 1

    valid_amount = int(count_images * valid_x)
    test_amount = int(count_images * test_x)
    train_amount = count_images - valid_amount - test_amount

    return train_amount, valid_amount, test_amount


# Split images by folders
def create_sets_folders(path_to_folder, valid_part, test_part, classes):
    train_amount, valid_amount, test_amount = get_sets_amount(valid_part, test_part, path_to_folder)
    print(f'Train images: {train_amount}\nValid images: {valid_amount}\nTest images: {test_amount}')

    os.chdir(path_to_folder)
    if os.path.isdir('train') is False:

        os.mkdir('valid')
        os.mkdir('test')

        for name in classes:
            shutil.copytree(f'{name}', f'train/{name}')
            os.mkdir(f'valid/{name}')
            os.mkdir(f'test/{name}')

            valid_samples = random.sample(os.listdir(f'train/{name}'), round(valid_amount / len(classes)))
            for j in valid_samples:
                shutil.move(f'train/{name}/{j}', f'valid/{name}')

            test_samples = random.sample(os.listdir(f'train/{name}'), round(test_amount / len(classes)))
            for k in test_samples:
                shutil.move(f'train/{name}/{k}', f'test/{name}')

        print('Created train, valid and test directories')


# Load images to Torch and preprocess them
def load_data(path, im_size, batch_size):
    transform = transforms.Compose([transforms.Resize(im_size),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def get_splited_data(path_to_folder, valid_part, test_part, classes, im_size, batch_size):
    create_sets_folders(path_to_folder, valid_part, test_part, classes)

    train_data = load_data(os.path.join(path_to_folder, 'train'), im_size, batch_size)
    valid_data = load_data(os.path.join(path_to_folder, 'valid'), im_size, batch_size)
    test_data = load_data(os.path.join(path_to_folder, 'test'), im_size, batch_size)

    return train_data, valid_data, test_data


# get model predictions
def get_preds(model, testing_data, device):
    val_preds = []
    val_labels = []
    model.eval() # set mode

    with torch.no_grad():
        for data, target in testing_data:
            images, labels = data.to(device), target.to(device)
            outputs = model.forward(images)
            val_preds.extend(torch.max(outputs.data, 1).indices.tolist())
            val_labels.extend(labels.tolist())

    return val_preds, val_labels


class Gun_classifier(nn.Module):
    def __init__(self):
        super(Gun_classifier, self).__init__()

        # input channels, output channels, kernel size, padding
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.pool1 = nn.MaxPool2d((3, 3))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.activ1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.pool2 = nn.MaxPool2d((3, 3))
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.activ2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, (3, 3))
        self.pool3 = nn.MaxPool2d((3, 3))
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.activ3 = nn.ReLU()

        self.glob_pool = nn.AdaptiveAvgPool2d((1, 1)) # Global avarage pooling

        self.fc1 = nn.Linear(128, 64)
        self.activ4 = nn.ReLU()

        self.fc2 = nn.Linear(64, 2) # n_classes
        self.sm = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.batch_norm1(x)
        x = self.activ1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.batch_norm2(x)
        x = self.activ2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.batch_norm3(x)
        x = self.activ3(x)

        x = self.glob_pool(x).reshape(-1, 128)

        x = self.fc1(x)
        x = self.activ4(x)

        x = self.fc2(x)
        x = self.sm(x)
        return x


def train(train_data, device, optimizer, model, loss_func, valid_data, epochs, path_to_save):
    best_metric = 0

    for epoch in range(1, epochs + 1):
        model.train() # set mode
        with tqdm(train_data, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{epochs}")

                images, labels = data.to(device), target.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + loss + backward + optimize
                outputs = model.forward(images)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

        # Get metrics after an epoch
        preds, valid_labels = get_preds(model, valid_data, device)
        f1 = round(f1_score(preds, valid_labels), 2)
        accuracy = round(accuracy_score(preds, valid_labels), 2)

        print(f'Valid accuracy: {accuracy}, valid f1: {f1}')

        # Save best model
        if f1 > best_metric:
            best_metric = f1
            torch.save(model.state_dict(), os.path.join(path_to_save, 'model.pt'))


def main():
    classes = ['small_gun', 'umbrella']
    im_size = (224, 224)
    path_to_folder = ''
    path_to_save = ''
    valid_part = 0.15
    test_part = 0.05
    batch_size = 32
    epochs = 5

    device = torch.device('mps')
    # device = torch.device('cpu')

    train_data, valid_data, test_data = get_splited_data(path_to_folder, valid_part, test_part,
                                                         classes, im_size, batch_size)

    model = Gun_classifier().to(device) # build the model
    loss_func = nn.CrossEntropyLoss() # init loss function
    optimizer = torch.optim.Adam(model.parameters()) # init optimiser

    train(train_data, device, optimizer, model, loss_func, valid_data, epochs, path_to_save)

    test_accuracy = round(accuracy_score(*get_preds(model, test_data, device)), 2)
    print(f'Finished training, test accuracy: {test_accuracy}')


if __name__ == '__main__':
    main()
