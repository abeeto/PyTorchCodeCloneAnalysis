import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
from PIL import Image
import numpy as np
import os.path


def save_model(model):
    torch.save(model.state_dict(),'model.pth')

def load_model():
    model = AlexNet()
    model.load_state_dict(torch.load('model.pth'))
    return model



# Возвращает название предсказанного класса
def get_img_class(image_path, model):
    image = Image.open(image_path)

    # Преобразование изображения в тензор
    transform_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), # Превращает PIL Image (H x W x C) в torch.FloatTensor размера (C x H x W) заполенный [0.0, 1.0]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = transform_image(image)

    # Необходимо обернуть тензор изображения в ещё один, чтобы он соответсвовал принимаемому типу
    # tensor([1, 2, 3, 4]) -> tensor([[1, 2, 3, 4]])
    input_batch = img.unsqueeze(0)
    # Если cuda есть, выполняем вычисления на графическом процессоре
    if torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()

    # Получение вероятностей классов из модели
    output = model(input_batch)

    # Получаем наиболее вероятный класс
    preds = torch.topk(output, k=1).indices.squeeze(0)

    labels = ['another', 'cat', 'dog']
    
    return labels[preds]


class AlexNet(nn.Module):
    def __init__(self):

        dropout_rate = 0.5
        amount_of_classes = 3

        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, amount_of_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
       
        
    def forward(self, inputs):
        x = self.features(inputs)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_model():
    transform_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), # Превращает PIL Image (H x W x C) в torch.FloatTensor размера (C x H x W) заполенный [0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root = 'dataset\\training_set',
                                        transform = transform_image)

    # Чтобы узнать какое число соответствует классу
    # print(train_dataset.class_to_idx)

    test_dataset = datasets.ImageFolder(root = 'dataset\\test_set',
                                        transform = transform_image)

    batch_size = 30
    train_load = torch.utils.data.DataLoader(dataset = train_dataset, 
                                            batch_size = batch_size,
                                            shuffle = True)
    
    test_load = torch.utils.data.DataLoader(dataset = test_dataset, 
                                            batch_size = batch_size,
                                            shuffle = False)


    # Создание модели и загрузка весов
    model = AlexNet()
    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth'))
        print('Pre-trained model is loaded')

    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()    
    
    loss_fn = nn.CrossEntropyLoss()     

    # SGD действительно не обучает 
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.08)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    # Через каждые step_size эпох умножает lr на gamma
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    amount_of_epochs = 101

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    #Обучение
    for epoch in range(amount_of_epochs):

        correct = 0
        iterations = 0
        iter_loss = 0.0
        
        model.train() # метод train() доступен из-за наследования от nn.Module
        
        for i, (inputs, labels) in enumerate(train_load):

            inputs = Variable(inputs)
            labels = Variable(labels)

            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            optimizer.zero_grad()    
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)  
            iter_loss += loss.data 

            loss.backward()                 # Backpropagation 
            optimizer.step()                # Update the weights
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            iterations += 1

        # 
        lr_scheduler.step()
        #

        train_loss.append(iter_loss/iterations)

        train_accuracy.append((100 * correct / len(train_dataset)))
    
        #Тестовый проход
        loss = 0.0
        correct = 0
        iterations = 0

        model.eval()
        
        for i, (inputs, labels) in enumerate(test_load):
            
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            outputs = model(inputs)     
            loss = loss_fn(outputs, labels)
            loss += loss.data
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            
            iterations += 1

        test_loss.append(loss/iterations)

        test_accuracy.append((100 * correct / len(test_dataset)))
        
        print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}'
            .format(epoch+1, amount_of_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]))
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(),'model.pth')

if __name__ == "__main__":
    
    model = load_model()
    print(get_img_class('dataset/test/1.jpg', model))
    
    #train_model()
    
