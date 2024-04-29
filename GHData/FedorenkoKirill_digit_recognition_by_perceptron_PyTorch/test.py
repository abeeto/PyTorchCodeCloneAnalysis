import torchvision as tv

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import Net
from transforms import make_transform


def get_dataset():

    # Тестовый набор
    testset = tv.datasets.MNIST(
        root='data/',
        train=False,
        download=False,
        transform=make_transform(),
    )
    testloadter = DataLoader(
        dataset=testset,
        batch_size=4,
        shuffle=False,
    )

    return len(testset), testloadter


def main():
    #Номер категории
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # Инициализировать параметры сети
    model = Net()

    # Загрузить модель
    model.load_state_dict(torch.load('model.pth'))

    if torch.cuda.is_available():
        # Используйте GPU
        model.cuda()

    # get dataloader
    data_len, dataloader = get_dataset()

    # Предсказать правильный номер
    correct_num = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)

        # Возьмите максимальное значение в качестве результата прогноза
        _, predicted = torch.max(outputs, 1)

        for j in range(len(predicted)):
            predicted_num = predicted[j].item()
            label_num = labels[j].item()
            # Сравните прогнозируемое значение со значением метки
            if predicted_num == label_num:
                correct_num += 1

    # Расчет точности прогноза
    correct_rate = correct_num / data_len
    print('correct rate is {:.3f}%'.format(correct_rate * 100))


if __name__ == "__main__":
    main()

