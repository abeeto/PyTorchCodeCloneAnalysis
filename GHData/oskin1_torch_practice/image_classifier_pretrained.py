import torch
import torchvision

from torch_conv import ConvNet
from image_classifier import test_loader, im_show, classes


if __name__ == '__main__':

    # load trained model
    model = ConvNet()
    model.load_state_dict(torch.load('./model/image_classifier.pth'))
    model.eval()

    data_iter = iter(test_loader)
    images, labels = data_iter.next()

    # print images
    im_show(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))
