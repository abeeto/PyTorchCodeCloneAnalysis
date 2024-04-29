import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.autograd.variable as variable
from DeXpression.Model import Block1, Block2, Block3

BATCH_SIZE = 8

transform = transforms.Compose(
    [transforms.CenterCrop(480),
     transforms.Resize(224),
     transforms.Grayscale(3),
     transforms.ToTensor()])

TEST_DATA_PATH = "./test_set"
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)
test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
classes = ('Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise',)

block1 = Block1()
block2 = Block2()
block3 = Block3()
checkpoint = torch.load('last_model_state.pth')
block1.load_state_dict(checkpoint['BLOCK_1_state_dict'])
block1.eval()
block2.load_state_dict(checkpoint['BLOCK_2_state_dict'])
block2.eval()
block3.load_state_dict(checkpoint['BLOCK_3_state_dict'])
block3.eval()

dataiter = iter(test_data_loader)
images, labels = dataiter.next()
real = variable(images)
outputs = block1(real)
outputs = block2(outputs, outputs)
outputs = block3(outputs, outputs)
_, predicted = torch.max(outputs, 1)
print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(8)))
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(8)))

correct = 0
total = 0
with torch.no_grad():
    for data in test_data_loader:
        images, labels = data
        images = variable(images)
        labels = variable(labels)
        outputs = block1(real)
        outputs = block2(outputs, outputs)
        outputs = block3(outputs, outputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        len_list = list(labels.size())
        if len_list[0] < 8:
            l = []
            for i in range(len_list[0]):
                l.append(i)
            indices = torch.tensor(l)
            predicted = torch.index_select(predicted, 0, indices)
        correct += float((predicted == labels).sum().item())

print('Accuracy of the network: %f %%' % (100*correct/total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_data_loader:
        images, labels = data
        images = variable(images)
        labels = variable(labels)
        outputs = block1(real)
        outputs = block2(outputs, outputs)
        outputs = block3(outputs, outputs)
        _, predicted = torch.max(outputs, 1)
        len_list = list(labels.size())
        if len_list[0] < 8:
            # print("label length", len_list[0])
            l = []
            for i in range(len_list[0]):
                l.append(i)
            indices = torch.tensor(l)
            predicted = torch.index_select(predicted, 0, indices)
        c = (predicted == labels).squeeze()
        range_ = len(list(labels))
        for i in range(range_):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
