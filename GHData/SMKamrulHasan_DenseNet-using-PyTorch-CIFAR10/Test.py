correct = 0
total = 0

for data in testloader:
        images, labels = data
        images = Variable(images).cuda()
        outputs = densenet(images)
        _, predicted = torch.max(outputs.cpu().data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        
print('Accuracy on 10000 test images: %d %%' % (
    100 * correct / total))
