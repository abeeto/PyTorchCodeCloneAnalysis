loaded_net = torch.load("./cifar_two_net.pth")

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = loaded_net(images)
        outputs = torch.flatten(outputs)
        outputs = np.round(outputs)
        total += labels.size(0)
        correct += (outputs == labels).sum().item()
print("Accuracy over the {} test images : {:.2f} %".format(len(tensor_test_images), 100*correct/total))