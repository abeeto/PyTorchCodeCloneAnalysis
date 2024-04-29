import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

testset = torchvision.datasets.CIFAR10(
    root="../../datasets", download=False,
    train=False, transform=transform
)

print(f"There are {len(testset)} images and labels in our test set")

# create a mapper for filtering data from the test set
mapper = {0 : "plane",2 : "bird"}
# filtering data for plane and birds from the test set

test_images = np.array([np.asarray(image) for image, label in testset if label in list(mapper.keys())])
test_labels = np.array([label for image, label in testset if label in list(mapper.keys())])


# changing labels of birds from 2 to 1
test_labels = np.where(test_labels==2, 1, test_labels)

# new mapper
data_map = {0: "plane", 1:"bird"}

print("There are {} planes and birds in or test dataset".format(len(test_images)))

tensor_test_images = torch.Tensor(test_images)
tensor_test_labels = torch.Tensor(test_labels)

print(tensor_test_images.shape)
print(tensor_test_labels.shape)

# Print the first ten images
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(tensor_test_images[i].permute(1, 2, 0))
    plt.title(data_map[test_labels[i]])
plt.show()



tensor_test_dataset = torch.utils.data.TensorDataset(tensor_test_images, tensor_test_labels)
test_loader = torch.utils.data.DataLoader(
    tensor_test_dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=2
    )


correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        outputs = torch.flatten(outputs)
        outputs = np.round(outputs)
        total += labels.size(0)
        correct += (outputs == labels).sum().item()
print("Accuracy over the {} test images : {:.2f} %".format(len(tensor_test_images), 100*correct/total))


# Visualize the test images
inv_normalize = transforms.Normalize(
    (-1, -1, -1),
    (1/0.5, 1/0.5, 1/0.5)
)

# General formula for inverse normalization 
# is as follows
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# Given normalization is :
# norm = transforms.Normalize(
#               mean, 
#               std)
# then Inverse normalization is :
# inverse_norm = transforms.Normalize(
#               [-m/s for m, s in zip(mean, std)], 
#               [1/s for s in std]
#               )

visualized_correct = 0
visualized_num = 0
with torch.no_grad():
    fig, axes = plt.subplots(5, 5, figsize=(8,8))
    for i in range(5):
        for j in range(5):
            visualized_num += 1
            value = i * 5 + j
            # plt.subplot(5, 5, i + 1)
            predicted = net(tensor_test_images[value].unsqueeze(0))
            predicted = np.round(predicted).numpy().astype(int).item()
            correct = tensor_test_labels[value].numpy().astype(int).item()

            if predicted == tensor_test_labels[value]:
                visualized_correct += 1
                color = "green"
            else:
                color = "red"
            axes[i, j].set_title(f"Predicted: {data_map[predicted]}", color=color)
            axes[i, j].set_xlabel(f"Correct: {data_map[correct]}", color=color)
            temp_img = inv_normalize(tensor_test_images[value])
            axes[i, j].imshow(temp_img.permute(1, 2, 0))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    # plt.subplots_adjust(hspace=0.8, wspace=0.3)
    fig.suptitle("{}/{} ({:.2f} %) images predicted correctly"
    .format(
        visualized_correct, 
        visualized_num, 
        visualized_correct/visualized_num * 100))
    plt.tight_layout()
    plt.show()
