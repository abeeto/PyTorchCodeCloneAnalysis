import numpy as np
import matplotlib.pyplot as plt
import foolbox as fb
import eagerpy as ep
import torch
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet18(pretrained=True)
model = model.eval()

'''model = torchvision.models.resnet18(num_classes=10177)
model.load_state_dict(torch.load(weights, map_location=torch.device("cpu"))["weights"])
model.eval()
model.to(device=device)'''

preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
bounds = (0, 1)
fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

fmodel = fmodel.transform_bounds((0, 1))
assert fmodel.bounds == (0, 1)

images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=16)
print(images.shape, labels.shape)
accuracy = fb.utils.accuracy(fmodel, images, labels)
print(accuracy)

attack = fb.attacks.LinfDeepFoolAttack()
raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=0.03)
print(is_adv)


images = ep.astensor(images)
labels = ep.astensor(labels)

raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=0.03)

print(is_adv)

criterion = fb.criteria.Misclassification(labels)

raw, clipped, is_adv = attack(fmodel, images, criterion, epsilons=0.03)

print(is_adv, is_adv.float32().mean(axis=-1))

epsilons = np.linspace(0.0, 0.005, num=20)

raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=epsilons)

print(is_adv.float32().mean(axis=-1))

robust_accuracy = 1 - is_adv.float32().mean(axis=-1)

plt.plot(epsilons, robust_accuracy.numpy())
plt.grid()
plt.show()