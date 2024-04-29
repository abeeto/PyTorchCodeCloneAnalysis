#transformation function 
#transform function of pytorch support PIL image format. 
from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

#std is standard deviation
#mean is mean value
img = Image.open('PATH')
x = transform(img)

z = x * torch.tensor(std).view(3, 1, 1)
z = z + torch.tensor(mean).view(3, 1, 1)

img2 = transforms.ToPILImage(mode='RGB')(z)
plt.imshow(img2)
