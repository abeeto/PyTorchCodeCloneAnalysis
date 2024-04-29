from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("arsenal.jpg")

# ToTensor
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
trans_norm = transforms.Normalize([3, 2, 1], [1, 2, 3])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm, 2)

# Resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_toTensor(img_resize)
writer.add_image("Resize", img_resize, 0)

# Compose
trans_resize2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize2, trans_toTensor])
img_resize2 = trans_compose(img)
writer.add_image("Resize", img_resize2, 1)

#RandomCrop
trans_random=transforms.RandomCrop((64,128))
trans_compose2=transforms.Compose([trans_random,trans_toTensor])
for i in range(10):
    img_random=trans_compose2(img)
    writer.add_image("RandomCrop", img_random, i)
writer.close()
