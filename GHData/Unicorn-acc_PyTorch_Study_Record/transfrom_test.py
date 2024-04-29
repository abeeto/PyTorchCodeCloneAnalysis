from PIL import Image
from torchvision import transforms


img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

trans = transforms.ToTensor()
tensor_img = trans(img)