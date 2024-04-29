# 通过transforms.toTensor
from tensorboardX import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "hymenoptera_data/train/bees/90179376_abc234e5f4.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1.transforms如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# 2.为什么使用Tensor类型:包含后续所使用的神经网络所需的参数
# cv_img=cv2.imread(img_path)
writer.add_image("Tensor_img", tensor_img)
writer.close()
