from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import os

# 写自己的类继承Dataset
class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        # 所有image名称
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        # 获取具体image
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label # 返回图片和标签

    def __len__(self):
        return len(self.img_path)

#创建示例看看
root_dir = "dataset/train"
ants_label_dir = "ants_image"
bees_label_dir = "bees_image"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)

# 正常来说一个大的训练数据集是ant+bee
train_dataset = ants_dataset + bees_dataset
