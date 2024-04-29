import torch
from torchvision import Dataset
from torchvision import transforms as T
import random

class CustomDataset(Dataset):
  def __init__(self, image_folder):
    self.image_folder = image_folder
    self.transformations = T.Compose([
        T.ToTensor(),
        T.Resize((100,100)),
    ])

  def __getitem__(self, index):
    img, label = random.sample(list(self.image_folder),1)[0]
    get_the_same_class = random.randint(0, 1)

    if get_the_same_class:
      while True:
        pair_img, pair_label = random.sample(list(self.image_folder),1)[0]
        if pair_label == label:
          break
    else:
      while True:
        pair_img, pair_label = random.sample(list(self.image_folder),1)[0]
        if not(pair_label == label):
          break

    img = img.convert("L")
    pair_img = pair_img.convert("L")


    img = self.transformations(img)
    img_pair = self.transformations(pair_img)

    
    
    pair_class_label = torch.tensor(0., dtype=torch.float32) if get_the_same_class else torch.tensor(1., dtype=torch.float32)
    return img, img_pair, pair_class_label

  def __len__(self):
    return len(self.image_folder)