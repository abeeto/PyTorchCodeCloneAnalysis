from PIL import Image
import os.path as osp

from torch.utils.data import Dataset


def read_image(img_path):
    got_img = False
    img = None
    if not osp.exists(img_path):
        raise IOError('{} does not exist'.format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(img_path))
            pass
    return img


class ImageDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, v_id, color, model = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, color, model


def train_dataset(train_path, img_path):
    dataset = []
    for line in open(train_path, 'r'):
        img_name, v_id, color, model = line.strip().split(' ')
        img_name += '.jpg'
        img_name = osp.join(img_path, img_name)
        v_id = int(v_id)
        color = int(color)
        model = int(model)
        dataset.append((img_name, v_id, color, model))
    return dataset