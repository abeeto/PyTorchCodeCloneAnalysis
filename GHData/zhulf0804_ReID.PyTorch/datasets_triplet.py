import os
import glob
from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader
import random
import math
import collections
from torch.utils.data import sampler
from torchvision import transforms
from torch.utils.data import dataloader


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class RandomSampler(sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image):
        super(RandomSampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id

        self._id2index = collections.defaultdict(list)

        for idx, label in enumerate(data_source.labels):
            _id = label
            self._id2index[_id].append(idx)

    def __iter__(self):
        unique_ids = list(set(self.data_source.labels))
        random.shuffle(unique_ids)

        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.batch_image))
        return iter(imgs)

    def __len__(self):
        return len(self._id2index) * self.batch_image

    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        return random.sample(population, k)


class reid(dataset.Dataset):
    def __init__(self, args, transform, dtype):

        self.transform = transform
        self.loader = default_loader

        data_path = args.data_dir

        if dtype == 'train':
            data_path = os.path.join(data_path, 'train', 'train_list.txt')
            imgs, labels = [], []
            with open(data_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    imgs.append(os.path.join(args.data_dir, dtype, 'train_set', line.split()[0].split('/')[1]))
                    labels.append(int(line.strip().split()[1]))
        elif dtype == 'test':
            imgs, labels = [], []
            data_path = os.path.join(data_path, 'test', 'query_a_list.txt')
            with open(data_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    imgs.append(os.path.join(args.data_dir, dtype, 'query_a', line.split()[0].split('/')[1]))
                    labels.append(int(line.strip().split()[1]))
        else:
            data_path = os.path.join(data_path, 'test', 'gallery_a')
            imgs = glob.glob(os.path.join(data_path, '*.png'))
            labels = [0] * len(imgs)

        self.imgs = imgs
        self.labels = labels


    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.labels[index]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


class Data:
    def __init__(self, args):

        train_list = [
            transforms.Resize((384, 128), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if args.random_erasing:
            train_list.append(RandomErasing(probability=args.probability, mean=[0.0, 0.0, 0.0]))

        train_transform = transforms.Compose(train_list)

        test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if not args.test_only:

            self.trainset = reid(args, train_transform, 'train')
            self.train_loader = dataloader.DataLoader(self.trainset,
                                                      sampler=RandomSampler(self.trainset, args.batchid,
                                                                            batch_image=args.batchimage),
                                                      # shuffle=True,
                                                      batch_size=args.batchid * args.batchimage,
                                                      num_workers=args.nThread)
        else:
            self.train_loader = None


        self.testset = reid(args, test_transform, 'test')
        self.queryset = reid(args, test_transform, 'query')
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=args.batchtest, num_workers=2)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=args.batchtest, num_workers=2)