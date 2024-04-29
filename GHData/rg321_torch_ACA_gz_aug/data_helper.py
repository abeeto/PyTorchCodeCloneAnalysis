import glob
import csv
import numpy as np
import imageio

from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from config import classes_100, classes_20, classes_1554, classes_581

class DataHelper():
    def __init__(self, sequence_max_length=1024):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '
        self.char_dict = {}
        self.UNK = 68
        self.sequence_max_length = sequence_max_length
        for i,c in enumerate(self.alphabet):
            self.char_dict[c] = i+1

    def char2vec(self, text):
        data = np.zeros(self.sequence_max_length)
        for i in range(0, len(text)):
            if i >= self.sequence_max_length:
                return data
            elif text[i] in self.char_dict:
                data[i] = self.char_dict[text[i]]
            else:
                data[i] = self.UNK
        return np.array(data)

    def load_csv_file(self, filename, num_classes, train=True, one_hot=False):
        if train:
            s1 = 120000
        else:
            s1 = 7600
        all_data =np.zeros(shape=(s1, self.sequence_max_length), dtype=np.int)
        labels =np.zeros(shape=(s1, 1), dtype=np.int)

        # labels = []
        with open(filename) as f:
            reader = csv.DictReader(f, fieldnames=['class'], restkey='fields')
            # reader = np.genfromtxt(f)
            for i,row in enumerate(reader):
                if one_hot:
                    one_hot = np.zeros(num_classes)
                    one_hot[int(row['class']) - 1] = 1
                    labels[i] = one_hot
                else:
                    labels[i] = int(row['class']) - 1
                text = row['fields'][-1].lower()
                all_data[i] = self.char2vec(text)
        f.close()
        return all_data, labels

    def load_dataset(self, dataset_path):
        with open(dataset_path+"classes.txt") as f:
            classes = []
            for line in f:
                classes.append(line.strip())
        f.close()
        num_classes = len(classes)
        train_data, train_label = self.load_csv_file(dataset_path + 'train.csv', num_classes)
        test_data, test_label = self.load_csv_file(dataset_path + 'test.csv', num_classes, train=False)
        print(train_data.shape, test_data.shape)
        return train_data, train_label, test_data, test_label

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        # for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batch = shuffled_data[start_index:end_index]
            # batch_data, label = batch[:,  self.sequence_max_length-1], batch[:, -1]
            batch_data, label = np.split(batch, [self.sequence_max_length],axis=1)
            yield np.array(batch_data, dtype=np.int), label


def get_mnist_loaders(batch_size=128, test_batch_size=1000, perc=1.0):
    transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=0, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=0, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


def get_cifar_loaders(batch_size=128, test_batch_size=1000):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    train_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=0)

    test_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, None

def tiny_imagenet(batch_size=64, test_batch_size=64, dataset_source='', path_to_data='tiny-imagenet-200/',resize=32):
    """Tiny ImageNet dataloader.

    Parameters
    ----------
    batch_size : int

    path_to_data : string
        Path to Tiny ImageNet data files root folder.
    """
    if dataset_source == 'server_other':
        root = "/media/sdg/sravani/.gz/"
    imagenet_data = TinyImageNet(root_folder=root+path_to_data,
                                 transform=transforms.Compose([
                                            # transforms.ToTensor(),
                                            transforms.ToPILImage(),
                                            transforms.Resize((resize,resize)),
                                            transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    split_1 = .9
    split_2 = .1
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(imagenet_data)
    indices = list(range(dataset_size))
    split_1 = int(np.floor(split_1 * dataset_size))
    split_2 = int(np.floor(split_2 * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, eval_indices, test_indices = indices[:split_1], indices[split_1:(split_1+split_2)], indices[(split_1):]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    eval_sampler = SubsetRandomSampler(eval_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(imagenet_data
        ,batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, sampler=train_sampler)

    eval_loader = DataLoader(imagenet_data
        ,batch_size=test_batch_size, shuffle=False, num_workers=0, drop_last=True, sampler=eval_sampler)

    test_loader = DataLoader(imagenet_data
        ,batch_size=test_batch_size, shuffle=False, num_workers=0, drop_last=True, sampler=test_sampler)

    return train_loader, test_loader, imagenet_data


class TinyImageNet(Dataset):
    """Tiny ImageNet dataset (https://tiny-imagenet.herokuapp.com/), containing
    64 x 64 ImageNet images from 200 classes.

    Parameters
    ----------
    root_folder : string
        Root folder of Tiny ImageNet dataset.

    transform : torchvision.transforms
    """
    def __init__(self, root_folder='tiny-imagenet-200/', transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.imgs_and_classes = []  # Paths to images and their classes

        train_folder = root_folder + 'train/'
        class_folders = glob.glob(train_folder + '*')  # One folder for each image class

        for i, class_folder in enumerate(class_folders):
            image_paths = glob.glob(class_folder + '/images/*.JPEG')
            for image_path in image_paths:
                self.imgs_and_classes.append((image_path, i))

        self.transform = transform

    def __len__(self):
        return len(self.imgs_and_classes)

    def __getitem__(self, idx):
        img_path, label = self.imgs_and_classes[idx]
        img = imageio.imread(img_path)

        if self.transform:
            img = self.transform(img)

        # Some images are grayscale, convert to RGB
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        return img, label

# classes = classes_100
def form_classes(dataset_size='normal'):
    if dataset_size=='normal':
        classes = classes_100
    elif dataset_size=='large':
        classes = classes_581
    else:
        classes = classes_20


    anp_classes=dict((v,k) for k,v in classes.items())

    cls=anp_classes.values()
    a=set([x.split('_')[0] for x in cls])
    n=set([x.split('_')[1] for x in cls])

    adj_classes=dict([(v,i) for i,v in enumerate(a)])
    noun_classes=dict([(v,i) for i,v in enumerate(n)])
    return classes,anp_classes,adj_classes,noun_classes

def get_mtvso_loaders(batch_size=20, test_batch_size=20,
        dataset_size='normal', resize=400, network='sqnxt', dataset_type='anp',
        dataset_source='server_main'):
    # from torch.utils.data.sampler import SubsetRandomSampler
    # batch_size=training_config['batch_size']
    # test_batch_size=training_config['test_batch_size']
    # size=training_config['img_size']
    # crop_size=training_config['crop_size']
    # transform_train = transforms.Compose([
    #         transforms.Grayscale(num_output_channels=1),
    #         transforms.CenterCrop((crop_size,crop_size)),
    #         transforms.Resize(size),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (0.5,)),
    #     ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform = transforms.Compose([
        transforms.Resize((resize,resize)),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    classes,anp_classes,adj_classes,noun_classes=form_classes(dataset_size)

    def target_transform(x):
        anp=anp_classes[x]
        adj,noun=anp.split('_')
        if network in ('sqnxt','resnet'): 
            return classes[anp],adj_classes[adj],noun_classes[noun]
        else:
            if dataset_type == 'anp': return classes[anp]
            elif dataset_type == 'adj': return adj_classes[adj]
            elif dataset_type == 'noun': return noun_classes[noun]

    # transform_test = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     # transforms.CenterCrop((64,64)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,)),
    # ])

    # gz_root = '/home/cs19mtech11019/cs19mtech11024/imageFolder'
    # gz_root = '/content/drive/My Drive/imageFolder'
    if dataset_source == 'local': root = '/mnt/f/IITH/research/cs/'
    elif dataset_source == 'server_main': root = '/home/cs19mtech11019/cs19mtech11024/'
    elif dataset_source == 'server_nilesh': root = '/home/nilesh/raghav/'


    # gz_root = '/mnt/f/IITH/research/physics/galaxy_zoo/GalaxyClassification/imageFolder_small'
    gz_root = 'dataset'
    if dataset_size=='normal':
        gz_root = root + 'mtvso_task/dataset'
    elif dataset_size=='small':
        gz_root = root + 'mtvso_task/vso_dataset_20c_80i'
    elif dataset_size=='smallFull':
        gz_root = root + 'mtvso_task/vso_dataset_20c_240i'
    else:
        gz_root = '/raid/cs19mtech11019/bi_concepts1553'

    gz_root = '/raid/cs19mtech11019/' + 'imageFolder'
    gz_root = '/media/sdg/sravani/.gz/' + 'imageFolder'

    gz_dataset = datasets.ImageFolder(root=gz_root
            # ,train=True, download=True
            , transform=transform,
        target_transform=target_transform
        )

    # total_images = len(gz_dataset)

    # train_dataset, test_dataset = random_split(gz_dataset,[
    #     int(0.9*total_images),
    #     int(0.1*total_images)
    # ])

    split = .9
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(gz_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)



    train_loader = DataLoader(gz_dataset
        ,batch_size=batch_size,
        shuffle=False, num_workers=0, drop_last=True
        ,sampler=train_sampler
    )

    # train_eval_loader = DataLoader(validation_dataset
    #     ,batch_size=test_batch_size, shuffle=True, num_workers=0, drop_last=True
    # )

    test_loader = DataLoader(gz_dataset
        ,batch_size=test_batch_size,
        shuffle=False, num_workers=0, drop_last=True
        ,sampler=test_sampler
    )

    return train_loader, test_loader, gz_dataset

def get_galaxyzoo_loaders(batch_size=20, test_batch_size=20,
        dataset_size='normal', resize=400, crop_size=424, network='sqnxt', dataset_type='anp',
        dataset_source='server_main'):
    # batch_size=20, test_batch_size=20,
    #     dataset_size='normal', resize=400, network='sqnxt', dataset_type='anp',
    #     dataset_source='server_main'
    from torch.utils.data.sampler import SubsetRandomSampler
    # batch_size=training_config['batch_size']
    # test_batch_size=training_config['test_batch_size']
    # resize=training_config['img_size']
    # crop_size=training_config['crop_size']
    transform_train = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.CenterCrop((crop_size,crop_size)),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    gz_root = '/raid/cs19mtech11019/' + 'imageFolder'
    gz_root = '/media/sdg/sravani/.gz/' + 'imageFolder'
    
    gz_dataset = datasets.ImageFolder(root=gz_root
        ,transform=transform_train)

    split_1 = .9
    split_2 = .1
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(gz_dataset)
    indices = list(range(dataset_size))
    split_1 = int(np.floor(split_1 * dataset_size))
    split_2 = int(np.floor(split_2 * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, eval_indices, test_indices = indices[:split_1], indices[split_1:(split_1+split_2)], indices[(split_1):]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    eval_sampler = SubsetRandomSampler(eval_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(gz_dataset
        ,batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, sampler=train_sampler)

    eval_loader = DataLoader(gz_dataset
        ,batch_size=test_batch_size, shuffle=False, num_workers=0, drop_last=True, sampler=eval_sampler)

    test_loader = DataLoader(gz_dataset
        ,batch_size=test_batch_size, shuffle=False, num_workers=0, drop_last=True, sampler=test_sampler)
    # import pdb; pdb.set_trace()
    train_fileNames_classes=[gz_dataset.samples[i] for i in train_indices]
    test_fileNames_classes=[gz_dataset.samples[i] for i in test_indices]
    from datetime import datetime
    import json
    now_str=str(datetime.now().date())+'_'+str(datetime.now().time())
    with open('trainImages_'+now_str+'.json','a+') as fp:
        json.dump(train_fileNames_classes, fp)
    with open('testImages_'+now_str+'.json','a+') as fp:
        json.dump(test_fileNames_classes, fp)


if __name__ == '__main__':
    sequence_max_length = 1014
    batch_size = 32
    num_epochs = 32
    database_path = '.data/ag_news/'
    data_helper = DataHelper(sequence_max_length=sequence_max_length)
    train_data, train_label, test_data, test_label = data_helper.load_dataset(database_path)
    train_batches = data_helper.batch_iter(np.column_stack((train_data, train_label)), batch_size, num_epochs)
    for batch in train_batches:
        train_data_b,label = batch
        break
