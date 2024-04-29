import torch
from torch.utils.data import Dataset
from torchvision import transforms
# torch.manual_seed() is for forcing the random function to give
# the same number every time we try to recompile it.
torch.manual_seed(1)


# Define class for dataset
class toy_set(Dataset):
    # Constructor with default values
    def __init__(self, length=100, transform=None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform

    # Getter, redefine/override index function
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get Length, redefine len()
    def __len__(self):
        return self.len


# Create transform class add 1 to x and multiply y by 2
class add_mult:
    # Constructor
    def __init__(self, addx=1, muly=2):
        self.addx = addx
        self.muly = muly

    # Executor
    def __call__(self, sample):
        __x = sample[0]
        __y = sample[1]
        __x = __x + self.addx
        __y = __y * self.muly
        sample = __x, __y
        return sample


class mult(object):
    # Constructor
    def __init__(self, mult=100):
        self.mult = mult

    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample


# ------------------------#
# Create a dataset
# ------------------------#
our_dataset = toy_set()
print("Our toy_set object: ", our_dataset)
print("Value on index 0 of our toy_set object: ", our_dataset[0])
print("Our toy_set length: ", len(our_dataset), our_dataset.len)

for i in range(2):
    x, y = our_dataset[i]
    print("index: ", i, '; x:', x, '; y:', y)


# ------------------------#
# Transform a dataset
# ------------------------#
# Create an add_mult transform object, and an toy_set object
a_m = add_mult()
data_set = toy_set()
for i in range(2):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = a_m(data_set[i])
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

# Create a new data_set object with add_mult object as transform
cust_data_set = toy_set(transform=a_m)
for i in range(2):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

# ------------------------#
# Compose a dataset
# ------------------------#
# Compose multiple transforms on the dataset object
# Sample first goes through add_mult() and then mult()
data_transform = transforms.Compose([add_mult(), mult()])
x, y = data_set[0]
x_, y_ = data_transform(data_set[0])
print('Original x: ', x, 'Original y: ', y)
print('Transformed x_:', x_, 'Transformed y_:', y_)

# Create a new toy_set object with compose object as transform
compose_data_set = toy_set(transform=data_transform)
x, y = data_set[0]
print('Index: ', 0, 'Original x: ', x, 'Original y: ', y)
x_, y_ = cust_data_set[0]
print('Index: ', 0, 'Transformed x_:', x_, 'Transformed y_:', y_)
x_co, y_co = compose_data_set[0]
print('Index: ', 0, 'Compose Transformed x_co: ', x_co, 'Compose Transformed y_co: ', y_co)

