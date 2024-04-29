import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F#
from torchvision import transforms, models, datasets
from torchsummary import summary
import numpy as np, pandas as pd, os, glob, cv2
from torch.utils.data import TensorDataset, DataLoader, Dataset
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import cluster


device_type_str = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Cuda is available: {torch.cuda.is_available()}")

root_dir_type_str = 'data/P1_Facial_Keypoints/data/training'
all_img_paths_type_list = glob.glob(os.path.join(root_dir_type_str, '*.jpg'))

# obtain a dataframe with 3462 rows and 137 columns
data_type_DataFrame = pd.read_csv('data/P1_Facial_Keypoints/data/training_frames_keypoints.csv')


class FacesData(Dataset):
    def __init__(self, df_type_DataFrame):
        super(FacesData).__init__()
        self.df_type_DataFrame = df_type_DataFrame
        self.normalize_type_Normalize = transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )


    def __len__(self):
        return len(self.df_type_DataFrame)

    def __getitem__(self, ix):
        img_path = 'data/P1_Facial_Keypoints/data/training/' + self.df_type_DataFrame.iloc[ix, 0]

        img = cv2.imread(img_path)/255.

        kp = deepcopy(self.df_type_DataFrame.iloc[ix, 1:].tolist())
        kp_x = (np.array(kp[0::2]) / img.shape[1]).tolist()
        kp_y = (np.array(kp[1::2]) / img.shape[0]).tolist()

        kp2 = kp_x + kp_y
        kp2 = torch.tensor(kp2)
        img = self.preprocess_input(img)
        return img, kp2

    def preprocess_input(self, img):
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img).permute(2, 0, 1)
        img = self.normalize_type_Normalize(img).float()
        return img.to(device_type_str)

    def load_img(self, ix):
        img_path = 'data/P1_Facial_Keypoints/data/training/' + self.df_type_DataFrame.iloc[ix, 0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        img = cv2.resize(img, (224, 224))
        return img


from sklearn.model_selection import train_test_split

# load in the training and testing dataframes, each dataframe has 137 columns and train has a lot more rows than test
train_type_DataFrame, test_type_DataFrame = train_test_split(data_type_DataFrame, test_size=0.2, random_state=101)

train_dataset_type_FacesData = FacesData(train_type_DataFrame.reset_index(drop=True))
test_dataset_type_FacesData = FacesData(train_type_DataFrame.reset_index(drop=True))

train_loader_type_DataLoader = DataLoader(train_dataset_type_FacesData, batch_size=32)
test_loader_type_DataLoader = DataLoader(test_dataset_type_FacesData, batch_size=32)


def get_model():
    model_type_VGG = models.vgg16(pretrained=True)

    for parameter_type_Parameter in model_type_VGG.parameters():
        parameter_type_Parameter.requires_grad = False

    model_type_VGG.avgpool = nn.Sequential(
        nn.Conv2d(512, 512, 3),
        nn.MaxPool2d(2),
        nn.Flatten()
    )

    model_type_VGG.classifier = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 136),
        nn.Sigmoid()
    )

    criterion_type_L1Loss = nn.L1Loss()
    optimizer_type_Adam = torch.optim.Adam(model_type_VGG.parameters(), lr=1e-4)
    return model_type_VGG.to(device_type_str), criterion_type_L1Loss, optimizer_type_Adam

model_type_VGG, criterion_type_L1Loss, optimizer_type_Adam = get_model()

def train_batch(img_type_Tensor, kps_type_Tensor, model_type_VGG, optimizer_type_Adam, criterion_type_L1Loss):
    model_type_VGG.train()
    optimizer_type_Adam.zero_grad()
    _kps_type_Tensor = model_type_VGG(img_type_Tensor.to(device_type_str))
    loss_type_Tensor = criterion_type_L1Loss(_kps_type_Tensor, kps_type_Tensor.to(device_type_str))
    loss_type_Tensor.backward()
    optimizer_type_Adam.step()
    return loss_type_Tensor

def validate_batch(img_type_Tensor, kps_type_Tensor, model_type_VGG, criterion_type_L1Loss):
    model_type_VGG.eval()
    with torch.no_grad():
        _kps_type_Tensor = model_type_VGG(img_type_Tensor.to(device_type_str))
    loss_type_Tensor = criterion_type_L1Loss(_kps_type_Tensor, kps_type_Tensor.to(device_type_str))
    return kps_type_Tensor, loss_type_Tensor


train_loss_type_list, test_loss_type_list  = [], []
n_epochs_type_int = 50

for epoch_type_int in range(n_epochs_type_int):
    print(f" epoch {epoch_type_int+1} : 50")
    epoch_train_loss_type_float, epoch_test_loss_type_float = 0, 0
    # fix the issue with train_loader_type_DataLoader, fixed in the __len__ function
    for ix_type_int, (img_type_Tensor, kps_type_Tensor) in enumerate(train_loader_type_DataLoader):
        loss_type_Tensor = train_batch(img_type_Tensor, kps_type_Tensor, model_type_VGG, optimizer_type_Adam, criterion_type_L1Loss)
        epoch_train_loss_type_float += loss_type_Tensor.item()
    epoch_train_loss_type_float /= (ix_type_int+1)

    for ix_type_int, (img_type_Tensor, kps_type_Tensor) in enumerate(test_loader_type_DataLoader):
        ps_type_Tensor, loss_type_Tensor = validate_batch(img_type_Tensor, kps_type_Tensor, model_type_VGG, criterion_type_L1Loss)
        epoch_test_loss_type_float += loss_type_Tensor.item()
    epoch_test_loss_type_float /= (ix_type_int+1)

    train_loss_type_list.append(epoch_train_loss_type_float)
    test_loss_type_list.append(epoch_test_loss_type_float)

epochs_type_ndarray = np.arange(50) + 1
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.plot(epochs_type_ndarray, train_loss_type_list, 'bo', label='Training loss')
plt.plot(epochs_type_ndarray, test_loss_type_list, 'r', label='Test loss')
plt.title('Training and Test loss over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.show()


ix_type_int = 0
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.title('Original image')
im_type_ndarray = test_dataset_type_FacesData.load_img(ix_type_int)
plt.imshow(im_type_ndarray)
plt.grid(False)
plt.subplot(222)
plt.title('Image with facial keypoints')
x_type_Tensor, _ = test_dataset_type_FacesData[ix_type_int]
plt.imshow(im_type_ndarray)
kp_type_Tensor = model_type_VGG(x_type_Tensor[None]).flatten().detach().cpu()
plt.scatter(kp_type_Tensor[:68]*224, kp_type_Tensor[:68]*224, c='r')
plt.grid(False)
plt.show()

